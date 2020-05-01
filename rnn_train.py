import parse
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

embeddings = None
cmap = None


class CuisineRNN(nn.Module):
    def __init__(self, embedding_size, hidden_dim, layer_dim):
        super(CuisineRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding_size = embedding_size
        self.rnn = nn.RNN(embedding_size, hidden_dim, layer_dim,
                          batch_first=True, nonlinearity='relu')
        # readout layer
        self.fc = nn.Linear(hidden_dim, embedding_size)
        # self.fc = nn.Linear(hidden_dim, 20)
        # self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out, hn = self.rnn(x)
        # out, input_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # out = self.fc(out[:, -1, :])
        # return self.sm(out), hn
        """
        out = self.fc(hn[0, :, :])
        return self.sm(out), hn
        """
        return self.fc(hn[-1, :, :]), hn


class RecipeDataset(torch.utils.data.IterableDataset):
    def __init__(self, x, y):
        super(RecipeDataset).__init__()
        self.x = x
        self.y = y

    def __iter__(self):
        return iter([(r, self.y[i]) for i, r in enumerate(self.x)])


def condition_input(x):
    # pad all recipes to have the same number of rows
    out_rows = max([t.shape[0] for t in x])
    lens = torch.tensor([t.shape[0] for t in x])
    data = torch.stack([
        nn.functional.pad(item,
                          (0, 0, 0, out_rows-item.shape[0]),
                          mode='constant',
                          value=0)
        for item in x
    ])
    data = nn.utils.rnn.pack_padded_sequence(data, lens,
                                             batch_first=True,
                                             enforce_sorted=False)
    return data


# custom collation function to handle the ragged nature of recipes
def collate_recipes(batch):
    data = condition_input([item[0] for item in batch]).cuda()
    # stack the labels
    target = torch.stack([item[1] for item in batch])
    return [data, target]


# trains a CuisineRNN given the list of examples x and the
# tensor of ground truths y, returns the trained CuisineRNN
# optional parameters are hyperparameters to the training process
def train(x, y, batch_size, epochs, learning_rate, embedding_size, hidden_dim, layer_dim):
    train = RecipeDataset(x, y)
    train_loader = DataLoader(train, batch_size,
                              collate_fn=collate_recipes,
                              shuffle=False)
    model = CuisineRNN(embedding_size, hidden_dim, layer_dim).cuda()
    error = nn.SmoothL1Loss()
    # error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    count = 0
    for epoch in range(epochs):
        for i, (recipes, labels) in enumerate(train_loader):
            recipes = recipes.cuda()
            labels = labels.cuda()
            # zero the gradients
            optimizer.zero_grad()
            # forward propogate the model
            outputs, hn = model(recipes)
            # print(outputs.shape)
            # print(labels.shape)
            # calculate the loss, backpropgate
            """
            print(outputs)
            print(labels)
            """
            loss = error(outputs, labels)
            loss.backward()
            optimizer.step()
            count += 1

            if count % 100 == 0:
                print("Iteration:", count, "Loss:", loss.item(),
                      "% Accuracy:", calc_accuracy(labels, outputs) * 100)

            """
            if count % 100 == 0:
                corr = np.asanyarray(labels.detach().cpu()) == \
                       np.argmax(np.asanyarray(outputs.detach().cpu()), axis=1)
                print(outputs)
                print(np.argmax(np.asanyarray(outputs.detach().cpu()), axis=1))
                corr = np.mean(corr) * 100
                print("Iteration:", count, "Loss:", loss.item(), "%Correct", corr)
            """
    return model


def calc_accuracy(y, yhat):
    y = y.cpu()
    global cmap
    print("Y")
    print(y)
    print("YHAT")
    print(yhat)
    correct = 0
    names = {}
    for i, pred in enumerate(yhat):
        pred = np.asanyarray(pred.cpu().detach())
        closest_dist = None
        closest_name = None
        for key in cmap:
            dist = np.linalg.norm(pred - cmap[key])
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_name = key
        new_yhat = np.asanyarray(parse.get_cuisine(closest_name, embeddings))
        if closest_name not in names:
            names[closest_name] = 0
        names[closest_name] += 1
        correct += np.isclose(np.asanyarray(y[i]), new_yhat).astype(np.int)[0]
    print(names)
    return correct / y.shape[0]


if __name__ == "__main__":
    # Make sure CUDA is available
    if torch.cuda.is_available():
        print("CUDA Available")
    else:
        print("No CUDA Available")
    # load npy and pt files for training, make sure these exist first
    # the embedding file and the one used to generate the training/testing
    # set must match
    embeddings = parse.load_glove_embeddings("glove.6B.50d.txt")
    cmap = parse.get_cuisine_mapping("train.json", embeddings)
    x, y, ids = parse.load_data_tensor("train")
    """
    c = np.asanyarray(list(cmap.keys()))
    y = parse.load_labels("train.json", c)
    y = torch.tensor(np.argmax(y, axis=1))
    """

    model = train(x, y, 128, 25, 1e-2, 50, 30, 3)

    sys.exit(0)

    model = model.cpu()
    x_test, _, ids_test = parse.load_data_tensor("test")
    outputs, _ = model(condition_input(x_test).cpu())
    outputs = np.asanyarray(outputs.detach().cpu())
    outputs = (outputs == np.max(outputs, axis=1)[:, None]).astype(int)

    print("Writing final guesses to csv...")
    parse.create_output(ids_test, c, outputs, "rnn_guesses.csv")
