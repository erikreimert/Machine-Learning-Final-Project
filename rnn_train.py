import parse
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from ax import optimize
import math

embeddings = None
cmap = None


class CuisineRNN(nn.Module):
    def __init__(self, embedding_size, hidden_dim, layer_dim):
        super(CuisineRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.embedding_size = embedding_size
        self.rnn = nn.RNN(embedding_size, hidden_dim, layer_dim,
                          batch_first=True, nonlinearity='relu',
                          bidirectional=True)
        # readout layer
        self.fc = nn.Linear(hidden_dim, embedding_size)

    def forward(self, x):
        out, hn = self.rnn(x)
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


losses = []


# trains a CuisineRNN given the list of examples x and the
# tensor of ground truths y, returns the trained CuisineRNN
# optional parameters are hyperparameters to the training process
def train(x, y, batch_size, epochs, learning_rate,
          embedding_size, hidden_dim, layer_dim, rec_losses=True):
    train = RecipeDataset(x, y)
    train_loader = DataLoader(train, batch_size,
                              collate_fn=collate_recipes,
                              shuffle=False)
    model = CuisineRNN(embedding_size, hidden_dim, layer_dim).cuda()
    error = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    count = 0
    for epoch in range(epochs):
        running_loss = 0
        for i, (recipes, labels) in enumerate(train_loader):
            recipes = recipes.cuda()
            labels = labels.cuda()
            # zero the gradients
            optimizer.zero_grad()
            # forward propogate the model
            outputs, hn = model(recipes)
            # calculate the loss, backpropgate
            loss = error(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            count += 1

            if count % 1000 == 0:
                print("Epoch:", epoch, "Iteration:", count, "Loss:", loss.item(),
                      "%Correct:", calc_accuracy(labels, outputs) * 100)
        if rec_losses:
            losses.append(running_loss)
    return model


EMBEDDING_SIZE = 50


def evaluate(params, x_train, y_train, x_valid, y_valid, rec_losses=False):
    print("Evaluating with parameters:", params)
    batch_size = params["batch_size"]; epochs = params["epochs"]
    learning_rate = params["learning_rate"]; hidden_dim = params["hidden_dim"]
    layer_dim = params["layer_dim"]
    model = train(x_train, y_train, batch_size, epochs,
                  learning_rate, EMBEDDING_SIZE,
                  hidden_dim, layer_dim, rec_losses)
    valid = RecipeDataset(x_valid, y_valid)
    valid_loader = DataLoader(valid, batch_size,
                              collate_fn=collate_recipes,
                              shuffle=False)
    error = nn.SmoothL1Loss()
    running_loss = []
    for i, (recipes, labels) in enumerate(valid_loader):
        recipes = recipes.cuda()
        labels = labels.cuda()
        outputs, _ = model(recipes)
        loss = error(outputs, labels)
        running_loss.append(loss.item())
    print("Validation Loss:", np.mean(running_loss))
    return {"validation_loss": (np.mean(running_loss), np.std(running_loss)), "model": model}


def cuisine_from_out(yhat):
    new_yhat = []
    for i, pred in enumerate(yhat):
        pred = np.asanyarray(pred.cpu().detach())
        closest_dist = None
        closest_name = None
        for key in cmap:
            dist = np.linalg.norm(pred - cmap[key])
            if closest_dist is None or dist < closest_dist:
                closest_dist = dist
                closest_name = key
        new_yhat.append(closest_name)
    return np.vstack(new_yhat)


def calc_accuracy(y, yhat):
    y = y.cpu()
    global cmap
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
    c = np.asanyarray(list(cmap.keys()))

    # create a validation set
    valid_size = int(len(x) * 0.2)
    x_valid = x[0:valid_size]
    y_valid = y[0:valid_size, :]
    x_train = x[valid_size:]
    y_train = y[valid_size:, :]

    def eval_with_data(params):
        ret = evaluate(params, x_train, y_train, x_valid, y_valid)
        return {"validation_loss": ret["validation_loss"]}

    """
    best_parameters, best_values, _, _ = optimize(
        parameters=[
            {"name": "batch_size",
             "type": "range",
             "bounds": [16, 512]},
            {"name": "epochs",
             "type": "range",
             "bounds": [1, 400]},
            {"name": "learning_rate",
             "type": "range",
             "bounds": [1e-6, 0.01],
             "log_scale": True},
            {"name": "hidden_dim",
             "type": "range",
             "bounds": [10, 200]},
            {"name": "layer_dim",
             "type": "range",
             "bounds": [1, 8]}
        ],
        evaluation_function=eval_with_data,
        minimize=True,
        objective_name="validation_loss"
    )

    print("Best parameters:", best_parameters)
    print("Best values:", best_values)
    """

    """
    # Best parameters found on my laptop
    best_parameters = {'batch_size': 372, 'epochs': 183,
                       'learning_rate': 0.0009060921606804382,
                       'hidden_dim': 114, 'layer_dim': 3}
    """

    # Fine tune hyperparameters for spell check
    best_parameters = {'batch_size': 372, 'epochs': 220,
                       'learning_rate': 0.0009060921606804382,
                       'hidden_dim': 150, 'layer_dim': 4}

    """
    # found on Google Cloud for 200 embedding
    best_parameters = {"batch_size": 41, "epochs": 160, # 177,
                       "learning_rate": 1.45e-6,
                       "hidden_dim": 138, "layer_dim": 7}
    """

    model = evaluate(best_parameters, x_train, y_train, x_valid, y_valid, rec_losses=True)["model"]

    print("Saving trained model to file")
    torch.save(model, "trained_model.pt")

    print("Showing loss plot:")
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(np.arange(len(losses)), losses,
            color="red",
            linewidth=1.0)
    plt.show()


    print("Running model on test set")
    model = model.cpu()
    x_test, _, ids_test = parse.load_data_tensor("test")
    outputs, _ = model(condition_input(x_test).cpu())
    outputs = cuisine_from_out(outputs)
    print(outputs)

    print("Writing final guesses to csv...")
    parse.create_output(ids_test, outputs, "rnn_guesses.csv")
