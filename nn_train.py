import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader
from ax import optimize
from sklearn.model_selection import train_test_split


class CuisineNN(nn.Module):
    def __init__(self, in_size, l1size, l2size, out_size):
        super(CuisineNN, self).__init__()
        self.lin1 = nn.Linear(in_size, l1size)
        self.lin2 = nn.Linear(l1size, l2size)
        self.lin3 = nn.Linear(l2size, out_size)
        self.smax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        # x = self.lin2(x)
        x = self.smax(x)
        return x


# imports data, returns as pandas DataFrame
# handles data conditioning
def import_data(filename):
    df = pd.read_json(filename)
    # df["ingredients"] = df["ingredients"].apply(lambda a: (",".join(a).lower()))
    df["ingredients"] = df["ingredients"].apply(lambda a: " ".join(a))
    return df


# takes in a list of pandas data frames containing the table of data
# outputs a tfidf and a svd, tfidf is for fitting to the embedding
# svd is for dimensionality reduction
def gen_ing_embedding(all_data):
    all_ings = pd.concat(list(map(lambda a: a.ingredients, all_data)))
    tfidf = TfidfVectorizer(sublinear_tf=True, norm='l2', encoding='latin-1',
                            ngram_range=(1, 2), stop_words='english')
    print("Fitting tfidf vectorizer...")
    trans_all_ings = tfidf.fit_transform(all_ings)
    svd = TruncatedSVD(n_components=8000)
    print("Fitting TruncatedSVD to all ingredients...")
    svd.fit(trans_all_ings)
    return tfidf, svd


# takes in the data frame containing the data as well as the trained tfidf
# containing the ingredients, outputs x, y if available and a list of ids in order,
# along with the cuisine idx if available
def gen_training_data(data, tfidf, svd):
    print("Transforming based on tfidf...")
    # get matrix from tfidf
    x = tfidf.transform(data.ingredients)
    print("Doing dimensionality reduction...")
    # reduce dimensionality
    x = svd.transform(x)
    x = torch.tensor(x)
    if "cuisine" in data:
        y, cidx = pd.factorize(data.cuisine, sort=True)
        y = torch.tensor(np.reshape(y, (y.size, 1)))
    else:
        y = None
        cidx = None
    return x, y, data.id, cidx


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(RecipeDataset).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, pos):
        return [self.x[pos], self.y[pos]]

    def __len__(self):
        return self.x.shape[0]


# trains a model given the x data and y labels and the hyperparameters
# outputs a trained model
def train_nn(x, y, hyps):
    model = CuisineNN(hyps["in_size"], hyps["l1size"],
                      hyps["l2size"], hyps["out_size"]).cuda()
    train = RecipeDataset(x, y)
    train_loader = DataLoader(train, batch_size=hyps["batch_size"],
                              shuffle=False,
                              num_workers=8)
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyps["learning_rate"])
    count = 0
    for epoch in range(hyps["epochs"]):
        running_loss = 0
        for i, (recipes, labels) in enumerate(train_loader):
            recipes = recipes.cuda()
            labels = labels.cuda()
            labels = torch.flatten(labels)
            optimizer.zero_grad()
            outputs = model(recipes.float())
            loss = error(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            count += 1
            if count % 1000 == 0:
                correct = np.argmax(outputs.detach().cpu(), axis=1) == labels.cpu()
                perc_corr = np.mean(np.asanyarray(correct)) * 100
                print("Epoch:", epoch, "Iteration:", count * hyps["batch_size"],
                      "Loss:", loss.item(), "%Corr:", perc_corr)
    return model


def predict(model, x, cidx, ids):
    model = model.cpu()
    labels = model(x.float())
    labels = np.argmax(np.asanyarray(labels.detach()), axis=1)
    guesses = cidx[np.ndarray.flatten(labels)]
    return pd.DataFrame(guesses, index=ids)


if __name__ == "__main__":
    """
    train = import_data("train.json")
    test = import_data("test.json")

    tfidf, svd = gen_ing_embedding([train, test])

    x_train, y_train, _, cidx = gen_training_data(train, tfidf, svd)
    x_test, _, ids_test, _ = gen_training_data(test, tfidf, svd)

    # save the training matrices and testing matrices
    torch.save(x_train, "nn-x-train.pt")
    torch.save(y_train, "nn-y-train.pt")
    torch.save(x_test, "nn-x-test.pt")
    torch.save(ids_test, "nn-ids-test.pt")
    np.save("nn-cidx.npy", cidx)
    """

    x_train = torch.load("nn-x-train.pt")
    y_train = torch.load("nn-y-train.pt")
    x_test = torch.load("nn-x-test.pt")
    ids_test = torch.load("nn-ids-test.pt")
    cidx = np.load("nn-cidx.pt.npy")

    """
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(ids_test.size)
    print(cidx.size)
    print(y_train)
    """

    """
    model = train_nn(x_train, y_train, {
        "in_size": x_train.shape[1],
        "l1size": 2000,
        "l2size": 1000,
        "out_size": cidx.size,
        "batch_size": 1024,
        "epochs": 300,
        "learning_rate": 1e-4
    })
    """

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.2
    )

    def run_training(params):
        print("Running Parameters:", params)
        params["in_size"] = x_train.shape[1]
        params["out_size"] = cidx.size
        model = train_nn(x_train, y_train, params)
        error = nn.CrossEntropyLoss()
        valid = RecipeDataset(x_valid, y_valid)
        valid_loader = DataLoader(valid,
                                  batch_size=512,
                                  shuffle=False)
        avg_loss = 0
        batches = 0
        for (recipes, labels) in valid_loader:
            output = model(recipes.cuda().float())
            loss = error(output, torch.flatten(labels.cuda())).cpu().item()
            avg_loss += loss
            batches += 1
        avg_loss /= batches
        print("Validation Loss:", avg_loss)
        return avg_loss


    """
    best_params, best_values, _ = optimize(
        parameters=[
            {"name": "batch_size",
             "type": "range",
             "bounds": [16, 1024]},
            {"name": "epochs",
             "type": "range",
             "bounds": [64, 512]},
            {"name": "learning_rate",
             "type": "range",
             "bounds": [1e-6, 1e-2],
             "log_scale": True},
            {"name": "l1size",
             "type": "range",
             "bounds": [512, 4096]},
            {"name": "l2size",
             "type": "range",
             "bounds": [512, 2048]}
        ],
        evaluation_function=run_training,
        minimize=True
    )

    print("Best Parameters:", best_params)
    print("Best Values:", best_values)
    """

    """
    # found using partial ax search
    best_params =  {'batch_size': 16, 'epochs': 210,
                    'learning_rate': 0.000904888672109361,
                    'l1size': 2730, 'l2size': 1126}
    """


    # fine tune
    best_params = {'batch_size': 2048, 'epochs': 100,
                   'learning_rate': 0.0001,
                   'l1size': 2700, 'l2size': 1024,
                   "in_size": x_train.shape[1], "out_size": cidx.size}



    """
    best_params = {'batch_size': 512, 'epochs': 1000,
                   'learning_rate': 0.001,
                   'l1size': 100, 'l2size': cidx.size,
                   "in_size": x_train.shape[1], "out_size": cidx.size}
    """

    model = train_nn(x_train, y_train, best_params)

    torch.save(model, "nn_model.pt")

    preds = predict(model, x_test, cidx, ids_test)
    print(preds)

    preds.to_csv("nn_guesses.csv")
