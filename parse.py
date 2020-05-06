import json
import numpy as np
from scipy import spatial
import torch
import re
from nltk.stem import PorterStemmer
from autocorrect import Speller

"""
Parses input json files into matrices and handles
creating files suitable for submission onto Kaggle
"""


"""
The following is the 1 hot encoded version
"""


# gets the ingredient vector from the list of files
def get_ing_vector(files):
    ing = set()
    for name in files:
        with open(name, "r") as f:
            data = json.load(f)
            for recipe in data:
                ing.update(recipe["ingredients"])
    return np.asanyarray(list(ing))


# parses input JSON file and ingredient vector and turns it into the outputs:
# x (rows are examples, columns are the 1 hot encoded ingredients)
# y (rows are examples, columns are 1 hot encoded cusine types)
# c, cuisine names in order
# ids, id names in order
# outputs in order x, y, c, ids
def parse_input(path, ing):
    with open(path, "r") as f:
        data = json.load(f)
    cuisines = set()
    training = "cuisine" in data[0]
    ids = []
    for recipe in data:
        recipe["ingredients"] = \
            list(map(lambda a: a.lower(),
                     recipe["ingredients"]))
        if training:
            cuisines.add(recipe["cuisine"])
        ids.append(recipe["id"])
    c = np.asanyarray(list(cuisines))
    xs = []
    ys = []
    for recipe in data:
        v = np.vstack(list(map(lambda a: ing == a, recipe["ingredients"])))
        v = np.bitwise_or.reduce(v, axis=0).astype(np.float)
        xs.append(v)
        if training:
            ys.append((c == recipe["cuisine"]).astype(np.float))
    x = np.vstack(xs)
    if training:
        y = np.vstack(ys)
    else:
        y = None
    return x, y, c, ids


# saves a given set of data to a base name in this directory
def save(x, y, ing, c, ids, name):
    for n, data in [("x", x), ("y", y),
                    ("ingredients", ing), ("cuisine", c),
                    ("ids", ids)]:
        np.save(name+"-"+n+".npy", data)


# reloads a given data set back into data
def load(name):
    return (np.load(name+"-x.npy", allow_pickle=True),
            np.load(name+"-y.npy", allow_pickle=True),
            np.load(name+"-ingredients.npy", allow_pickle=True),
            np.load(name+"-cuisine.npy", allow_pickle=True),
            np.load(name+"-ids.npy", allow_pickle=True))


"""
Utility methods
"""


# creates the output csv file for Kaggle submissions
def create_output(ids, c, yhat, name):
    out_data = [["id", "cuisine\n"]]
    for i in range(yhat.shape[0]):
        cur_cuisine = str(c[np.argmax(yhat[i, :])])
        out_data.append([str(ids[i]), cur_cuisine + "\n"])
    out_data = list(map(lambda a: ",".join(a), out_data))
    with open(name, "w") as f:
        f.writelines(out_data)


# output returned by rnn
def create_output(ids, yhat, name):
    out_data = [["id", "cuisine\n"]]
    for i in range(yhat.shape[0]):
        out_data.append([str(ids[i]), str(yhat[i][0])+"\n"])
    out_data = list(map(lambda a: ",".join(a), out_data))
    with open(name, "w") as f:
        f.writelines(out_data)


"""
These following functions are for the neural network approach for the Kaggle problem
"""

# download this to use glove code:
# nlp.stanford.edu/data/glove.6B.zip


def get_cuisine(cuisine, embeddings):
    if cuisine == "southern_us":
        return embeddings["american"]
    elif cuisine == "cajun_creole":
        return embeddings["cajun"]
    else:
        # general case, just use the embeddings for the raw word
        return embeddings[cuisine]


# loads all possible cuisine types and their
# embedding mappings using the training json file
# and the embedding mappings
def get_cuisine_mapping(filename, embeddings):
    with open(filename, "r") as f:
        data = json.load(f)
    cuisines = {}
    for recipe in data:
        cuisine = recipe["cuisine"]
        if cuisine not in cuisines:
            cuisines[cuisine] = get_cuisine(cuisine, embeddings)
    return cuisines


# loads the training/testing data in based on the filename
# this is for the neural network approach
# returns a list x containing the training examples encoded
# using glove, and a tensor y containing the ground truth if applicable
# returns:
# x, list of tensors that contain the training example for each recipe
#    rows are ingredients columns are the glove embedding data
# y, ground truth tensor if present, rows are each example
#    columns are the embedding values
# ids, ids in order specified in x
def parse_data_tensor(filename, embeddings, cmap):
    with open(filename, "r") as f:
        data = json.load(f)
    ids = []
    y = None
    x = []
    unknown_words = set()
    special_chars = str.maketrans({
        "®": "",
        ",": "",
        ".": "",
        ")": "",
        "(": "",
        "™": "",
        "!": "",
    })
    porter = PorterStemmer()
    check = Speller(lang='en')
    for recipe in data:
        ids.append(recipe["id"])
        if "cuisine" in recipe:
            if y is None:
                y = []
            y.append(cmap[recipe["cuisine"]])
        ings = recipe["ingredients"]
        # sort the ingredient list
        ings = list(sorted(ings))
        # turn the ingredient list into a sentence the RNN can understand
        # separate all the spaces in all the ingredients so they can be looked up separately
        recipe_sentence = []
        for ing in ings:
            # condition the ingredient list by taking out special characters
            # converting everything to lower case and splitting on spaces and hyphens
            for s in re.split(r'[\s-]\s*', ing.translate(special_chars).lower()):
                # spell check the word
                recipe_sentence.append(check(s))
        # look up all these words in the embeddings dictionary and build the tensor
        recipe_vecs = []
        for word in recipe_sentence:
            if word not in embeddings:
                unknown_words.add(word)
                continue
            recipe_vecs.append(np.asanyarray(embeddings[word]))
        if len(recipe_vecs) == 0:
            print("No valid words for:", ings)
        x.append(torch.from_numpy(np.vstack(recipe_vecs)))
    print("Unknown Words:", unknown_words)
    if y is None:
        y = torch.tensor([])
    else:
        y = torch.from_numpy(np.vstack(y))
    return x, y, np.asanyarray(ids)


def save_data_tensor(name, x, y, ids):
    torch.save(x, name + "-x.pt")
    torch.save(y, name + "-y.pt")
    np.save(name + "-ids.npy", ids, allow_pickle=True)


def load_data_tensor(name):
    return (
        torch.load(name + "-x.pt"),
        torch.load(name + "-y.pt"),
        np.load(name + "-ids.npy")
    )


def load_glove_embeddings(filename):
    embeddings_dict = {}
    with open(filename, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    return embeddings_dict


def find_closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))


def load_labels(file, cvec):
    with open(file, "r") as f:
        data = json.load(f)
    y = []
    for recipe in data:
        y.append((recipe["cuisine"] == cvec).astype(np.int))
    return np.vstack(y)


# use this to test stuff since this won't be run by itself
if __name__ == "__main__":
    # loads them back in
    """
    x, y, ing, c, ids = load("train")
    print(c)
    print(y.shape)
    # number of recipes for each cuisine
    print(y.sum(axis=0))
    # create output file to test
    create_output(ids, c, y, "out_test.csv")
    """
    embeddings_dict = load_glove_embeddings("glove.6B.50d.txt")
    cmap = get_cuisine_mapping("train.json", embeddings_dict)
    x, y, ids = parse_data_tensor("train.json", embeddings_dict, cmap)
    print("X:", len(x), "Y:", y.shape, "ids:", ids.shape)
    save_data_tensor("train", x, y, ids)
    x, y, ids = parse_data_tensor("test.json", embeddings_dict, cmap)
    save_data_tensor("test", x, y, ids)