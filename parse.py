import json
import numpy as np

"""
Parses input json files into matrices and handles
creating files suitable for submission onto Kaggle
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


# creates the output csv file for Kaggle submissions
def create_output(ids, c, yhat, name):
    out_data = [["id", "cuisine\n"]]
    for i in range(yhat.shape[0]):
        cur_cuisine = str(c[np.argwhere(yhat[i, :])[0, 0]])
        out_data.append([str(ids[i]), cur_cuisine + "\n"])
    out_data = list(map(lambda a: ",".join(a), out_data))
    with open(name, "w") as f:
        f.writelines(out_data)


# use this to test stuff since this won't be run by itself
if __name__ == "__main__":
    # loads them back in
    x, y, ing, c, ids = load("train")
    print(c)
    print(y.shape)
    # number of recipes for each cuisine
    print(y.sum(axis=0))
    # create output file to test
    create_output(ids, c, y, "out_test.csv")
