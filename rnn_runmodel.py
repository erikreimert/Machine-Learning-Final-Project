import rnn_train
from rnn_train import CuisineRNN
import torch
import parse
import numpy as np

# Allows running the rnn model on arbitrary recipes
# put the recipes in input_recipes.json and run this

if __name__ == "__main__":
    embeddings = parse.load_glove_embeddings("glove.6B.50d.txt")
    cmap = parse.get_cuisine_mapping("train.json", embeddings)
    rnn_train.embeddings = embeddings
    rnn_train.cmap = cmap
    parse.embeddings_dict = embeddings
    x, _, _ = parse.parse_data_tensor("input_recipes.json", embeddings, cmap)
    model = torch.load("trained_model_laptopwithspell.pt")
    output, _ = model(rnn_train.condition_input(x).cuda())
    # Use euclidean distance to known categories
    cat_out = rnn_train.cuisine_from_out(output)
    print("Using euclidean distances to known categories:")
    print(cat_out)
    # find the closest word in the corpus
    closest_out = []
    for row in output.detach().cpu():
        closest_out.append(parse.find_closest_embeddings(row)[0:10])
    print("Closest 10 words in the corpus:")
    print(np.vstack(closest_out))
