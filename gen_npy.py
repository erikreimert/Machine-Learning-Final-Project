from parse import get_ing_vector, parse_input, save

# Generates the npy files needed to run the algorithm

if __name__ == "__main__":
    ing = get_ing_vector(["train.json", "test.json"])
    print(ing)
    print(ing.shape)
    print("Generated ing vector...")
    x, y, c, ids = parse_input("train.json", ing)
    save(x, y, ing, c, ids, "train")
    print("Saved training data...")
    x, y, c, ids = parse_input("test.json", ing)
    save(x, y, ing, c, ids, "test")
    print("Saved testing data...")