import numpy as np
import parse as ps

# Main code (do stuff here)
if __name__ == "__main__":
    x_train, y_train, ing, c_train, ids_train = ps.load("train")
    x_test, _, _, _, ids_test = ps.load("test")

    print(ing)
