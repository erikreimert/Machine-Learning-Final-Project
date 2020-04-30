import numpy as np
import parse as ps
import sim_train as st

# Main code (do stuff here)
if __name__ == "__main__":
    x_train, y_train, ing, c_train, ids_train = ps.load("train")
    x_test, _, _, _, ids_test = ps.load("test")

    predictions = st.train_correlation(x_train, ing, c_train)
    y_train_list = []
    for el in y_train:
        y_train_list.append(''.join(str(e) for e in st.un_onehot(el, c_train)))
    acc = st.accuracy_corrolation(predictions, y_train_list)
    print("Accuracy: {}%".format(acc * 100))
