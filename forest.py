import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor
import tree


class Forest:
    def __init__(self, count=100, max_depth=10):
        self.count = count
        self.max_depth = max_depth

    def data(self, X, Y):
        arr_X = []
        arr_Y = []
        for i in range(self.count):
            list_x = []
            list_y = []
            for k in range(X.shape[0]):
                rand = random.randint(0, X.shape[0] - 1)
                list_x.append(X[rand])
                list_y.append(Y[rand])
            list_x = np.asarray(list_x)
            list_y = np.asarray(list_y)
            arr_X.append(list_x)
            arr_Y.append(list_y)
        arr_X = np.asarray(arr_X)
        arr_Y = np.asarray(arr_Y)
        return arr_X, arr_Y

    def prediction(self, X, x_test, x_pred, Y):
        arr_X, arr_Y = self.data(X, Y)
        arr_y = [[], [], []]
        for i in range(self.count):
            print(i)
            rtf = tree.TreeReg(max_depth=self.max_depth)  # Работает медленно
            rtf = DecisionTreeRegressor(max_depth=self.max_depth)
            rtf.fit(arr_X[i], arr_Y[i])
            arr_y[0].append(np.asarray(rtf.predict(X)))
            arr_y[1].append(np.asarray(rtf.predict(x_test)))
            arr_y[2].append(np.asarray(rtf.predict(x_pred)))
        prediction_1 = np.asarray(arr_y[0]).mean(axis=0)
        prediction_2 = np.asarray(arr_y[1]).mean(axis=0)
        prediction_3 = np.asarray(arr_y[2]).mean(axis=0)
        return (prediction_1, prediction_2, prediction_3)
