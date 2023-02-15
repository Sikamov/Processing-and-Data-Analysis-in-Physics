import numpy as np
import sklearn.model_selection


class TreeReg:
    def __init__(self, max_depth=1):
        self.first = None
        self.max_depth = max_depth
        self.koeff_split = None
        self.column = None
        self.left = None
        self.right = None
        self.res = None

    def fit(self, X, y):
        self.first = TreeReg(self.max_depth)
        self.first.depth = 0
        self.first.fiting_in_rec(X, y)

    def fiting_in_rec(self, X, y, depth=0):
        def mse(y1):
            return np.mean((y1 - np.mean(y1)) ** 2)

        data_shape = X.shape[0]
        if len(np.unique(y)) == 1:
            self.res = np.mean(y)
            return
        loss = mse(y)
        best_loss = 0
        best_col, best_koeff = None, None
        for input_col in range(X.shape[1]):
            feature_level = np.unique(X[:, input_col])
            koeff = (feature_level[:-1] + feature_level[1:]) / 2
            for i in koeff:
                y_l = y[X[:, input_col] >= i]
                y_r = y[X[:, input_col] < i]
                mse_l = mse(y_l)
                mse_r = mse(y_r)
                n_l = (y_l.shape[0]) / data_shape
                n_r = (y_r.shape[0]) / data_shape
                new_loss = loss - (n_l * mse_l + n_r * mse_r)
                if new_loss > best_loss:
                    best_koeff = i
                    best_loss = new_loss
                    best_col = input_col
        self.column = best_col
        self.koeff_split = best_koeff
        if depth >= self.max_depth:
            self.column = None
            self.res = np.mean(y)
            return
        self.left = TreeReg()
        self.right = TreeReg()
        self.left.max_depth = self.max_depth
        self.right.max_depth = self.max_depth
        left = (X[X[:, self.column] >= self.koeff_split], y[X[:, self.column] >= self.koeff_split])
        right = (X[X[:, self.column] < self.koeff_split], y[X[:, self.column] < self.koeff_split])
        self.left.fiting_in_rec(*left, depth + 1)
        self.right.fiting_in_rec(*right, depth + 1)

    def predict(self, X):
        return np.array([self.first.predicting(i) for i in X])

    def predicting(self, X_new):
        if self.column is None:
            return self.res
        else:
            if X_new[self.column] >= self.koeff_split:
                return self.left.predicting(X_new)
            else:
                return self.right.predicting(X_new)


from csv import reader


if __name__ == "__main__":
    X = []
    y = []
    with open('sdss_redshift.csv', 'r') as file:
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            X.append(np.asarray(row[:5]))
            y.append(row[5])
    X = np.asarray(X[1:]).astype(np.float)
    y = np.asarray(y[1:]).astype(np.float)
    X, test_X, Y, test_Y = sklearn.model_selection.train_test_split(X, y, test_size=0.25)
    rgf = TreeReg(max_depth=10)
    rgf.fit(X, Y)
    print(round(np.sqrt(np.mean((test_Y - rgf.predict(test_X)) ** 2)), 5))

