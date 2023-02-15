import numpy as np
import json
from csv import writer, reader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import forest


count_tree = 100


X = []
Y = []
with open('sdss_redshift.csv', 'r') as file:
    rd = reader(file, delimiter=',')
    for row in rd:
        X.append(np.asarray(row[0:5]))
        Y.append(row[5])
X = np.asarray(X[1:]).astype(np.float)
Y = np.asarray(Y[1:]).astype(np.float)
X, x_test, Y, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


x_pred = []
with open('sdss.csv', 'r') as file:
    rd = reader(file, delimiter=',')
    for row in rd:
        x_pred.append(np.asarray(row[0:5]))
x_pred = np.asarray(x_pred[1:]).astype(np.float)


Y_new_1, Y_new_2, Y_new_3 = forest.Forest(count_tree, 10).prediction(X, x_test, x_pred, Y)


plt.plot(np.concatenate((Y, y_test)) - np.concatenate((Y_new_1, Y_new_2)))
plt.savefig('redshift.png')


with open('redhsift.json', 'w') as file:
    json.dump({"train": np.sqrt(np.mean(((Y - Y_new_1) ** 2))),
               "test": np.sqrt(np.mean((y_test - Y_new_2) ** 2))}, file)


new = []
for i in range(x_pred.shape[0]):
    new.append(np.asarray([*x_pred[i], Y_new_3[i]]))
new = np.asarray(new)


with open('sdss_predict.csv', "w", newline='') as file:
    wr = writer(file, delimiter=',')
    for line in new:
        wr.writerow(line)

