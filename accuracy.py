import numpy as np
import pandas as pd

def intializeDataset():   
    dataset = pd.read_csv("data/wdbc.data")
    y = np.array(dataset.iloc[:,1])
    for i in range(y.shape[0]):
        if (y[i] == 'M'):
            y[i] = 1
        else:
            y[i] = 0
    x = np.array(dataset.iloc[:,2:])
    return x, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predictSingle(x, w, b,sigmas, means):
    newx = np.zeros(x.shape)
    for i in range(x.shape[0]):
        newx[i] = (x[i] - means[i]) / sigmas[i]
    y = sigmoid(np.dot(w, newx) + b)
    if(y > 0.5):
        return 1
    else:
        return 0

x, y = intializeDataset()
saved = np.load("saved.npz")
w = saved['w']
b = saved['b']
sigmas = saved['sigmas']
means = saved['means']
m = x.shape[0]
n = x.shape[1]
correct = 0

for i in range(m):
    if(y[i] == predictSingle(x[i], w, b, sigmas, means)):
        correct += 1

accuracy = correct / m
accuracy = accuracy * 100
print(f"accuracy = {accuracy:.2f}%")