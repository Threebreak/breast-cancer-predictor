import pandas as pd
import numpy as np

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

def scale(x):
    m = x.shape[1]
    means = np.zeros(m)
    sigmas = np.zeros(m)
    for i in range(m):
        std = np.std(x[..., i])
        sigmas[i] = std
        mean = np.mean(x[..., i])
        means[i] = mean
        x[..., i] = (x[..., i] - mean) / std
    return x, sigmas, means

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def computeCost(x, y, w, b):
    cost = 0.0
    m = x.shape[0]
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_i = sigmoid(z_i)
        if(f_i != 1 and f_i != 0):
            cost += -y[i]*np.log(f_i) - (1-y[i])*np.log(1-f_i)
    
    cost = cost / m
    return cost


def computeGradient(x, y, w, b):
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        z_i = np.dot(w, x[i]) + b
        f_i = sigmoid(z_i)
        error = f_i - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i][j]
        dj_db += error
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db


def gradientDescent(x, y, w, b, alpha, num_iters):
    for i in range(num_iters):
        dj_dw , dj_db = computeGradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if(i % 1000 == 0):
            print(f"{i}- cost = {computeCost(x, y, w, b)}")
    return w, b


x, y = intializeDataset()
x, sigmas, means = scale(x)
w = np.zeros(x.shape[1])
b = 0
w, b = gradientDescent(x, y, w, b, 30, 10000)
np.savez("saved.npz", w=w, b=b, sigmas=sigmas, means=means)