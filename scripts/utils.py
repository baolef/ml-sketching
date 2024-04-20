# Created by Baole Fang at 4/20/24

import numpy as np

def load(path, idx: int = 0, n=10000, seed=0):
    data = np.load(path, allow_pickle=True)
    X = []
    Y = []
    for x, y in zip(data['X'], data['Y']):
        ys = y.split(';')
        if idx < len(ys):
            X.append(x)
            Y.append(ys[idx])
    np.random.seed(seed)
    idx = np.random.choice(len(Y), n, replace=False)
    return np.array(X)[idx], np.array(Y)[idx]
