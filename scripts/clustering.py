# Created by Baole Fang at 4/18/24

import argparse
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_rand_score as ari


def load(path, idx: int = 0, n=10000):
    data = np.load(path, allow_pickle=True)
    X = []
    Y = []
    for x, y in zip(data['X'], data['Y']):
        ys = y.split(';')
        if idx < len(ys):
            X.append(x)
            Y.append(ys[idx])
    # sampler = RandomUnderSampler(random_state=0)
    # counter = Counter(Y)
    # keys = {k for k, v in counter.most_common(n)}
    # X_, Y_=[], []
    # for x, y in zip(X, Y):
    #     if y in keys:
    #         X_.append(x)
    #         Y_.append(y)
    # X, Y = np.array(X_), np.array(Y_)
    # X, Y = sampler.fit_resample(X, Y)
    # return X, Y
    np.random.seed(args.seed)
    idx = np.random.choice(len(Y), n, replace=False)
    return np.array(X)[idx], np.array(Y)[idx]


def jaccard_distance(x, y):
    x = set(x)
    y = set(y)
    return 1 - len(x.intersection(y)) / len(x.union(y))


def main(args):
    print("Loading data...")
    X, y = load(args.input, args.column, args.n)
    n_classes = len(set(y))
    print(f'Number of samples: {len(y)}')
    print(f'Number of classes: {n_classes}')
    distance = pairwise_distances(X, metric=args.distance, n_jobs=-1)
    print(f'Clustering with {args.distance} distance...')
    model = AgglomerativeClustering(n_clusters=n_classes, metric='precomputed', linkage='average')
    model.fit(distance)
    score = ari(y, model.labels_)
    print(f'Adjusted Rand Index: {score}')
    with open(args.output, 'a') as f:
        f.write(
            f'input: {args.input}, distance: {args.distance}, level: {args.column}, n_samples: {len(y)}, n_classes: {n_classes}, ari: {score}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train the classification problem')
    parser.add_argument('-i', '--input', help='path of the data', type=str, required=True)
    parser.add_argument('-d', '--distance', help='distance metric', type=str, default='euclidean')
    parser.add_argument('-c', '--column', help='column/index of label', type=int, default=0)
    parser.add_argument('-t', '--test', help='test set size', type=float, default=0.2)
    parser.add_argument('-o', '--output', help='path of output model', type=str, default='outputs/clustering.txt')
    parser.add_argument('-n', '--n', help='number of samples', type=int, default=10000)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
