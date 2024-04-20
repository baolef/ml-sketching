# Created by Baole Fang at 4/18/24

import argparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import adjusted_rand_score as ari
from utils import load


def main(args):
    print("Loading data...")
    X, y = load(args.input, args.column, args.n, args.seed)
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
        f.write(f'input: {args.input}, distance: {args.distance}, level: {args.column}, n_samples: {len(y)}, n_classes: {n_classes}, ari: {score}\n')


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
