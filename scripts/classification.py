# Created by Baole Fang at 10/31/23

import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import numpy as np
import pickle
import xgboost
import pandas as pd
import matplotlib.pyplot as plt
from utils import load


def main(args):
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    X, y = load(args.input, args.column, args.n, args.seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test, random_state=args.seed)
    model = xgboost.XGBClassifier(n_jobs=os.cpu_count())
    # model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).astype(int)
    report = classification_report(y_test, y_pred)
    path = os.path.join(args.output, os.path.basename(args.input).split('.')[0]+'_'+str(args.column))
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(path, 'metrics.txt'), 'w') as f:
        f.write(report)
    df = pd.DataFrame(cm)
    df.to_csv(os.path.join(path, 'confusion.csv'), index=False, header=False)

    fig, ax = plt.subplots(figsize=(len(model.classes_) // 4, len(model.classes_) // 4))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    plt.savefig(os.path.join(path, 'confusion.png'))

    n_classes = len(set(y))
    score = model.score(X_test, y_test)
    with open(os.path.join(args.output, 'classification.txt'), 'a') as f:
        f.write(f'input: {args.input}, level: {args.column}, n_samples: {len(y)}, n_classes: {n_classes}, acc: {score}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train the classification problem')
    parser.add_argument('-i', '--input', help='path of the data', type=str, required=True)
    parser.add_argument('-c', '--column', help='column/index of label', type=int, default=0)
    parser.add_argument('-t', '--test', help='test set size', type=float, default=0.2)
    parser.add_argument('-o', '--output', help='path of output model', type=str, default='outputs')
    parser.add_argument('-n', '--n', help='number of samples', type=int, default=10000)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
