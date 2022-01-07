# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import argparse

import numpy as np
import pandas as pd


def get_word_acc(data, recall, predcol):
    threshold = np.quantile(data['prob'], 1 - recall)
    sub = data[data['prob'] >= threshold]

    realized_recall = len(sub) / len(data)

    is_correct = 0
    counter = 0

    for pred, label in zip(sub[predcol], sub['label']):
        pred, label = pred.split(' '), label.split(' ')

        nb_names_label = len(label)
        nb_names_pred = len(pred)

        if nb_names_label == nb_names_pred:
            for i in range(nb_names_label):
                is_correct += pred[i] == label[i]
                counter += 1
        elif nb_names_label > nb_names_pred: # too few predictions. 1 error pr. too few
            counter += nb_names_label - nb_names_pred # mistake cases since pred did not include all words
            for i in range(nb_names_pred - 1): # not all the way through since last is compared to last always
                is_correct += pred[i] == label[i]
                counter += 1
            # finally handle last name
            is_correct += pred[-1] == label[-1]
            counter += 1
        elif nb_names_pred > nb_names_label: # now too many predictions! one error pr. too many
            counter += nb_names_pred - nb_names_label # mistake cases since pred includes too many
            for i in range(nb_names_label - 1): # not all the way through since last is compared to last always
                is_correct += pred[i] == label[i]
                counter += 1
            # finally handle last name
            is_correct += pred[-1] == label[-1]
            counter += 1

    acc = is_correct / counter

    print(f'Recall: {round(100 * realized_recall, 2)}%. Word accuracy: {round(100 * acc, 2)}%.')


def parse_args():
    parser = argparse.ArgumentParser(description='Get word accuracies')

    parser.add_argument('--fn-preds', type=str)
    parser.add_argument('--recall', type=float, default=None)

    args = parser.parse_args()

    if args.recall is None:
        args.recall = 0.9
    else:
        if 1 < args.recall < 100:
            args.recall /= 100
        assert 0 < args.recall < 1, args.recall

    return args


def main():
    args = parse_args()

    preds = pd.read_csv(args.fn_preds)

    get_word_acc(preds, 1, 'pred')
    get_word_acc(preds, args.recall, 'pred')

    if 'pred_m' in preds.columns:
        print('\nUsing matched predictions:')
        get_word_acc(preds, 1, 'pred_m')
        get_word_acc(preds, args.recall, 'pred_m')


if __name__ == '__main__':
    main()
