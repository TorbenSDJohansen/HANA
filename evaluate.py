# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import os
import pickle

import argparse
import torch

from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from networks.constructor import load_models
from networks.util.setup_functions import prepare_labels
from networks.util.pytorch_functions import (
    predict_sequence, eval_sequence,
    )
from networks.util.pytorch_datasets import SequenceDataset

from util import get_pipe
from settings import SETTINGS


def dict_to_str(clean: np.ndarray):
    if isinstance(clean[0], str):
        return clean

    assert isinstance(clean[0], dict)

    return np.array([' '.join((x['firstnames'], x['lastnames'])) for x in clean])


def evaluate(
        data_info: dict,
        model_info: dict,
        device: torch.device, # pylint: disable=E1101
        root: str,
        fn_results: str = None,
        fn_preds: str = None,
        ):
    """
    Perform evaluation by calculating accuracy on the validation data. Also
    produces predictions on the validation data.

    Parameters
    ----------
    data_info : dict
        Dictionary contains all information on the data, i.e. where to locate
        it, which batch size to use, and more.
    model_info : dict
        Dictionary contains information on how to build the models.
    device : torch.device
        Controls whether GPU or CPU is used.
    root: str.
        Directory where the models are loaded from.
    fn_results : str, optional
        Name used to save the results (models used, number of observation
        evaluated on, and accuracy). The default is None, in which case the
        results are printed but not saved.
    fn_preds : str, optional
        Name of the file with the predictions.

    Returns
    -------
    None.

    """
    print('Initializing model(s).')
    models = load_models(model_info, root, device)

    print('Initializing generator and dataset.')
    generator = SequenceDataset(
        labels_raw=prepare_labels(
            cells=data_info['cells'],
            root_labels=data_info['root_labels'],
            root_images=data_info['root_images'],
            ),
        transform_to_label=data_info['transform_label'],
        val_size=1.0,
        transform_pipe=get_pipe(),
        )
    generator.status = 'sample_val'

    dataset = DataLoader(generator, data_info['batch_size'], shuffle=False, num_workers=8)

    print('Predicting on validation data!')
    preds, seq_prob, files, labels = predict_sequence(
        models=models,
        dataset=dataset,
        device=device,
        retrieve_labels=True,
        nb_passes=1,
        )

    print('Cleaning predictions and labels.')
    preds_clean = np.array(list(map(data_info['clean_pred'], preds)))
    labels_clean = np.array(list(map(data_info['clean_pred'], labels.astype(int))))

    preds_clean = dict_to_str(preds_clean)
    labels_clean = dict_to_str(labels_clean)

    acc, _, _ = eval_sequence(labels_clean, preds_clean, seq_prob)

    results = {
        'Models used': list(models.keys()),
        'Number of observations for testing': len(generator),
        'Accuracy': acc,
        }
    pickle.dump(results, open(fn_results, 'wb'))

    pred_df = pd.DataFrame({
        'filename_full': files,
        'label': labels_clean,
        'pred': preds_clean,
        'prob': seq_prob,
        })

    pred_df.to_csv(fn_preds, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')

    parser.add_argument('--settings', type=str, choices=SETTINGS.keys())
    parser.add_argument('--root', type=str)
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--fn_results', type=str, default=None)
    parser.add_argument('--fn_preds', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1024)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    settings = args.settings
    root = args.root
    fn_results = args.fn_results
    fn_preds = args.fn_preds

    if fn_results is None:
        fn_results = os.path.join(root, 'eval_results.pkl')
    if fn_preds is None:
        fn_preds = os.path.join(root, 'preds.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pylint: disable=E1101

    data_info = SETTINGS[settings]['data_info']
    model_info = SETTINGS[settings]['model_info']

    data_info['root_labels'] = data_info['root_labels'].format(args.datadir)
    data_info['root_images'] = data_info['root_images'].format(args.datadir)

    data_info['batch_size'] = args.batch_size

    # Want to study test performance.
    data_info['cells'] = [(data_info['cells'][0][0].replace('train', 'test'), data_info['cells'][0][1])]

    evaluate(data_info, model_info, device, root, fn_results, fn_preds)


if __name__ == '__main__':
    main()
