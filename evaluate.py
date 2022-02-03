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
from networks.util.setup_functions import prepare_labels_csv
from networks.util.pytorch_functions import (
    predict_sequence, eval_sequence,
    )
from networks.util.pytorch_datasets import SequenceDataset

from util import PipeConstructor
from settings import SETTINGS


def evaluate(
        data_info: dict,
        model_info: dict,
        device: torch.device, # pylint: disable=E1101
        root: str,
        fn_results: str = None,
        fn_preds: str = None,
        ):
    """
    Perform evaluation by calculating accuracy on the test data. Also produces
    predictions on the test data.

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
        Directory where the models are loaded from. If model_info contains URL
        to model weights, the weights will be loaded from there instead.
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

    print('Initializing generator.')
    labels_raw = prepare_labels_csv(
        cells=data_info['cells'],
        root_labels=data_info['root_labels'],
        root_images=data_info['root_images'],
        )
    if 'debug' in data_info.keys():
        labels_raw = labels_raw[:data_info['debug']]

    generator = SequenceDataset(
        labels_raw=labels_raw,
        transform_to_label=data_info['transform_label'],
        val_size=1.0,
        transform_pipe=PipeConstructor().get_pipe_from_settings(data_info),
        )
    generator.status = 'sample_val'

    dataset = DataLoader(generator, data_info['batch_size'], shuffle=False, num_workers=8)

    print('Predicting on test data!')
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

    acc, _, _ = eval_sequence(labels_clean, preds_clean, seq_prob)

    results = {
        'Models used': list(models.keys()),
        'Number of observations for testing': len(generator),
        'Full sequence accuracy': acc,
        }
    pickle.dump(results, open(fn_results, 'wb'))
    print(results)

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
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--fn-results', type=str, default=None)
    parser.add_argument('--fn-preds', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--debug', type=int, default=None, help='Keep only specified number of obs. for debugging.')
    parser.add_argument('--custom-name', type=str, default=None)
    parser.add_argument('--model-from-url', type=str, default=None, help='Load model weights from URL.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    settings = args.settings
    root = args.root
    fn_results = args.fn_results
    fn_preds = args.fn_preds

    if args.root is None:
        for _required in (args.model_from_url, args.fn_results, args.fn_preds):
            assert _required is not None, 'if root not specified, must specify model url and name of output files'

    if fn_results is None:
        fn_results = os.path.join(root, 'eval_results.pkl')
    if fn_preds is None:
        fn_preds = os.path.join(root, 'preds.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pylint: disable=E1101

    data_info = SETTINGS[settings]['data_info'].copy()
    model_info = SETTINGS[settings]['model_info'].copy()

    data_info['root_labels'] = data_info['root_labels'].format(args.datadir, 'test')
    data_info['root_images'] = data_info['root_images'].format(args.datadir)

    data_info['batch_size'] = args.batch_size

    model_name = list(model_info.keys())[0]

    if args.model_from_url is not None:
        model_info[model_name]['url'] = args.model_from_url

    if args.custom_name is not None:
        model_info[args.custom_name] = model_info.pop(model_name)
        model_name = args.custom_name

    if args.debug is not None:
        print(f'Debug mode using {args.debug} number of observations.')
        data_info['debug'] = args.debug

    evaluate(data_info, model_info, device, root, fn_results, fn_preds)


if __name__ == '__main__':
    main()
