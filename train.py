# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import argparse
import torch

from torch.utils.data import DataLoader

from networks.experiment import NetworkExperimentSequences
from networks.util.setup_functions import prepare_labels
from networks.util.pytorch_functions import (
    sequence_loss, sequence_acc, sequence_loss_eval,
    )
from networks.util.pytorch_datasets import SequenceDataset

from util import get_pipe, _setup_model_optimizer
from settings import SETTINGS


def train(
        data_info: dict,
        model_info: dict,
        device: torch.device, # pylint: disable=E1101
        root: str,
        ):
    """
    "Wrapper" function to run. Allows one to easily change a lot of the
    settings by simply passing new arguments.

    Parameters
    ----------
    data_info : dict
        Dictionary contains all information on the data, i.e. where to locate
        it, which batch size and how many epochs to train, and more.
    model_info : dict
        Dictionary contains information on how to build the model, including
        optimization and scheduler info.
    device : torch.device
        Controls whether GPU or CPU is used.
    root : str
        Directory where the checkpoints and the logs are saved.

    Returns
    -------
    None.

    """
    print('Initializing generator.')
    labels_raw = prepare_labels(
        cells=data_info['cells'],
        root_labels=data_info['root_labels'],
        root_images=data_info['root_images'],
        )
    if 'debug' in data_info.keys():
        labels_raw = labels_raw[:data_info['debug']]

    generator = SequenceDataset(
        labels_raw=labels_raw,
        transform_to_label=data_info['transform_label'],
        val_size=0.05,
        transform_pipe=get_pipe(),
        )

    print('Generating validation data.')
    generator.status = 'sample_val'
    ds_val = DataLoader(generator, batch_size=len(generator), shuffle=False)
    val_data = next(iter(ds_val))
    del ds_val

    print('Initializing training loader.')
    generator.status = 'sample_train'
    ds_train = DataLoader(generator, data_info['batch_size'], shuffle=True, num_workers=8)

    print('Initializing models, optimizers, and schedulers.')
    models = dict()
    for model_name, info in model_info.items():
        models[model_name] = _setup_model_optimizer(
            info=info,
            batch_size=data_info['batch_size'],
            epoch_len=len(generator),
            device=device,
            )

    experiment = NetworkExperimentSequences(
        dataset=ds_train,
        models=models,
        loss_function=sequence_loss,
        eval_loss=sequence_loss_eval,
        acc_function=sequence_acc,
        epochs=data_info['nb_epochs'],
        save_interval=min(10 * len(ds_train), 10_000),
        log_interval=min(len(ds_train), 1_000),
        root=root,
        device=device,
        )

    experiment.run(val_data)


def parse_args():
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--settings', type=str, choices=SETTINGS.keys())
    parser.add_argument('--root', type=str)
    parser.add_argument('--datadir', type=str)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--debug', type=int, default=None, help='Keep only specified number of obs. for debugging.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    settings = args.settings
    root = args.root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # pylint: disable=E1101

    data_info = SETTINGS[settings]['data_info']
    model_info = SETTINGS[settings]['model_info']

    data_info['root_labels'] = data_info['root_labels'].format(args.datadir)
    data_info['root_images'] = data_info['root_images'].format(args.datadir)

    data_info['batch_size'] = args.batch_size or data_info['batch_size']

    if args.debug is not None:
        print(f'Debug mode using {args.debug} number of observations.')
        data_info['debug'] = args.debug

    train(data_info, model_info, device, root)


if __name__ == '__main__':
    main()
