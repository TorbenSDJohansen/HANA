# -*- coding: utf-8 -*-
"""
@author: tsdj

Script to contain classes to be used for pytorch data sets used in training and
predicting of neural networks.
"""

import types

import torch

from PIL import Image
from sklearn.model_selection import train_test_split

import numpy as np

def split_sample(labels: np.ndarray, val_size: int or float) -> tuple:
    ''' Split based on integer or float. '''
    if isinstance(val_size, float) and val_size == 0.0:
        return labels, None
    if isinstance(val_size, float) and val_size == 1.0:
        return None, labels

    return train_test_split(labels, test_size=val_size, random_state=1)


class AbstractDataset(torch.utils.data.Dataset):
    ''' Abstract dataset for torch, only ourpose of which is to be inherited.
    The class inherits from `torch.utils.data.Dataset` to create a dataset for
    images, where the images are loaded into memory as needed. This is useful
    for image classification tasks, including classification of sequences.

    An explanation of `self.status` is warrented. This MUST be one of:
        1) "sample_train".
        2) "sample_val".
    It is used to control from whether sampling is done on the training or
    validation data - which are created by splitting the labels, as seen in
    the classes inheriting from this class. The purpose of distinguishing
    between the two are primarily:
        1) To be able to log validation performance while training.
        2) The be able to distinguish between image preprocessing - as we may
        want to apply distortions to the training images.
    '''
    # To be defined in derived classes
    labels_train = None
    labels_val = None

    def __init__(self, transform_pipe: dict):
        self.transform_pipe = transform_pipe
        self.status = 'sample_train'

    def _instantiate_labels(self, labels_raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError('Abstract method meant to be overwritten.')

    def _len(self):
        if self.status == 'sample_train':
            return len(self.labels_train)
        if self.status == 'sample_val':
            return len(self.labels_val)

        raise Exception('Unrecognized sample length requested.')

    def _getitem(self, idx):
        if self.status == 'sample_train':
            return self.labels_train[idx, :]
        if self.status == 'sample_val':
            return self.labels_val[idx, :]

        raise Exception('Unrecognized sample requested.')

    def __len__(self):
        return self._len()

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self._getitem(idx)

        image = Image.open(sample[0])
        label = sample[1]

        if self.transform_pipe:
            image = self.transform_pipe[self.status](image)

        return {'image': image, 'label': label, 'fname': sample[0]}


class SequenceDataset(AbstractDataset): # pylint: disable=R0903
    ''' Prepares dataset for torch, taking as labels the information specified
    in `labels_raw` and transforms them by using `transform_to_label`.

    `labels_raw`: the np.array with the labels.
    `transform_to_label`: function to transform the raw labels into a sequence
    of integers.
    `val_size`: info to construct hold-out set, int if number desired
    or float if ratio desired
    `transform_pipe`: A dictionary of instances of
    torchvision.transforms.Compose, providing details on the chain of image
    transformation (seperately for train and eval).

    '''
    nb_dropped = None

    def __init__(
            self,
            labels_raw: str,
            transform_to_label: types.FunctionType,
            val_size: int or float = 1000,
            transform_pipe: dict = None,
        ):
        super().__init__(transform_pipe=transform_pipe)

        self.transform_to_label = transform_to_label

        labels = self._instantiate_labels(labels_raw)
        self.labels_train, self.labels_val = split_sample(labels, val_size)

    def _instantiate_labels(self, labels_raw: np.ndarray) -> np.ndarray:
        ''' This method instantiates the labels by loading and transforming
        them from raw sequences to lists of floats.
        '''
        labels_raw[:, 1] = list(map(self.transform_to_label, labels_raw[:, 1]))
        # It is assumed that bad labels are returned as None from
        # `self.transform_to_label`, and hence None are now removed
        self.nb_dropped = len([x for x in labels_raw[:, 1] if x is None])
        print(f'Dropped {self.nb_dropped} observations due to bad labels.')
        good_idxs = [i for i, x in enumerate(labels_raw[:, 1]) if x is not None]
        labels = labels_raw[good_idxs]

        return labels
