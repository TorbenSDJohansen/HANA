# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:54:58 2022

@author: sa-tsdj
"""


import string
import functools
import re

import torch

from torchvision import transforms

import numpy as np

from networks.constructor import SequenceNet
from networks.augment.pytorch_randaugment.rand_augment import RandAugment
from networks.augment.augmenters import to_col, ResizePad


LETTERS = list(string.ascii_lowercase) + ['æ', 'ø', 'å']
MAP_LETTER_IDX = {letter: idx for idx, letter in enumerate(LETTERS)}
MAP_IDX_LETTER = {v: k for k, v in MAP_LETTER_IDX.items()}

MISSING_INDICATOR = 29
MAX_INDIVIDUAL_NAME_LEN = 18
MAX_NB_NAMES = 10

def _transform_label_individual_name(raw_input: str) -> np.ndarray:
    '''
    Formats an individual name to array of floats representing characters. The
    floats are just integers cast to float, as that format is used for training
    the neural networks.

    '''
    assert isinstance(raw_input, str)

    name_len = len(raw_input)

    # assert MAX_INDIVIDUAL_NAME_LEN >= name_len
    if  MAX_INDIVIDUAL_NAME_LEN < name_len:
        return None

    label = []

    for char in raw_input:
        label.append(MAP_LETTER_IDX[char])

    label += (MAX_INDIVIDUAL_NAME_LEN - name_len) * [MISSING_INDICATOR]

    label = np.array(label)

    # Assert cycle consistency
    assert raw_input == _clean_pred_individual_name(label, False)

    label = label.astype('float')

    return label


def _clean_pred_individual_name(raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
    '''
    Maps predictions back from integer to string representation.

    '''
    non_missings = []

    for i, val in enumerate(raw_pred):
        if val != MISSING_INDICATOR:
            non_missings.append(i)

    pred = np.concatenate([
        raw_pred[non_missings],
        np.ones(MAX_INDIVIDUAL_NAME_LEN - len(non_missings), dtype=int) * MISSING_INDICATOR,
        ])

    clean = []

    for idx in pred:
        if idx == MISSING_INDICATOR:
            continue
        clean.append(MAP_IDX_LETTER[idx])

    clean = ''.join((clean))

    # Need to be cycle consistent - however, the function may be called from
    # `transform_label`, and we do not want infinite recursion, hence the if.
    if assert_consistency:
        transformed_clean = _transform_label_individual_name(clean)

        if not all(pred.astype('float') == transformed_clean):
            raise Exception(raw_pred, pred, clean, transformed_clean)

    return clean


def transform_label_last_name(raw_input: str) -> np.ndarray:
    last_name = raw_input.split(' ')[-1]

    return _transform_label_individual_name(last_name)


def clean_pred_last_name(raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
    return _clean_pred_individual_name(raw_pred, assert_consistency)


def transform_label_full_name(raw_input: str) -> np.ndarray:
    names = raw_input.split(' ')
    last_name = names[-1:]
    remaining = names[:-1]

    nb_names = len(names)
    full_name = remaining + [''] * (MAX_NB_NAMES - nb_names) + last_name

    label = []

    for name in full_name:
        label.append(_transform_label_individual_name(name))

    label = np.concatenate(label)

    # Assert cycle consistency
    assert raw_input == clean_pred_full_name(label.astype(int), False)

    return label


def clean_pred_full_name(raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
    sub_preds = [
        raw_pred[i * MAX_INDIVIDUAL_NAME_LEN:(i + 1) * MAX_INDIVIDUAL_NAME_LEN]
        for i in range(MAX_NB_NAMES)
        ]

    sub_preds_reordered = []
    empty_name = np.array([MISSING_INDICATOR] * MAX_INDIVIDUAL_NAME_LEN)

    for i, sub_pred in enumerate(sub_preds):
        sub_pred_reordered = []
        for element in sub_pred:
            if element != MISSING_INDICATOR:
                sub_pred_reordered.append(element)
        sub_pred_reordered += [MISSING_INDICATOR] * (MAX_INDIVIDUAL_NAME_LEN - len(sub_pred_reordered))

        if i + 1 == MAX_NB_NAMES:
            sub_preds_reordered.extend([empty_name for _ in range(MAX_NB_NAMES - 1 - len(sub_preds_reordered))])
            sub_preds_reordered.append(np.array(sub_pred_reordered))
        else:
            if all(sub_pred_reordered == empty_name):
                continue
            sub_preds_reordered.append(np.array(sub_pred_reordered))

    raw_pred_reordered = np.concatenate(sub_preds_reordered)

    names = []

    for sub_pred in sub_preds_reordered:
        names.append(_clean_pred_individual_name(sub_pred, assert_consistency))

    clean = re.sub(' +', ' ', ' '.join(names)).strip()

    # Need to be cycle consistent - however, the function may be called from
    # `transform_label`, and we do not want infinite recursion, hence the if.
    if assert_consistency:
        transformed_clean = transform_label_full_name(clean)

        if not all(raw_pred_reordered.astype('float') == transformed_clean):
            raise Exception(raw_pred, names, clean, transformed_clean)

    return clean


def transform_label_first_and_last_name(raw_input: str) -> np.ndarray:
    names = raw_input.split(' ')
    first_name = names[0]
    last_name = names[-1]

    mod_input = ' '.join((first_name, last_name))

    label = []

    for name in (first_name, last_name):
        label.append(_transform_label_individual_name(name))

    label = np.concatenate(label)

    # Assert cycle consistency
    assert mod_input == clean_pred_first_and_last_name(label.astype(int), False)

    return label


def clean_pred_first_and_last_name(raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
    sub_preds = [
        raw_pred[i * MAX_INDIVIDUAL_NAME_LEN:(i + 1) * MAX_INDIVIDUAL_NAME_LEN]
        for i in range(2)
        ]

    sub_preds_reordered = []

    for sub_pred in sub_preds:
        sub_pred_reordered = []
        for element in sub_pred:
            if element != MISSING_INDICATOR:
                sub_pred_reordered.append(element)
        sub_pred_reordered += [MISSING_INDICATOR] * (MAX_INDIVIDUAL_NAME_LEN - len(sub_pred_reordered))
        sub_preds_reordered.append(np.array(sub_pred_reordered))

    raw_pred_reordered = np.concatenate(sub_preds_reordered)

    names = []

    for sub_pred in sub_preds_reordered:
        names.append(_clean_pred_individual_name(sub_pred, assert_consistency))

    clean = ' '.join(names)

    if assert_consistency:
        transformed_clean = transform_label_first_and_last_name(clean)

        if not all(raw_pred_reordered.astype('float') == transformed_clean):
            raise Exception(raw_pred, names, clean, transformed_clean)

    return clean


class PipeConstructor(): # pylint: disable=C0115
    _ALLOWED_RESIZE_METHODS = (
        'resize_pad', 'resize',
        )
    resizer = None

    def init_resizer(self, resize_method, target_h, target_w):
        """
        Setup for image transformation resizer.

        Parameters
        ----------
        resize_method : str
            The resize method uses to achieve desired image size.
        target_h : int
            The desired height.
        target_w : int
            The desired width.

        """

        assert resize_method in self._ALLOWED_RESIZE_METHODS
        assert isinstance(target_h, int) and target_h > 0
        assert isinstance(target_w, int) and target_w > 0

        if resize_method == 'resize':
            self.resizer = transforms.Resize((target_h, target_w))
        elif resize_method == 'resize_pad':
            # Maybe allow for type of padding to be controlled.
            self.resizer = ResizePad(target_h, target_w, ('left', 'bottom'), 'constant')
        else:
            err_msg = (
                'Must specify a valid resize method. Requested: '
                + f'"{resize_method}". Allowed: {self._ALLOWED_RESIZE_METHODS}.'
            )
            raise Exception(err_msg)

    def get_pipe(self, nb_augments: int = 3, magnitude: int = 5) -> dict: # pylint: disable=C0116
        pipe = {
            'sample_train': transforms.Compose([
                self.resizer,
                to_col,
                RandAugment(n=nb_augments, m=magnitude),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]),
            'sample_val': transforms.Compose([
                self.resizer,
                to_col,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            }

        return pipe

    def get_pipe_from_settings(self, data_setting: dict) -> dict: # pylint: disable=C0116
        self.init_resizer(
            resize_method=data_setting['resize_method'],
            target_h=data_setting['height'],
            target_w=data_setting['width'],
            )

        return self.get_pipe()


def _rgetattr(obj, attr, *args):
    # https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
    # Simple way to retrieve attributes of attributes (i.e. solve the problem
    # of nested objects). Useful to freeze (and unfreeze) layers that consists
    # of other layers.
    # This way, a PART of a submodule can be frozen. For example, if a module
    # contains a submodule for feature extraction, a submodule of the feature
    # extraction submodule can be frozen (rather than only the entire feature
    # extraction submodule)
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def _setup_model_optimizer(
        info: dict,
        batch_size: int,
        epoch_len: int,
        device: torch.device, # pylint: disable=E1101
    ) -> dict:
    model = SequenceNet(
        feature_extractor=info['feature_extractor'],
        classifier=info['classifier'],
        output_sizes=info['output_sizes'],
        ).to(device)

    if 'fn_pretrained' in info.keys():
        model.load_state_dict(torch.load(info['fn_pretrained']), strict=False)
    if 'url' in info:
        model.load_state_dict(torch.hub.load_state_dict_from_url(info['url']))

    if 'to_freeze' in info.keys():
        for layer in info['to_freeze']:
            print(f'Freezing {layer}!')
            params = _rgetattr(model, layer).parameters()
            for param in params:
                param.requires_grad = False

    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=info['learning_rate'] * batch_size / 256,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
        )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=epoch_len // batch_size * info['epochs_between_anneal'],
        gamma=info['anneal_rate'],
        )

    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'state_objects': ['model', 'optimizer', 'scheduler'],
        'step_objects': ['optimizer', 'scheduler'],
        }
