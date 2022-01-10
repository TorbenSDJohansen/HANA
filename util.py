# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 13:54:58 2022

@author: sa-tsdj
"""


import string

import torch

from torchvision import transforms

import numpy as np

from networks.constructor import SequenceNet
from networks.augment.rand_augment import RandAugment
from networks.augment.augmenters import to_col


LETTERS = list(string.ascii_lowercase) + ['æ', 'ø', 'å']
MAP_LETTER_IDX = {letter: idx for idx, letter in enumerate(LETTERS)}
MAP_IDX_LETTER = {v: k for k, v in MAP_LETTER_IDX.items()}

MISSING_INDICATOR = 29
MAX_INDIVIDUAL_NAME_LEN = 18
MAX_NB_FIRST_NAMES = 9
MAX_NB_LAST_NAMES = 1
MAX_NB_MIDDLE_NAMES = MAX_NB_FIRST_NAMES - 1

def transform_label_individual_name(raw_input: dict or str) -> np.ndarray:
    '''
    Formats the last name to array of floats representing characters. The
    floats are just integers cast to float, as that format is used for training
    the neural networks.

    '''
    if isinstance(raw_input, dict):
        raw_input = raw_input['lastnames']

    assert isinstance(raw_input, str)

    name_len = len(raw_input)

    assert MAX_INDIVIDUAL_NAME_LEN >= name_len

    label = []

    for char in raw_input:
        label.append(MAP_LETTER_IDX[char])

    label += (MAX_INDIVIDUAL_NAME_LEN - name_len) * [MISSING_INDICATOR]

    label = np.array(label)

    # Assert cycle consistency
    assert raw_input == clean_pred_individual_name(label, False)

    label = label.astype('float')

    return label


def clean_pred_individual_name(raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
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
        transformed_clean = transform_label_individual_name(clean)

        if not all(pred.astype('float') == transformed_clean):
            raise Exception(raw_pred, pred, clean, transformed_clean)

    return clean


def transform_label_full_name(raw_input: dict) -> np.ndarray:
    '''
    Formats the name to array of floats representing characters. The floats are
    just integers cast to float, as that format is used for training
    the neural networks.

    '''
    first_names = raw_input['firstnames'].split(' ')
    last_names = raw_input['lastnames'].split(' ')

    nb_first_names = len(first_names)
    nb_last_name = len(last_names)

    full_name = first_names + [''] * (MAX_NB_FIRST_NAMES - nb_first_names) + \
        last_names + [''] * (MAX_NB_LAST_NAMES - nb_last_name)

    assert len(full_name) == MAX_NB_FIRST_NAMES + MAX_NB_LAST_NAMES

    label = []

    for name in full_name:
        label.append(transform_label_individual_name(name))

    label = np.concatenate(label)

    # Assert cycle consistency
    assert raw_input == clean_pred_full_name(label.astype(int), False)

    return label


def clean_pred_full_name(raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
    '''
    Maps predictions back from integer to string representation.

    '''
    sub_preds = [raw_pred[i * MAX_INDIVIDUAL_NAME_LEN:(i + 1) * MAX_INDIVIDUAL_NAME_LEN]
                 for i in range(MAX_NB_FIRST_NAMES + MAX_NB_LAST_NAMES)]

    sub_preds_reordered = []
    empty_name = np.array([MISSING_INDICATOR] * MAX_INDIVIDUAL_NAME_LEN)

    for i, sub_pred in enumerate(sub_preds):
        sub_pred_reordered = []
        for element in sub_pred:
            if element != MISSING_INDICATOR:
                sub_pred_reordered.append(element)
        sub_pred_reordered += [MISSING_INDICATOR] * (MAX_INDIVIDUAL_NAME_LEN - len(sub_pred_reordered))

        if i + 1 == MAX_NB_FIRST_NAMES + MAX_NB_LAST_NAMES:
            sub_preds_reordered.extend([empty_name for _ in range(MAX_NB_FIRST_NAMES + MAX_NB_LAST_NAMES - 1 - len(sub_preds_reordered))])
            sub_preds_reordered.append(np.array(sub_pred_reordered))
        else:
            if all(sub_pred_reordered == empty_name):
                continue
            sub_preds_reordered.append(np.array(sub_pred_reordered))

    raw_pred_reordered = np.concatenate(sub_preds_reordered)

    names = []

    for sub_pred in sub_preds_reordered:
        names.append(clean_pred_individual_name(sub_pred))

    first_name = ' '.join((x for x in names[:MAX_NB_FIRST_NAMES] if x != ''))
    last_name = ' '.join((x for x in names[MAX_NB_FIRST_NAMES:] if x != ''))

    clean = {'firstnames': first_name, 'lastnames': last_name}

    # Need to be cycle consistent - however, the function may be called from
    # `transform_label`, and we do not want infinite recursion, hence the if.
    if assert_consistency:
        transformed_clean = transform_label_full_name(clean)

        if not all(raw_pred_reordered.astype('float') == transformed_clean):
            raise Exception(raw_pred, names, clean, transformed_clean)

    return clean


def transform_label_first_and_last_name(raw_input: dict) -> np.ndarray:
    '''
    Formats the name to array of floats representing characters. The floats are
    just integers cast to float, as that format is used for training
    the neural networks.

    Specifically for first and last name only (i.e. 2 names).

    '''
    first_name = raw_input['firstnames'].split(' ')[0]
    last_name = raw_input['lastnames'].split(' ')[0]
    full_name = [first_name] + [last_name]

    mod = {'firstnames': first_name, 'lastnames': last_name}

    label = []

    for name in full_name:
        label.append(transform_label_individual_name(name))

    label = np.concatenate(label)

    # Assert cycle consistency
    assert mod == clean_pred_first_and_last_name(label.astype(int), False)

    return label


def clean_pred_first_and_last_name(raw_pred: np.ndarray, assert_consistency: bool = True) -> str:
    '''
    Maps predictions back from integer to string representation for first and
    last name (i.e. 2 names).

    '''

    sub_preds = [raw_pred[i * MAX_INDIVIDUAL_NAME_LEN:(i + 1) * MAX_INDIVIDUAL_NAME_LEN]
                 for i in range(2)]

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
        names.append(clean_pred_individual_name(sub_pred))

    first_name = names[0]
    last_name = names[1]

    clean = {'firstnames': first_name, 'lastnames': last_name}

    if assert_consistency:
        transformed_clean = transform_label_first_and_last_name(clean)

        if not all(raw_pred_reordered.astype('float') == transformed_clean):
            raise Exception(raw_pred, names, clean, transformed_clean)

    return clean


def get_pipe(nb_augments: int = 3, magnitude: int = 5) -> dict:
    '''
    Convinience for image transformation pipe construction.
    Useful to have as a funciton with NO parameters (or at least defaults) to
    import this to other files and ensure identical image preprocessing. This
    is useful when the network is used for some other purpose, such as
    prediciton on new images.

    '''
    im_h, im_w = 160, 1045
    scale_h, scale_w = 0.5, 0.5
    resizer = transforms.Resize((int(im_h * scale_h), int(im_w * scale_w)))
    pipe = {
        'sample_train': transforms.Compose([
            resizer,
            to_col,
            RandAugment(n=nb_augments, m=magnitude),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'sample_val': transforms.Compose([
            resizer,
            to_col,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        }

    return pipe


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
