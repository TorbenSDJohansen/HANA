# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


from util import (
    MISSING_INDICATOR,
    MAX_INDIVIDUAL_NAME_LEN,
    MAX_NB_FIRST_NAMES,
    MAX_NB_LAST_NAMES,
    transform_label_individual_name,
    clean_pred_individual_name,
    transform_label_full_name,
    clean_pred_full_name,
    transform_label_first_and_last_name,
    clean_pred_first_and_last_name,
    )


DATA_DEFAULT_INFO = {
    'cells': [('HANA_train', 'HANA')],
    'root_labels': '{}/',
    'root_images': '{}/',
    'batch_size': 256,
    'nb_epochs': 100,
    }
MODEL_DEFAULT_INFO = {
    'feature_extractor': 'resnet50',
    'classifier': 'multi-branch',
    'learning_rate': 0.05,
    'epochs_between_anneal': 30,
    'anneal_rate': 0.1,
    }

SETTINGS = {
    'ln': {
        'data_info': {
            **DATA_DEFAULT_INFO,
            'transform_label': transform_label_individual_name,
            'clean_pred': clean_pred_individual_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                },
            },
        },
    'fn': {
        'data_info': {
            **DATA_DEFAULT_INFO,
            'transform_label': transform_label_full_name,
            'clean_pred': clean_pred_full_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN * (MAX_NB_FIRST_NAMES + MAX_NB_LAST_NAMES),
                },
            },
        },
    'fln': {
        'data_info': {
            **DATA_DEFAULT_INFO,
            'transform_label': transform_label_first_and_last_name,
            'clean_pred': clean_pred_first_and_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN * 2,
                },
            },
        },
    }
