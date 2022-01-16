# -*- coding: utf-8 -*-
"""
@author: sa-tsdj
"""


import pickle

from util import (
    MISSING_INDICATOR,
    MAX_INDIVIDUAL_NAME_LEN,
    MAX_NB_NAMES,
    transform_label_last_name,
    clean_pred_last_name,
    transform_label_full_name,
    clean_pred_full_name,
    transform_label_first_and_last_name,
    clean_pred_first_and_last_name,
    )


with open(r'./data/cells-uscensus.pkl', 'rb') as f:
    CELLS_US_CENSUS = pickle.load(f)

_DATA_DEFAULT_INFO = {
    'root_labels': '{}/labels/{}/',
    'root_images': '{}/minipics/',
    'batch_size': 256,
    'nb_epochs': 100,
    'resize_method': 'resize',
    }
MODEL_DEFAULT_INFO = {
    'feature_extractor': 'resnet50',
    'classifier': 'multi-branch',
    'learning_rate': 0.05,
    'epochs_between_anneal': 30,
    'anneal_rate': 0.1,
    }

DATA_DEFAULT_INFO_HANA = {
    **_DATA_DEFAULT_INFO,
    'cells': ['HANA'],
    'height': 80,
    'width': 522,
    }
DATA_DEFAULT_INFO_DANISH_CENSUS_SMALL = {
    **_DATA_DEFAULT_INFO,
    'cells': [f'name-{x}' for x in range(1, 26)],
    'height': 65,
    'width': 465,
    }
DATA_DEFAULT_INFO_DANISH_CENSUS_LARGE = {
    **_DATA_DEFAULT_INFO,
    'cells': [f'name-{x}' for x in range(1, 26)],
     'height': 65,
    'width': 465,
    }
DATA_DEFAULT_INFO_US_CENSUS_SMALL = {
    **_DATA_DEFAULT_INFO,
    'cells': CELLS_US_CENSUS,
    'height': 95,
    'width': 350,
    }
DATA_DEFAULT_INFO_US_CENSUS_LARGE = {
     **_DATA_DEFAULT_INFO,
    'cells': CELLS_US_CENSUS,
    'height': 95,
    'width': 350,
    }

SETTINGS = {
    # SETTINGS FOR NETWORKS ON HANA DATABASE
    'hana-last-name': {
        'data_info': {
            **DATA_DEFAULT_INFO_HANA,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                },
            },
        },
    'hana-full-name': {
        'data_info': {
            **DATA_DEFAULT_INFO_HANA,
            'transform_label': transform_label_full_name,
            'clean_pred': clean_pred_full_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN * MAX_NB_NAMES,
                },
            },
        },
    'hana-first-and-last-name': {
        'data_info': {
            **DATA_DEFAULT_INFO_HANA,
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
    # SETTINGS FOR NETWORKS ON DANISH CENSUS
    'danish-census-small-last-name': {
        'data_info': {
            **DATA_DEFAULT_INFO_DANISH_CENSUS_SMALL,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                },
            },
        },
    'danish-census-small-last-name-tl': {
        'data_info': {
            **DATA_DEFAULT_INFO_DANISH_CENSUS_SMALL,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                'learning_rate': 0.001,
                },
            },
        },
    'danish-census-large-last-name': {
        'data_info': {
            **DATA_DEFAULT_INFO_DANISH_CENSUS_LARGE,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                },
            },
        },
    'danish-census-large-last-name-tl': {
        'data_info': {
            **DATA_DEFAULT_INFO_DANISH_CENSUS_LARGE,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                'learning_rate': 0.001,
                },
            },
        },
    # SETTINGS FOR NETWORKS ON US CENSUS
    'US-census-small-last-name': {
        'data_info': {
            **DATA_DEFAULT_INFO_US_CENSUS_SMALL,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                },
            },
        },
    'US-census-small-last-name-tl': {
        'data_info': {
            **DATA_DEFAULT_INFO_US_CENSUS_SMALL,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                'learning_rate': 0.001,
                },
            },
        },
    'US-census-large-last-name': {
        'data_info': {
            **DATA_DEFAULT_INFO_US_CENSUS_LARGE,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                },
            },
        },
    'US-census-large-last-name-tl': {
        'data_info': {
            **DATA_DEFAULT_INFO_US_CENSUS_LARGE,
            'transform_label': transform_label_last_name,
            'clean_pred': clean_pred_last_name,
            },
        'model_info': {
            'resnet50-multi-branch': {
                **MODEL_DEFAULT_INFO,
                'output_sizes': [MISSING_INDICATOR + 1] * MAX_INDIVIDUAL_NAME_LEN,
                'learning_rate': 0.001,
                },
            },
        },
    }
