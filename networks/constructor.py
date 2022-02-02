# -*- coding: utf-8 -*-
"""
@author: tsdj
"""

import torch

from torch import nn

from networks.util.setup_functions import get_model_file
from networks.util.torchvision import resnet
from networks.util.pytorch_modules import SequenceEstimator


RESNETS = {
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'wide_resnet50_2',
    'wide_resnet101_2',
    'resnext101_32x8d_wsl',
    'resnext101_32x16d_wsl',
    'resnext101_32x32d_wsl',
    'resnext101_32x48d_wsl',
    }
FEATURE_EXTRACTORS = {
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnext50_32x4d': resnet.resnext50_32x4d,
    'resnext101_32x8d': resnet.resnext101_32x8d,
    'wide_resnet50_2': resnet.wide_resnet50_2,
    'wide_resnet101_2': resnet.wide_resnet101_2,
    'resnext101_32x8d_wsl': resnet.resnext101_32x8d_wsl,
    'resnext101_32x16d_wsl': resnet.resnext101_32x16d_wsl,
    'resnext101_32x32d_wsl': resnet.resnext101_32x32d_wsl,
    'resnext101_32x48d_wsl': resnet.resnext101_32x48d_wsl,
    }
assert set(FEATURE_EXTRACTORS.keys()) == RESNETS

CLASSIFIERS = {
    'multi-branch': SequenceEstimator,
    }

INPUT_TRANSFORMS = {
    'map-1-3': nn.Conv2d,
    }

def _build_feature_extractor(feature_extractor: str, pretrained: bool):
    assert feature_extractor in FEATURE_EXTRACTORS.keys()
    assert isinstance(pretrained, bool)

    builder = FEATURE_EXTRACTORS[feature_extractor]
    model = builder(pretrained=pretrained)

    return model, model.fc.in_features


def _build_classifier(classifier: str, input_size: int, output_sizes: list or tuple):
    assert classifier in CLASSIFIERS.keys()

    builder = CLASSIFIERS[classifier]

    return builder(input_size=input_size, output_sizes=output_sizes)


def _build_input_transform(input_transform: str):
    if input_transform is None:
        return None

    assert input_transform in INPUT_TRANSFORMS.keys()

    builder = INPUT_TRANSFORMS[input_transform]

    if builder == 'map-1-3':
        return builder(1, 3, kernel_size=3, stride=1, padding=1, bias=False)

    return builder


class SequenceNet(nn.Module):
    def __init__( # pylint: disable=R0913
            self,
            feature_extractor: str,
            classifier: str,
            output_sizes: list or tuple,
            pretrained: bool = True,
            input_transform: str = None,
        ):
        super().__init__()

        self.input_transform = _build_input_transform(input_transform)
        self.feature_extractor, out_nodes = _build_feature_extractor(
            feature_extractor, pretrained
            )
        self.classifier = _build_classifier(classifier, out_nodes, output_sizes)

    def forward(self, x): # pylint: disable=W0221
        if self.input_transform:
            x = self.input_transform(x)
        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x


def _load_model(model_name: str, root: str, info: dict, device: torch.device): # pylint: disable=E1101
    model = SequenceNet(
        feature_extractor=info['feature_extractor'],
        classifier=info['classifier'],
        output_sizes=info['output_sizes'],
        ).to(device)

    if 'url' in info:
        model.load_state_dict(torch.hub.load_state_dict_from_url(info['url']))
    else:
        model.load_state_dict(torch.load(get_model_file(root, model_name)))

    model.eval()

    return model


def load_models(model_info: dict, root: str, device: torch.device) -> dict: # pylint: disable=E1101
    """
    Loads multiple trained models (must be of type SequenceNet) and puts them
    in eval mode.

    Parameters
    ----------
    model_info : dict
        Dictionary with the names of the models as well as the info used to
        construct them.
    root : str
        The directory in which the models are saved (they will be in the sub-
        folder "./logs/"). If model_info contains URL to model weights, the
        weights will be loaded from there instead.
    device : torch.device
        The device (CPU/GPU) on which to load the models.

    Returns
    -------
    dict
        Dictionary of (model name, model) pairs.

    """
    models = dict()

    for model_name, info in model_info.items():
        models[model_name] = _load_model(model_name, root, info, device)

    return models


if __name__ == '__main__':
    pass
