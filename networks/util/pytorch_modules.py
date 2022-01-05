# -*- coding: utf-8 -*-
"""
@author: tsdj
"""

from torch import nn


class SequenceEstimator(nn.Module):
    ''' Creates a dense layer for each item in output_sizes with nodes defined
    by the items in output_sizes, then applies a log_softmax to each layer
    '''
    def __init__(self, input_size, output_sizes):
        super(SequenceEstimator, self).__init__()
        for size in output_sizes:
            assert isinstance(size, int) and size > 0
        self.fc_layers = nn.ModuleList(
            [nn.Linear(input_size, size) for size in output_sizes])

    def forward(self, x):
        affine_transform = [fc(x) for fc in self.fc_layers]
        x = [nn.functional.log_softmax(x, dim=1) for x in affine_transform]
        x = tuple(x)
        return x
