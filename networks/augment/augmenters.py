# -*- coding: utf-8 -*-
"""
@author: tsdj
"""


import random

from torchvision import transforms

class ResizePad():
    PLACEMENTS_X = ('left', 'right', 'middle', 'random')
    PLACEMENTS_Y = ('top', 'bottom', 'middle', 'random')

    def __init__(
            self,
            target_h: int,
            target_w: int,
            placement: str = ('middle', 'middle'),
            padding_mode: str = 'edge',
        ):
        placements = [(x, y) for x in self.PLACEMENTS_X for y in self.PLACEMENTS_Y]
        assert placement in placements
        assert padding_mode in ('constant', 'edge')

        self.target_h = target_h
        self.target_w = target_w
        self.placement = placement
        self.padding_mode = padding_mode

    def _get_padding(self, delta_w, delta_h):
        ratios = {
            'w': {
                'left': 0.0,
                'right': 1.0,
                'middle': 0.5,
                'random': random.random(),
                },
            'h': {
                'top': 0.0,
                'bottom': 1.0,
                'middle': 0.5,
                'random': random.random(),
                },
            }

        pad_left = int(delta_w * ratios['w'][self.placement[0]])
        pad_top = int(delta_h * ratios['h'][self.placement[1]])
        pad_right = delta_w - pad_left
        pad_bottom = delta_h - pad_top

        return pad_left, pad_top, pad_right, pad_bottom

    def __call__(self, image):
        assert image.mode in ('L', 'RGB')

        width, height = image.size
        fill = 255 if image.mode == 'L' else (255, 255, 255)

        ratio_h = self.target_h / height
        ratio_w = self.target_w / width
        min_ratio = min(ratio_h, ratio_w)

        if min_ratio < 1:
            image = transforms.functional.resize(
                image,
                (int(height * min_ratio), int(width * min_ratio)),
                )

        width, height = image.size
        padding = self._get_padding(self.target_w - width, self.target_h - height)

        image = transforms.functional.pad(image, padding, fill=fill, padding_mode=self.padding_mode)

        return image

    def __repr__(self):
        return self.__class__.__name__ + '(target_h={0}, target_w={1}, placement={2}, padding_mode={3})'.format(
            self.target_h, self.target_w, self.placement, self.padding_mode,
            )


def to_col(image):
    return image.convert('RGB')
