from typing import List

import torch
import torch.nn as nn
from torch import device


class ClipFiller(nn.Module):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            clip_model_name: str,
            classes: List[str],
            classes_clip: List[str],
            templates: List[str],
            device: device):

        super(ClipFiller, self).__init__()

    def forward(self, x, o):
        '''
        non-standard hack around an nn, really should be more principled here
        '''
        b, c, h, w = x.shape
        assert b == 1 and h == 224 and w == 224 and c == 3

        with torch.no_grad():

            return torch.zeros((h, w)).float()
