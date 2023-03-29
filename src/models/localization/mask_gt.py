import torch
import torch.nn as nn


class MaskGt(nn.Module):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(self):

        super(MaskGt, self).__init__()

    def forward(self, x, o):

        image_relevance = torch.zeros((224, 224))

        return image_relevance