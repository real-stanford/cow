# modified from: https://colab.research.google.com/github/ashkamath/mdetr/blob/colab/notebooks/MDETR_demo.ipynb#scrollTo=5Tv4-AYncCSP

from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
from torch import device
import numpy as np
import torch.nn.functional as nnf


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

class MdetrSeg(nn.Module):
    '''
    NOTE: this class is kinda just meant for inference
    '''

    def __init__(
            self,
            clip_model_name: str,
            classes: List[str],
            classes_clip: List[str],
            templates: List[str],
            threshold: float,
            device: device,
            center_only: bool = False):

        super(MdetrSeg, self).__init__()

        # phrase cut fine-tuned mdetr
        model_name = 'mdetr_efficientnetB3_phrasecut'

        self.model = torch.hub.load(
            'ashkamath/mdetr:main',
            'mdetr_efficientnetB3_phrasecut',
            pretrained=True,
            return_postprocessor=False
        ).eval().to(device)

        self.center_only = center_only

        self.target_classes = classes
        self.target_classes_clip = classes_clip
        self.device = device

        self.sentence_lookup = {}
        for i, c in enumerate(self.target_classes):
            self.sentence_lookup[c] = self.target_classes_clip[i]
        self.count = 0
        self.threshold = threshold

        self.center_only = center_only

    def forward(self, x, o):
        outputs = self.model(x.to(self.device), [self.sentence_lookup[o]])
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > self.threshold)

        masks = nnf.interpolate(outputs["pred_masks"], size=(224, 224), mode="bilinear", align_corners=False)
        masks = masks.cpu()[0, keep].sigmoid() > 0.5

        if masks.shape[0] == 0:
            return torch.zeros((224, 224)).float()

        return masks[0].cpu().float()

    def remap_classes(self, remap):
        remapped_classes = []
        for c in self.target_classes:
            remapped_classes.append(remap[c])
        self.target_classes_clip = remapped_classes

        self.sentence_lookup = {}
        for i, c in enumerate(self.target_classes):
            self.sentence_lookup[c] = self.target_classes_clip[i]
