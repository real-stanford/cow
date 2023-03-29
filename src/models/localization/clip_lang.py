from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import device

from src.clip import clip
from src.shared.utils import find_centroid, zeroshot_classifier


class ClipLang(nn.Module):
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

        super(ClipLang, self).__init__()

        self.clip_model_name = clip_model_name
        self.model, _ = clip.load(clip_model_name, device=device)
        self.target_classes = classes
        directional_classes_clip = [
            [
                f'{c} in the top left', f'{c} in the top center', f'{c} in the top right',
                f'{c} in the center left', f'{c} in the center', f'{c} in the center right',
                f'{c} in the bottom left', f'{c} in the bottom center', f'{c} in the bottom right',
            ] for c in classes_clip]

        self.templates = templates
        self.device = device
        self.class_to_language_feature = {}
        self.center_only = center_only
        self.threshold = threshold

        for i, c in enumerate(self.target_classes):
            # shape: [dim, n classes]
            self.class_to_language_feature[c] = zeroshot_classifier(
                self.model, directional_classes_clip[i], self.templates, self.device)

    def forward(self, x, o):
        '''
        non-standard hack around an nn, really should be more principled here
        '''
        b, c, h, w = x.shape
        assert b == 1 and h == 224 and w == 224 and c == 3

        with torch.no_grad():
            img_feature = self.model.encode_image(x.to(self.device))
            img_feature /= img_feature.norm(dim=-1, keepdim=True)
            zeroshot_weights = self.class_to_language_feature[o]
            logits = 100. * img_feature @ zeroshot_weights
            probs = logits.squeeze().softmax(dim=-1)
            attention = probs.reshape(3, 3).unsqueeze(0).unsqueeze(0)

            image_relevance = (nnf.interpolate(attention, size=(h, w), mode='nearest').squeeze() > self.threshold).cpu().float()

            if self.center_only:
                return find_centroid(image_relevance)

            return image_relevance

    def remap_classes(self, remap):
        remapped_classes = []
        for c in self.target_classes:
            remapped_classes.append(remap[c])
        self.target_classes_clip = remapped_classes

        directional_classes_clip = [
            [
                f'{c} in the top left', f'{c} in the top center', f'{c} in the top right',
                f'{c} in the center left', f'{c} in the center', f'{c} in the center right',
                f'{c} in the bottom left', f'{c} in the bottom center', f'{c} in the bottom right',
            ] for c in remapped_classes]

        for i, c in enumerate(self.target_classes):
            # shape: [dim, n classes]
            self.class_to_language_feature[c] = zeroshot_classifier(
                self.model, directional_classes_clip[i], self.templates, self.device)

    def load_weight_from_open_clip(self, open_clip_path, alpha):

        # alpha interpolation factor between zs clip and ft clip
        open_clip_ft = torch.load(open_clip_path)
        for key in list(open_clip_ft.keys()):
            if 'model.visual.' in key:
                open_clip_ft[key.replace('model.visual.', '')] = open_clip_ft.pop(
                    key).to(self.device)
            elif 'model.' in key:
                open_clip_ft[key.replace('model.', '')] = open_clip_ft.pop(
                    key).to(self.device)

        tmp = deepcopy(self.model.visual.state_dict())
        for key in list(self.model.visual.state_dict().keys()):
            tmp[key] = (1 - alpha) * self.model.visual.state_dict()[key] + \
                alpha * open_clip_ft[key]

        for k in tmp:
            self.model.visual.state_dict()[k] = tmp[k]
