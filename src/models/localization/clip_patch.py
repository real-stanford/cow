from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
from src.clip import clip
from src.shared.utils import find_centroid, zeroshot_classifier
from src.simulation.constants import PATCH_TO_ACTION_THOR
from torch import device
import torch.nn.functional as nnf
# import cv2
# import numpy as np
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import torchvision.transforms as T


class ClipPatch(nn.Module):
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

        super(ClipPatch, self).__init__()

        self.clip_model_name = clip_model_name
        self.model, _ = clip.load(clip_model_name, device=device)
        self.target_classes = classes
        self.templates = templates
        self.device = device
        language_features = zeroshot_classifier(
            self.model, classes_clip, self.templates, self.device)

        self.class_to_language_feature = {}
        for i, c in enumerate(self.target_classes):
            self.class_to_language_feature[c] = language_features[:, [i]]

        self.count = 0
        self.center_only = center_only
        self.threshold = threshold

    def forward(self, x, o):
        '''
        non-standard hack around an nn, really should be more principled here
        '''
        patches = self.patch_batch(x.to(self.device))
        b, p, c, h, w = patches.shape

        assert b == 1
        patches = patches.reshape(b*p, c, h, w)
        patches = nnf.interpolate(patches, size=(
            224, 224), mode='bicubic', align_corners=False)

        with torch.no_grad():
            patch_features = self.model.encode_image(patches)
            patch_features /= patch_features.norm(dim=-1, keepdim=True)
            zeroshot_weights = self.class_to_language_feature[o]
            patch_logits = 100. * patch_features @ zeroshot_weights
            patch_probs = patch_logits.squeeze().softmax(dim=-1)

            attention = patch_probs.reshape(3, 3).unsqueeze(0).unsqueeze(0)
            attention = (nnf.interpolate(attention, size=(h, w), mode='nearest').squeeze() > self.threshold).cpu().float()

            if self.center_only:
                return find_centroid(attention)

            # t = T.Resize(224, interpolation=BICUBIC)

            # image = t(x)[0].permute(1, 2, 0).cpu().numpy()
            # image = (image - image.min()) / (image.max() - image.min())
            # vis = self.show_cam_on_image(image, attention)
            # vis = np.uint8(255 * vis)

            # cv2.imwrite(f'tmp2/{self.count}.png', vis)
            # self.count += 1

            return attention

    def patch_batch(self, x):
        patch_dim_h = int(x.shape[-2] / 3)
        patch_dim_w = int(x.shape[-1] / 3)
        kc, kh, kw = 3, patch_dim_h, patch_dim_w  # kernel size
        dc, dh, dw = 3, patch_dim_h, patch_dim_w  # stride
        patches = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)

        # [b, p, c, h, w]
        return patches.contiguous().view(patches.size(0), -1, kc, kh, kw)

    # def show_cam_on_image(self, img, mask):
    #     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #     heatmap = np.float32(heatmap) / 255
    #     cam = heatmap + np.float32(img)
    #     cam = cam / np.max(cam)
    #     return cam

    def remap_classes(self, remap):
        remapped_classes = []
        for c in self.target_classes:
            remapped_classes.append(remap[c])
        self.target_classes_clip = remapped_classes

        language_features = zeroshot_classifier(
            self.model, self.target_classes_clip, self.templates, self.device)

        self.class_to_language_feature = {}
        for i, c in enumerate(self.target_classes):
            self.class_to_language_feature[c] = language_features[:, [i]]

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
