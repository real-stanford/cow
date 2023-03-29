import math
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch import device

from src.clip import clip, clip_explainability
from src.shared.utils import find_centroid, zeroshot_classifier

# import cv2


class ClipGrad(nn.Module):
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

        super(ClipGrad, self).__init__()

        self.clip_model_name = clip_model_name

        self.is_vit = False
        if 'ViT' in clip_model_name:
            self.is_vit = True

        # codebase only supports ViTs at the moment
        assert self.is_vit

        if self.is_vit:
            self.model, self.preprocess = clip_explainability.load(
                clip_model_name, device=device)
        else:
            # different implementation of Grad-CAM for RN than ViT
            self.model, self.preprocess = clip.load(
                clip_model_name, device=device)
        self.model.float()


        self.target_classes = classes
        self.target_classes_clip = classes_clip
        self.templates = templates
        self.device = device
        language_features = zeroshot_classifier(
            self.model, self.target_classes_clip, self.templates, self.device)

        self.class_to_language_feature = {}
        for i, c in enumerate(self.target_classes):
            self.class_to_language_feature[c] = language_features[:, [i]]
        self.count = 0
        self.gradient_scalar = 10
        self.threshold = threshold
        self.center_only = center_only

    def forward(self, x, o):
        '''
        non-standard hack around an nn, really should be more principled here
        '''
        assert x.shape[0] == 1

        image_features = self.model.encode_image(x.to(self.device))
        image_features = image_features / \
            image_features.norm(dim=-1, keepdim=True)
        zeroshot_weights = self.class_to_language_feature[o]
        logits_per_image = 100. * image_features @ zeroshot_weights.float()

        return self.interpret_vit(x, logits_per_image, self.model, self.device)

    def interpret_vit(self, image, logits_per_image, model, device, num_layers=10):
        # modified from: https://colab.research.google.com/github/hila-chefer/Transformer-MM-Explainability/blob/main/CLIP_explainability.ipynb#scrollTo=fWKGyu2YAeSV
        batch_size = logits_per_image.shape[0]
        index = [i for i in range(batch_size)]
        one_hot = np.zeros(
            (logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * logits_per_image)
        model.zero_grad()

        image_attn_blocks = list(
            dict(model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens,
                      dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, block in enumerate(image_attn_blocks):
            if i <= num_layers:
                continue
            grad = torch.autograd.grad(
                one_hot, [block.attn_probs], retain_graph=True)[0].detach()
            cam = block.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)

        image_relevance = R[:, 0, 1:]
        res = int(math.sqrt(image_relevance.shape[-1]))
        image_relevance = image_relevance.reshape(1, 1, res, res)
        image_relevance = nnf.interpolate(
            image_relevance, size=224, mode='bilinear', align_corners=False)
        image_relevance = image_relevance.reshape(224, 224)

        # image = image[0].permute(1, 2, 0).cpu().numpy()
        # image = (image - image.min()) / (image.max() - image.min())
        # vis = self.show_cam_on_image(image, torch.clamp(image_relevance * self.gradient_scalar, min=0., max=1.))
        # vis = np.uint8(255 * vis)

        # cv2.imwrite(f'tmp2/{self.count}.png', vis)
        # self.count += 1

        image_relevance = (image_relevance * self.gradient_scalar > self.threshold).cpu().float()

        # NOTE: uncomment for fig
        # image_relevance = (torch.clamp(image_relevance * self.gradient_scalar,
        #                    min=0., max=1.)).cpu().float()

        if self.center_only:
            image_relevance = find_centroid(image_relevance)

        return image_relevance

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
                open_clip_ft[key.replace('model.visual.', '')] = open_clip_ft.pop(key).to(self.device)
            elif 'model.' in key:
                open_clip_ft[key.replace('model.', '')] = open_clip_ft.pop(key).to(self.device)

        for p in self.model.parameters():
            p.data = p.data.float()
            if p.grad:
                p.grad.data = p.grad.data.float()

        tmp = deepcopy(self.model.visual.state_dict())
        for key in list(self.model.visual.state_dict().keys()):
            tmp[key] = ((1 - alpha) * self.model.visual.state_dict()[key]) + (alpha *  open_clip_ft[key])

        self.model.visual.load_state_dict(tmp)
