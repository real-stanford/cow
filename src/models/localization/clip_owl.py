from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import device
from transformers import OwlViTForObjectDetection, OwlViTProcessor, OwlViTConfig
from transformers.models.owlvit.feature_extraction_owlvit import center_to_corners_format

def post_process(outputs, target_sizes):
    # NOTE: transformers/models/owlvit/feature_extraction_owlvit.py to put scale_fct on the correct device
    """
    Converts the output of [`OwlViTForObjectDetection`] into the format expected by the COCO api.

    Args:
        outputs ([`OwlViTObjectDetectionOutput`]):
            Raw outputs of the model.
        target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
            Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
            image size (before any data augmentation). For visualization, this should be the image size after data
            augment, but before padding.
    Returns:
        `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
        in the batch as predicted by the model.
    """
    logits, boxes = outputs.logits, outputs.pred_boxes

    if len(logits) != len(target_sizes):
        raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
    if target_sizes.shape[1] != 2:
        raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

    probs = torch.max(logits, dim=-1)
    scores = torch.sigmoid(probs.values)
    labels = probs.indices

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners_format(boxes)

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(logits.device)
    boxes = boxes * scale_fct[:, None, :]

    results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

    return results

class MyOwlViTForObjectDetection(OwlViTForObjectDetection):
    def __init__(self, config: OwlViTConfig):
        super().__init__(config)

    # NOTE: override so things are on the correct device, seems to be bug in
    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # Computes normalized xy corner coordinates from feature_map.
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_channels, height, width]")

        height, width = feature_map.shape[1:3]

        box_coordinates = np.stack(np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1)), axis=-1).astype(
            np.float32
        )
        box_coordinates /= np.array([width, height], np.float32)

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.reshape(
            box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2]
        )
        box_coordinates = torch.from_numpy(box_coordinates)

        return box_coordinates.to(feature_map.device)

class ClipOwl(nn.Module):
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

        super(ClipOwl, self).__init__()

        self.clip_model_name = clip_model_name

        assert 'ViT' in clip_model_name
        owl_from_pretrained = None
        if clip_model_name == 'ViT-B/32':
            owl_from_pretrained = 'google/owlvit-base-patch32'
        elif clip_model_name == 'ViT-B/16':
            owl_from_pretrained = 'google/owlvit-base-patch16'
        elif clip_model_name == 'ViT-L/14':
            owl_from_pretrained = 'google/owlvit-large-patch14'
        else:
            raise ValueError('gotta be a clip vit')

        self.model = MyOwlViTForObjectDetection.from_pretrained(owl_from_pretrained).eval().to(device)
        self.model.owlvit = self.model.owlvit.eval().to(device)
        self.model.class_head = self.model.class_head.eval().to(device)
        self.model.box_head = self.model.box_head.eval().to(device)
        self.center_only = center_only

        self.processor = OwlViTProcessor.from_pretrained(owl_from_pretrained)

        self.target_classes = classes
        self.target_classes_clip = classes_clip
        self.templates = templates
        self.device = device

        self.sentence_lookup = {}
        for i, c in enumerate(self.target_classes):
            self.sentence_lookup[c] = f'a photo of a {self.target_classes_clip[i]}.'
        self.count = 0
        self.threshold = threshold

        self.center_only = center_only

    def forward(self, x, o):
        texts = [[self.sentence_lookup[o],],]
        inputs = self.processor(text=texts, images=x, return_tensors="pt", truncation=True)
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)

        outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        # target_sizes = torch.Tensor([x.size[::-1]])
        results = post_process(outputs=outputs, target_sizes=torch.tensor([[224., 224.],]))

        # Retrieve predictions for the first image for the corresponding text queries
        boxes, scores = results[0]["boxes"], results[0]["scores"]

        image_relevance = torch.zeros((224, 224))

        for box, score in zip(boxes, scores):
            if score >= self.threshold:
                box = [int(round(i, 2)) for i in box.tolist()]

                if self.center_only:
                    u = int(round((box[1]+box[3]) / 2, 2))
                    v = int(round((box[0]+box[2]) / 2, 2))

                    image_relevance[u, v] = 1.

                else:
                    image_relevance[box[1]:box[3], box[0]:box[2]] = 1.

        return image_relevance

    def remap_classes(self, remap):
        remapped_classes = []
        for c in self.target_classes:
            remapped_classes.append(remap[c])
        self.target_classes_clip = remapped_classes

        self.sentence_lookup = {}
        for i, c in enumerate(self.target_classes):
            self.sentence_lookup[c] = f'a photo of a {self.target_classes_clip[i]}.'
