import json
import os
import pdb
import platform
import random
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.clip import clip
from src.simulation.constants import THOR_OBJECT_TYPES
from tqdm import tqdm

import Xlib
import Xlib.display
import glob
import torch.nn as nn
import torch.nn.functional as nnf

def seed_everything(seed: int):
    print(f"setting seed: {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_iou(bb1, bb2):
    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def zeroshot_classifier(clip_model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname)
                     for template in templates]  # format with class
            texts = clip.tokenize(texts).to(device)  # tokenize
            class_embeddings = clip_model.encode_text(
                texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights  # shape: [dim, n classes]

def plot_category_accuracy(config_path, figure_out_path):

    data = None
    with open(config_path, 'r') as f:
        data = json.load(f)

    accs = []
    for c in THOR_OBJECT_TYPES:
        accs.append(data["num_successful_samples_per_category"]
                    [c] / data["num_total_samples_per_category"][c])

    x = np.arange(len(THOR_OBJECT_TYPES))

    plt.title(
        f'Action Acc. (Avg. {data["num_successful_samples"] / data["num_total_samples"]:.3f})\nExperiment name: {data["experiment_meta"]}')
    plt.ylabel('Acc.')
    plt.xlabel('Categories / Sample Counts')
    xt = [f"{c} / {data['num_total_samples_per_category'][c]}" for c in THOR_OBJECT_TYPES]
    plt.xticks(x, xt, rotation=90)
    plt.bar(x, height=accs)
    plt.tight_layout()

    plt.savefig(f'{figure_out_path}/{data["experiment_meta"]}')
    plt.cla()


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end

    return out


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def get_open_x_displays(throw_error_if_empty: bool = False) -> Sequence[str]:
    # NOTE: modified from allenact as out server has X-displays open not suitable for rendering
    assert platform.system() == "Linux", "Can only get X-displays for Linux systems."

    displays = []

    open_display_strs = [
        os.path.basename(s)[1:] for s in glob.glob("/tmp/.X11-unix/X*")
    ]

    for open_display_str in sorted(open_display_strs):
        try:
            open_display_str = str(int(open_display_str))


            display = Xlib.display.Display(":{}".format(open_display_str))

            displays.extend(
                [f"{open_display_str}.{i}" for i in range(display.screen_count())]
            )
        except Exception:
            continue

    if throw_error_if_empty and len(displays) == 0:
        raise IOError(
            "Could not find any open X-displays on which to run AI2-THOR processes. "
            " Please see the AI2-THOR installation instructions at"
            " https://allenact.org/installation/installation-framework/#installation-of-ithor-ithor-plugin"
            " for information as to how to start such displays."
        )

    return displays

def make_latex_table(cols: str, rows: str, title: str, caption: str, label: str):

    header = \
f"""
\\begin{{table}}
\\centering
\\begin{{tabular}}{{l?{'c'*(len(cols)-1)}}}
\\toprule
"""

    body = ' & '.join(cols) + '\\\\\\midrule\n'
    for row in rows:
        body += ' & '.join(row) + '\\\\\n'


    footer = \
f"""
\\bottomrule
\\end{{tabular}}
\\caption{{\\textbf{{{title}}} {caption}}}
\\label{{{label}}}
\\end{{table}}
"""

    return header + body + footer

def find_centroid(object_mask):
    us, vs = np.where(object_mask)

    if len(us) == 0:
        return object_mask

    mean_u = np.mean(us)
    mean_v = np.mean(vs)

    index = np.argmin((us - mean_u)**2 + (vs - mean_v)**2, axis=None)

    ret = torch.zeros_like(object_mask)
    ret[us[index], vs[index]] = 1

    return ret