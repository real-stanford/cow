import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from src.shared.data_split import DataSplit
from src.simulation.constants import (OBJECT_TYPES_WITH_PROPERTIES,
                                      RENDERING_BOX_FRAC_THRESHOLD,
                                      THOR_OBJECT_TYPES)
from PIL import Image
from torchvision.transforms.functional import hflip



def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


def get_device(device_number):
    device = torch.device("cpu")
    if device_number >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{0}".format(device_number))

    return device


def compute_3d_dist(p1, p2):
    p1_np = np.array([p1['x'], p1['y'], p1['z']])
    p2_np = np.array([p2['x'], p2['y'], p2['z']])

    squared_dist = np.sum((p1_np-p2_np)**2, axis=0)

    return np.sqrt(squared_dist)


def tile_image(im, height_pieces=3, width_pieces=3):
    del_h = im.shape[0]//height_pieces
    del_w = im.shape[1]//width_pieces

    # NOTE: coords in thor order so gotta do some flips
    coords = [
        {'y1': x, 'x1': y, 'y2': x+del_h, 'x2': y+del_w}
        for x in range(0, im.shape[0], del_h)
        for y in range(0, im.shape[1], del_w)]

    tiles = [im[x:x+del_h, y:y+del_w]
             for x in range(0, im.shape[0], del_h) for y in range(0, im.shape[1], del_w)]

    return tiles, coords


def get_roi_patches(im, height_pieces=3, width_pieces=3):
    del_h = im.shape[0]//height_pieces
    del_w = im.shape[1]//width_pieces

    # NOTE: coords in thor order so gotta do some flips
    coords = [
        {'y1': x, 'x1': y, 'y2': x+del_h, 'x2': y+del_w}
        for x in range(0, im.shape[0], del_h)
        for y in range(0, im.shape[1], del_w)]

    return coords


def depth_frame_to_camera_space_xyz_thor_grid(
    depth_frame: torch.Tensor, mask: Optional[torch.Tensor], fov: float = 90
) -> torch.Tensor:
    """Transforms a input depth map into a collection of xyz points (i.e. a
    point cloud) in the camera's coordinate frame.

    NOTE (samirg): copied from allenact to reduce # of dependencies, easier debugging,
    to ensure compatibility as allenact might change. This function is designed for
    Unity left-handed coordinate system.

    # Parameters
    depth_frame : A square depth map, i.e. an MxM matrix with entry `depth_frame[i, j]` equaling
        the distance from the camera to nearest surface at pixel (i,j).
    mask : An optional boolean mask of the same size (MxM) as the input depth. Only values
        where this mask are true will be included in the returned matrix of xyz coordinates. If
        `None` then no pixels will be masked out (so the returned matrix of xyz points will have
        dimension 3x(M*M)
    fov: The field of view of the camera.

    # Returns

    A 3xN matrix with entry [:, i] equalling a the xyz coordinates (in the camera's coordinate
    frame) of a point in the point cloud corresponding to the input depth frame.
    """
    assert (
        len(
            depth_frame.shape) == 2 and depth_frame.shape[0] == depth_frame.shape[1]
    ), f"depth has shape {depth_frame.shape}, we only support (N, N) shapes for now."



    depth_frame=hflip(depth_frame)

    shape = depth_frame.shape

    resolution = depth_frame.shape[0]
    if mask is None:
        mask = torch.ones_like(depth_frame, dtype=bool)

    # pixel centers
    camera_space_yx_offsets = (
        torch.stack(torch.where(mask))
        + 0.5  # Offset by 0.5 so that we are in the middle of the pixel
    )

    # Subtract center
    camera_space_yx_offsets -= resolution / 2.0

    # Make "up" in y be positive
    camera_space_yx_offsets[0, :] *= -1

    # Put points on the clipping plane
    camera_space_yx_offsets *= (2.0 / resolution) * \
        math.tan((fov / 2) / 180 * math.pi)

    camera_space_xyz = torch.cat(
        [
            camera_space_yx_offsets[1:, :],  # This is x
            camera_space_yx_offsets[:1, :],  # This is y
            torch.ones_like(camera_space_yx_offsets[:1, :]),
        ],
        axis=0,
    )

    points = camera_space_xyz * depth_frame[mask][None, :]

    points = points.reshape(3, shape[0], shape[1])
    points = hflip(points)

    return points
