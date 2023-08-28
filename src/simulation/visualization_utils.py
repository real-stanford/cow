# modified from: https://github.com/allenai/cordial-sync/blob/master/utils/visualization_utils.py
from __future__ import print_function, division

import math
from typing import Optional, List

import matplotlib

# matplotlib.use("TkAgg", force=False)
import matplotlib.pyplot as plt
from matplotlib import animation
import pylab


from PIL import Image, ImageDraw

import copy
import numpy as np
import textwrap
import re
import itertools


def partition(n, num_parts):
    m = n // num_parts
    parts = [m] * num_parts
    num_extra = n % num_parts
    for i in range(num_extra):
        parts[i] += 1
    return parts


def outline_square(frame, row, col, num_grids=7, stroke=2):
    f = copy.copy(frame)
    row_inds = [0] + np.cumsum(partition(f.shape[0], num_grids)).tolist()
    col_inds = [0] + np.cumsum(partition(f.shape[1], num_grids)).tolist()

    row_start = row_inds[row]
    row_end = row_inds[row + 1]
    col_start = col_inds[col]
    col_end = col_inds[col + 1]

    col = np.array([[[255, 0, 0]]])
    offset = stroke
    f[row_start : (row_start + offset), col_start:col_end, :] = col
    f[(row_end - offset) : row_end, col_start:col_end, :] = col

    f[row_start:row_end, col_start : (col_start + offset), :] = col
    f[row_start:row_end, (col_end - offset) : col_end, :] = col

    return f


def PIL2array(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def save_frames_to_mp4(frames, file_name, fps=3):
    h, w, _ = frames[0].shape
    aspect_ratio = w / h
    fig = plt.figure(figsize=(5 * aspect_ratio, 5))
    ax = fig.add_subplot(111)
    ax.set_frame_on(False)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(frames[0], cmap="gray", interpolation="nearest")
    im.set_clim([0, 255])

    pylab.tight_layout()

    def update_img(n):
        if n >= len(frames):
            im.set_data(frames[-1])
        else:
            im.set_data(frames[n])
        # k += 1
        return im

    ani = animation.FuncAnimation(fig, update_img, len(frames) - 1, interval=200)
    writer = animation.writers["ffmpeg"](fps=fps)

    ani.save(file_name, writer=writer, dpi=300)


class ThorPositionTo2DFrameTranslator(object):
    def __init__(self, frame_shape, cam_position, orth_size):
        self.frame_shape = frame_shape
        self.lower_left = np.array((cam_position[0], cam_position[2])) - orth_size
        self.span = 2 * orth_size

        # print(self.frame_shape)
        # print(self.lower_left)
        # print(self.span)

    def __call__(self, position):
        if len(position) == 3:
            x, _, z = position
        else:
            x, z = position

        camera_position = (np.array((x, z)) - self.lower_left) / self.span
        return np.array(
            (
                round(self.frame_shape[0] * (1.0 - camera_position[1])),
                round(self.frame_shape[1] * camera_position[0]),
            ),
            dtype=int,
        )


def position_to_tuple(position):
    if "position" in position:
        position = position["position"]
    return (position["x"], position["y"], position["z"])


def get_agent_map_data(env):
    env.step({"action": "ToggleMapView", "agentId": 0})
    cam_position = env.last_event.metadata["cameraPosition"]
    cam_orth_size = env.last_event.metadata["cameraOrthSize"]
    # print(cam_position)
    # print(cam_orth_size)
    pos_translator = ThorPositionTo2DFrameTranslator(
        env.last_event.events[0].frame.shape,
        position_to_tuple(cam_position),
        cam_orth_size,
    )
    to_return = {
        "frame": env.last_event.events[0].frame,
        "cam_position": cam_position,
        "cam_orth_size": cam_orth_size,
        "pos_translator": pos_translator,
    }
    env.step({"action": "ToggleMapView", "agentId": 0})
    return to_return


def add_agent_view_triangle(
    position, rotation, frame, pos_translator, scale=1.0, opacity=0.1
):
    p0 = np.array((position[0], position[2]))
    p1 = copy.copy(p0)
    p2 = copy.copy(p0)

    theta = -2 * math.pi * (rotation / 360.0)
    rotation_mat = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    offset1 = scale * np.array([-1, 1]) * math.sqrt(2) / 2
    offset2 = scale * np.array([1, 1]) * math.sqrt(2) / 2

    p1 += np.matmul(rotation_mat, offset1)
    p2 += np.matmul(rotation_mat, offset2)

    img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", frame.shape[:-1])  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    points = [tuple(reversed(pos_translator(p))) for p in [p0, p1, p2]]
    draw = ImageDraw.Draw(img2)
    draw.polygon(points, fill=(255, 255, 255, opacity))

    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))


def draw_line_with_rounded_ends(draw, xy, fill, width):
    draw.line(xy, fill=fill, width=width)
    for c in [xy[:2], xy[2:]]:
        draw.ellipse(
            (
                c[0] - width / 2 + 1,
                c[1] - width / 2 + 1,
                c[0] + width / 2 - 1,
                c[1] + width / 2 - 1,
            ),
            fill=fill,
            outline=None,
        )


def add_line_to_map(p0, p1, frame, pos_translator, opacity, color=None):
    if p0 == p1:
        return frame
    if color is None:
        color = (255, 0, 0)

    input_was_rgba = frame.shape[-1] == 4
    if input_was_rgba:
        img1 = Image.fromarray(frame.astype("uint8"), "RGBA")
    else:
        img1 = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.new("RGBA", (frame.shape[:-1][1], frame.shape[:-1][0]))  # Use RGBA

    opacity = int(round(255 * opacity))  # Define transparency for the triangle.
    draw = ImageDraw.Draw(img2)
    # draw.line(
    draw_line_with_rounded_ends(
        draw,
        tuple(reversed(pos_translator(p0))) + tuple(reversed(pos_translator(p1))),
        fill=color + (opacity,),
        width=int(frame.shape[0] / 50),
    )

    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB" if not input_was_rgba else "RGBA"))


def overlay_rgba_onto_rgb(rgb, rgba):
    img1 = Image.fromarray(rgb.astype("uint8"), "RGB").convert("RGBA")
    img2 = Image.fromarray(rgba.astype("uint8"), "RGBA")
    img = Image.alpha_composite(img1, img2)
    return np.array(img.convert("RGB"))


def visualize_agent_path(
    positions,
    frame,
    pos_translator,
    color_pair_ind: Optional[int] = None,
    colors: Optional[List] = None,
    show_vis_cone=True,
    show_visibility_cone_marked_points=True,
    only_show_last_visibility_cone=False,
    position_mark_colors: Optional[List[Optional[str]]] = None,
    opacity: float = 1.0,
    max_colors: int = None,
):
    import colour as col

    if colors is None:
        c0, c1 = [("red", "#ffc8c8"), ("green", "#c8ffc8"), ("blue", "#c8c8ff")][
            (color_pair_ind % 3)
        ]
        if max_colors is not None:
            colors = list(col.Color(c0).range_to(col.Color(c1), max_colors - 1))
        else:
            colors = list(col.Color(c0).range_to(col.Color(c1), len(positions) - 1))

    if opacity != 0:
        lines_frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
    else:
        lines_frame = frame

    for i in range(len(positions) - 1):
        lines_frame = add_line_to_map(
            position_to_tuple(positions[i]),
            position_to_tuple(positions[i + 1]),
            lines_frame,
            pos_translator,
            opacity=1.0,
            color=tuple(map(lambda x: int(round(255 * x)), colors[i].rgb)),
        )

    if opacity != 0:
        lines_frame[:, :, 3] = np.array(
            (lines_frame[:, :, 3] * opacity).round(), dtype=np.uint8
        )
        frame = overlay_rgba_onto_rgb(rgb=frame, rgba=lines_frame)
    else:
        frame = lines_frame

    mark_positions = []
    if position_mark_colors is not None:
        assert len(position_mark_colors) == len(positions)
        mark_positions = [
            p
            for p, mark_col in zip(positions, position_mark_colors)
            if mark_col is not None
        ]

        offsets = [(0.1, 0), (0, -0.1), (-0.1, 0), (0, 0.1)]
        offset_mark_positions = []
        mark_colors = []
        for i in range(len(positions)):
            if position_mark_colors[i] is not None:
                offset_ind = (int(positions[i]["rotation"]) % 360) // 90
                offset = offsets[offset_ind]
                mp = copy.copy(positions[i])
                mp["x"] = offset[0] + mp["x"]
                mp["z"] = offset[1] + mp["z"]
                offset_mark_positions.append(mp)
                mark_colors.append(position_mark_colors[i])

        frame = mark_positions_with_color(
            offset_mark_positions,
            frame,
            pos_translator,
            mark_colors,
            radius_frame_percent=0.02,
        )

    agent_view_triangle_positions = positions
    if only_show_last_visibility_cone:
        agent_view_triangle_positions = [positions[-1]]
    elif show_visibility_cone_marked_points:
        agent_view_triangle_positions = copy.copy(mark_positions)

    if show_vis_cone:
        for i, position in enumerate(agent_view_triangle_positions):
            frame = add_agent_view_triangle(
                position_to_tuple(position),
                rotation=position["rotation"],
                frame=frame,
                pos_translator=pos_translator,
                scale=1.5,
                opacity=0.35,
            )

    return frame


def visualize_agent_path_video(
    positions,
    frame,
    pos_translator,
    color_pair_ind: int,
    show_visibility_cone_marked_points=True,
    only_show_last_visibility_cone=False,
    position_mark_colors: Optional[List[Optional[str]]] = None,
):
    import colour as col

    c0, c1 = [("red", "red"), ("green", "green"), ("orange", "black")][
        (color_pair_ind % 3)
    ]
    if len(positions) > 1:
        colors = list(col.Color(c0).range_to(col.Color(c1), len(positions) - 1))
    else:
        colors = list(col.Color(c0).range_to(col.Color(c1), len(positions)))

    for i in range(len(positions) - 1):
        frame = add_line_to_map(
            position_to_tuple(positions[i]),
            position_to_tuple(positions[i + 1]),
            frame,
            pos_translator,
            opacity=1.0,  # 0.5,
            color=tuple(map(lambda x: int(round(255 * x)), colors[i].rgb)),
        )

    mark_positions = []
    if position_mark_colors is not None:
        if not len(position_mark_colors) == len(positions):
            raise RuntimeError
        mark_positions = [
            p
            for p, mark_col in zip(positions, position_mark_colors)
            if mark_col is not None
        ]

        offsets = [(0.1, 0), (0, -0.1), (-0.1, 0), (0, 0.1)]
        offset_mark_positions = []
        mark_colors = []
        for i in range(len(positions)):
            if position_mark_colors[i] is not None:
                offset_ind = (int(positions[i]["rotation"]) % 360) // 90
                offset = offsets[offset_ind]
                mp = copy.copy(positions[i])
                mp["x"] = offset[0] + mp["x"]
                mp["z"] = offset[1] + mp["z"]
                offset_mark_positions.append(mp)
                mark_colors.append(position_mark_colors[i])

        frame = mark_positions_with_color(
            offset_mark_positions,
            frame,
            pos_translator,
            mark_colors,
            radius_frame_percent=0.02,
        )

    agent_view_triangle_positions = positions
    if only_show_last_visibility_cone:
        agent_view_triangle_positions = [positions[-1]]
    elif show_visibility_cone_marked_points:
        agent_view_triangle_positions = copy.copy(mark_positions)

    for i, position in enumerate(agent_view_triangle_positions):
        frame = add_agent_view_triangle(
            position_to_tuple(position),
            rotation=position["rotation"],
            frame=frame,
            pos_translator=pos_translator,
            scale=1.5,
            opacity=0.15,
        )

    return frame


def mark_positions_with_color(
    positions, frame, pos_translator, color, radius_frame_percent: float = 0.01
):
    if len(positions) == 0:
        return frame

    if type(color) == list:
        assert len(positions) % len(color) == 0
        colors = color * (len(positions) // len(color))
    else:
        colors = [color] * len(positions)

    radius = int(frame.shape[0] * radius_frame_percent)

    img = Image.fromarray(frame.astype("uint8"), "RGB").convert("RGBA")
    draw = ImageDraw.Draw(img)

    for i, p in enumerate(positions):
        ptuple = tuple(reversed(pos_translator(position_to_tuple(p))))
        draw.ellipse(
            (
                ptuple[0] - radius / 2 + 1,
                ptuple[1] - radius / 2 + 1,
                ptuple[0] + radius / 2 - 1,
                ptuple[1] + radius / 2 - 1,
            ),
            fill=colors[i],
            outline=None,
        )
    return np.array(img.convert("RGB"))


def plot_confusion_matrix(cm, labels):
    """
    :param labels:
    :param cm: A confusion matrix: A square ```numpy array``` of the same size as labels
    :return:  A ``matplotlib.figure.Figure`` object with a numerical and graphical representation of the cm array
    """
    numClasses = len(labels)

    fig = matplotlib.figure.Figure(
        figsize=(numClasses, numClasses), dpi=100, facecolor="w", edgecolor="k"
    )
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap="Oranges")

    classes = [
        re.sub(r"([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))", r"\1 ", x) for x in labels
    ]
    classes = ["\n".join(textwrap.wrap(l, 20)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel("Agent 1")
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=-90, ha="center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.set_ylabel("Agent 0")
    ax.set_yticks(tick_marks)
    ax.set_ylim(-0.5, len(classes) - 0.5)
    ax.set_yticklabels(classes, va="center")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(numClasses), range(numClasses)):
        ax.text(
            j,
            i,
            round(cm[i, j] * 100, 2) if cm[i, j] != 0 else ".",
            horizontalalignment="center",
            verticalalignment="center",
            color="black",
        )
    fig.set_tight_layout(True)
    return fig
