import json
import math
from copy import copy, deepcopy
from enum import IntEnum
from heapq import heappop, heappush
from math import ceil, floor
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as nnf
import trimesh
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import AgglomerativeClustering
from torch import device

from src.shared.utils import tensor_linspace
from src.simulation.constants import (ACTION_SEQUENCES, ROTATION_MATRICIES,
                                      VOXEL_SIZE_M)
from src.simulation.utils import depth_frame_to_camera_space_xyz_thor_grid

cmap = plt.get_cmap('jet')

try:
    import wandb
except ModuleNotFoundError:
    print("wandb not properly installed")


class PQNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    # compares the second value
    def __lt__(self, other):
        return self.key < other.key

    def __str__(self):
        return str("{} : {}".format(self.key, self.value))


class VoxelType(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    WALL = 3
    FRONTIER = 4
    # ROI = 5
    DBG = 5

    def color(self, rgba=False):

        c = None

        if self.value == VoxelType.UNKNOWN:
            c = [0, 225, 225]
        elif self.value == VoxelType.FREE:
            c = [255, 255, 255]
        elif self.value == VoxelType.OCCUPIED:
            c = [255, 0, 0]
        elif self.value == VoxelType.WALL:
            c = [255, 0, 0]
        elif self.value == VoxelType.FRONTIER:
            # c = [225, 0, 225]
            c = [255, 255, 255]
        # elif self.value == VoxelType.ROI:
        #     c = [255, 0, 255]
        elif self.value == VoxelType.DBG:
            c = [0, 255, 0]
        else:
            raise ValueError("Not supported enum")

        if rgba:
            c.append(225)

        return c


class FrontierBasedExploration(object):

    def __init__(
            self,
            fov: float,
            device: device,
            max_ceiling_height: float,
            agent_height: float,
            agent_height_tolerance: float,
            rotation_deg: float,
            forward_dist: float,
            voxel_size_m: float,
            in_cspace: bool,
            wandb_log: bool,
            negate_action: bool,
            fail_stop: bool):

        super(FrontierBasedExploration, self).__init__()
        self.fov = fov
        self.device = device
        self.camera_to_agent = torch.eye(4)
        self.max_ceiling_height = max_ceiling_height
        self.agent_height = agent_height
        self.agent_height_tolerance = agent_height_tolerance
        self.rotation_deg = rotation_deg
        self.forward_dist = forward_dist
        self.voxel_size_m = voxel_size_m
        self.in_cspace = in_cspace
        self.voxels = nx.Graph()
        self.floor_height_vox = int(
            math.floor(-self.agent_height / self.voxel_size_m))

        self.agent_voxel = None
        self.lookat_direction = torch.tensor([0., 0., 1.])
        self.exploration_targets = []
        self.visited_frontiers = set()

        self.run = None
        if wandb_log:
            self.run = wandb.init()

        self.reverse = True
        self.roi_targets = []
        self.failed_action = None
        self.last_observation = None
        self.negate_action = negate_action
        self.fail_stop = fail_stop

    def reset(self):
        self.camera_to_agent = torch.eye(4)
        self.voxels = nx.Graph()
        self.reverse = not self.reverse
        self.visited_frontiers = set()
        self.exploration_targets = []
        self.roi_targets = []
        self.failed_action = None
        self.last_observation = None

    def poll_roi_exists(self):
        return len(self.roi_targets) != 0

    def voxel_to_pointcloud(self, rgba=False, viz_edges=False, viz_roi_count=False, height_aware=False):
        voxel_groups = {}

        for v in self.voxels:
            voxel_type = self.voxels.nodes[v]["voxel_type"]

            if self.voxels.nodes[v]["obj_conf"] is None:
                obj_conf = 0.
            else:
                obj_conf = self.voxels.nodes[v]["obj_conf"]

            if voxel_type in voxel_groups:
                voxel_groups[voxel_type].append(
                    (v, obj_conf, self.voxels.nodes[v]["roi_count"]))
            else:
                voxel_groups[voxel_type] = [
                    (v, obj_conf, self.voxels.nodes[v]["roi_count"])]
        if viz_edges:
            for e in self.voxels.edges:
                if VoxelType.DBG in voxel_groups:
                    voxel_groups[VoxelType.DBG].append(
                        ((np.array(e[0]) + np.array(e[1])) / 2, 0, 0))
                else:
                    voxel_groups[VoxelType.DBG] = [
                        ((np.array(e[0]) + np.array(e[1])) / 2, 0, 0)]

        xyz = []
        color = []

        for t in voxel_groups:
            if t == VoxelType.OCCUPIED:
                for p, c, roi_count in voxel_groups[t]:
                    if height_aware:
                        h = self.voxels.nodes[p]['height']
                        if h is None:
                            h = self.floor_height_vox
                        xyz.append((p[0], h, p[2]))
                    else:
                        xyz.append(p)

                    if viz_roi_count:
                        if roi_count > 0:
                            color.append([int(255 * x) for x in cmap(1.0)])
                        else:
                            color.append([int(255 * x) for x in cmap(0.0)])
                    else:
                        color.append([int(255 * x) for x in cmap(c)])
            else:
                for p, _, _ in voxel_groups[t]:
                    xyz.append(p)
                    color.append(t.color(rgba=rgba))

        return xyz, color

    def dump_voxel_pointcloud(self, out_path):
        vertices, colors = self.voxel_to_pointcloud(rgba=True)

        ply = trimesh.points.PointCloud(
            vertices=np.array(vertices), colors=np.array(colors))

        # NOTE: uncomment for fig
        # with open('map.json', 'w') as f:
        #     json.dump(
        #         {
        #             'points': vertices,
        #             'colors': colors
        #         },
        #         f
        #     )
        ply.export(out_path)

    def log_voxel_pointcloud(self, suffix):
        if self.run is not None:
            vertices, colors = self.voxel_to_pointcloud(rgba=False)
            points_rgb = np.array([[p[0], p[1], p[2], c[0], c[1], c[2]]
                                  for p, c in zip(vertices, colors)])
            self.run.log({f"point_cloud{suffix}": wandb.Object3D(points_rgb)})

    def update_map(self, observations, attention, last_action):
        """Updates the map for frontier based exploration.

        Args:
            observations ([type]): [description]
            roi_mask ([type]): [description]
        """
        new_obs = torch.as_tensor(observations["depth"])
        new_obs.squeeze_()

        if self.last_observation is not None and self.fail_stop:
            abs_diff = torch.abs(self.last_observation-new_obs)
            m_pix = torch.mean(abs_diff)
            s_pix = torch.std(abs_diff)
            if m_pix.item() < 0.09 and s_pix.item() < 0.09:
                self.failed_action = last_action
            else:
                self.failed_action = None

        self.last_observation = new_obs

        # hflip as depth_frame_to_camera_space_xyz assumes left handed coordinate system
        depth_lowres = nnf.interpolate(torch.as_tensor(observations["depth"]).squeeze().unsqueeze(0).unsqueeze(0),
                                       size=(224, 224),
                                       mode='nearest').squeeze()


        # [3, 224*224]
        points_in_camera = depth_frame_to_camera_space_xyz_thor_grid(
            depth_frame=depth_lowres, mask=None, fov=self.fov
        )

        no_holes = depth_lowres > 0.051

        # [1, 224*224]
        clip_confidence = nnf.interpolate(attention.unsqueeze(0).unsqueeze(0),
                                          size=depth_lowres.shape,
                                          mode='nearest')[0]

        not_ceiling_mask = points_in_camera[1, :, :] < (
            self.max_ceiling_height-self.agent_height-self.agent_height_tolerance)

        composite_mask = torch.logical_and(no_holes, not_ceiling_mask)

        points_in_camera = points_in_camera[:, composite_mask]

        # [224*224, 1]
        clip_confidence = clip_confidence[:, composite_mask].transpose(0, 1)

        floor_mask = torch.logical_and(points_in_camera[1, :] > (-self.agent_height-self.agent_height_tolerance),
                                       points_in_camera[1, :] < (-self.agent_height+self.agent_height_tolerance))

        camera_new_to_old = torch.eye(4)

        if self.failed_action is None:
            camera_new_to_old = self._action_to_movement_matrix(last_action)

        self.camera_to_agent = self.camera_to_agent @ camera_new_to_old
        self.lookat_direction = self.camera_to_agent[:3,
                                                     :3] @ torch.tensor([0., 0., 1.])

        points_in_camera = torch.cat(
            (points_in_camera, torch.ones(1, points_in_camera.shape[1])), 0)
        agent_position = self.camera_to_agent[:3, 3]
        agent_voxel = agent_position / self.voxel_size_m
        agent_voxel[1] = self.floor_height_vox
        self.agent_voxel = tuple([v.item() for v in agent_voxel.int()])

        padding_in_camera = self._get_floor_padding_points(
            points_in_camera[:, points_in_camera[1, :] < 0])

        floor_mask = torch.logical_and(points_in_camera[1, :] > (-self.agent_height-self.agent_height_tolerance),
                                       points_in_camera[1, :] < (-self.agent_height+self.agent_height_tolerance))

        points_in_agent = self.camera_to_agent @ points_in_camera
        local_voxels = self._voxelize_points(points_in_agent)
        heights = torch.clone(local_voxels[:, 1])

        # NOTE: uncomment for fig
        local_voxels[:, 1] = self.floor_height_vox

        padding_in_agent = self.camera_to_agent @ padding_in_camera
        padding_voxels = self._voxelize_points(padding_in_agent)
        padding_voxels[:, 1] = self.floor_height_vox

        # NOTE: order of calls important here as state variables modified
        self._reset_dbg_to_free()
        self._reset_frontier()
        self._fill_free(local_voxels[floor_mask])
        self._fill_free(padding_voxels, only_non_empty=True)
        self._fill_occupied(local_voxels, clip_confidence, heights, floor_mask)
        self._fill_frontier()
        self._update_exploration_targets()
        self._update_roi_targets()

        # after all updating make sure that the agent is in a free space
        if self.agent_voxel not in self.voxels.nodes \
            or self.voxels.nodes[self.agent_voxel]["voxel_type"] != VoxelType.FREE:
            self._fill_free(torch.tensor(self.agent_voxel).int().unsqueeze(0))

    def actions_toward_next_frontier(self):
        high_level_path = None
        for target in self.exploration_targets:
            if target in self.visited_frontiers:
                continue
            try:
                high_level_path = nx.astar_path(
                    self.voxels, self.agent_voxel, target, heuristic=self._node_dist, weight="weight")
                break
            except:
                continue

        if high_level_path is None:
            return []

        for n in high_level_path:
            self.voxels.nodes[n]['voxel_type'] = VoxelType.DBG

        curr = self.camera_to_agent[:3, 3]
        curr_lookat = self.lookat_direction

        target = None
        for waypoint in high_level_path:
            tmp = torch.tensor(
                waypoint, dtype=torch.float32) * self.voxel_size_m
            if self._node_dist(curr, tmp) > self.voxel_size_m * 3.5:
                target = tmp
                break
        if target is None:
            self.visited_frontiers.update(
                self._get_neighbors_twentyfour(high_level_path[-1]))
            self.visited_frontiers.add(high_level_path[-1])
            self.exploration_targets.pop(0)
            return self.actions_toward_next_frontier()

        curr[1] = target[1]

        local_expansion = self.low_level_planner(
            curr, curr_lookat, target)

        if local_expansion is None:
            return []

        return local_expansion.value["seq"][1:]

    def action_towards_next_roi(self):
        high_level_path = None
        for true_target, target in self.roi_targets:
            # assert target is not None
            if target is None:
                return ['Stop', ]

            if self._node_dist(self.agent_voxel, true_target) < 1/VOXEL_SIZE_M:
                bearing = np.array(true_target)
                bearing[1] = self.voxels.nodes[true_target]['height']

                agent_point = np.array(self.agent_voxel)
                agent_point[1] = 0

                correct_direction = bearing - agent_point
                correct_direction = correct_direction / np.sqrt(np.sum(correct_direction**2))
                lookat_direction = self.lookat_direction.numpy()
                y_rotation_options = [rot_mat @ lookat_direction for rot_mat in ROTATION_MATRICIES]
                cosines = [np.dot(option, correct_direction) for option in y_rotation_options]
                index1 = np.argmax(cosines)

                seq = list(ACTION_SEQUENCES[index1][0][:-1])

                return seq + ['Stop', ]

        # for true_target, target in self.roi_targets:
            try:
                high_level_path = nx.astar_path(
                    self.voxels, self.agent_voxel, target, heuristic=self._node_dist, weight="weight")
                break
            except:
                continue

        if high_level_path is None:
            for true_target, target in self.roi_targets:
                try:
                    exploration_targets = copy(self.exploration_targets)
                    exploration_targets = sorted(
                        exploration_targets,
                        key=lambda x: self._node_dist(true_target, x),
                    )

                    found_path = False
                    for exploration_target in exploration_targets:
                        try:
                            high_level_path = nx.astar_path(
                                self.voxels, self.agent_voxel, exploration_target, heuristic=self._node_dist, weight="weight")
                            found_path = True
                            break
                        except:
                            continue

                    if found_path:
                        break
                except Exception as e:
                    continue

        # have tried to nav to target directly and to a frontier, both unsuccessfully so give up
        if high_level_path is None:
            return []

        for n in high_level_path:
            self.voxels.nodes[n]['voxel_type'] = VoxelType.DBG

        curr = self.camera_to_agent[:3, 3]
        curr_lookat = self.lookat_direction

        target = None
        for waypoint in high_level_path:
            tmp = torch.tensor(
                waypoint, dtype=torch.float32) * self.voxel_size_m
            if self._node_dist(curr, tmp) > self.voxel_size_m * 3.5:
                target = tmp
                break

        if target is not None:
            curr[1] = target[1]

        if target is None:
            # NOTE: case where there is not path to the target
            return ['RotateLeft', ]

        local_expansion = self.low_level_planner(
            curr, curr_lookat, target)

        if local_expansion is None:
            return []

        return local_expansion.value["seq"][1:]

    def low_level_planner(self, curr, curr_lookat, target, max_tree_depth=50):
        pq = []

        heappush(pq, PQNode(self._node_dist(curr, target).item(),
                 {"seq": ["Start"], "seq_position": [torch.clone(curr)], "position": torch.clone(curr), "lookat": torch.clone(curr_lookat)}))

        visited = set()

        for i in range(max_tree_depth):
            node = None
            try:
                node = deepcopy(heappop(pq))
            except:
                return None

            # expand the neighbors and if valid queue them up
            for seq, (rot_deg, trans_m) in ACTION_SEQUENCES:
                if self.failed_action is not None:
                    if seq[0] == self.failed_action:
                        continue

                value = deepcopy(node.value)
                value["seq"] = value["seq"] + list(seq)

                rot_mat = torch.tensor(R.from_euler(
                    "y", rot_deg, degrees=True).as_matrix(), dtype=torch.float32)

                value["lookat"] = rot_mat @ value["lookat"]
                value["position"] += (value["lookat"] * trans_m)

                subvox = tuple(
                    (value["position"] / self.voxel_size_m).tolist())
                vox = (round(subvox[0]), round(subvox[1]), round(subvox[2]))
                if vox in visited:
                    continue
                visited.add(vox)

                surrounding_ops = (
                    # (ceil, int, floor),
                    # (ceil, int, ceil),
                    # (floor, int, floor),
                    # (floor, int, ceil),
                    (round, round, round),
                )

                valid_position = True
                for op in surrounding_ops:
                    candidate = (op[0](subvox[0]), op[1](
                        subvox[1]), op[2](subvox[2]))

                    if candidate in self.voxels and self.voxels.nodes[candidate]["voxel_type"] in [VoxelType.OCCUPIED, VoxelType.WALL]:
                        valid_position = False

                if not valid_position:
                    continue

                key = self._node_dist(value["position"], target).item()

                if key <= (self.voxel_size_m * 3.5):
                    self.failed_action = None
                    return PQNode(key, value)

                heappush(pq, PQNode(key, value))

            # reset failed action past level 1 depth.
            self.failed_action = None

        return None

    def _action_to_movement_matrix(self, action):
        action_delta = torch.eye(4)

        negation_factor = 1
        if self.negate_action:
            negation_factor = -1

        if action == "RotateLeft":
            action_delta[:3, :3] = torch.tensor(
                R.from_euler("y", negation_factor * self.rotation_deg, degrees=True).as_matrix())
        elif action == "RotateRight":
            action_delta[:3, :3] = torch.tensor(
                R.from_euler("y", negation_factor * -self.rotation_deg, degrees=True).as_matrix())
        elif action == "MoveAhead":
            action_delta[2, 3] = self.forward_dist
        elif action is None:
            pass
        elif action == "TeleportFull":
            pass
        elif action == "Teleport":
            pass
        else:
            raise ValueError("unsupported action type")

        return action_delta

    def _update_exploration_targets(self):
        frontiers = self._cluster_frontiers()
        frontier_means = {k: np.mean(frontiers[k], axis=0) for k in frontiers}
        frontier_diffs = {k: [self._node_dist(
            p, frontier_means[k]) for p in frontiers[k]] for k in frontiers}
        exploration_targets = {k: frontiers[k][np.argmin(
            frontier_diffs[k])] for k in frontiers}

        self.exploration_targets = sorted(
            exploration_targets.values(),
            key=lambda x: self._node_dist(self.agent_voxel, x),
            reverse=self.reverse)

    def _update_roi_targets(self):
        for i in range(len(self.roi_targets)):
            roi, closest_free = self.roi_targets[i]
            none_check = closest_free is None

            existence_check = False
            if not none_check:
                existence_check = closest_free in self.voxels.nodes

            not_free_check = False
            if existence_check:
                not_free_check = self.voxels.nodes[closest_free]["voxel_type"] in [
                    VoxelType.OCCUPIED, VoxelType.WALL]

            if none_check or not_free_check:

                dist = self._node_dist(self.agent_voxel, roi)
                line = tensor_linspace(torch.tensor(roi).float(),
                                       torch.tensor(self.agent_voxel).float(),
                                       ceil(dist*2))
                for j in range(line.shape[1]):
                    candidate = tuple(torch.round(line[:, j]).int().tolist())
                    if candidate in self.voxels.nodes and self.voxels.nodes[candidate]["voxel_type"] not in [VoxelType.OCCUPIED, VoxelType.WALL]:
                        self.roi_targets[i][1] = candidate
                        break

    def _node_dist(self, a, b):
        (x1, _, z1) = a
        (x2, _, z2) = b
        return ((x1 - x2) ** 2 + (z1 - z2) ** 2) ** 0.5

    def _voxelize_points(self, points):
        voxels = torch.round(points[:3, :] / self.voxel_size_m).int()
        voxels = torch.transpose(voxels, 0, 1)

        return voxels

    def _get_floor_padding_points(self, points_in_camera):
        voxels_in_camera = self._voxelize_points(points_in_camera)
        voxels_in_camera[:, 1] = self.floor_height_vox
        voxels_in_camera = torch.cat(
            (voxels_in_camera, torch.tensor([list(self.agent_voxel)])), dim=0)

        voxels = torch.unique(voxels_in_camera, dim=0)
        min_z = {}

        for i in range(voxels.shape[0]):
            if voxels[i][0].item() not in min_z:
                min_z[voxels[i][0].item()] = voxels[i].float()
            else:
                if voxels[i][2].item() < min_z[voxels[i][0].item()][2]:
                    min_z[voxels[i][0].item()] = voxels[i].float()

        step = 0.25
        interpolated_points = None
        for k in min_z:
            dist = torch.norm(min_z[k])
            num_steps = torch.ceil(dist/step).int()
            padding = tensor_linspace(torch.tensor(
                [0., 0., 0.]), min_z[k], num_steps)
            if interpolated_points is None:
                interpolated_points = padding
            else:
                interpolated_points = torch.cat(
                    (interpolated_points, padding), 1)

        interpolated_points *= self.voxel_size_m
        interpolated_points[1, :] = -self.agent_height
        interpolated_points = torch.cat(
            (interpolated_points, torch.ones(1, interpolated_points.shape[1])), 0)

        return interpolated_points

    def _cluster_frontiers(self):
        """Runs agglomerative clustering on frontiers
        """
        dust = []
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FRONTIER:
                count = 0
                neighbors, _ = self._get_neighbors_eight(v)
                for n in neighbors:
                    if n in self.voxels and self.voxels.nodes[n]["voxel_type"] == VoxelType.FREE:
                        count += 1

                if count == 0 and v != self.agent_voxel:
                    dust.append(v)

        for d in dust:
            self.voxels.remove_node(d)

        candidates = []
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FRONTIER:
                candidates.append(v)

        frontiers = {}

        if len(candidates) < 3:
            for v in candidates:
                self.voxels.nodes[v]["voxel_type"] = VoxelType.FREE
            return frontiers

        clustering = AgglomerativeClustering(
            n_clusters=None,
            linkage="single",
            distance_threshold=1.5
        ).fit(candidates)

        for i, c in enumerate(clustering.labels_):
            if c in frontiers:
                frontiers[c].append(candidates[i])
            else:
                frontiers[c] = [candidates[i]]

        return frontiers

    def _fill_occupied(self, local_voxels, conf, heights, mask):

        if local_voxels is None:
            return

        local_vox_unique = None
        inverse_indices = None
        conf_masked = None
        heights_masked = None

        if mask is not None:
            local_vox_unique, inverse_indices = torch.unique(local_voxels[~mask], dim=0, return_inverse=True)
            conf_masked = conf[~mask]
            heights_masked = heights[~mask]
        else:
            local_vox_unique, inverse_indices = torch.unique(local_voxels, dim=0, return_inverse=True)
            conf_masked = conf
            heights_masked = heights

        # NOTE: this throws for some reason in some cases: IndexError: too many indices for tensor of dimension 1
        #       debug after deadline

        groups = []
        local_voxels_unique = local_vox_unique.int()
        for i in range(local_voxels_unique.shape[0]):
            k1 = tuple(local_voxels_unique[i, :].tolist())

            inv_mask = inverse_indices == i

            conf_inv_mask = conf_masked[inv_mask]
            height_inv_mask = heights_masked[inv_mask]

            j = torch.argmax(conf_inv_mask)
            k2 = conf_inv_mask[j]
            # j = torch.argmax(height_inv_mask)
            k3 = height_inv_mask[j]

            groups.append((k1, k2, k3))

        observation_rois = []
        for v_tuple, v_conf, v_height in groups:

            if v_tuple in self.voxels.nodes:
                if self.voxels.nodes[v_tuple]["voxel_type"] == VoxelType.WALL:  # or\
                    # self.voxels.nodes[v_tuple]["voxel_type"] == VoxelType.ROI:
                    continue

            self._add_node_conditional(
                v_tuple, v_conf.item(), v_height.item(), VoxelType.OCCUPIED)

            # now check if we have an ROI so that we can switch to exploit mode
            if self.voxels.nodes[v_tuple]["roi_count"] > 0:
                observation_rois.append(v_tuple)

            for edge in list(self.voxels.edges(v_tuple)):
                self.voxels.remove_edge(*edge)

            if self.in_cspace:
                neighbors, _ = self._get_neighbors_four(v_tuple)
                for vc_tuple in neighbors:
                    if vc_tuple in self.voxels.nodes:
                        if self.voxels.nodes[vc_tuple]["voxel_type"] == VoxelType.WALL:  # or\
                            # self.voxels.nodes[vc_tuple]["voxel_type"] == VoxelType.ROI:
                            continue

                    self._add_node_conditional(
                        vc_tuple, None, self.floor_height_vox, VoxelType.OCCUPIED)

                    for edge in list(self.voxels.edges(vc_tuple)):
                        self.voxels.remove_edge(*edge)

        if len(observation_rois):
            # sort in order of most seen roi, break ties with euclidean distance to the point.
            # grab first entry as the target
            observation_roi_target = sorted(observation_rois, key=lambda x: (
                self._node_dist(self.agent_voxel, x), -self.voxels.nodes[x]["roi_count"]), reverse=False)[0]
            exists = [k1 for k1, _ in self.roi_targets]
            if observation_roi_target not in exists:
                self.roi_targets.append([observation_roi_target, None])
                self.roi_targets = sorted(self.roi_targets, key=lambda x: self._node_dist(self.agent_voxel, x[0]), reverse=False)

    def _fill_free(self, local_voxels, only_non_empty=False):
        edges_to_add = {}
        local_voxels_unique = torch.unique(local_voxels, dim=0)
        for v in local_voxels_unique:
            v_tuple = tuple(v.tolist())
            if only_non_empty and v_tuple in self.voxels.nodes:
                continue
            # if v_tuple in self.voxels.nodes and self.voxels.nodes[v_tuple]["voxel_type"] == VoxelType.ROI:
            #     continue

            edges_to_add[v_tuple] = self._get_neighbors_eight(v_tuple)

            self._add_node_conditional(v_tuple, 0., self.floor_height_vox, VoxelType.FREE)

        if self.agent_voxel not in self.voxels.nodes:
            edges_to_add[self.agent_voxel] = self._get_neighbors_eight(
                self.agent_voxel)
            self._add_node_conditional(self.agent_voxel, 0., self.floor_height_vox, VoxelType.FREE)

        for src in edges_to_add:
            for i in range(len(edges_to_add[src][0])):
                sink = edges_to_add[src][0][i]
                edge_weight = edges_to_add[src][1][i]

                if sink in self.voxels and self.voxels.nodes[sink]["voxel_type"] == VoxelType.FREE:
                    self.voxels.add_edge(src, sink, weight=edge_weight)

    def _reset_frontier(self):
        """Sets all frontier space to free space.
        """
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FRONTIER:
                self.voxels.nodes[v]["voxel_type"] = VoxelType.FREE

    def _reset_dbg_to_free(self):
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.DBG:
                self.voxels.nodes[v]["voxel_type"] = VoxelType.FREE

    def _fill_frontier(self):
        """Looks for the edge of free space and unknown to set fontier.
        """
        for v in self.voxels:
            if self.voxels.nodes[v]["voxel_type"] == VoxelType.FREE:
                count = 0
                neighbors, _ = self._get_neighbors_four(v)
                for n in neighbors:
                    if n in self.voxels:
                        count += 1
                if count != 4:
                    self.voxels.nodes[v]["voxel_type"] = VoxelType.FRONTIER

    def _get_neighbors_four(self, voxel_tuple: Tuple[float]) -> List[Tuple[float]]:
        """Gets four (plus) neighbors of a voxel in xz space.

        Args:
            voxel_tuple (Tuple[float]): center voxel.

        Returns:
            List[Tuple[float]]: neighbors
        """
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        edge_costs = (1., 1., 1., 1.)
        nodes = []
        for o in offsets:
            nodes.append(
                (voxel_tuple[0] + o[0], voxel_tuple[1], voxel_tuple[2] + o[1]))

        return nodes, edge_costs

    def _get_neighbors_eight(self, voxel_tuple: Tuple[float]) -> List[Tuple[float]]:
        """Gets eight (cross and plus) neighbors of a voxel in xz space.

        Args:
            voxel_tuple (Tuple[float]): center voxel.

        Returns:
            List[Tuple[float]]: neighbors
        """
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (-1, 1), (-1, -1), (1, -1))
        edge_costs = (1., 1., 1., 1., 1.41, 1.41, 1.41, 1.41)

        nodes = []
        for o in offsets:
            nodes.append(
                (voxel_tuple[0] + o[0], voxel_tuple[1], voxel_tuple[2] + o[1]))

        return nodes, edge_costs

    def _get_neighbors_twentyfour(self, voxel_tuple: Tuple[float]) -> List[Tuple[float]]:
        """Gets eight (cross and plus) neighbors of a voxel in xz space.

        Args:
            voxel_tuple (Tuple[float]): center voxel.

        Returns:
            List[Tuple[float]]: neighbors
        """
        nodes = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i == 0 and j == 0:
                    continue
                nodes.append(
                    (voxel_tuple[0] + i, voxel_tuple[1], voxel_tuple[2] + j))

        return nodes

    def _add_node_conditional(self, voxel_tuple, obj_conf, height, type):

        is_roi = False
        if obj_conf is not None:
            is_roi = obj_conf > 0.9

        if voxel_tuple not in self.voxels:
            self.voxels.add_node(voxel_tuple,
                                 voxel_type=type,
                                 obj_conf=obj_conf,
                                 roi_count=int(is_roi),
                                 height=height)
        else:
            self.voxels.nodes[voxel_tuple]["voxel_type"] = type
            if self.voxels.nodes[voxel_tuple]["obj_conf"] is None:
                self.voxels.nodes[voxel_tuple]["obj_conf"] = obj_conf
                self.voxels.nodes[voxel_tuple]["height"] = height
            else:
                if obj_conf is not None:
                    if obj_conf >= self.voxels.nodes[voxel_tuple]["obj_conf"]:
                        self.voxels.nodes[voxel_tuple]["obj_conf"] = obj_conf
                        self.voxels.nodes[voxel_tuple]["height"] = height

            self.voxels.nodes[voxel_tuple]["roi_count"] += int(is_roi)
