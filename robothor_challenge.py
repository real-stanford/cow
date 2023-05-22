# modified from https://github.com/allenai/robothor-challenge/blob/main/robothor_challenge/challenge.py
import glob
import gzip
import json
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from copy import deepcopy
from typing import Any, Dict

import ai2thor.controller
import ai2thor.util.metrics
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from ai2thor.util import metrics
from ai2thor.util.metrics import compute_single_spl

from scripts.startx import startx
from src.shared.utils import seed_everything
from src.simulation.sim_enums import ClassTypes, EnvTypes
from src.simulation.utils import get_device

torch.multiprocessing.set_start_method('spawn', force=True)


logger = logging.getLogger(__name__)
ch = logging.StreamHandler(sys.stdout)
ch.flush = sys.stdout.flush
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

ALLOWED_ACTIONS = ["MoveAhead", "RotateLeft",
                   "RotateRight", "Stop", "LookUp", "LookDown"]


def get_object_by_type(event_objects, object_type):
    for obj in event_objects:
        if obj['objectId'].split("|")[0] == object_type:
            return obj
    return None

def get_object_by_id(event_objects, object_id):
    for obj in event_objects:
        if obj['objectId'] == object_id:
            return obj
    return None

def query_instance_segmentation_for_object(instance_masks, object_type, centroid=True):
    object_mask = None
    for oid in instance_masks:
        if object_mask is None:
            object_mask = np.zeros_like(instance_masks[oid])

        if oid.split("|")[0] == object_type:
            object_mask = np.logical_or(object_mask, instance_masks[oid])

    assert object_mask is not None

    # coords = np.where(object_mask)

    if np.sum(object_mask) < 0.5:
        return object_mask

    if centroid:
        us, vs = np.where(object_mask)
        mean_u = np.mean(us)
        mean_v = np.mean(vs)

        index = np.argmin((us - mean_u)**2 + (vs - mean_v)**2, axis=None)

        object_mask = np.zeros_like(object_mask)
        object_mask[us[index], vs[index]] = 1

    return object_mask

def query_instance_segmentation_for_object_id(instance_masks, object_id, centroid=True):
    object_mask = None
    for oid in instance_masks:
        if object_mask is None:
            object_mask = np.zeros_like(instance_masks[oid])

        if oid == object_id:
            object_mask = np.logical_or(object_mask, instance_masks[oid])
            break

    assert object_mask is not None

    if np.sum(object_mask) < 0.5:
        return object_mask

    if centroid:
        us, vs = np.where(object_mask)
        mean_u = np.mean(us)
        mean_v = np.mean(vs)

        index = np.argmin((us - mean_u)**2 + (vs - mean_v)**2, axis=None)

        object_mask = np.zeros_like(object_mask)
        object_mask[us[index], vs[index]] = 1

    return object_mask

def path_from_point_to_object_type(
        controller: ai2thor.controller.Controller, point: Dict[str, float], object_type: str, allowed_error: float
    ):
        event = controller.step(
            action="GetShortestPath",
            objectType=object_type,
            position=point,
            allowedError=allowed_error,
        )
        if event.metadata["lastActionSuccess"]:
            return event.metadata["actionReturn"]["corners"]
        else:
            return None

def distance_from_point_to_object_type(
        controller: ai2thor.controller.Controller, point: Dict[str, float], object_type: str, allowed_error: float
    ) -> float:
        """Minimal geodesic distance from a point to an object of the given
        type.
        It might return inf for unreachable targets.
        """
        path = path_from_point_to_object_type(controller, point, object_type, allowed_error)
        print(path)
        if path:
            # Because `allowed_error != 0` means that the path returned above might not start
            # at `point`, we explicitly add any offset there is.
            s_dist = math.sqrt(
                (point["x"] - path[0]["x"]) ** 2 + (point["z"] - path[0]["z"]) ** 2
            )
            return metrics.path_distance(path) + s_dist
        return float('inf')

class RobothorChallenge:

    def __init__(self, cfg_file, agent_class, agent_kwargs, experiment_name, no_grad, env_type, class_type, render_depth=False, render_segmentation=False, class_remap=None, seed=0):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs

        self.config = self.load_config(cfg_file, render_depth)

        self.setup_env()

        if render_segmentation:
            self.config['initialize']['renderInstanceSegmentation'] = True

        self.controller_kwargs = {
            "width": self.config["width"],
            "height": self.config["height"],
            **self.config["initialize"]
        }

        if env_type == EnvTypes.ROBOTHOR:
            self.controller_kwargs["commit_id"] = self.config["thor_build_id"]
        elif env_type == EnvTypes.NORMAL:
            self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_normal/normal.x86_64'
        elif env_type == EnvTypes.DUP:
            self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_dup/dup.x86_64'
        elif env_type == EnvTypes.REMOVE:
            self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_remove/remove.x86_64'
        elif env_type == EnvTypes.LONGTAIL:
            self.controller_kwargs["local_executable_path"] = './pasture_builds/thor_build_longtail/longtail.x86_64'
        else:
            raise ValueError('unsupported env_type')

        self.env_type = env_type
        self.class_type = class_type

        self.current_scene = None
        self.reachable_positions_per_scene = {}
        self.class_remap = class_remap
        self.experiment_name = experiment_name
        self.seed = seed
        self.no_grad = no_grad

        if not os.path.exists(self.experiment_name):
            os.mkdir(self.experiment_name)

    @staticmethod
    def load_config(cfg_file, render_depth):
        logger.info("Loading configuration from: %s" % cfg_file)
        with open(cfg_file, "r") as f:
            config = yaml.safe_load(f.read())
        if render_depth:
            config["initialize"]["renderDepthImage"] = True
        return config

    @staticmethod
    def setup_env():
        if "DISPLAY" not in os.environ:
            xthread = threading.Thread(target=startx)
            xthread.daemon = True
            xthread.start()
            import time

            # XXX change this to use xdpyinfo
            time.sleep(4)

    @staticmethod
    def load_split(dataset_dir, split):
        split_paths = os.path.join(dataset_dir, split, "episodes", "*.json.gz")
        split_paths = sorted(glob.glob(split_paths))

        episode_list = []
        dataset = {}

        for split_path in split_paths:
            logger.info("Loading: {path}".format(path=split_path))

            with gzip.GzipFile(split_path, "r") as f:
                episodes = json.loads(f.read().decode("utf-8"))

                # Build a dictionary of the dataset indexed by scene, object_type
                curr_scene = None
                curr_object = None
                points = []
                scene_points = {}
                for data_point in episodes:

                    if curr_object != data_point["object_type"]:
                        scene_points[curr_object] = points
                        curr_object = data_point["object_type"]
                        points = []
                    if curr_scene != data_point["scene"]:
                        dataset[curr_scene] = scene_points
                        curr_scene = data_point["scene"]
                        scene_points = {}

                    points.append(data_point)

                episode_list += episodes

        return episode_list, dataset

    @staticmethod
    def inference_worker(
        worker_ind: int,
        in_queue: mp.Queue,
        out_queue: mp.Queue,
        agent_class: Any,
        agent_kwargs: Dict[str, Any],
        controller_kwargs: Dict[str, Any],
        max_steps: int,
        test: bool,
        device: Any,
        class_remap: Dict,
        experiment_name: str,
        no_grad: bool,
        seed: int,
        env_type: EnvTypes,
        class_type: ClassTypes,
    ):
        seed_everything(seed)
        agent_kwargs['device'] = device
        agent = agent_class(**agent_kwargs)
        controller = ai2thor.controller.Controller(**controller_kwargs)
        assert controller.step(
            'SetRandomSeed', seed=seed).metadata["lastActionSuccess"]

        while True:
            try:
                e = in_queue.get(timeout=1)
            except queue.Empty:
                break

            logger.info(
                "Task Start id:{id} scene:{scene} target_object:{object_type} initial_position:{initial_position} rotation:{initial_orientation}".format(**e))
            controller.initialization_parameters["robothorChallengeEpisodeId"] = e["id"]
            print(e["scene"], e["object_type"])
            controller.reset(e["scene"])
            event = controller.step(
                'SetRandomSeed', seed=seed)
            assert event.metadata["lastActionSuccess"]

            if 'object_id' in e:
                controller.step('SetObjectFilter', objectIds=[e['object_id'],])
            else:
                object_id = get_object_by_type(event.metadata["objects"], e["object_type"])
                controller.step('SetObjectFilter', objectIds=[object_id,])

            if class_remap is not None:
                agent.clip_module.remap_classes(class_remap[e["scene"]])

            teleport_action = {
                "action": "TeleportFull",
                **e["initial_position"],
                "rotation": {"x": 0, "y": e["initial_orientation"], "z": 0},
                "horizon": 0,
            }
            controller.step(action=teleport_action)

            total_steps = 0
            agent.reset()

            episode_metrics = {
                "trajectory": [{
                    **e["initial_position"],
                    "rotation": float(e["initial_orientation"]),
                    "horizon": 0,
                }],
                "actions_taken": []
            }

            stopped = False
            # NOTE: uncomment for fig
            # while total_steps < 13 and stopped is False:
            vision_error_in_case_of_fail = False
            while total_steps < max_steps and stopped is False:

                if env_type == EnvTypes.ROBOTHOR or env_type == EnvTypes.LONGTAIL:
                    target_obj = get_object_by_type(
                        event.metadata["objects"], e["object_type"])
                else:
                    target_obj = get_object_by_id(
                        event.metadata["objects"], e["object_id"])

                assert target_obj is not None
                target_visible = target_obj["visible"]
                if target_visible:
                    vision_error_in_case_of_fail = True

                total_steps += 1
                event = controller.last_event

                observations = {
                    "object_goal": e["object_type"],
                    "depth": np.copy(event.depth_frame),
                    "rgb": np.copy(event.frame),
                }

                if 'renderInstanceSegmentation' in controller_kwargs and controller_kwargs['renderInstanceSegmentation']:
                    if env_type == EnvTypes.ROBOTHOR or env_type == EnvTypes.LONGTAIL:
                        observations['target_mask'] = query_instance_segmentation_for_object(event.instance_masks, e["object_type"])
                    else:
                        observations['target_mask'] = query_instance_segmentation_for_object_id(event.instance_masks, e["object_id"])

                action = None
                if no_grad:
                    with torch.no_grad():
                        action = agent.act(observations)
                else:
                    action = agent.act(observations)

                if action not in ALLOWED_ACTIONS:
                    raise ValueError(
                        "Invalid action: {action}".format(action=action))

                event = controller.step(action=action)
                episode_metrics["trajectory"].append({
                    **event.metadata["agent"]["position"],
                    "rotation": event.metadata["agent"]["rotation"]["y"],
                    "horizon": event.metadata["agent"]["cameraHorizon"]
                })
                episode_metrics["actions_taken"].append({
                    "action": action,
                    "success": event.metadata["lastActionSuccess"]
                })
                stopped = action == "Stop"

                # NOTE: uncomment for fig
                # agent.fbe.dump_voxel_pointcloud(
                #     f'tmp/{agent.timesteps-1}.ply')
                # Image.fromarray(observations["rgb"]).save(
                #     f'tmp/{agent.timesteps-1}.png')
                # with open(f'tmp/{agent.timesteps-1}_info.json', 'w') as f:
                #     json.dump(agent.debug_data, f, indent=4)

            if not test:
                target_obj = None

                if env_type == EnvTypes.ROBOTHOR or env_type == EnvTypes.LONGTAIL:
                    target_obj = get_object_by_type(
                        event.metadata["objects"], e["object_type"])
                else:
                    target_obj = get_object_by_id(
                        event.metadata["objects"], e["object_id"])

                assert target_obj is not None
                target_visible = target_obj["visible"]
                episode_metrics["success"] = stopped and target_visible
                episode_metrics["vision_error_in_case_of_fail"] = vision_error_in_case_of_fail

            if not test:
                episode_result = {
                    "path": episode_metrics["trajectory"],
                    "shortest_path": e["shortest_path"],
                    "success": episode_metrics["success"],
                }
            else:
                episode_result = None

            with open(f'{experiment_name}/{e["id"]}.json', 'w') as f:
                json.dump(
                    {
                        "episode_metrics": episode_metrics,
                        "episode_result": episode_result,
                    },
                    f
                )

            controller.step('ResetObjectFilter')

            out_queue.put((e["id"], episode_metrics, episode_result))

        controller.stop()
        print(f"Worker {worker_ind} Finished.")

    def inference(self, episodes, nprocesses=1, test=False):
        send_queue = mp.Queue()
        receive_queue = mp.Queue()

        expected_count = len(episodes)

        # randomize evaluation to get a better signal of what is going on
        random.shuffle(episodes)

        for e in episodes:
            send_queue.put(e)

        processes = []
        num_devices = torch.cuda.device_count()
        for worker_ind in range(nprocesses):
            class_remap = None
            if self.class_remap is not None:
                class_remap = deepcopy(self.class_remap)
            thread_controller_kwargs = deepcopy(self.controller_kwargs)
            thread_controller_kwargs['gpu_device'] = worker_ind % num_devices
            thread_controller_kwargs['x_display'] = f':0.{worker_ind % num_devices}'

            p = mp.Process(
                target=self.inference_worker,
                kwargs=dict(
                    worker_ind=worker_ind,
                    in_queue=send_queue,
                    out_queue=receive_queue,
                    agent_class=self.agent_class,
                    agent_kwargs=self.agent_kwargs,
                    controller_kwargs=thread_controller_kwargs,
                    max_steps=self.config["max_steps"],
                    test=test,
                    device=get_device(worker_ind % num_devices),
                    class_remap=class_remap,
                    experiment_name=self.experiment_name,
                    no_grad=self.no_grad,
                    seed=self.seed,
                    env_type=self.env_type,
                    class_type=self.class_type,
                ),
            )
            p.start()
            processes.append(p)
            time.sleep(0.2)

        metrics = {"episodes": {}}
        episode_results = []

        while len(metrics["episodes"]) < expected_count:
            try:
                ep_id, episode_metrics, episode_result = receive_queue.get(
                    timeout=1000)
                metrics["episodes"][ep_id] = episode_metrics
                if not test:
                    episode_results.append(episode_result)
            except TimeoutError:
                print("Went 1000 seconds without a new episode result.")
                if all(not p.is_alive() for p in processes):
                    try:
                        ep_id, episode_metrics, episode_result = receive_queue.get(
                            timeout=1)
                        metrics["episodes"][ep_id] = episode_metrics
                        if not test:
                            episode_results.append(episode_result)
                    except TimeoutError:
                        raise RuntimeError(
                            "All processes dead but nothing in queue!")

        for p in processes:
            p.join(timeout=2)

        if len(metrics["episodes"]):
            metrics["ep_len"] = sum(
                [len(em["trajectory"]) for em in metrics["episodes"].values()]) / len(metrics["episodes"])

            if not test:
                metrics["success"] = sum(
                    [r["success"] for r in episode_results]) / len(episode_results)
                metrics["spl"] = ai2thor.util.metrics.compute_spl(episode_results)

            if not test:
                logger.info("Total Episodes: {episode_count} Success:{success} SPL:{spl} Episode Length:{ep_len}".format(
                    episode_count=len(episodes), success=metrics["success"], spl=metrics["spl"], ep_len=metrics["ep_len"]))
            else:
                logger.info("Total Episodes: {episode_count} Episode Length:{ep_len}".format(
                    episode_count=len(episodes), ep_len=metrics["ep_len"]))

            return metrics

    def _change_scene(self, scene, seed):
        if self.current_scene != scene:
            self.current_scene = scene
            self.controller.reset(scene)
            assert self.controller.step(
                'SetRandomSeed', seed=seed).metadata["lastActionSuccess"]

            logger.info("Changed to scene: '{scene}'".format(scene=scene))

    def move_to_point(self, datapoint, seed):
        self._change_scene(datapoint["scene"], seed)
        logger.info("Moving to position: {p}, y-rotation: {rot}, horizon: {hor}".format(
            p=datapoint["initial_position"],
            rot=datapoint["initial_orientation"],
            hor=0  # datapoint["initial_horizon"]
        ))
        return self.controller.step(
            action="TeleportFull",
            x=datapoint["initial_position"]["x"],
            y=datapoint["initial_position"]["y"],
            z=datapoint["initial_position"]["z"],
            rotation={"x": 0, "y": datapoint["initial_orientation"], "z": 0},
            horizon=0,
        )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
