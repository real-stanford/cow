import argparse
import copy
import hashlib
import importlib
import json
import logging
import os
import sys
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import torch
import torch.multiprocessing as mp
from allenact_plugins.habitat_plugin.habitat_environment import \
    HabitatEnvironment

import habitat
from src.allenact_experiments.shared.objectnav_habitat_base_config import \
    ObjectNavHabitatBaseConfig
from src.shared.utils import seed_everything
from src.simulation.constants import HABITAT_OBJECT_TYPES, VOXEL_SIZE_M
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

def get_instance_to_semantic_map(scene):
    instance_to_semantic = {}
    semantic_to_instance = {}

    for level in scene.levels:
        for region in level.regions:
            for obj in region.objects:
                if obj.category.name() in HABITAT_OBJECT_TYPES:
                    instance_to_semantic[int(obj.id.split('_')[-1])] = obj.category.name()
                    if obj.category.name() in semantic_to_instance:
                        semantic_to_instance[obj.category.name()].append(int(obj.id.split('_')[-1]))
                    else:
                        semantic_to_instance[obj.category.name()] = [int(obj.id.split('_')[-1]),]

    return instance_to_semantic, semantic_to_instance

def query_instance_segmentation_for_object(instance_masks, object_type, centroid=False):
    object_mask = np.isin(instance_masks, object_type)

    # NOTE: tuned and 200 seems reasonable to max perf for GT oracle baseline
    if np.sum(object_mask) < 200:
        return np.zeros_like(object_mask)

    if centroid:
        us, vs = np.where(object_mask)
        mean_u = np.mean(us)
        mean_v = np.mean(vs)

        index = np.argmin((us - mean_u)**2 + (vs - mean_v)**2, axis=None)

        object_mask = np.zeros_like(object_mask)
        object_mask[us[index], vs[index]] = 1

    return object_mask

class HabitatChallenge:

    def __init__(self,
                 habitat_config,
                 agent_class,
                 agent_kwargs,
                 experiment_name,
                 nprocesses,
                 fail_stop,
                 run_exploration_split,
                 render_depth=False,
                 class_remap=None,
                 no_grad=False,
                 gt_localization=False,
                 seed=0):

        if not os.path.exists(experiment_name):
            os.mkdir(experiment_name)

        num_devices = torch.cuda.device_count()
        print(f'num devices: {num_devices}')

        dataset = habitat.make_dataset(
            habitat_config.DATASET.TYPE, config=habitat_config.DATASET)

        cache = set()
        for stuff in os.listdir(experiment_name):
            cache.add(stuff.split('.')[0])

        configs = []

        exploration_keys = set()
        if run_exploration_split:
            with open('habitat_exploration_episode_keys.json', 'r') as f:
                exploration_keys = json.load(f)
                exploration_keys = set(exploration_keys)

        if len(exploration_keys):
            new_episodes = []
            for e in dataset.episodes:
                env_id = e.scene_id.split('/')[-2]
                position_id = f'{e.start_position[0]}|{e.start_position[1]}|{e.start_position[2]}'
                rotation_id = f'{e.start_rotation[0]}|{e.start_rotation[1]}|{e.start_rotation[2]}|{e.start_rotation[3]}'
                key = hashlib.md5(f'{env_id}_{position_id}_{rotation_id}'.encode('utf-8')).hexdigest()

                if key in exploration_keys:
                    new_episodes.append(e)
            dataset = copy.copy(dataset)
            dataset.episodes = new_episodes

            print(len(dataset.episodes))


        dataset_splits = dataset.get_splits(
            nprocesses, allow_uneven_splits=True)

        for i, d in enumerate(dataset_splits):

            dn = i % num_devices

            shard = deepcopy(habitat_config)
            shard.defrost()
            shard.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = dn
            shard.freeze()

            configs.append(
                {
                    "habitat_config": shard,
                    "dataset_split": d,
                    "seed": seed,
                    "agent_class": agent_class,
                    "agent_kwargs": deepcopy(agent_kwargs),
                    "experiment_name": experiment_name,
                    "device_num": dn,
                    "cache": deepcopy(cache),
                    "fail_stop": fail_stop,
                    "exploration_keys": exploration_keys,
                    "no_grad": no_grad,
                    "gt_localization": gt_localization,
                }
            )

        # create thread pool
        pool = Pool(processes=nprocesses)
        pool.map(self.worker, configs)

    def worker(self, config):
        habitat_config = config['habitat_config']
        dataset = config['dataset_split']
        seed = config['seed']
        agent_class = config['agent_class']
        agent_kwargs = config['agent_kwargs']
        experiment_name = config['experiment_name']
        device_num = config['device_num']
        cache = config['cache']
        no_grad = config['no_grad']

        env = HabitatEnvironment(config=habitat_config, dataset=dataset)

        seed_everything(seed)
        agent_kwargs['device'] = get_device(device_num)
        agent = agent_class(**agent_kwargs)
        num_episodes = env.num_episodes

        assert num_episodes > 0, "num_episodes should be greater than 0"

        count_episodes = 0

        while count_episodes < num_episodes:

            agent.reset()
            env.reset()
            observations = env.current_frame
            observations['object_goal'] = HABITAT_OBJECT_TYPES[observations['objectgoal'].item()]

            semantic_to_instance = None
            if config['gt_localization']:
                _, semantic_to_instance = get_instance_to_semantic_map(env.env.sim.semantic_scene)
                if observations['object_goal'] not in semantic_to_instance:
                    raise ValueError(f"{observations['object_goal']} not in the map")

                observations['target_mask'] = query_instance_segmentation_for_object(observations['semantic'], semantic_to_instance[observations['object_goal']])

            env.env.seed(seed)

            curr_ep = env.get_current_episode()

            env_id = curr_ep.scene_id.split('/')[-2]
            position_id = f'{curr_ep.start_position[0]}|{curr_ep.start_position[1]}|{curr_ep.start_position[2]}'
            rotation_id = f'{curr_ep.start_rotation[0]}|{curr_ep.start_rotation[1]}|{curr_ep.start_rotation[2]}|{curr_ep.start_rotation[3]}'

            q = env.get_rotation()
            curr_position = env.get_location()
            curr_orientation = np.array([q.w, q.x, q.y, q.z])

            state = (np.round(curr_position[0]/VOXEL_SIZE_M).astype(int).item(), np.round(curr_position[2]/VOXEL_SIZE_M).astype(int).item())
            visited_states = set()
            visited_states.add(state)
            action_success_count = 0
            total_actions = 0

            key = hashlib.md5(f'{env_id}_{position_id}_{rotation_id}'.encode('utf-8')).hexdigest()

            actions = []
            error = ''

            if f'{observations["object_goal"]}_{key}' in cache:
                count_episodes += 1
                continue
            try:
                while not env.env.episode_over:
                    action = None

                    if no_grad:
                        with torch.no_grad():
                            action = agent.act(observations)
                    else:
                        action = agent.act(observations)

                    action_mapping = {
                        "Stop": 0,
                        "MoveAhead": 1,
                        "RotateLeft": 2,
                        "RotateRight": 3,
                    }
                    actions.append(action)
                    action_dict = {"action": action_mapping[action]}

                    observations = env.step(action_dict)
                    observations['object_goal'] = HABITAT_OBJECT_TYPES[observations['objectgoal'].item()]

                    if config['gt_localization']:
                        observations['target_mask'] = query_instance_segmentation_for_object(observations['semantic'], semantic_to_instance[observations['object_goal']])

                    q = env.get_rotation()
                    next_position = env.get_location()
                    next_orientation = np.array([q.w, q.x, q.y, q.z])

                    state = (np.round(next_position[0]/VOXEL_SIZE_M).astype(int).item(), np.round(next_position[2]/VOXEL_SIZE_M).astype(int).item())
                    visited_states.add(state)

                    if np.all(np.isclose(next_position, curr_position)) and np.all(np.isclose(next_orientation, curr_orientation)):
                        pass
                    else:
                        action_success_count += 1
                    total_actions += 1

                    curr_position = next_position
                    curr_orientation = next_orientation

            except Exception as e:
                print(e)
                error = str(e)

            metrics = env.env.get_metrics()
            agg_metrics = {}
            for m, v in metrics.items():
                agg_metrics[m] = v
            agg_metrics['successful_actions'] = action_success_count / \
                total_actions
            count_episodes += 1
            agg_metrics['env'] = env_id
            agg_metrics['start_position'] = curr_ep.start_position
            agg_metrics['start_rotation'] = curr_ep.start_rotation
            agg_metrics['visited_state'] = len(visited_states)
            agg_metrics['total_actions'] = total_actions
            agg_metrics['error'] = error
            agg_metrics['actions'] = actions

            if os.path.exists(f'{experiment_name}/{observations["object_goal"]}_{key}.json'):
                print('repeated episode in dataset:')
                print(f'{experiment_name}/{observations["object_goal"]}_{key}.json')
            with open(f'{experiment_name}/{observations["object_goal"]}_{key}.json', 'w') as f:
                json.dump(agg_metrics, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Inference script for Habitat MP3D ObjectNav challenge.")

    parser.add_argument(
        "--agent", "-a",
        required=True,
        help="Relative module for agent definition (e.g., src.models.agent_fbe_patch).",
    )

    parser.add_argument(
        "--template", "-t",
        required=False,
        default="prompt_templates/imagenet_template.json",
        help="Prompt template json (e.g., prompt_templates/imagenet_template.json).",
    )

    parser.add_argument(
        "--cfg", "-c",
        default="config_cow.yaml",
        help="Filepath to challenge config.",
    )

    parser.add_argument(
        "--nprocesses", "-n",
        default=1,
        type=int,
        help="Number of parallel processes used to compute inference.",
    )

    parser.add_argument(
        "--model-path", "-p",
        default='',
        help="Path to exploration agent checkpoint in cases of using learned models.",
    )

    parser.add_argument(
        "-nfs",
        default=False,
        action='store_true',
        help="nfs stands for no failure stopping, passing this flag results in a cow not employing a depth heursitic to check for failed action.",
    )

    parser.add_argument(
        "--depth",
        default=False,
        action='store_true',
        help="Flag to run a depth ONLY cow.",
    )

    parser.add_argument(
        "--semantic",
        default=False,
        action='store_true',
        help="Retain GT semantic labels, used for a cow with GT vision.",
    )

    parser.add_argument(
        "-exp",
        default=False,
        action='store_true',
        help="Run exploration only (not target driven)."
    )

    parser.add_argument(
        "--gpt",
        default=False,
        action='store_true',
        help="Support for a GPT-3 class mapping"
    )

    parser.add_argument(
        '--arch',
        default='B32',
        action='store',
        type=str,
        help='Architecture for the cow. Supported: B32, B16, L14'
    )

    parser.add_argument(
        '--center',
        default=False,
        action='store_true',
        help='Use only the center location of a localized region as a target.'
    )

    args = parser.parse_args()

    no_grad = True
    if 'grad' in args.agent:
        no_grad = False

    agent_type_to_explore_localize = {
        'src.models.agent_fbe_patch': ['fbe', 'patch'],
        'src.models.agent_fbe_grad': ['fbe', 'grad'],
        'src.models.agent_fbe_lang': ['fbe', 'lang'],
        'src.models.agent_fbe_owl': ['fbe', 'owl'],
        'src.models.agent_fbe_owl_subgoal': ['fbe', 'owl'],
        'src.models.agent_fbe_gt': ['fbe', 'gt'],
    }
    assert args.agent in agent_type_to_explore_localize

    assert args.arch in ['B32', 'B16', 'L14']

    clip_model_name = None
    if args.arch == 'B32':
        clip_model_name = 'ViT-B/32'
    elif args.arch == 'B16':
        clip_model_name = 'ViT-B/32'
    elif clip_model_name == 'L14':
        clip_model_name = 'ViT-L/14'

    hparams = None
    with open('hparams/habitat.json', 'r') as f:
        hparams = json.load(f)

    explore, loc = agent_type_to_explore_localize[args.agent]
    loc_name = f'{loc}-{args.arch.lower()}-openai'
    assert loc_name in hparams

    mp.set_start_method('spawn', force=True)
    config = habitat.get_config(
        config_paths='datasets/habitat/configs/tasks/objectnav_mp3d.yaml')
    config.defrost()
    config.DATASET.DATA_PATH = 'datasets/habitat/datasets/objectnav/mp3d/v1/val/val.json.gz'
    config.DATASET.SCENES_DIR = 'datasets/habitat/scene_datasets/'
    config.freeze()

    config = ObjectNavHabitatBaseConfig.CONFIG.clone()
    config.defrost()
    config.DATASET.DATA_PATH = ObjectNavHabitatBaseConfig.VALID_SCENES
    config.MODE = "validate"
    config.ENVIRONMENT.MAX_STEPS = 250
    config.ENVIRONMENT.MAX_EPISODE_STEPS = 250

    class_type = ClassTypes.REGULAR
    if args.gpt:
        class_type = ClassTypes.GPT

    agent = importlib.import_module(args.agent)
    device_num = -1
    if torch.cuda.is_available():
        device_num = 0

    fail_stop = not args.nfs
    run_exploration_split = args.exp

    agent_class, agent_kwargs, render_depth = None, None, None
    threshold = hparams[loc_name]

    if args.model_path == '':
        agent_class, agent_kwargs, render_depth = agent.build(fail_stop=fail_stop, prompts_path=args.template, env_type=EnvTypes.HABITAT, threshold=threshold, class_type=class_type, device_num=device_num, clip_model_name=clip_model_name, center_only=args.center)
    else:
        if run_exploration_split:
            agent_class, agent_kwargs, render_depth = agent.build(fail_stop=fail_stop, prompts_path=args.template, exploration_model_path=args.model_path, env_type=EnvTypes.HABITAT, threshold=threshold, class_type=class_type, device_num=device_num, depth_only=args.depth, clip_model_name=clip_model_name, center_only=args.center)
        else:
            agent_class, agent_kwargs, render_depth = agent.build(fail_stop=fail_stop, prompts_path=args.template, exploration_model_path=args.model_path, env_type=EnvTypes.HABITAT, threshold=threshold, class_type=class_type, device_num=device_num, clip_model_name=clip_model_name, center_only=args.center)

    if render_depth:
        config.SIMULATOR.AGENT_0.SENSORS.append('DEPTH_SENSOR')

    if args.semantic:
        config.SIMULATOR.AGENT_0.SENSORS.append('SEMANTIC_SENSOR')

    if 'gt' in args.agent:
        assert args.semantic

    config.freeze()

    if not os.path.exists('results'):
        os.mkdir('results')

    experiment_name = f'results/habitat_{class_type.name.lower()}_{explore}_{loc_name}'
    if args.center:
        experiment_name = experiment_name + '-center'
    if 'subgoal' in args.agent:
        experiment_name = experiment_name + '-subgoal'

    HabitatChallenge(config, agent_class, agent_kwargs,
                     experiment_name, args.nprocesses,
                     fail_stop, run_exploration_split, no_grad=no_grad,
                     gt_localization=args.semantic)
