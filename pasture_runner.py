import os
from robothor_challenge import RobothorChallenge
import argparse
import importlib
import json
import logging
import torch

from src.simulation.sim_enums import ClassTypes, EnvTypes, POSIBLE_CONFIGS
logging.getLogger().setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Inference script for RoboThor ObjectNav challenge.")

    parser.add_argument(
        "--agent", "-a",
        required=True,
        help="Relative module for agent definition.",
    )

    parser.add_argument(
        "--template", "-t",
        required=False,
        default="prompt_templates/imagenet_template.json",
        help="Prompt template json.",
    )

    parser.add_argument(
        "--cfg", "-c",
        default="config_cow.yaml",
        help="Filepath to challenge config.",
    )

    parser.add_argument(
        "--output", "-o",
        default="metrics.json.gz",
        help="Filepath to output results to.",
    )

    parser.add_argument(
        "--remap-class-json",
        action='store',
        type=str,
        required=False,
        default=''
    )

    parser.add_argument(
        "--nprocesses", "-n",
        default=1,
        type=int,
        help="Number of parallel processes used to compute inference.",
    )

    parser.add_argument(
        "--model-path", "-p",
        default=''
    )

    parser.add_argument(
        "--semantic",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "--seed", "-s",
        default=0,
        required=False
    )

    parser.add_argument(
        "-nfs",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "-exp",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        "--depth",
        default=False,
        action='store_true'
    )

    parser.add_argument(
        '--arch',
        default='B32',
        action='store',
        type=str,
    )

    parser.add_argument(
        '--center',
        default=False,
        action='store_true'
    )

    args = parser.parse_args()

    if 'gt' in args.agent:
        assert args.semantic

    experiments = [
        (EnvTypes.LONGTAIL, ClassTypes.LONGTAIL),
        (EnvTypes.NORMAL, ClassTypes.APPEARENCE),
        (EnvTypes.NORMAL, ClassTypes.HIDDEN),
        (EnvTypes.DUP, ClassTypes.APPEARENCE),
        (EnvTypes.REMOVE, ClassTypes.HIDDEN),
        (EnvTypes.DUP, ClassTypes.SPATIAL),
        (EnvTypes.NORMAL, ClassTypes.SPATIAL),
        (EnvTypes.ROBOTHOR, ClassTypes.REGULAR),
    ]

    no_grad = True
    if 'grad' in args.agent:
        no_grad = False

    learned_suffix = ''
    if args.model_path != '':
        if 'RoboThor' in args.model_path:
            learned_suffix = '-robo'
        else:
            learned_suffix = '-hab'

    agent_type_to_explore_localize = {
        'src.models.agent_fbe_patch': ['fbe', 'patch'],
        'src.models.agent_fbe_grad': ['fbe', 'grad'],
        'src.models.agent_fbe_lang': ['fbe', 'lang'],
        'src.models.agent_fbe_owl': ['fbe', 'owl'],
        'src.models.agent_map_learned_owl': [f'learned{learned_suffix}', 'owl'],
        'src.models.agent_fbe_gt': ['fbe', 'gt'],
        'src.models.agent_fbe_mdetr_seg': ['fbe', 'mdetrseg'],
    }

    hparams = None
    with open('hparams/robo.json', 'r') as f:
        hparams = json.load(f)

    assert args.agent in agent_type_to_explore_localize

    assert args.arch in ['B32', 'B16', 'L14', 'ENB3']

    clip_model_name = None
    if args.arch == 'B32':
        clip_model_name = 'ViT-B/32'
    elif args.arch == 'B16':
        clip_model_name = 'ViT-B/32'
    elif clip_model_name == 'L14':
        clip_model_name = 'ViT-L/14'

    for env_type, class_type in experiments:

        assert class_type in POSIBLE_CONFIGS[env_type]

        explore, loc = agent_type_to_explore_localize[args.agent]
        loc_name = f'{loc}-{args.arch.lower()}-openai'
        print(loc_name)
        assert loc_name in hparams

        if not os.path.exists('results'):
            os.mkdir('results')

        experiment_name = f'results/{env_type.name.lower()}_{class_type.name.lower()}_{explore}_{loc_name}'
        if args.center:
            experiment_name = experiment_name + '-center'

        cache = set()
        if os.path.exists(experiment_name):
            for p in os.listdir(experiment_name):
                cache.add(p.split('.')[0])
        if ('robothor' in experiment_name and len(cache) == 1800):
            continue
        elif len(cache) == 360:
            continue

        fail_stop = not args.nfs
        run_exploration_split = args.exp

        agent = importlib.import_module(args.agent)
        agent_class, agent_kwargs, render_depth = None, None, None
        threshold = hparams[loc_name]

        if args.model_path == '':
            agent_class, agent_kwargs, render_depth = agent.build(
                fail_stop=fail_stop, prompts_path=args.template, threshold=threshold, class_type=class_type, env_type=env_type, clip_model_name=clip_model_name, center_only=args.center)
        else:
            with torch.no_grad():
                if run_exploration_split:
                    agent_class, agent_kwargs, render_depth = agent.build(fail_stop=fail_stop, prompts_path=args.template, threshold=threshold, exploration_model_path=args.model_path,
                                                                          class_type=class_type, env_type=env_type, depth_only=args.depth, clip_model_name=clip_model_name, center_only=args.center)
                else:
                    agent_class, agent_kwargs, render_depth = agent.build(fail_stop=fail_stop, prompts_path=args.template, threshold=threshold,
                                                                          exploration_model_path=args.model_path, class_type=class_type, env_type=env_type, clip_model_name=clip_model_name, center_only=args.center)

        class_remap = None
        if class_type in [ClassTypes.SPATIAL, ClassTypes.APPEARENCE]:
            class_remap = {}
            raw_annotation = None
            with open('class_templates/spatial_appearence_map.json', 'r') as f:
                raw_annotation = json.load(f)
            for scene in raw_annotation:
                class_remap[scene] = {}
                for object in raw_annotation[scene]:
                    if class_type == ClassTypes.SPATIAL:
                        class_remap[scene][object] = raw_annotation[scene][object][0]
                    else:
                        class_remap[scene][object] = raw_annotation[scene][object][1]
        elif class_type == ClassTypes.HIDDEN:
            with open('class_templates/hidden_map.json', 'r') as f:
                class_remap = json.load(f)

        r = RobothorChallenge(
            args.cfg,
            agent_class,
            agent_kwargs,
            experiment_name,
            no_grad,
            env_type,
            class_type,
            render_depth=render_depth,
            render_segmentation=args.semantic,
            class_remap=class_remap)

        challenge_metrics = {}
        dataset_dir = None

        if env_type == EnvTypes.ROBOTHOR:
            dataset_dir = 'datasets/robothor-objectnav'
        elif env_type == EnvTypes.LONGTAIL:
            dataset_dir = 'datasets/robothor-objectnav-longtail'
        elif env_type == EnvTypes.NORMAL:
            if class_type == ClassTypes.SPATIAL or class_type == ClassTypes.APPEARENCE:
                dataset_dir = 'datasets/robothor-objectnav-normal'
            else:
                # hidden case
                dataset_dir = 'datasets/robothor-objectnav-hidden'
        elif env_type == EnvTypes.DUP:
            dataset_dir = 'datasets/robothor-objectnav-dup'
        elif env_type == EnvTypes.REMOVE:
            dataset_dir = 'datasets/robothor-objectnav-hidden'

        assert dataset_dir is not None

        val_episodes, val_dataset = r.load_split(dataset_dir, "val")

        if run_exploration_split:
            refined_val_episodes = []
            subsampled_episodes = None
            with open('robothor_exploration_episode_keys.json', 'r') as f:
                subsampled_episodes = set(json.load(f))

            for e in val_episodes:
                if e['id'] in subsampled_episodes:
                    refined_val_episodes.append(e)

            val_episodes = refined_val_episodes

        refined_val_episodes = []
        for v in val_episodes:
            if v['id'] not in cache:
                refined_val_episodes.append(v)

        challenge_metrics["val"] = r.inference(
            refined_val_episodes,
            nprocesses=args.nprocesses,
            test=False
        )


if __name__ == "__main__":
    main()
