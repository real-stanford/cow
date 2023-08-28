<h1> CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation</h1>
<div style="text-align: center;">

[Samir Yitzhak Gadre](https://sagadre.github.io/), [Mitchell Wortsman](https://mitchellnw.github.io/), [Gabriel Ilharco](https://gabrielilharco.com/), [Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/), [Shuran Song](https://www.cs.columbia.edu/~shurans/)

[Project Page](https://cow.cs.columbia.edu/) | [arXiv](https://arxiv.org/abs/2203.10421)


<div style="margin:50px; text-align: justify;">
<img style="width:100%;" src="media/gingerbread_example.gif">

We present baselines and benchmarks for langauge-driven zero-shot object navigation (L-ZSON). In this task, an agent must find an object based on a natural language description, which may contain different levels of specificity (e.g., "gingerbread house", "gingerbread house on the TV stand", or "brown gingerbread house"). Since L-ZSON is "zero-shot", we consider agents that do not have access to navigation training on the target objects or domains. This reflects realistic application scenarios, where the environment and object set may not be known a priori.

</div>
</div>

<br>

This repository contains code for CoWs on Pasture.

If you have any questions, please contact [Samir](https://sagadre.github.io) at `sy [at] cs [dot] columbia [dot] edu`.

**Table of Contents**

- [Base Environment](#base-environment)
- [Pasture Benchmark Setup](#pasture-benchmark-setup)
- [Evaluation on Pasture and RoboTHOR](#evaluation-on-pasture-and-robothor)
- [Habitat MP3D Benchmark Setup](#habitat-mp3d-benchmark-setup)
- [Evaluation on Habitat MP3D](#evaluation-on-habitat-mp3d)
- [Helpful Pointers](#helpful-pointers)
- [Codebase Overview](#codebase-overview)
- [Acknowledgements](#acknowledgements)


# Base Environment

Create the conda environment:
```sh
conda env create environment.yml
```
Activate the environment:
```sh
conda activate cow
```

Note: environment is tested with `CUDA Version: 11.2` on an 8 GPU node with `RTX 2080s`. Run the following to ensure that torch is properly installed.
```
python scripts/test_torch_download.py
```

# Pasture Benchmark Setup

To download the Pasture THOR binaries (~4GB) see below. This is a required step to run evaluations. Navigate to the repo root directory (`cow/`) and run the following:
```sh
wget https://cow.cs.columbia.edu/downloads/pasture_builds.tar.gz
```
```sh
tar -xvf pasture_builds.tar.gz
```
This should create a folder called `pasture_builds/`

To download episode targets and ground truth for evaluation, run the following:
```sh
wget https://cow.cs.columbia.edu/downloads/datasets.tar.gz
```
```sh
tar -xvf datasets.tar.gz
```
This should create a folder called `datasets/`

Additionally, THOR rendering requires that `Xorg` processes are running on all GPUs. If processes are not already running, run the following:
```
sudo python scripts/startx.py
```

# Evaluation on Pasture and RoboTHOR

Note: it is recommended to run evaluations in a `tmux` session as they are long running jobs.

For Pasture and RoboTHOR, to evaluate the OWL-ViT B-32 default cow, run:

```
python pasture_runner.py -a src.models.agent_fbe_owl -n 8 --arch B32 --center
```

Note: this automatically evaluates all Pasture splits and RoboTHOR. If the script is stopped, it will resume where it left off. If you want to re-evaluate from scratch, remove the results subfolder associated with the agent being evaluated in `results/`.

# Habitat MP3D Benchmark Setup

Unfortunately, we are not legally allowed to re-distrubute the MP3D scene assets.
Please follow the instructions posted [here](https://niessner.github.io/Matterport/) to download Habitat MP3D (under the "Dataset Download" section). After you have been granted access to the official `download_mp.py` script, run the following in `cow/`:
```
python2 download_mp.py --task habitat -o .
```
Note: the above script will take ~30min to download, so consider running in a `tmux` session.

Now unpack the downloaded scene data:
```sh
unzip v1/tasks/mp3d_habitat.zip -d datasets/habitat/scene_datasets
```

Our habitat evalutation additionally relies on [Habitat-Lab](https://github.com/facebookresearch/habitat-lab), which calls for its own download procedure based on the one described [here](https://github.com/facebookresearch/habitat-lab#installation):

```sh
wget https://github.com/facebookresearch/habitat-lab/archive/refs/tags/v0.2.1.tar.gz
tar -xzvf v0.2.1.tar.gz
mv habitat-lab-0.2.1 src/habitat
cd src/habitat
pip install -e .
```

To verify that your habitat setup is correct navigate back to `cow/` and run the following:
```python
python scripts/test_habitat_download.py
```

# Evaluation on Habitat MP3D

Note: it is recommended to run evaluations in a `tmux` session as they are long running jobs.

For Habitat MP3D, to evaluate the OWL-ViT B-32 default cow, run:

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export HABITAT_BASE_DIR=datasets/habitat
ln -s datasets/habitat data
python habitat_mp3d_runner.py -a src.models.agent_fbe_owl -n 8 --arch B32 --center
```

If the script is stopped, it will resume where it left off. If you want to re-evaluate from scratch, remove the results subfolder associated with the agent being evaluated in `results/`.

# Visualization on Pasture
To visualize both an egocentric trajectory view and a top-down path as in the teaser gif above, run:

```
python path_visualization.py --out-dir viz/ --thor-floor FloorPlan_Val3_5 --result-json media/media_data/FloorPlan_Val3_5_GingerbreadHouse_1.json --thor-build pasture_builds/thor_build_longtail/longtail.x86_64
```

The script outputs 1) egocentric pngs for each view, 2) an mp4 for the egocentric feed, 3) top-down pngs for each pose, 4) an mp4 for the top-down feed.

Note: flag arguments should be swapped accordinly for the floor plan and trajectory you wish to visualize. This script provides functionality to visualize RoboTHOR or Pasture evaluations.

Script based on code sourced from [here](https://github.com/allenai/cordial-sync/issues/5).

# Helpful Pointers

Evaluation is often long running. Each time an evaluation episode completes, a `json` with information about the trajectory is stored in the `results/` folder. For example, for the default agent on the Pasture uncommon object split: `results/longtail_longtail_fbe_owl-b32-openai-center/*.json`. This allows for printing the completed evaluations, e.g.,

```
python success_agg.py --result-dir results/longtail_longtail_fbe_owl-b32-openai-center/
```

For completed Habitat evaluations, e.g.,

```
python success_agg.py --result-dir results/habitat_regular_fbe_owl-b32-openai-center/ --hab
```

# Codebase Overview

We hope this repo will make it easy to run experiments on Pasture, RoboTHOR, Habitat, etc. in a relatively painless way. We hope this will encourage people to create new cows and to evaluate single agents on multiple benchmarks!

At a high level, `pasture_runner.py` supports Pasture and RoboTHOR evaluation and `habitat_mp3d_runner.py` supports evaluation on Habitat.

Here are the most relevant modules for extending the repo:

```
src
├── models
│   ├── agent_build_utils.py
│   ├── agent_fbe_explorer.py
│   ├── agent_fbe_grad.py
│   ├── agent_fbe_gt.py
│   ├── agent_fbe_lang.py
│   ├── agent_fbe_mdetr_seg.py
│   ├── agent_fbe_owl.py
│   ├── agent_fbe_owl_subgoal.py
│   ├── agent_fbe_patch.py
│   ├── agent_fbe.py
│   ├── agent_learned.py
│   ├── agent_map_learned_explorer.py
│   ├── agent_map_learned_grad.py
│   ├── agent_map_learned_owl.py
│   ├── agent_map_learned.py
│   ├── agent_mode.py
│   ├── agent.py
│   ├── exploration
│   │   ├── frontier_based_exploration.py
│   ├── localization
│   │   ├── clip_filler.py
│   │   ├── clip_grad.py
│   │   ├── clip_lang.py
│   │   ├── clip_owl.py
│   │   ├── clip_patch.py
│   │   ├── mask_gt.py
│   │   └── mdetr_seg.py
├── shared
│   ├── data_split.py
│   └── utils.py
└── simulation
    ├── constants.py
    ├── sim_enums.py
    ├── utils.py
    └── visualization_utils.py
```
all agents extend `src/models/agent.py`, which is the base abstraction borrowed from [RoboTHOR](https://github.com/allenai/robothor-challenge/blob/main/robothor_challenge/agent.py).

Our implementation of Frontier based exploration (FBE) is here: `src/models/exploration/frontier_based_exploration.py`

FBE is incorporated into a template cow, which explores via FBE and exploits when its has found the target. This basic algorithm is coded here: `src/models/agent_fbe.py`, which several other agents extend.

A localization module found here, `src/models/localization/`, provides the implementation of what it means to be confident that a target has been localized. Object localization modules take image input and provide a per-pixel confidence value. This abstraction natural encorporates detection and segmentation.

# Acknowledgements

We would like to thank Jessie Chapman, Cheng Chi, Huy Ha, Zeyi Liu, Sachit Menon, and Sarah Pratt for valuable feedback. We would also like to thank Luca Weihs for technical help with AllenAct and Cheng Chi for help speeding up code. This work was supported in part by NSF CMMI-2037101, NSF IIS-2132519, and an Amazon Research Award. SYG is supported by a NSF Graduate Research Fellowship. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of the sponsors.

Code acknowledgements:
 - `src/clip/` modified from [Chefer et al.'s codebase](https://github.com/hila-chefer/Transformer-MM-Explainability), which was in turn modified from the original [CLIP codebase](https://github.com/openai/CLIP).
 - Our repo depends on [AllenAct](https://allenact.org/). Thanks to Luca Weihs and co. for developing this!
 - `robothor_challenge.py` is modified from the [2021 RoboTHOR Challenge](https://github.com/allenai/robothor-challenge) repo.
 - `habitat_challenge.py` is modified from the [2021 Habitat Challenge](https://github.com/facebookresearch/habitat-challenge) repo and made to be consistent with the abstractions used in the [2021 RoboTHOR Challenge](https://github.com/allenai/robothor-challenge) repo.
 - `README.md` format borrowed from the [SemAbs codebase](https://github.com/columbia-ai-robotics/semantic-abstraction). Thanks to Huy Ha!


# Citation

If you find this codebase useful, consider citing:

```bibtex
@article {gadre2022cow,
	title={CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation},
	author={Gadre, Samir Yitzhak and Wortsman, Mitchell and Ilharco, Gabriel and Schmidt, Ludwig and Song, Shuran},
	journal={CVPR},
	year={2023}
}
```
Note: while we changed the title to be more discriptive, this is an updated version of the original arXiv submission, "CLIP on Wheels: Zero-Shot Object Navigation as Object Localization and Exploration". Thanks to all of the reviewers along the way that helped us improve our work!
