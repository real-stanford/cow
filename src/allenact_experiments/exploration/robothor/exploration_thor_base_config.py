# modified from: https://github.com/allenai/allenact/blob/main/projects/objectnav_baselines/experiments/objectnav_thor_base.py

from abc import ABC
from typing import Optional, Sequence

import torch
from allenact.base_abstractions.task import TaskSampler
from src.allenact_experiments.exploration.robothor.exploration_dataset_task_sampler import \
    ExplorationDatasetTaskSampler
from src.allenact_experiments.shared.objectnav_thor_base_config import ObjectNavThorBaseConfig


class ExplorationThorBaseConfig(ObjectNavThorBaseConfig, ABC):
    """The base config for all AI2-THOR Exploration experiments."""

    DEFAULT_NUM_TRAIN_PROCESSES: Optional[int] = None
    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = (torch.cuda.device_count() - 1,)
    DEFAULT_TEST_GPU_IDS = (torch.cuda.device_count() - 1,)

    TRAIN_DATASET_DIR: Optional[str] = None
    VAL_DATASET_DIR: Optional[str] = None
    TEST_DATASET_DIR: Optional[str] = None

    AGENT_MODE = "default"

    TARGET_TYPES: Optional[Sequence[str]] = None

    THOR_COMMIT_ID: Optional[str] = None
    THOR_IS_HEADLESS: bool = False

    def __init__(
        self,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: bool = False,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
        randomize_train_materials: bool = False,
    ):
        super().__init__(
            num_train_processes=num_train_processes,
            num_test_processes=num_test_processes,
            test_on_validation=test_on_validation,
            train_gpu_ids=train_gpu_ids,
            val_gpu_ids=val_gpu_ids,
            test_gpu_ids=test_gpu_ids,
            randomize_train_materials=randomize_train_materials,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ExplorationDatasetTaskSampler(**kwargs)