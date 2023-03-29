# modified from: https://github.com/allenai/allenact/blob/main/projects/objectnav_baselines/experiments/objectnav_thor_base.py

from abc import ABC
from typing import Optional, Sequence

import torch
from allenact.base_abstractions.task import TaskSampler
from src.allenact_experiments.exploration.habitat.exploration_dataset_task_sampler import \
    ExplorationDatasetTaskSampler
from src.allenact_experiments.shared.objectnav_habitat_base_config import \
    ObjectNavHabitatBaseConfig


class ExplorationHabitatBaseConfig(ObjectNavHabitatBaseConfig, ABC):

    def __init__(
        self,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: bool = True,
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
