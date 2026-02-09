#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
import numpy as np
from lerobot.datasets.utils import write_json
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.io_utils import deserialize_json_into_object
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(
        self, optimizer: Optimizer, num_training_steps: int
    ) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup_flower")
@dataclass
class CosineDecayWithWarmupSchedulerFLOWERConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_plateau_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_plateau_steps = self.num_plateau_steps
        decay_start_step = actual_warmup_steps + actual_plateau_steps

        # The remaining steps are dedicated to the cosine decay
        actual_decay_duration = max(0, num_training_steps - decay_start_step)

        if num_training_steps < self.num_decay_steps:  # TODO: think about this
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_plateau_steps = int(actual_plateau_steps * scale_factor)
            actual_decay_steps = num_training_steps - actual_plateau_steps

            logging.info(  # TODO: improve logging
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                # step = min(current_step, actual_decay_steps)
                # cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                # alpha = self.decay_lr / self.peak_lr
                # decayed = (1 - alpha) * cosine_decay + alpha
                # return decayed
                # 3. Cosine Decay Phase (Decay from peak_lr to decay_lr)
                # The step count for decay must be relative to the decay start
                step_in_decay = current_step - decay_start_step

                # The total number of steps in the *cosine function* itself is
                # now actual_decay_duration, ensuring the decay completes at the end.
                t = np.clip(step_in_decay / actual_decay_duration, 0.0, 1.0)

                # Cosine decay from 1.0 down to alpha (relative decay ratio)
                alpha = self.decay_lr / self.peak_lr
                cosine_output = 0.5 * (1.0 + math.cos(math.pi * t))

                # Interpolate between 1.0 (start of decay) and alpha (end of decay)
                decayed_multiplier = (1.0 - alpha) * cosine_output + alpha

                return decayed_multiplier

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            if current_step < decay_start_step:
                return 1.0

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(
        save_dir / SCHEDULER_STATE, scheduler.state_dict()
    )
    scheduler.load_state_dict(state_dict)
    return scheduler
