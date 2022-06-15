# Modified by Suraj Pai to include wandb support for VISSL
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from classy_vision import tasks

from vissl.utils.misc import is_wandb_available
from classy_vision.generic.distributed_util import is_primary
from classy_vision.hooks.classy_hook import ClassyHook
from vissl.utils.tensorboard import create_visual

if not is_wandb_available():
    raise RuntimeError(
                "wandb not installed, cannot use WandbHook"
            )
else:
    import wandb

class WandbHook(ClassyHook):
    """
    Wandb hook to log info similar to tensorboard hook
    """

    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_phase_start = ClassyHook._noop

    def __init__(
        self, cfg
    ) -> None:
        """The constructor method of WandbHook.

        Args:
            cfg: A config that defines the wandb hook configuration.
        """
        super().__init__()
        self.cfg = cfg

    def on_start(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the start of training. Wandb init is called here to
        allow logging only the primary rank.
        """
        if not is_primary():
            return
        
        logging.info("Setting up Wandb Hook...")
        wandb.init(name=self.cfg.NAME, project=self.cfg.PROJECT, config=task.config)
        logging.info(
            f"Wandb config: {self.cfg}"
        )

        # Wandb watch handles watching the model parameters and gradients
        if self.cfg.WATCH_MODEL:
            wandb.watch(task.model, log_freq=self.cfg.LOG_FREQ, log="all", log_graph=True)

    def on_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of training.
        """
        if not is_primary():
            return

        wandb.finish()


    def on_phase_end(self, task: "tasks.ClassyTask") -> None:
        """
        Called at the end of every epoch if the tensorboard hook is
        enabled.
        Log model parameters and/or parameter gradients as set by user
        in the tensorboard configuration. Also resents the CUDA memory counter.
        """
        # Log train/test accuracy
        if is_primary():
            phase_type = "Training" if task.train else "Testing"
            for meter in task.meters:
                for metric_name, vals in meter.value.items():
                    for i, val in vals.items():
                        tag_name = f"{phase_type}/{meter.name}_{metric_name}_Output_{i}"
                        wandb.log({f"{tag_name}":  round(val, 5)}, step=task.train_phase_idx)


            # Reset the GPU Memory counter
            if torch.cuda.is_available():
                torch.cuda.reset_max_memory_allocated()
                torch.cuda.reset_max_memory_cached()

    def on_update(self, task: "tasks.ClassyTask") -> None:
        """
        Called after every parameters update if tensorboard hook is enabled.
        Logs the parameter gradients if they are being set to log,
        log the scalars like training loss, learning rate, average training
        iteration time, batch size per gpu, img/sec/gpu, ETA, gpu memory used,
        peak gpu memory used.
        """

        if not is_primary():
            return

        iteration = task.iteration

        if iteration % task.config["LOG_FREQUENCY"] == 0 or (
            iteration <= 100 and iteration % 5 == 0
        ):
            logging.info(f"Logging metrics. Iteration {iteration}")
            wandb.log({"Training/Loss": round(task.last_batch.loss.data.cpu().item(), 5)}, step=iteration)
            wandb.log({"Training/Learning_rate": round(task.optimizer.options_view.lr, 5)}, step=iteration)

            # Batch processing time
            if len(task.batch_time) > 0:
                batch_times = task.batch_time
            else:
                batch_times = [0]
            batch_time_avg_s = sum(batch_times) / max(len(batch_times), 1)
            wandb.log({"Speed/Batch_processing_time_ms": int(1000.0 * batch_time_avg_s)}, step=iteration)

            # Images per second per replica
            pic_per_batch_per_gpu = task.config["DATA"]["TRAIN"][
                "BATCHSIZE_PER_REPLICA"
            ]
            pic_per_batch_per_gpu_per_sec = (
                int(pic_per_batch_per_gpu / batch_time_avg_s)
                if batch_time_avg_s > 0
                else 0.0
            )
            wandb.log({"Speed/img_per_sec_per_gpu": pic_per_batch_per_gpu_per_sec}, step=iteration)

            # ETA
            avg_time = sum(batch_times) / len(batch_times)
            eta_secs = avg_time * (task.max_iteration - iteration)
            wandb.log({"Speed/ETA_hours": eta_secs / 3600.0}, step=iteration)

            # Log training sample images
            if self.cfg.LOG_IMAGES:
                visual = task.last_batch.sample["input"].clone()
                out = create_visual(visual)
                wandb.log({'visuals': wandb.Image(out)}, step=iteration)

                