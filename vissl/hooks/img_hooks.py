from classy_vision.hooks.classy_hook import ClassyHook
from classy_vision.generic.distributed_util import is_primary
from classy_vision import tasks
import torch
from vissl.utils.tensorboard import create_visual
import torchvision
import gc
import logging
from pathlib import Path

class ImgScreenshotHook(ClassyHook):
    """
    Saves img screenshots on test split 
    """
    on_backward = ClassyHook._noop
    on_step = ClassyHook._noop
    on_phase_end = ClassyHook._noop
    on_end = ClassyHook._noop
    on_phase_start = ClassyHook._noop

    def __init__(self, save_dir) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        logging.info("Setting up Img Screenshot Hook...")


    def on_start(self, task: "tasks.ClassyTask") -> None:
        if is_primary() and not(task.train):
            # Log training sample images
            data_iterator = iter(task.dataloaders["test"])
            sample_count = 0
            while True:
                try:
                    sample = next(data_iterator)
                    assert isinstance(sample, dict)
                    input = torch.cat(sample["data"])

                    for n in range(input.size(0)):
                        img = input.unbind(dim=2) # Expand over slice dimensions
                        img = img[len(img) // 2] # Take middle slice
                        img = img[n].unsqueeze(0)
                        torchvision.utils.save_image(img, f"{self.save_dir}/{sample_count}.png")
                        sample_count += 1

                except StopIteration:
                    del data_iterator
                    gc.collect()
                    break


