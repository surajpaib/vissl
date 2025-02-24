# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("ImgReplicate")
class ImgReplicate(ClassyTransform):
    """
    Adds the same image multiple times to the batch K times so that the batch.
    Size is now N*K. Use the simclr_collator to convert into batches.

    This transform is useful when generating multiple copies of the same image,
    for example, when training contrastive methods.
    """

    def __init__(self, num_times: int = 2):
        """
        Args:
            num_times (int): how many times should the image be replicated.
        """
        assert isinstance(
            num_times, int
        ), f"num_times must be an integer. Found {type(num_times)}"
        assert num_times > 0, f"num_times {num_times} must be greater than zero."
        self.num_times = num_times

    def __call__(self, image):
        output = []
        for _ in range(self.num_times):
            output.append(image.copy())
        return output

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgReplicatePil":
        """
        Instantiates ImgReplicatePil from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgReplicatePil instance.
        """
        num_times = config.get("num_times", 2)
        logging.info(f"ImgReplicatePil | Using num_times: {num_times}")
        return cls(num_times=num_times)
