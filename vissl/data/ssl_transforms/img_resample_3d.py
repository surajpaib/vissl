# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from typing import Any, Dict, List

from monai.transforms import Spacing

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("RandomResample3D")
class RandomResample3D(ClassyTransform):
    """
    Uses monai's Spacing transform to randomly resample an image.

    Modification: 
    1. 
    """

    def __init__(self, spacing, prob):
        """
        Args:
            scale (List[int]): Specifies the lower and upper bounds for the random area of the crop,
             before resizing. The scale is defined with respect to the area of the original image.
        """
        self.spacing = spacing
        self.prob = prob

    def __call__(self, image):

        transforms = []
        if torch.rand(1) < self.prob:
            transforms.append(Spacing(self.spacing, image_only=True))

        for transform in transforms:
            image = transform(image)

        return image
        

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomResample3D":
        """
        Instantiates RandomResizedCrop3D from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            RandomResizedCrop3D instance.
        """
        spacing = config.get("spacing", [1, 1, 1])
        prob = config.get("prob", 0)
        logging.info(f"RandomResample3D | Using spacing: {spacing} and size: {prob}")
        return cls(spacing=spacing, prob=prob)
