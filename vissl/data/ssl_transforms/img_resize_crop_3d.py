# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from typing import Any, Dict, List

from monai.transforms import RandScaleCrop, Resize

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform


@register_transform("RandomResizedCrop3D")
class RandomResizedCrop3D(ClassyTransform):
    """
    Combines monai's random spatial crop followed by resize to the desired size.

    Modification: 
    1. The spatial crop is done with same dimensions for all the axes 
    2. Handles cases where the image_size is less than the crop_size by choosing 
        the smallest dimension as the random scale.

    """

    def __init__(self, size, scale: List[float] = [0.5, 1.0]):
        """
        Args:
            scale (List[int]): Specifies the lower and upper bounds for the random area of the crop,
             before resizing. The scale is defined with respect to the area of the original image.
        """
        self.scale = scale
        self.size = [size] * 3


    def __call__(self, image):
        random_scale = torch.empty(1).uniform_(*self.scale).item()
        rand_cropper = RandScaleCrop(random_scale, random_size=False)
        resizer = Resize(self.size, mode="trilinear")

        for transform in [rand_cropper, resizer]:
            image = transform(image)

        return image
        

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomResizedCrop3D":
        """
        Instantiates RandomResizedCrop3D from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            RandomResizedCrop3D instance.
        """
        scale = config.get("scale", [0.5, 1.0])
        size = config.get("size", 50)
        logging.info(f"RandomResizedCrop3D | Using roi_size: {scale} and size: {size}")
        return cls(size=size, scale=scale)
