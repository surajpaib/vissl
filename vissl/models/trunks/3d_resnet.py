# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import List

import monai
import torch
import torch.nn as nn
from monai.networks.nets.resnet import ResNetBottleneck as Bottleneck
from vissl.config import AttrDict
from vissl.data.collators.collator_helper import MultiDimensionalTensor
from vissl.models.model_helpers import (
    Flatten,
    _get_norm,
    get_trunk_forward_outputs,
    get_tunk_forward_interpolated_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk


# For more depths, add the block config here
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
    200: (3, 24, 36, 3),
}

class INPUT_CHANNEL(int, Enum):
    lab = 1
    bgr = 3
    rgb = 3
    gray = 1

class SUPPORTED_DEPTHS(int, Enum):
    RN50 = 50
    RN101 = 101
    RN152 = 152
    RN200 = 200


class SUPPORTED_L4_STRIDE(int, Enum):
    one = 1
    two = 2


@register_model_trunk("resnet_3d")
class ResNet3D(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(ResNet3D, self).__init__()
        self.model_config = model_config
        logging.info(
            "3DResNet trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.trunk_config = self.model_config.TRUNK.RESNETS_3D
        self.depth = SUPPORTED_DEPTHS(self.trunk_config.DEPTH)
        self.width_multiplier = self.trunk_config.WIDTH_MULTIPLIER
        self.shortcut_type = self.trunk_config.SHORTCUT_TYPE
        self._norm_layer = nn.BatchNorm3d
        self.first_kernel_size = self.trunk_config.FIRST_KERNEL_SIZE
        self.first_kernel_stride = self.trunk_config.FIRST_KERNEL_STRIDE    

        self.spatial_dims = self.trunk_config.SPATIAL_DIMS
        in_planes = (64, 128, 256, 512)

        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        (n1, n2, n3, n4) = BLOCK_CONFIG[self.depth]
        logging.info(
            f"Building model: 3DResNet"
            f"{self.depth}-{in_planes}x{self.width_multiplier}"
            f"{self._norm_layer.__name__}"
        )

        model = monai.networks.nets.resnet.ResNet(
            block=Bottleneck,
            layers=(n1, n2, n3, n4),
            block_inplanes=in_planes,
            spatial_dims=self.spatial_dims,
            n_input_channels=INPUT_CHANNEL[self.model_config.INPUT_TYPE],
            conv1_t_stride=self.first_kernel_stride,
            conv1_t_size=self.first_kernel_size,
            widen_factor=self.width_multiplier,
        )

        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1", model.conv1),
                ("bn1", model.bn1),
                ("conv1_relu", model.relu),
                ("maxpool", model.maxpool),
                ("layer1", model.layer1),
                ("layer2", model.layer2),
                ("layer3", model.layer3),
                ("layer4", model.layer4),
                ("avgpool", model.avgpool),
                ("flatten", Flatten(1)),
            ]
        )

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "conv1": "conv1_relu",
            "res1": "maxpool",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
            "res5avg": "avgpool",
            "flatten": "flatten",
        }

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        if isinstance(x, MultiDimensionalTensor):
            out = get_tunk_forward_interpolated_outputs(
                input_type=self.model_config.INPUT_TYPE,
                interpolate_out_feat_key_name="res5",
                remove_padding_before_feat_key_name="avgpool",
                feat=x,
                feature_blocks=self._feature_blocks,
                feature_mapping=self.feat_eval_mapping,
                use_checkpointing=self.use_checkpointing,
                checkpointing_splits=self.num_checkpointing_splits,
            )
        else:
            model_input = transform_model_input_data_type(
                x, self.model_config.INPUT_TYPE
            )
            out = get_trunk_forward_outputs(
                feat=model_input,
                out_feat_keys=out_feat_keys,
                feature_blocks=self._feature_blocks,
                feature_mapping=self.feat_eval_mapping,
                use_checkpointing=self.use_checkpointing,
                checkpointing_splits=self.num_checkpointing_splits,
            )
        return out
