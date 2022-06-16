# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import List
import monai
import torch
import torch.nn as nn
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
class INPUT_CHANNEL(int, Enum):
    lab = 1
    bgr = 3
    rgb = 3
    gray = 1

VARIANT = {
    "SEResNeXt50" : monai.networks.nets.SEResNeXt50,
    "SEResNeXt101" : monai.networks.nets.SEResNeXt101,
    "SEResNet50" : monai.networks.nets.SEResNet50,
    "SEResNet101" : monai.networks.nets.SEResNet101,
    "SEResNet152" : monai.networks.nets.SEResNet152,
    "SENet154" : monai.networks.nets.SENet154,
}

class SUPPORTED_L4_STRIDE(int, Enum):
    one = 1
    two = 2

@register_model_trunk("senet_3d")
class SENet3D(nn.Module):
    """
    Wrapper for TorchVison ResNet Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """
    def __init__(self, model_config: AttrDict, model_name: str):
        super(SENet3D, self).__init__()
        self.model_config = model_config
        logging.info(
            "SEResNeXt3D trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.trunk_config = self.model_config.TRUNK.SENET_3D

        self.spatial_dims = self.trunk_config.SPATIAL_DIMS

        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        model = VARIANT[self.trunk_config.VARIANT](spatial_dims=self.spatial_dims,
                                in_channels=INPUT_CHANNEL[self.model_config.INPUT_TYPE])

        self._feature_blocks = nn.ModuleDict(
            [
                ("layer0", model.layer0),
                ("layer1", model.layer1),
                ("layer2", model.layer2),
                ("layer3", model.layer3),
                ("layer4", model.layer4),
                ("avgpool", model.adaptive_avg_pool),
                ("flatten", Flatten(1)),
            ]
        )

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "res1": "layer0",
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
