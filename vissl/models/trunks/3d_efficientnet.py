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

class VARIANT(str, Enum):
    B0 = "efficientnet-b0"
    B1 = "efficientnet-b1"
    B2 = "efficientnet-b2"
    B3 = "efficientnet-b3"
    B4 = "efficientnet-b4"
    B5 = "efficientnet-b5"
    B6 = "efficientnet-b6"
    B7 = "efficientnet-b7"

class INPUT_CHANNEL(int, Enum):
    lab = 1
    bgr = 3
    rgb = 3
    gray = 1


@register_model_trunk("efficientnet_3d")
class EfficientNet3D(nn.Module):
    """
    """
    def __init__(self, model_config: AttrDict, model_name: str):
        super(EfficientNet3D, self).__init__()
        self.model_config = model_config
        logging.info(
            "EfficientNet3D trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.trunk_config = self.model_config.TRUNK.EFFICIENTNET_3D

        self.spatial_dims = self.trunk_config.SPATIAL_DIMS

        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )

        model = monai.networks.nets.EfficientNetBN(VARIANT[self.trunk_config.VARIANT], spatial_dims=self.spatial_dims,
                                in_channels=INPUT_CHANNEL[self.model_config.INPUT_TYPE])

        self._feature_blocks = nn.ModuleDict(
            [
                ("padding", model._conv_stem_padding),
                ("conv_stem", model._conv_stem),
                ("bn0", model._bn0),
                ("swish", model._swish),
                ("blocks", model._blocks),
                ("padding", model._conv_head_padding),
                ("conv_head", model._conv_head),
                ("bn1", model._bn1),
                ("swish", model._swish),                
                ("avg_pool", model._avg_pooling),
                ("flatten", Flatten(1)),
            ]
        )

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
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
