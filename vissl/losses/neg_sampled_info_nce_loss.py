# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pprint

import numpy as np
import torch
from classy_vision.generic.distributed_util import get_cuda_device_index, get_rank
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
from vissl.config import AttrDict
from vissl.utils.distributed_utils import gather_from_all


@register_loss("neg_sampled_info_nce_loss")
class NegativeSampledInfoNCELoss(ClassyLoss):
    """
    This is the loss which was proposed in SimCLR https://arxiv.org/abs/2002.05709 paper.
    See the paper for the details on the loss.

    Config params:
        temperature (float): the temperature to be applied on the logits
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(self, loss_config: AttrDict, device: str = "gpu"):
        super(NegativeSampledInfoNCELoss, self).__init__()

        self.loss_config = loss_config
        # loss constants
        self.temperature = self.loss_config.temperature
        self.buffer_params = self.loss_config.buffer_params
        self.info_criterion = NegativeSampledInfoNCECriterion(
            self.buffer_params, self.temperature
        )

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates SimclrInfoNCELoss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            SimclrInfoNCELoss instance.
        """
        return cls(loss_config)

    def forward(self, output, target):
        normalized_output = nn.functional.normalize(output, dim=1, p=2)
        loss = self.info_criterion(normalized_output)
        return loss

    def __repr__(self):
        repr_dict = {"name": self._get_name(), "info_average": self.info_criterion}
        return pprint.pformat(repr_dict, indent=2)


class NegativeSampledInfoNCECriterion(nn.Module):
    """
    The criterion corresponding to the SimCLR loss as defined in the paper
    https://arxiv.org/abs/2002.05709.

    Args:
        temperature (float): the temperature to be applied on the logits
        buffer_params:
            world_size (int): total number of trainers in training
            embedding_dim (int): output dimensions of the features projects
            effective_batch_size (int): total batch size used (includes positives)
    """

    def __init__(self, buffer_params, temperature: float, balanced: bool = True):
        super(NegativeSampledInfoNCECriterion, self).__init__()

        self.use_gpu = get_cuda_device_index() > -1
        self.temperature = temperature
        self.num_pos = 2
        self.buffer_params = buffer_params
        self.criterion = nn.CrossEntropyLoss()
        self.dist_rank = get_rank()
        self.pos_mask = None
        self.neg_mask = None

        self.balanced = balanced
        self.precompute_pos_neg_mask()
        logging.info(f"Creating Info-NCE loss on Rank: {self.dist_rank}")

    def precompute_pos_neg_mask(self):
        """
        We precompute the positive and negative masks to speed up the loss calculation
        """
        # computed once at the begining of training

        # total_images is x2 SimCLR Info-NCE loss 
        # as we have negative samples for each positive sample
        total_images = self.buffer_params.effective_batch_size * 2
        world_size = self.buffer_params.world_size

        # Batch size computation is different from SimCLR paper
        batch_size = (self.buffer_params.effective_batch_size // world_size)
        orig_images = batch_size // self.num_pos
        rank = self.dist_rank

        pos_mask = torch.zeros(batch_size * 2, total_images)
        neg_mask = torch.zeros(batch_size * 2, total_images)

        all_indices = np.arange(total_images // 2)
        pos_members = orig_images * np.arange(self.num_pos)
        orig_members = torch.arange(orig_images)



        for anchor in np.arange(self.num_pos):
            for img_idx in range(orig_images):
                delete_inds = batch_size * rank + img_idx + pos_members
                neg_inds = torch.tensor(np.delete(all_indices, delete_inds)).long() + 2 * orig_images
                neg_mask[anchor * orig_images + img_idx, neg_inds] = 1

            for pos in np.delete(np.arange(self.num_pos), anchor):
                pos_inds = batch_size * rank + pos * orig_images + orig_members
                pos_mask[
                    torch.arange(
                        anchor * orig_images, (anchor + 1) * orig_images
                    ).long(),
                    pos_inds.long(),
                ] = 1

        self.pos_mask = pos_mask.cuda(non_blocking=True) if self.use_gpu else pos_mask
        self.neg_mask = neg_mask.cuda(non_blocking=True) if self.use_gpu else neg_mask

    def forward(self, embedding: torch.Tensor):
        """
        Calculate the loss. Operates on embeddings tensor.
        """
        assert embedding.ndim == 2
        assert embedding.shape[1] == int(self.buffer_params.embedding_dim)

        batch_size = embedding.shape[0]
        T = self.temperature
        num_pos = self.num_pos
        assert batch_size % num_pos == 0, "Batch size should be divisible by num_pos"

        # Step 1: gather all the embeddings. Shape example: 4096 x 128
        embeddings_buffer = self.gather_embeddings(embedding)

        # Step 2: matrix multiply: 64 x 128 with 4096 x 128 = 64 x 4096 and
        # divide by temperature.
        similarity = torch.exp(torch.mm(embedding, embeddings_buffer.t()) / T)
        pos = torch.sum(similarity * self.pos_mask, 1)
        neg = torch.sum(similarity * self.neg_mask, 1)

        # Ignore the negative samples as entries for loss calculation
        pos = pos[: batch_size // 2]
        neg = neg[: batch_size // 2]

        loss = -(torch.mean(torch.log(pos / (pos + neg) )))
        return loss

    def __repr__(self):
        num_negatives = self.buffer_params.effective_batch_size - 2
        T = self.temperature
        num_pos = self.num_pos
        repr_dict = {
            "name": self._get_name(),
            "temperature": T,
            "num_negatives": num_negatives,
            "num_pos": num_pos,
            "dist_rank": self.dist_rank,
        }
        return pprint.pformat(repr_dict, indent=2)

    @staticmethod
    def gather_embeddings(embedding: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            embedding_gathered = gather_from_all(embedding)
        else:
            embedding_gathered = embedding
        return embedding_gathered
