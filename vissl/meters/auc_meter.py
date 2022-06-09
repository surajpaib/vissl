import logging

from classy_vision.meters import ClassyMeter, register_meter
from classy_vision.generic.distributed_util import all_reduce_sum, gather_from_all

from vissl.config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, \
                            balanced_accuracy_score, \
                            recall_score, precision_score, confusion_matrix

@register_meter("auc_meter")
class AUCMeter(ClassyMeter):
    """
    Add docuementation on what this meter does

    Args:
        add documentation about each meter parameter
    """

    def __init__(self, meters_config: AttrDict):
        # implement what the init method should do like
        # setting variable to update etc.
        self.num_classes = meters_config.get("num_classes")
        self._total_sample_count = None
        self._curr_sample_count = None
        # self._meters = ?
        self.reset()

    @classmethod
    def from_config(cls, meters_config: AttrDict):
        """
        Get the AUCMeter instance from the user defined config
        """
        return cls(meters_config)

    @property
    def name(self):
        """
        Name of the meter
        """
        return "auc_binary_meter"

    @property
    def value(self):
        """
        Value of the meter globally synced. mean AP and AP for each class is returned
        """

        def sigmoid(X):
            return 1 / (1 + np.exp(-X))

        _, distributed_rank = get_machine_local_and_dist_rank()
        logging.info(
            f"Rank: {distributed_rank} AUC meter: "
        )


        y_true_hotencoded = self._targets.detach().numpy() # [n_samples, n_classes]
        y_score_logits = self._scores.detach().numpy() # [n_samples, n_classes] in logits
        y_true = np.argmax(y_true_hotencoded, axis = 1) # [n_samples,]
        y_score = sigmoid(y_score_logits)[:, 1] # [n_samples,]

        auroc = roc_auc_score(y_true, y_score)
        return {"AUC": {"0": auroc}}

    def gather_scores(self, scores: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            scores_gathered = gather_from_all(scores)
        else:
            scores_gathered = scores
        return scores_gathered

    def gather_targets(self, targets: torch.Tensor):
        """
        Do a gather over all embeddings, so we can compute the loss.
        Final shape is like: (batch_size * num_gpus) x embedding_dim
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # gather all embeddings.
            targets_gathered = gather_from_all(targets)
        else:
            targets_gathered = targets
        return targets_gathered

    def sync_state(self):
        """
        Globally syncing the state of each meter across all the trainers.
        Should perform distributed communications like all_gather etc
        to correctly gather the global values to compute the metric
        """
        # Communications
        self._curr_sample_count = all_reduce_sum(self._curr_sample_count)
        self._scores = self.gather_scores(self._scores)
        self._targets = self.gather_targets(self._targets)

        # Store results
        self._total_sample_count += self._curr_sample_count

        # Reset values until next sync
        self._curr_sample_count.zero_()

    def reset(self):
        """
        Reset the meter. Should reset all the meter variables, values.
        """
        self._scores = torch.zeros(0, self.num_classes, dtype=torch.float32)
        self._targets = torch.zeros(0, self.num_classes, dtype=torch.int8)
        self._total_sample_count = torch.zeros(1)
        self._curr_sample_count = torch.zeros(1)

    def __repr__(self):
        # implement what information about meter params should be
        # printed by print(meter). This is helpful for debugging
        return repr({"name": self.name, "value": self.value})

    def set_classy_state(self, state):
        """
        Set the state of meter. This is the state loaded from a checkpoint when the model
        is resumed
        """
        """
                Set the state of meter
                """
        # assert (
        #         self.name == state["name"]
        # ), f"State name {state['name']} does not match meter name {self.name}"
        # assert self.num_classes == state["num_classes"], (
        #     f"num_classes of state {state['num_classes']} "
        #     f"does not match object's num_classes {self.num_classes}"
        # )

        # Restore the state -- correct_predictions and sample_count.
        # self.reset()
        # self._total_sample_count = state["total_sample_count"].clone()
        # self._curr_sample_count = state["curr_sample_count"].clone()
        # self._scores = state["scores"]
        # self._targets = state["targets"]
        pass

    def get_classy_state(self):
        """
        Returns the states of meter that will be checkpointed. This should include
        the variables that are global, updated and affect meter value.
        """
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "scores": self._scores,
            "targets": self._targets,
            "total_sample_count": self._total_sample_count,
            "curr_sample_count": self._curr_sample_count,
        }

    def verify_target(self, target):
        """
        Verify that the target contains {0, 1} values only
        """
        assert torch.all(
            torch.eq(target, 0) + torch.eq(target, 1)
        ), "Target values should be either 0 OR 1"

    def validate(self, model_output, target) -> None:
        pass

    def update(self, model_output, target):
        """
        Update the meter every time meter is calculated
        """
        target = F.one_hot(target, num_classes=self.num_classes)
        self.validate(model_output, target)
        self.verify_target(target)

        self._curr_sample_count += model_output.shape[0]

        # scores as in logits I think
        curr_scores, curr_targets = self._scores, self._targets
        sample_count_so_far = curr_scores.shape[0]
        self._scores = torch.zeros(
            int(self._curr_sample_count[0]), self.num_classes, dtype=torch.float32
        )
        self._targets = torch.zeros(
            int(self._curr_sample_count[0]), self.num_classes, dtype=torch.int8
        )

        if sample_count_so_far > 0:
            self._scores[:sample_count_so_far, :] = curr_scores
            self._targets[:sample_count_so_far, :] = curr_targets
        self._scores[sample_count_so_far:, :] = model_output
        self._targets[sample_count_so_far:, :] = target
        del curr_scores, curr_targets