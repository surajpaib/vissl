# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import queue
import PIL

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def get_mean_image(crop_size):
    """
    Helper function that returns a gray PIL image of the size specified by user.

    Args:
        crop_size (int): used to generate (crop_size x crop_size x 3) image.

    Returns:
        img: PIL Image
    """
    img = Image.fromarray(128 * np.ones((crop_size, crop_size, 3), dtype=np.uint8))
    return img


@contextlib.contextmanager
def with_temporary_numpy_seed(sampling_seed: int):
    """
    Context manager to run a specific portion of code with a given seed:
    resumes the numpy state after the execution of the block
    """
    original_random_state = np.random.get_state()
    np.random.seed(sampling_seed)
    yield
    np.random.set_state(original_random_state)


def unbalanced_sub_sampling(
    total_num_samples: int, num_samples: int, skip_samples: int = 0, seed: int = 0
) -> np.ndarray:
    """
    Given an original dataset of size 'total_size', sub_sample part of the dataset such that
    the sub sampling is deterministic (identical across distributed workers)
    Return the selected indices
    """
    with with_temporary_numpy_seed(seed):
        return np.random.choice(
            total_num_samples, size=skip_samples + num_samples, replace=False
        )[skip_samples:]


def balanced_sub_sampling(
    labels: np.ndarray, num_samples: int, skip_samples: int = 0, seed: int = 0
) -> np.ndarray:
    """
    Given all the labels of a dataset, sub_sample a part of the labels such that:
    - the number of samples of each label differs by at most one
    - the sub sampling is deterministic (identical across distributed workers)
    Return the indices of the selected labels
    """
    groups = {}
    for i, label in enumerate(labels):
        groups.setdefault(label, []).append(i)

    unique_labels = sorted(groups.keys())
    skip_quotient, skip_rest = divmod(skip_samples, len(unique_labels))
    sample_quotient, sample_rest = divmod(num_samples, len(unique_labels))
    assert (
        sample_quotient > 0
    ), "the number of samples should be at least equal to the number of classes"

    with with_temporary_numpy_seed(seed):
        for i, label in enumerate(unique_labels):
            label_indices = groups[label]
            num_label_samples = sample_quotient + (1 if i < sample_rest else 0)
            skip_label_samples = skip_quotient + (1 if i < skip_rest else 0)
            permuted_indices = np.random.choice(
                label_indices,
                size=skip_label_samples + num_label_samples,
                replace=False,
            )
            groups[label] = permuted_indices[skip_label_samples:]

    return np.concatenate([groups[label] for label in unique_labels])


class StatefulDistributedSampler(DistributedSampler):
    """
    More fine-grained state DataSampler that uses training iteration and epoch
    both for shuffling data. PyTorch DistributedSampler only uses epoch
    for the shuffling and starts sampling data from the start. In case of training
    on very large data, we train for one epoch only and when we resume training,
    we want to resume the data sampler from the training iteration.
    """

    def __init__(self, dataset, batch_size=None, seed: int = 0):
        """
        Initializes the instance of StatefulDistributedSampler. Random seed is set
        for the epoch set and data is shuffled. For starting the sampling, use
        the start_iter (set to 0 or set by checkpointing resuming) to
        sample data from the remaining images.

        Args:
            dataset (Dataset): Pytorch dataset that sampler will shuffle
            batch_size (int): batch size we want the sampler to sample
            seed (int): Seed for the torch generator.
        """
        super().__init__(dataset, shuffle=False, seed=seed)

        self.start_iter = 0
        self.batch_size = batch_size
        self.total_size = len(dataset) - (len(dataset) % self.num_replicas)
        self.num_samples = self.total_size // self.num_replicas
        logging.info(f"rank: {self.rank}: Sampler created...")

    def __iter__(self):
        # partition data into num_replicas and optionally shuffle within a rank
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)
        shuffling = torch.randperm(self.num_samples, generator=g).tolist()
        indices = np.array(
            list(
                range(
                    (self.rank * self.num_samples), (self.rank + 1) * self.num_samples
                )
            )
        )[shuffling].tolist()

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)

    def set_start_iter(self, start_iter):
        """
        Set the iteration number from which the sampling should start. This is
        used to find the marker in the data permutation order from where the
        sampler should start sampling.
        """
        self.start_iter = start_iter


class DeterministicDistributedSampler(StatefulDistributedSampler):
    """
    StatefulDistributedSampler that does not generates random permutations but
    instead uses a fixed assignment of samples for each rank.

    Useful for debugging: it gives 100% reproducible sample assignments.

    Args:
        dataset (Dataset): Pytorch dataset from which to sample elements
        batch_size (int): batch size we want the sampler to sample
    """

    def __init__(self, dataset, batch_size=None):
        super().__init__(dataset, batch_size=batch_size)
        logging.info(f"rank: {self.rank}: DEBUGGING sampler created...")

    def __iter__(self):
        # Cut the dataset in deterministic parts
        indices = list(
            range((self.rank * self.num_samples), (self.rank + 1) * self.num_samples)
        )

        # make sure we have correct number of samples per replica
        assert len(indices) == self.num_samples
        assert self.batch_size > 0, "batch_size not set for the sampler"

        # resume the sampler
        start_index = self.start_iter * self.batch_size
        indices = indices[start_index:]
        return iter(indices)


class QueueDataset(Dataset):
    """
    This class helps dealing with the invalid images in the dataset by using
    two queue. One queue is used to enqueue seen and valid images from previous
    batches. The other queue is used to dequeue. The class is implemented such
    that the same batch will never have duplicate images. If we can't dequeue a
    valid image, we return None for that instance.

    Args:
        queue_size: size the the queue (ideally set it to batch_size). Both queues
                    will be of the same size
    """

    def __init__(self, queue_size):

        self.queue_size = queue_size
        # we create a CPU queue to buffer the valid seen images. We use these
        # images to replace the invalid images in the minibatch
        # 2 queues (FIFO) per gpu of size = batch size per gpu (64 img):
        #   a) 1st queue is used only to dequeue seen images. We get images from
        #      this queue only to backfill.
        #   b) 2nd queue is used only to add the new incoming valid seen images
        self.queue_init = False
        self.dequeue_images_queue = None
        self.enqueue_images_queue = None

    def _init_queues(self):
        self.dequeue_images_queue = queue.Queue(maxsize=self.queue_size)
        self.enqueue_images_queue = queue.Queue(maxsize=self.queue_size)
        self.queue_init = True
        logging.info(f"QueueDataset enabled. Using queue_size: {self.queue_size}")

    def _refill_dequeue_buffer(self):
        dequeue_qsize = self._get_dequeue_buffer_size()
        for _ in range(self.queue_size - dequeue_qsize):
            try:
                self.dequeue_images_queue.put(
                    self.enqueue_images_queue.get(), block=True
                )
            except Exception:
                continue

    def _enqueue_valid_image(self, img):
        if self._get_enqueue_buffer_size() >= self.queue_size:
            return
        try:
            self.enqueue_images_queue.put(img, block=True, timeout=0.1)
            return
        except queue.Full:
            return

    def _dequeue_valid_image(self):
        if self._get_dequeue_buffer_size() == 0:
            return
        try:
            return self.dequeue_images_queue.get(block=True, timeout=0.1)
        except queue.Empty:
            return None

    def _get_enqueue_buffer_size(self):
        return self.enqueue_images_queue.qsize()

    def _get_dequeue_buffer_size(self):
        return self.dequeue_images_queue.qsize()

    def _is_large_image(self, sample):

        if isinstance(sample, PIL.Image.Image):
            h, w = sample.size
            if h * w > 10000000:
                return True
            return False

        elif isinstance(sample, np.ndarray):
            if sample.size > 10000000:
                return True
            return False

        return False

    def on_sucess(self, sample):
        """
        If we encounter a successful image and the queue is not full, we store it
        in the queue. One consideration we make further is: if the image is very
        large, we don't add it to the queue as otherwise the CPU memory will grow
        a lot.
        """
        if self._is_large_image(sample):
            return
        self._enqueue_valid_image(sample)
        if self.enqueue_images_queue.full() and not self.dequeue_images_queue.full():
            self._refill_dequeue_buffer()

    def on_failure(self):
        """
        If there was a failure in getting the origin image, we look into the queue
        if there is any valid seen image available. If yes, we dequeue and use this
        image in place of the failed image.
        """
        sample, is_success = None, False
        if self._get_dequeue_buffer_size() > 0:
            sample = self._dequeue_valid_image()
            if sample is not None:
                is_success = True
        return sample, is_success
