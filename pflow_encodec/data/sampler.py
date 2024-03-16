from typing import List

import torch
from torch.utils.data.distributed import DistributedSampler

from pflow_encodec.utils.pylogger import RankedLogger

logger = RankedLogger(__name__)


class DistributedBucketSampler(DistributedSampler):
    """Distributed Bucket Sampler for dynamic batching.

    it gathers samples with similar length into a batch. each batch comes from a single bucket.
    bucket[i] contains samples with length in (boundaries[i], boundaries[i+1]]. samples with length < first bucket and
    length > last bucket will be discarded. dataset.lengths should be sorted in ascending order.
    Args:
        dataset: dataset to sample from. it should have a lengths attribute
        batch_durations: number of frames in a batch
        boundaries: a list of boundaries for bucketing, samples with length in (boundaries[i], boundaries[i+1]]
                    will be put into bucket i.
    """

    def __init__(
        self,
        dataset,
        batch_durations: float,
        boundaries: List[float],
        num_replicas=None,
        rank=None,
        shuffle: bool = True,
        drop_last: bool = True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, drop_last=drop_last)

        self.durations = dataset.audio_durations
        self.batch_durations = batch_durations
        self.boundaries = boundaries

        self.buckets = self._create_bucket()
        logger.info(f"Created {len(self.buckets)} buckets")
        logger.info(f"Boundaries: {self.boundaries}")
        bucket_sizes = [len(bucket) for bucket in self.buckets]
        logger.info(f"Bucket sizes: {bucket_sizes}")
        self.batches = self._create_batches()

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def _create_bucket(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.durations)):
            length = self.durations[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        return buckets

    def _create_batches(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        buckets = self.buckets
        indices = []
        if self.shuffle:
            for bucket in buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in buckets:
                indices.append(list(range(len(bucket))))
        batches = []
        for bucket_id, bucket in enumerate(buckets):
            bucket_indices = indices[bucket_id]
            bucket_batches = []
            current_batch = []
            durations = 0
            # since we can not guarantee every process has the same number of batches, iterate all samples to generate batches
            for bucket_idx in bucket_indices:
                sample_idx = bucket[bucket_idx]
                sample_duration = self.durations[sample_idx]
                if durations + sample_duration > self.batch_durations and current_batch:
                    bucket_batches.append(current_batch)
                    current_batch = []
                    durations = 0
                durations += sample_duration
                current_batch.append(sample_idx)

            if not bucket_batches:
                # there's no batch made, just append the current batch
                bucket_batches.append(current_batch)
            elif current_batch and not self.drop_last:
                # there's still samples left in the current batch
                bucket_batches.append(current_batch)
            if len(bucket_batches) % self.num_replicas != 0:
                # the number of batches should be a multiple of num_replicas, duplicate random batches to make it so
                remainder = self.num_replicas - (len(bucket_batches) % self.num_replicas)
                assert remainder > 0
                for _ in range(remainder):
                    random_idx = torch.randint(0, len(bucket_batches), (1,), generator=g).item()
                    bucket_batches.append(bucket_batches[random_idx])
            batches.extend(bucket_batches)
        assert len(batches) % self.num_replicas == 0
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        return batches

    def __iter__(self):
        """# of batches should be multiple of num_replicas"""
        self.batches = self._create_batches()
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
