"""Sparse-task normalization over a full optimizer update, not each microbatch."""
from __future__ import annotations
import torch


def pooled_task_term(mean_loss, present_count, update_count):
    """Convert a microbatch task mean into its share of the update's task mean.

    Sum this across microbatches; do not divide by accumulation_steps again.
    Counts use training-label availability only. BatchNorm still sees each
    microbatch independently. This pools loss normalization, not BatchNorm.
    """
    if update_count <= 0:
        raise ValueError('An observed task requires a positive update count')
    return mean_loss * (present_count / update_count)
