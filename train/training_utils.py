from collections import deque

import torch
from torch import distributed as dist

from utils.dist_utils import is_dist_avail_and_initialized


class SmoothedValue(object):
    """
    Provides functionality for tracking and smoothing numerical values using a sliding window.

    This class is used to compute and track smoothed statistics (such as median, average, etc.)
    over a sliding window of numerical inputs. It is particularly useful in scenarios
    where real-time or batch statistics need to be computed and displayed, such as monitoring
    training progress in machine learning workflows.

    Attributes:
        deque (deque): A deque object to store recent values up to a maximum specified window size.
        total (float): The cumulative sum of all values added to the object.
        count (int): The cumulative count of values added.
        fmt (str): A format string used for displaying the object as a string.

    Initializer:
        __init__(window_size: int = 20, fmt: str = None)
            Creates a SmoothedValue instance with an optional window size and format string.

    Methods:
        update(value: Union[int, float], n: int = 1)
            Adds a new value to the tracking statistics, optionally incrementing by a specified amount.

        synchronize_between_processes()
            Synchronizes cumulative statistics (count and total) across multiple processes
            when distributed processing is enabled. Note: the sliding window is not synchronized.

    Properties:
        median
            Returns the median of the tracked values in the sliding window.

        avg
            Returns the mean of the tracked values in the sliding window.

        global_avg
            Returns the global average of all tracked values across the entire lifespan of the object.

        max
            Returns the maximum value within the sliding window.

        value
            Returns the most recent value in the sliding window.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
