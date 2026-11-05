import torch
import numpy as np
from Distributed_Danilo.config import device


def to_tensor(x):
    return torch.from_numpy(x).contiguous().float().to(device) if isinstance(x, np.ndarray) else x


class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = (logger is not None)
        self.logger = logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)
