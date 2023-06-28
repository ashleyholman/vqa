import torch
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_step=0):
        self.warmup_steps = warmup_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]