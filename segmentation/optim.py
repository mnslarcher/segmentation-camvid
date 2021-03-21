from torch.optim.lr_scheduler import _LRScheduler


# controlla implementazione con end lr
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_step, power=0.9):
        super.__init__(optimizer)
        self.max_step = max_step
        self.power = power
        self.last_step = 0

    def get_lr(self):
        return [
            max(0.0, base_lr * (1.0 - self.last_step / self.max_step) ** self.power)
            for base_lr in self.base_lrs
        ]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1

        self.last_step = step if step else 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
