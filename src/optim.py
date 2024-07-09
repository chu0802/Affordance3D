import torch
import torch.nn as nn


class LearnableParams(nn.Module):
    def __init__(self, init_val=None, shape=None, dtype=torch.float32, func=None):
        super().__init__()
        self.func = func
        if init_val is not None:
            self.param = nn.Parameter(init_val).to(dtype=dtype)
        elif shape is not None:
            self.param = nn.Parameter(torch.randn(*shape)).to(dtype=dtype)
        else:
            raise RuntimeError("[!] init_val and shape cannot be both None!")

    def forward(self):
        if self.func is not None:
            return self.func(self.param)
        else:
            return self.param


class OptimWrapper(object):
    def __init__(
        self, params, lrs, optim_steps, momentum=0.9, optim_type="sgd", name=""
    ):
        self.params = params
        self.momentum = momentum
        self.optim_type = optim_type
        self.name = name
        if params is not None and len(params) > 0:
            if optim_type == "sgd":
                self.optimizer = torch.optim.SGD(params, lr=lrs[0], momentum=momentum)
            elif optim_type == "adam":
                self.optimizer = torch.optim.Adam(params, lr=lrs[0])
            elif optim_type == "adamw":
                self.optimizer = torch.optim.AdamW(params, lr=lrs[0])
            else:
                raise RuntimeError(f"{optim_type} is not supported!")

            assert len(lrs) == len(optim_steps)
            self.lrs = list()
            for idx in range(len(lrs)):
                self.lrs.extend([lrs[idx]] * optim_steps[idx])
        else:
            self.optimizer = None
            self.lrs = list()

    def zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def step_params(self):
        if self.optimizer is not None:
            self.optimizer.step()

    def step_lr(self):
        if self.optimizer is not None:
            lr = self.lrs.pop(0)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def get_lr(self):
        if self.optimizer is not None:
            return self.optimizer.param_groups[0]["lr"]
        else:
            return 0

    def get_name(self):
        return self.name
