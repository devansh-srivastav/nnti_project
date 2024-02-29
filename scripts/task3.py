import argparse
import torch 

import datasets
import numpy as np
import transformers
import wandb

MODEL_NAME = "facebook/xglm-564M"

########################################################
# Entry point
########################################################


import torch
from torch import nn
from torch.optim import Optimizer

class BitFitOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, pretrained_params=None):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if pretrained_params is None:
            raise ValueError("pretrained_params must be provided for BitFit")

        defaults = dict(lr=lr, pretrained_params=pretrained_params)
        super(BitFitOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p, pretrained_p in zip(group['params'], group['pretrained_params']):
                if p.grad is None:
                    continue
                d_p = p.grad

                # BitFit: only update if gradient sign matches pretrained weight sign
                mask = torch.sign(d_p) == torch.sign(pretrained_p.data)
                p.data.add_(d_p, alpha=-group['lr'] * mask.float())

# Usage
model = ...  # your model here
pretrained_model = ...  # your pretrained model here

optimizer = BitFitOptimizer(model.parameters(), lr=0.01, pretrained_params=pretrained_model.parameters())


if __name__ == "__main__":
    # TODO: your code goes here
    pass
