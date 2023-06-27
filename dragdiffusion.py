import torch, math
import numpy as np
from model import Diffusion
import torch.nn.functional as functional

def motion_supervision():
    ...

@torch.no_grad()
def point_tracking():
    ...

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class DragDiffusion():
    def __init__(self, device):
        self._device = device

    def load_ckpt(self, path):
        print(f'loading checkpoint from {path}')
        ...
        print('loading checkpoint successed!')

    def to(self, device):
        if self._device != device:
            ...
            self._device = device

    @torch.no_grad()
    def generate_image(self, seed):
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, 512).astype(np.float32)
        ).to(self._device)
        ...

    @property
    def device(self):
        return self._device

    def __call__(self, *args, **kwargs):
        ...

    def step(self, points):
        ...
