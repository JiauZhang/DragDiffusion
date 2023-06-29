import torch, math
import numpy as np
from model import Diffusion
import torch.nn.functional as functional
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler

def motion_supervision():
    ...

@torch.no_grad()
def point_tracking():
    ...

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class DragDiffusion():
    def __init__(self, device, cache_dir, model_id='runwayml/stable-diffusion-v1-5'):
        self._device = device

        ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.model = Diffusion.from_pretrained(
            model_id, scheduler=ddim_scheduler, torch_dtype=torch.float32,
            cache_dir=cache_dir,
        ).to(device)

    def to(self, device):
        if self._device != device:
            self.model = self.model.to(device)
            self._device = device

    @torch.no_grad()
    def generate_image(self, seed, text):
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
