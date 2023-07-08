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
        requires_grad(self.model, False)

    def to(self, device):
        if self._device != device:
            self.model = self.model.to(device)
            self._device = device

    @torch.no_grad()
    def generate_image(self, prompt, seed, steps, guidance_scale=7.5, time_step=40):
        generator = torch.Generator(self._device).manual_seed(seed)
        images = self.model(
            prompt, generator=generator, num_inference_steps=steps,
            guidance_scale=guidance_scale, time_step=time_step,
        ).images
        return images[0]

    @property
    def device(self):
        return self._device

    def __call__(self, *args, **kwargs):
        ...

    def step(self, points):
        ...
