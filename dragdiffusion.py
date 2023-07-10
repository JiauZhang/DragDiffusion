import torch, math
import numpy as np
from model import Diffusion
import torch.nn.functional as functional
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler

def motion_supervision(F0, F, pi, ti, r1=1, M=None):
    width = F0.shape[-1]
    pi = (pi[0] // 8, pi[1] // 8)
    ti = (ti[0] // 8, ti[1] // 8)
    dw, dh = ti[0] - pi[0], ti[1] - pi[1]
    norm = math.sqrt(dw**2 + dh**2)
    w = (max(0, pi[0] - r1), min(width, pi[0] + r1))
    h = (max(0, pi[1] - r1), min(width, pi[1] + r1))
    d = torch.tensor(
        (dw / norm, dh / norm),
        dtype=F.dtype, device=F.device,
    ).reshape(1, 1, 1, 2)
    grid_h, grid_w = torch.meshgrid(
        torch.tensor(range(h[0], h[1])),
        torch.tensor(range(w[0], w[1])),
        indexing='xy',
    )
    grid = torch.stack([grid_w, grid_h], dim=-1).unsqueeze(0)
    grid = (grid.to(F.device) / width - 0.5) * 2
    grid_d = grid + 2 * d / width

    sample = functional.grid_sample(
        F, grid, mode='bilinear', padding_mode='border',
        align_corners=True,
    )
    sample_d = functional.grid_sample(
        F, grid_d, mode='bilinear', padding_mode='border',
        align_corners=True,
    )

    loss = (sample_d - sample.detach()).abs().mean(1).sum()

    return loss

@torch.no_grad()
def point_tracking(F0, F, pi, p0, r2=3):
    width = F0.shape[-1]
    pi = (pi[0] // 8, pi[1] // 8)
    p0 = (p0[0] // 8, p0[1] // 8)
    x = (max(0, pi[0] - r2), min(width, pi[0] + r2))
    y = (max(0, pi[1] - r2), min(width, pi[1] + r2))
    base = F0[..., p0[1], p0[0]].reshape(1, -1, 1, 1)
    diff = (F[..., y[0]:y[1], x[0]:x[1]] - base).abs().mean(1)
    idx = diff.argmin()
    dy = int(idx / (x[1] - x[0]))
    dx = int(idx % (x[1] - x[0]))
    npi = ((x[0] + dx) * 8, (y[0] + dy) * 8)
    return npi

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class DragDiffusion():
    def __init__(self, device, cache_dir, model_id='runwayml/stable-diffusion-v1-5'):
        self._device = device
        self.optimizer = None

        ddim_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.model = Diffusion.from_pretrained(
            model_id, scheduler=ddim_scheduler, torch_dtype=torch.float32,
            cache_dir=cache_dir,
        ).to(device)
        requires_grad(self.model.unet, False)

    def to(self, device):
        if self._device != device:
            self.model = self.model.to(device)
            self._device = device

    @torch.no_grad()
    def generate_image(self, prompt, seed, steps, guidance_scale=7.5, time_step=40):
        generator = torch.Generator(self._device).manual_seed(seed)
        self.guidance_scale = guidance_scale
        self.steps = steps
        images = self.model(
            prompt, generator=generator, num_inference_steps=steps,
            guidance_scale=guidance_scale, time_step=time_step,
        ).images
        return images[0]

    @property
    def device(self):
        return self._device

    def step(self, points):
        if self.optimizer is None:
            len_pts = (len(points) // 2) * 2
            if len_pts == 0:
                print('Select at least one pair of points')
                return False, None
            self.z_t = self.model.time_step_latent.detach().requires_grad_(True)
            self.optimizer = torch.optim.Adam([self.z_t], lr=2e-3)
            with torch.no_grad():
                self.F0 = self.model.one_step(self.z_t).detach()
            points = points[:len_pts]
            self.p0 = points[::2]
        self.optimizer.zero_grad()
        z_t_1 = self.model.one_step(self.z_t)
        loss = 0
        for i in range(len(self.p0)):
            loss += motion_supervision(self.F0, z_t_1, points[2*i], points[2*i+1])
        print(loss)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            z_t_1 = self.model.one_step(self.z_t)
        for i in range(len(self.p0)):
            points[2*i] = point_tracking(self.F0, z_t_1, points[2*i], self.p0[i])
        image = self.model.latent_to_image(
            self.z_t, guidance_scale=self.guidance_scale, steps=self.steps,
        )
        return True, (points, image)
