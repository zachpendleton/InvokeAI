"""wrapper around part of Katherine Crowson's k-diffusion library, making it call compatible with other Samplers"""
import k_diffusion as K
import torch
import torch.nn as nn
from ldm.dream.devices import choose_torch_device
from ldm.models.diffusion.sampler import Sampler

# just for debugging
from PIL import Image
from einops import rearrange, repeat
import numpy as np

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


class KSampler(Sampler):
    def __init__(self, model, schedule='lms', device=None, **kwargs):
        denoiser = K.external.CompVisDenoiser(model)
        super().__init__(
            denoiser,
            schedule,
            steps=model.num_timesteps,
        )
        self.ds    = None
        self.s_in  = None

        def forward(self, x, sigma, uncond, cond, cond_scale):
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)
            cond_in = torch.cat([uncond, cond])
            uncond, cond = self.inner_model(
                x_in, sigma_in, cond=cond_in
            ).chunk(2)
            return uncond + (cond - uncond) * cond_scale

    def make_schedule(
            self,
            ddim_num_steps,
            ddim_discretize='uniform',
            ddim_eta=0.0,
            model=None,
            verbose=False,
    ):
        super().make_schedule(
            ddim_num_steps,
            ddim_discretize='uniform',
            ddim_eta=0.0,
            model=self.model.inner_model,   # use the inner model to make the schedule, not the denoiser wrapped model
            verbose=False,
        )            

    def do_sampling(
            self,
            cond,
            shape,
            **kwargs
    ):
        # callback = kwargs['img_callback']
        # def route_callback(k_callback_values):
        #     if callback is not None:
        #         callback(k_callback_values['x'], k_callback_values['i'])

        # kwargs['img_callback']=route_callback
        return super().do_sampling(cond,shape,**kwargs)

    # most of these arguments are ignored and are only present for compatibility with
    # other samples
    @torch.no_grad()
    def p_sample(
            self,
            img,
            cond,
            ts,
            index,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            **kwargs,
    ):
        model_wrap_cfg = CFGDenoiser(self.model)
        extra_args = {
            'cond': cond,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        if self.s_in is None:
            self.s_in  = img.new_ones([img.shape[0]])
        if self.ds is None:
            self.ds = []
        img =  K.sampling.__dict__[f'_{self.schedule}'](
            model_wrap_cfg,
            img,
            self.sigmas,
            len(self.sigmas)-index-1,  # adjust for reverse index in ddim/plms and 1-based indexing in trange
            s_in = self.s_in,
            ds   = self.ds,
            extra_args=extra_args,
        )

        return img, None, None

    def get_initial_image(self,x_T,shape,steps):
        if x_T is None:
            return (
                torch.randn(shape, device=self.device)
                * self.sigmas[0]
            )   # for GPU draw
        else:
            return x_T * self.sigmas[0]
    
    def prepare_to_sample(self,steps):
        self.sigmas = self.model.get_sigmas(steps)
        self.ds    = None
        self.s_in  = None
