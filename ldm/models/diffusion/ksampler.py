"""wrapper around part of Katherine Crowson's k-diffusion library, making it call compatible with other Samplers"""
import k_diffusion as K
import torch
import torch.nn as nn
from ldm.dream.devices import choose_torch_device
from ldm.models.diffusion.sampler import Sampler

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
#        print(f'ddim_num_steps={ddim_num_steps}') # want total steps here (50)
#        self.sigmas = self.model.get_sigmas(ddim_num_steps)
        
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
        if self.model_wrap is None:
            self.model_wrap = CFGDenoiser(self.model)
        extra_args = {
            'cond': cond,
            'uncond': unconditional_conditioning,
            'cond_scale': unconditional_guidance_scale,
        }
        if self.s_in is None:
            self.s_in  = img.new_ones([img.shape[0]])
        if self.ds is None:
            self.ds = []

        # terrible, confusing names here
        steps = self.ddim_num_steps
        t_enc = self.ddim_steps
        
        # sigmas is a full steps in length, but t_enc might
        # be less. We start in the middle of the sigma array
        # and work our way to the end after t_enc steps.
        # index starts at t_enc and works its way to zero,
        # so the actual formula for indexing into sigmas:
        # sigma_index = (steps-index)
        s_index = t_enc - index -1
        img =  K.sampling.__dict__[f'_{self.schedule}'](
            self.model_wrap,
            img,
            self.sigmas,
            s_index,
            s_in = self.s_in,
            ds   = self.ds,
            extra_args=extra_args,
        )

        return img, None, None

    def get_initial_image(self,x_T,shape,steps):
        if x_T is not None:
            return x_T + x_T * self.sigmas[0]
        else:
            return (torch.randn(shape, device=self.device) * self.sigmas[0])
        
    def prepare_to_sample(self,steps):
        self.ddim_steps = steps
        self.model_wrap = None
        self.ds         = None
        self.s_in       = None
        print(f'steps={steps}') # want total steps here (37)
        self.sigmas = self.model.get_sigmas(steps)

 # unused code
            # def do_sampling(
    #         self,
    #         cond,
    #         shape,
    #         **kwargs
    # ):
    #     # callback = kwargs['img_callback']
    #     # def route_callback(k_callback_values):
    #     #     if callback is not None:
    #     #         callback(k_callback_values['x'], k_callback_values['i'])

    #     # kwargs['img_callback']=route_callback
    #     return super().do_sampling(cond,shape,**kwargs)

