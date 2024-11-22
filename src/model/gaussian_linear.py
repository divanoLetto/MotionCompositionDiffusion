from collections import defaultdict

import torch

from ..data.collate import collate_tensor_with_padding
from src.stmc import SubMotion
from .gaussian import GaussianDiffusion
from abc import ABC, abstractmethod


class LinearGaussianDiffusion(GaussianDiffusion):
    name = "linear wights gaussian"

    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        prediction: str = "x",
        lr: float = 2e-4,
        weight = 1,
        badabim = True
    ):
        super().__init__(denoiser, schedule, timesteps, motion_normalizer, text_normalizer, prediction, lr, weight, badabim)

    
    def p_sample_stmc(self, xt, y, t_seq, t_bs):
        n_seq = y["infos"]["n_seq"]
        all_intervals = y["infos"]["all_intervals"]
        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        x_lst = []
        for idx, intervals in enumerate(all_intervals):
            x_lst.extend([xt[idx, x.start : x.end] for x in intervals])

        lengths = [len(x) for x in x_lst]
        # assert lengths == y["length"]

        xx = collate_tensor_with_padding(x_lst)
        output = self.denoiser(xx, y, t_bs) # GPU memory

        output_cond = output
        y_uncond = y.copy()  # not a deep copy
        y_uncond["tx"] = y_uncond["tx_uncond"]

        output_uncond = self.denoiser(xx, y_uncond, t_bs) # GPU memory

        if guidance_weight != 1.0:
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        output_xt = 0 * xt

        if not self.mcd:
            raise NameError("Not implemented")
        else:
            y_core = y.copy()  # not a deep copy
            y_core["tx"] = y_core["tx_core"]
            y_core["length"] = y_core["length_core"]
            y_core["mask"] = y_core["mask_core"]
            output_core = self.denoiser(xt, y_core, t_bs[:n_seq])

            y["infos"]["diffusion_step"] = t_seq[0].item()
            output_xt = self.compose_diffusion(output_core, output_cond, output_uncond, y["infos"], output=output_xt)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(
            output_xt, xt, t_seq
        )

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise

        xstart = output_xt
        return x_out, xstart

    
    def compose_diffusion(self, output_core, output_cond, output_uncond, infos, output): 
        # FULL AND
        all_intervals =  infos["all_intervals"]
        lengths = infos["lengths"]

        real_nseq = len(output)

        # base Uncond 
        output += output_core[:real_nseq] 

        # I use an offset because we have several timeline per batch
        offset = 0
        for idx in range(real_nseq):

            for ii, x in enumerate(all_intervals[idx]):
                end = x.end - x.start
                start = 0
                ii_offset = ii + offset
                if hasattr(x, 'text'):
                    # Spatial stiching
                    current_weight = self.weight * (1-infos["diffusion_step"]/99)
                    val = current_weight * (output_cond[ii_offset, start : end, :] - output_uncond[ii_offset, start : end, :])
                    output[idx, x.start : x.end, :] += val
            
            offset += len(lengths[idx])

        return output