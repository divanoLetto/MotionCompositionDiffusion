## MYCODE

import logging
from collections import defaultdict
from tqdm import tqdm

import torch

from .diffusion_base import DiffuserBase
from ..data.collate import length_to_mask, collate_tensor_with_padding
from src.stmc import TextInterval, Interval, combine_features_intervals, interpolate_intervals
from .gaussian import GaussianDiffusion, masked
import random 


class GaussianDiffusionMultisteps(GaussianDiffusion):
    name = "gaussian_multisteps"

    def __init__(
        self,
        denoiser,
        schedule,
        timesteps,
        motion_normalizer,
        text_normalizer,
        prediction: str = "x",
        lr: float = 2e-4,
        weight_1 = 1, 
        weight_2 = 1, 
        weight_3 = 1,
        badabim = False,
        timesteps_pre = 20
    ):
        super().__init__(denoiser, schedule, timesteps, motion_normalizer, text_normalizer, prediction, lr, weight_1, weight_2, weight_3, badabim)
        self.timesteps_pre = timesteps_pre

    def stmc_forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        device = self.device

        # DEBUG! Togliere poi o stmc non fa
        if self.mcd and self.original_timeline:
            infos["all_intervals"] = infos["original_timeline"] 
            infos["all_lengths"] = [x.end -x.start for x in infos["all_intervals"][0]]

        xt_multistep = []
        xstart_multistep, xstart = [], None
        pre_steps_idx = find_separation_ids(infos["all_intervals"]) # [[0], [1, 2]]
        for indices in pre_steps_idx:
            infos_pre = infos.copy()
            infos_pre["all_intervals"] = [[infos["all_intervals"][0][idx] for idx in indices]]
            min_time = min([x.start for x in infos_pre["all_intervals"][0]])
            for int_id, interv in enumerate(infos_pre["all_intervals"][0]):
                infos_pre["all_intervals"][0][int_id] = TextInterval(text=interv.text, start=interv.start-min_time, end=interv.end-min_time, bodyparts=interv.bodyparts)
            infos_pre["all_lengths"] = [infos["all_lengths"][idx] for idx in indices]
            n_seq, nfeats = 1, 205
            lengths = infos_pre["all_lengths"]
            n_frames = max(lengths)

            mask = length_to_mask(lengths, device=device)
            length_core = [n_frames for _ in range(n_seq)]
            mask_core =  length_to_mask(length_core, device=device)
            tx_emb_ = {"x": tx_emb["x"][indices, :, :], "length":tx_emb["length"][indices]}
            tx_emb_uncond_ = {"x": tx_emb_uncond["x"][indices, :, :], "length":tx_emb_uncond["length"][indices]}
            y = {   
                "length": lengths,
                "mask": mask,
                "tx": self.prepare_tx_emb(tx_emb_),
                "tx_uncond": self.prepare_tx_emb(tx_emb_uncond_),
                "tx_core": self.prepare_tx_emb(infos_pre["tx_emb_core"]),
                "length_core": length_core,
                "mask_core": mask_core,
                "infos": infos_pre,
            }

            bs = len(lengths)
            shape = n_seq, n_frames, nfeats
            xt = torch.randn(shape, device=device)
            iterator = range(self.timesteps - 1, self.timesteps - self.timesteps_pre, -1)
            if progress_bar is not None:
                iterator = progress_bar(list(iterator), desc="Diffusion")
            for diffusion_step in iterator:
                t_seq = torch.full((n_seq,), diffusion_step)
                t_bs = torch.full((bs,), diffusion_step)
                xt, xstart = self.p_sample_stmc(xt, y, t_seq, t_bs)
            xt_multistep.append(xt)
            xstart_multistep.append(xstart)

         # the lengths of all the crops + uncondionnal
        lengths = infos["all_lengths"]
        n_frames = infos["n_frames"]
        n_seq = infos["n_seq"]

        mask = length_to_mask(lengths, device=device)

        length_core = [n_frames for _ in range(n_seq)]
        mask_core =  length_to_mask(length_core, device=device)

        y = {   
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "tx_core": self.prepare_tx_emb(infos["tx_emb_core"]),
            "length_core": length_core,
            "mask_core": mask_core,
            "infos": infos,
        }

        bs = len(lengths)
        nfeats = self.denoiser.nfeats

        shape = n_seq, n_frames, nfeats
        xt = torch.cat(xt_multistep, dim=1)

        xstart = None if xstart_multistep[0] is None else torch.cat(xstart_multistep, dim=1)
        iterator = range(self.timesteps - self.timesteps_pre - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t_seq = torch.full((n_seq,), diffusion_step)
            t_bs = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample_stmc(xt, y, t_seq, t_bs)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

def find_separation_ids(all_intervals):
    # metodo scemo che funziona solo per sequenza temporali divisibili perfettamente in due, senza sovrapposizione 
    mid_point = all_intervals[0][0].end
    division = [[],[]]
    for idx, interval in enumerate(all_intervals[0]):
        if interval.end <= mid_point:
            division[0].append(idx)
        else:
            division[1].append(idx)
    return division
