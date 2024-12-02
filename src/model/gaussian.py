import logging
from collections import defaultdict
from tqdm import tqdm

import torch

from .diffusion_base import DiffuserBase
from ..data.collate import length_to_mask, collate_tensor_with_padding
from src.stmc import combine_features_intervals, interpolate_intervals, bada_bim_core

import numpy as np
import os
import itertools

from src.tools.extract_joints import extract_joints
from src.tools.smpl_layer import SMPLH

smplh = SMPLH(
    path="deps/smplh",
    jointstype="both",
    input_pose_rep="axisangle",
    gender="male",
)

# Inplace operator: return the original tensor
# work with a list of tensor as well
def masked(tensor, mask):
    if isinstance(tensor, list):
        return [masked(t, mask) for t in tensor]
    tensor[~mask] = 0.0
    return tensor


logger = logging.getLogger(__name__)


def remove_padding_to_numpy(x, length):
    x = x.detach().cpu().numpy()
    return [d[:l] for d, l in zip(x, length)]


class GaussianDiffusion(DiffuserBase):
    name = "gaussian"

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
        mcd = False
    ):
        super().__init__(schedule, timesteps)

        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.lr = lr
        self.prediction = prediction

        self.reconstruction_loss = torch.nn.MSELoss(reduction="mean")

        # normalization
        self.motion_normalizer = motion_normalizer
        self.text_normalizer = text_normalizer

        self.weight = weight
        self.mcd = mcd
        self.original_timeline = True

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)

    def activate_composition_model(self):
        print(f"Num of trainable parametrers {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def configure_optimizers(self) -> None:
        return {"optimizer": torch.optim.AdamW(lr=self.lr, params=self.parameters())}

    def prepare_tx_emb(self, tx_emb):
        # Text embedding normalization
        if "mask" not in tx_emb:
            tx_emb["mask"] = length_to_mask(tx_emb["length"], device=self.device)
        tx = {
            "x": masked(self.text_normalizer(tx_emb["x"]), tx_emb["mask"]),
            "length": tx_emb["length"],
            "mask": tx_emb["mask"],
        }
        return tx

    def diffusion_step(self, batch, batch_idx, training=False):
        mask = batch["mask"]

        # normalization
        x = masked(self.motion_normalizer(batch["x"]), mask)
        y = {
            "length": batch["length"],
            "mask": mask,
            "tx": self.prepare_tx_emb(batch["tx"]),
            # the condition is already dropped sometimes in the dataloader
        }

        bs = len(x)
        # Sample a diffusion step between 0 and T-1
        # 0 corresponds to noising from x0 to x1
        # T-1 corresponds to noising from xT-1 to xT
        t = torch.randint(0, self.timesteps, (bs,), device=x.device)

        # Create a noisy version of x
        # no noise for padded region
        noise = masked(torch.randn_like(x), mask)
        xt = self.q_sample(xstart=x, t=t, noise=noise)
        xt = masked(xt, mask)

        # denoise it
        # no drop cond -> this is done in the training dataloader already
        # give "" instead of the text
        # denoise it
        output = masked(self.denoiser(xt, y, t), mask)

        # Predictions
        xstart = masked(self.output_to("x", output, xt, t), mask)
        xloss = self.reconstruction_loss(xstart, x)
        loss = {"loss": xloss}
        return loss

    def training_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx, training=True)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"train_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        bs = len(batch["x"])
        loss = self.diffusion_step(batch, batch_idx)
        for loss_name in sorted(loss):
            loss_val = loss[loss_name]
            self.log(
                f"val_{loss_name}",
                loss_val,
                on_epoch=True,
                on_step=False,
                batch_size=bs,
            )

        return loss["loss"]

    def on_train_epoch_end(self):
        dico = {
            "epoch": float(self.trainer.current_epoch),
            "step": float(self.trainer.global_step),
        }
        # reset losses
        self._saved_losses = defaultdict(list)
        self.losses = []
        self.log_dict(dico)

    # dispatch
    def forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        if "timeline" in infos:
            ff = self.stmc_forward
            if "baseline" in infos:
                if "sinc" in infos["baseline"]:
                    ff = self.sinc_baseline
            # for the other baselines stmc handle it
            # STMC generalize one text forward and DiffCollage
        else:
            ff = self.text_forward
        return ff(tx_emb, tx_emb_uncond, infos, progress_bar=progress_bar)

    def text_forward(
        self,
        tx_emb,
        tx_emb_uncond,
        infos,
        progress_bar=tqdm,
    ):
        # normalize text embeddings first
        device = self.device

        lengths = infos["all_lengths"]
        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        duration = max(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, duration, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

    def p_sample(self, xt, y, t):
        # guided forward
        output_cond = self.denoiser(xt, y, t)

        guidance_weight = y["infos"].get("guidance_weight", 1.0)

        if guidance_weight == 1.0:
            output = output_cond
        else:
            y_uncond = y.copy()  # not a deep copy
            y_uncond["tx"] = y_uncond["tx_uncond"]

            output_uncond = self.denoiser(xt, y_uncond, t)
            # classifier-free guidance
            output = output_uncond + guidance_weight * (output_cond - output_uncond)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(output, xt, t)

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise
        xstart = output
        return x_out, xstart

    def stmc_forward(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        device = self.device

        if self.mcd and self.original_timeline:
            infos["all_intervals"] = infos["original_timeline"] 
            infos["all_lengths"] = [x.end - x.start for x in list(itertools.chain.from_iterable(infos["all_intervals"]))]

        # the lengths of all the crops + uncondionnal
        lengths = infos["all_lengths"]
        n_frames = infos["n_frames"]
        n_seq = infos["n_seq"]

        mask = length_to_mask(lengths, device=device)

        y = {   
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }
        if self.mcd:
            length_core = [n_frames for _ in range(n_seq)]
            mask_core =  length_to_mask(length_core, device=device)
            y["tx_core"] = self.prepare_tx_emb(infos["tx_emb_core"])
            y["length_core"]= length_core
            y["mask_core"]= mask_core

        bs = len(lengths)
        nfeats = self.denoiser.nfeats

        shape = n_seq, n_frames, nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t_seq = torch.full((n_seq,), diffusion_step)
            t_bs = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample_stmc(xt, y, t_seq, t_bs)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart

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
            combine_features_intervals(output, y["infos"], output_xt)
        else:
            y_core = y.copy()  # not a deep copy
            y_core["tx"] = y_core["tx_core"]
            y_core["length"] = y_core["length_core"]
            y_core["mask"] = y_core["mask_core"]
            output_core = self.denoiser(xt, y_core, t_bs[:n_seq])
            output_xt = self.compose_diffusion(output_core, output_cond, output_uncond, y["infos"], output=output_xt)

        mean, sigma = self.q_posterior_distribution_from_output_and_xt(
            output_xt, xt, t_seq
        )

        noise = torch.randn_like(mean)
        x_out = mean + sigma * noise

        xstart = output_xt
        return x_out, xstart

    def sinc_baseline(self, tx_emb, tx_emb_uncond, infos, progress_bar=tqdm):
        device = self.device

        # the lengths of all the crops + uncondionnal
        lengths = infos["all_lengths"]
        n_frames = infos["n_frames"]
        n_seq = infos["n_seq"]

        mask = length_to_mask(lengths, device=device)

        y = {
            "length": lengths,
            "mask": mask,
            "tx": self.prepare_tx_emb(tx_emb),
            "tx_uncond": self.prepare_tx_emb(tx_emb_uncond),
            "infos": infos,
        }

        bs = len(lengths)
        nfeats = self.denoiser.nfeats

        shape = bs, max(lengths), nfeats
        xt = torch.randn(shape, device=device)

        iterator = range(self.timesteps - 1, -1, -1)
        if progress_bar is not None:
            iterator = progress_bar(list(iterator), desc="Diffusion")

        for diffusion_step in iterator:
            t_bs = torch.full((bs,), diffusion_step)
            xt, xstart = self.p_sample(xt, y, t_bs)

        # at the end recombine
        shape = n_seq, n_frames, nfeats
        output = torch.zeros(shape, device=device)

        xstart = combine_features_intervals(xstart, infos, output)

        if "lerp" in infos["baseline"] or "interp" in infos["baseline"]:
            # interpolate to smooth the results
            xstart = interpolate_intervals(xstart, infos)

        xstart = self.motion_normalizer.inverse(xstart)
        return xstart
    
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
                    val = self.weight * (output_cond[ii_offset, start : end, :] - output_uncond[ii_offset, start : end, :])
                    output[idx, x.start : x.end, :] += val
            
            offset += len(lengths[idx])

        return output
