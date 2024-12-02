import logging
import hydra
import os
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config import read_config
from src.tools.smpl_layer import SMPLH
from src.stmc import read_submotions, process_submotions

# avoid conflic between tokenizer and rendering
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="generate_mtt", version_base="1.3")
def generate_folder(c: DictConfig):
    logger.info("Prediction script")

    assert c.baseline in ["none", "sinc", "sinc_lerp", "singletrack", "onetext"]

    mtt_name = "mtt"
    if c.baseline == "onetext":
        mtt_file = "mtt/baselines/MTT_onetext.txt"
    elif c.baseline == "singletrack":
        mtt_file = "mtt/baselines/MTT_singletrack.txt"
    else:
        mtt_file = "mtt/MTT.txt"

    cfg = read_config(c.run_dir)
    cfg.diffusion.weight = c.diffusion.weight
    cfg.diffusion.mcd = c.diffusion.mcd
    fps = cfg.data.motion_loader.fps

    interval_overlap = int(fps * c.overlap_s)

    logger.info("Reading the timelines")
    all_timelines = read_submotions(mtt_file, fps)
    n_sequences = len(all_timelines)

    logger.info("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np
    import torch
    from src.model.text_encoder import TextToEmb
    from src.tools.extract_joints import extract_joints

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, f"logs/checkpoints/{ckpt_name}.ckpt")
    logger.info("Loading the checkpoint")

    ckpt = torch.load(ckpt_path, map_location=c.device)
    # Models
    logger.info("Loading the models")

    # Diffusion model
    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])

    # Evaluation mode
    diffusion.eval()
    diffusion.to(c.device)

    # in case we want to get the joints from SMPL
    smplh = SMPLH(
        path="deps/smplh",
        jointstype="both",
        input_pose_rep="axisangle",
        gender="neutral",
    )

    modelpath = cfg.data.text_encoder.modelname
    mean_pooling = cfg.data.text_encoder.mean_pooling
    text_model = TextToEmb(modelpath=modelpath, mean_pooling=mean_pooling, device=c.device)

    out_path = os.path.join(c.run_dir,f"generations_{mtt_name}_M" + str(ckpt_name) + "_D" + c.value_from + "_S" + (f'mcd_G{c.diffusion.weight}' if str(c.diffusion.mcd) else 'stmc'),)

    if c.baseline != "none":
        out_path += "_baseline_" + c.baseline

    if c.overlap_s != 0.5:
        out_path += "_intervaloverlap_" + str(c.overlap_s)

    os.makedirs(out_path, exist_ok=True)
    logger.info(f"The results (joints) will be saved in: {out_path}")

    if c.seed != -1:
        pl.seed_everything(c.seed)

    at_a_time = 50
    iterator = np.array_split(np.arange(n_sequences), n_sequences // at_a_time)

    with torch.no_grad():
        for x in iterator:
            timelines = [all_timelines[y] for y in x]
            npy_paths = [os.path.join(out_path, str(y).zfill(4) + ".npy") for y in x]

            if "sinc" in c.baseline:
                # No extension and no unconditional transitions
                infos = process_submotions(
                    timelines, interval_overlap, extend=False, uncond=False
                )
            else:
                infos = process_submotions(timelines, interval_overlap, uncond=(not cfg.diffusion.mcd), bodyparts=(not cfg.diffusion.mcd))

            infos["baseline"] = c.baseline
            infos["output_lengths"] = infos["max_t"]
            infos["featsname"] = cfg.motion_features
            infos["guidance_weight"] = c.guidance

            tx_emb = text_model(infos["all_texts"])
            tx_emb_uncond = text_model(["" for _ in infos["all_texts"]])

            if isinstance(tx_emb, torch.Tensor):
                tx_emb = {
                    "x": tx_emb[:, None],
                    "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(c.device),
                }
                tx_emb_uncond = {
                    "x": tx_emb_uncond[:, None],
                    "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(c.device),
                }
                n_motions = len(timelines)
                infos["tx_emb_core"]  = {
                    "x": tx_emb_uncond["x"][0].reshape(1, 1, -1).repeat(n_motions,1,1),# (M, 1, 512)
                    "length": torch.tensor([1 for _ in range(n_motions)]).to(c.device),
                }

            xstarts = diffusion(tx_emb, tx_emb_uncond, infos).cpu()

            for idx, (length, npy_path) in enumerate(zip(infos["max_t"], npy_paths)):
                xstart = xstarts[idx, :length]
                output = extract_joints(
                    xstart,
                    infos["featsname"],
                    fps=fps,
                    value_from=c.value_from,
                    smpl_layer=smplh,
                )
                joints = output["joints"]

                # shape T, F
                np.save(npy_path, joints)


if __name__ == "__main__":
    generate_folder()
