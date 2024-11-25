import logging
import hydra
import os
from os import listdir
from pathlib import Path
import json
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config import read_config
from src.tools.extract_joints import extract_joints
from src.stmc import read_submotions, process_submotions
from src.text import TextDuration
from src.tools.smpl_layer import SMPLH

def update_cfg(cfg, c):
    if hasattr(c, "diffusion") and hasattr(c["diffusion"], "weight"):
        cfg.diffusion.weight = c.diffusion.weight
    if hasattr(c, "diffusion") and hasattr(c["diffusion"], "mcd"):
        cfg.diffusion.mcd = c.diffusion.mcd

# avoid conflic between tokenizer and rendering
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logger = logging.getLogger(__name__)
fps = 20

@hydra.main(config_path="configs", config_name="generate", version_base="1.3")
def generate_testset(c: DictConfig):
    cfg = read_config(c.run_dir) # ho messo negli args di vscode, rompe tutto?
    update_cfg(cfg, c)

    logger.info("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np
    import torch
    
    ### Load the model
    logger.info("Loading the models")
    ckpt_path = c.ckpt
    split_dir = c.split.split("/")[0]

    logger.info("Loading the checkpoint")
    ckpt = torch.load(ckpt_path, map_location=c.device)
    
    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])
    # Evaluation mode
    diffusion.eval()
    diffusion.to(c.device)

    print(cfg)

    # Select only the files in the split
    with open(f"datasets/annotations/{cfg.dataset}/splits/{c.split}.txt", "r") as fr:
        split_lines = fr.readlines()
    testset_ids = [s.strip() for s in split_lines]
    # Load the TESTSET annotations file
    annotations = json.load(open(f"datasets/annotations/{cfg.dataset}/splits/{split_dir}/annotations_test.json"))
    if "smplrifke" in c.run_dir:# BRUTTO! è per dire che se prendo il task di multiset devo considerare un altro file
        annotations = json.load(open(f"datasets/annotations/{cfg.dataset}/annotations.json"))
    
    # Select only the files that are not yet generated if flag is set
    if hasattr(c, "only_not_generated") and c.only_not_generated:
        existing_files = [f"{Path(t).stem}" for t in listdir(f"{c.out_dir}") if t.endswith(".npy")]
        ids_to_generate = list(set([f"{l}" for l in testset_ids]) -  set(existing_files))
        testset_ids = ids_to_generate 

    testset_ids = sorted(testset_ids)

    ### Rendering
    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)
    # jointstype = "smpljoints"
    jointstype = "both"
    smplh = SMPLH(
        path="deps/smplh",
        jointstype=jointstype,
        input_pose_rep="axisangle",
        gender=c.gender,
    )

    n_sequences = len(testset_ids)
    at_a_time = 50 if 50 < n_sequences else n_sequences
    iterator = np.array_split(np.arange(n_sequences), n_sequences // at_a_time)

    with torch.no_grad():
        for x in iterator:

            batch_ids = [testset_ids[y] for y in x]

            # Se uso il testo posso direttamente usare il testloader 
            if c.input_type == "text":
                logger.info(f"Reading ({max(x)}/{n_sequences}) the texts {batch_ids}")

                texts_durations = []
                for idx in batch_ids:
                    text = annotations[idx]["annotations"][0]["text"]
                    duration_s = annotations[idx]["annotations"][0]["end"] - annotations[idx]["annotations"][0]["start"]
                    duration = int(fps * float(duration_s))
                    texts_durations.append(TextDuration(text, duration))

                n_motions = len(texts_durations)

                infos = {
                    "texts_durations": texts_durations,
                    "all_lengths": [x.duration for x in texts_durations],
                    "all_texts": [x.text for x in texts_durations],
                }
                infos["output_lengths"] = infos["all_lengths"]

            # Se invece uso le submotions devo prenderle dalla cartella relativa
            elif c.input_type == "submotions":
                submotions_dir = c.submotions_dir
                logger.info(f"Reading ({max(x)}/{n_sequences}) from {submotions_dir} the submotions ids {batch_ids}")
                interval_overlap = int(fps * c.overlap_s)

                submotions = []
                for idx in batch_ids:
                    timel = read_submotions(f"{submotions_dir}/{idx}.txt", fps)[0]
                    submotions.append(timel)

                n_motions = len(submotions)

                infos = process_submotions(submotions, interval_overlap, uncond=(not cfg.diffusion.mcd), bodyparts=(not cfg.diffusion.mcd)) 
                infos["output_lengths"] = infos["max_t"]

            infos["featsname"] = cfg.motion_features
            infos["guidance_weight"] = c.guidance

            import src.prepare  # noqa
            import pytorch_lightning as pl
            import numpy as np
            import torch
            torch.manual_seed(30)
            from src.model.text_encoder import TextToEmb

            modelpath = cfg.data.text_encoder.modelname
            mean_pooling = cfg.data.text_encoder.mean_pooling
            text_model = TextToEmb(modelpath=modelpath, mean_pooling=mean_pooling, device=c.device)

            logger.info("Generate the function")

            ### Output directory and files
            gen_dir = c.out_dir
            os.makedirs(gen_dir, exist_ok=True)
            for idx in batch_ids:
                original_text = annotations[idx]["annotations"][0]["text"]
                start = 0 
                end = annotations[idx]["annotations"][0]["end"] - annotations[idx]["annotations"][0]["start"]
                meta = f"{original_text} # {start} # {end}\n"
                if c.input_type == "submotions":
                    with open(f"{submotions_dir}/{idx}.txt", "r") as f:
                        lines = f.read()
                    meta += "-"*50 + "\n"+ lines

                with open(f"{gen_dir}/{idx}.txt", "w") as f:
                    f.write(meta)
            vext = ".mp4"
            joints_video_paths = [f"{gen_dir}/{idx}{vext}" for idx in batch_ids]
            smpl_video_paths = [f"{gen_dir}/{idx}{vext}" for idx in batch_ids]
            npy_paths = [f"{gen_dir}/{idx}.npy" for idx in batch_ids]
            logger.info(f"All the output videos will be saved in: {gen_dir}")
            #### 

            if c.seed != -1:
                pl.seed_everything(c.seed)
        
            ### Inference    
            tx_emb = torch.stack([ text_model(text) for text in infos["all_texts"]])
            tx_emb_uncond = torch.stack([text_model("") for _ in infos["all_texts"]])

            if isinstance(tx_emb, torch.Tensor):
                tx_emb = {
                    "x": tx_emb,
                    "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(c.device),
                }
                tx_emb_uncond = {
                    "x": tx_emb_uncond,
                    "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(c.device),
                }
                infos["tx_emb_core"]  = {
                    "x": tx_emb_uncond["x"][0].reshape(1, 1, -1).repeat(n_motions,1,1),# (M, 1, 512)
                    "length": torch.tensor([1 for _ in range(n_motions)]).to(
                        c.device
                    ),
                }

            xstarts = diffusion(tx_emb, tx_emb_uncond, infos).cpu()

            for idx, (xstart, length) in enumerate(zip(xstarts, infos["output_lengths"])):
                xstart = xstart[:length]

                output = extract_joints(
                    xstart,
                    infos["featsname"],
                    fps=fps,
                    value_from=c.value_from,
                    smpl_layer=smplh,
                )

                joints = output["joints"]
                path = npy_paths[idx]
                np.save(path, joints)

                path_vertices = path.replace(".npy", "_verts.npy")
                np.save(path_vertices, output["vertices"])

                if hasattr(c, "animations") and c.animations is True:
                    logger.info(f"Joints rendering {idx}")
                    joints_renderer(
                        joints, title="", output=joints_video_paths[idx], canonicalize=False
                    )
                    print(joints_video_paths[idx])

                    smpl_renderer(
                        output["vertices"], title="", output=smpl_video_paths[idx]
                    )


if __name__ == "__main__":
    generate_testset()