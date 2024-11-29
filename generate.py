import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from decompose.decompose_text2submotions import decompose, parser_json_MCD, parser_json_STMC
from src.config import read_config


# avoid conflic between tokenizer and rendering
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYOPENGL_PLATFORM"] = "egl"


@hydra.main(config_path="configs", config_name="generate", version_base="1.3")
def generate(c: DictConfig):
    print("Prediction script")

    assert c.input_type in ["text", "submotions", "auto"]

    exp_folder_name = os.path.splitext(os.path.split(c.submotions)[-1])[0]

    cfg = read_config(c.run_dir)
    cfg.diffusion.weight = c.diffusion.weight
    cfg.diffusion.mcd = c.diffusion.mcd

    fps = cfg.data.motion_loader.fps

    interval_overlap = int(fps * c.overlap_s)

    from src.stmc import read_submotions, process_submotions
    from src.text import read_texts

    if c.input_type == "auto" or "submotions":
        try:
            submotions = read_submotions(c.submotions, fps)
            print("Reading the submotions")
            n_motions = len(submotions)
            c.input_type = "submotions"
        except IndexError:
            c.input_type = "text"
    if c.input_type == "text":
        print("Reading the texts")
        texts_durations = read_texts(c.submotions, fps)
        n_motions = len(texts_durations)
        decomposed_dir = "decomposed_mcd" if cfg.diffusion.mcd else "decomposed_stmc"
        os.makedirs(f"{os.path.dirname(c.submotions)}/{decomposed_dir}", exist_ok=True)
        save_path = f"{os.path.dirname(c.submotions)}/{decomposed_dir}/{os.path.basename(c.submotions)}"

        # check if a decomposed file has already been saved
        if not os.path.exists(save_path):
            from openai import OpenAI
            from dotenv import load_dotenv
            # Load the .env file
            load_dotenv() 
            # Create the client using the API key
            client = OpenAI(api_key= os.getenv('OPENAI_API_KEY'),)
            gpt_model = "gpt-4o-mini"
            parser = parser_json_MCD if cfg.diffusion.mcd else parser_json_STMC
            instructions_path = "decompose/istructions/MCD_instruction.txt" if cfg.diffusion.mcd else "decompose/istructions/STMC_instruction.txt"
            train_file_path = f"datasets/annotations/{cfg.dataset}/splits/complex/gpt_train_texts.txt"
            examples_path = f"decompose/examples/MCD_examples_{cfg.dataset}.txt" if cfg.diffusion.mcd else f"decompose/examples/STMC_examples_{cfg.dataset}.txt"

            # Load istructions file
            with open(instructions_path, 'r') as f:
                instructions = f.read()

            ### CREATE ASSISTENT
            assistant = client.beta.assistants.create(
                model=gpt_model,
                instructions=instructions,
                name="Text to timeline assistant",
                tools=[{"type": "file_search"}],
                temperature=0.0,
            )
            ### UPLOAD FILE
            # Create a vector store
            vector_store = client.beta.vector_stores.create(name="texts_train")
            # Ready the files for upload to OpenAI
            file_paths = [train_file_path, examples_path]
            file_streams = [open(path, "rb") for path in file_paths]
            # Use the upload and poll SDK helper to upload the files, add them to the vector store,
            # and poll the status of the file batch for completion.
            print("Uploading file...")
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
            )
            # Print the status and the file counts of the batch to see the result of this operation.
            print(file_batch.status)
            print(file_batch.file_counts)

            assistant = client.beta.assistants.update(
                assistant_id=assistant.id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
            )
            m = f"{texts_durations[0].text} # 0 # {texts_durations[0].duration/fps}" 
            decomposed = decompose(m, assistant=assistant, parser=parser)

            # Save the decomposed file
            with open(f"{save_path}", 'w') as f:
                for submov in decomposed["decomposition"]:
                    text = submov["text"]
                    start = submov["start"]
                    end = submov["end"]
                    f.write(f"{text} # {start} # {end} # spine\n")
        
        submotions = read_submotions(save_path, fps)
        print("Reading the submotions")
        n_motions = len(submotions)
        c.input_type = "submotions"
    
    infos = process_submotions(submotions, interval_overlap, uncond=(not cfg.diffusion.mcd), bodyparts=(not cfg.diffusion.mcd))
    infos["output_lengths"] = infos["max_t"]
    if c.baseline != "none":
        infos["baseline"] = c.baseline

    print("Loading the libraries")
    import src.prepare  # noqa
    import pytorch_lightning as pl
    import numpy as np
    import torch

    infos["featsname"] = cfg.motion_features
    infos["guidance_weight"] = c.guidance

    ckpt_name = c.ckpt
    ckpt_path = os.path.join(c.run_dir, ckpt_name)
    print("Loading the checkpoint")

    ckpt = torch.load(ckpt_path, map_location=c.device)
    # Models
    print("Loading the models")

    # Rendering
    joints_renderer = instantiate(c.joints_renderer)
    smpl_renderer = instantiate(c.smpl_renderer)

    # Diffusion model
    # update the folder first, in case it has been moved
    cfg.diffusion.motion_normalizer.base_dir = os.path.join(c.run_dir, "motion_stats")
    cfg.diffusion.text_normalizer.base_dir = os.path.join(c.run_dir, "text_stats")

    print(cfg)

    diffusion = instantiate(cfg.diffusion)
    diffusion.load_state_dict(ckpt["state_dict"])

    # Evaluation mode
    diffusion.eval()
    diffusion.to(c.device)

    # jointstype = "smpljoints"
    jointstype = "both"

    from src.tools.smpl_layer import SMPLH

    smplh = SMPLH(
        path="deps/smplh",
        jointstype=jointstype,
        input_pose_rep="axisangle",
        gender=c.gender,
    )

    from src.model.text_encoder import TextToEmb

    modelpath = cfg.data.text_encoder.modelname
    mean_pooling = cfg.data.text_encoder.mean_pooling
    text_model = TextToEmb(
        modelpath=modelpath, mean_pooling=mean_pooling, device=c.device
    )

    print("Generate the function")

    sub_dir = "mcd" if cfg.diffusion.mcd else "stmc"
    video_dir = os.path.join(
        c.run_dir,
        f"generations_{sub_dir}",
        exp_folder_name,
    )
    os.makedirs(video_dir, exist_ok=True)

    with open(os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}.txt"), "w") as file:
        file.write(f"Original motion:\n- {texts_durations[0].text} - duration: {texts_durations[0].duration}\n")
        file.write(f"Decomposed submotions:\n")
        for sub in submotions[0]:
            file.write(f"- {sub.text} - start: {sub.start} - end {sub.end}\n")

    vext = ".mp4"

    joints_video_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}_joints{vext}")
        for idx in range(n_motions)
    ]

    smpl_video_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}_smpl{vext}")
        for idx in range(n_motions)
    ]

    npy_paths = [
        os.path.join(video_dir, f"{exp_folder_name}_{c.input_type}_{idx}.npy")
        for idx in range(n_motions)
    ]

    print(f"All the output videos will be saved in: {video_dir}")

    if c.seed != -1:
        pl.seed_everything(c.seed)
 
    with torch.no_grad():
        tx_emb = text_model(infos["all_texts"])
        tx_emb_uncond = text_model(["" for _ in infos["all_texts"]])

        if isinstance(tx_emb, torch.Tensor):
            tx_emb = {
                "x": tx_emb[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb))]).to(c.device),
            }
            tx_emb_uncond = {
                "x": tx_emb_uncond[:, None],
                "length": torch.tensor([1 for _ in range(len(tx_emb_uncond))]).to(
                    c.device
                ),
            }
            tx_emb_core = {
                "x": text_model([""])[:, None],
                "length": torch.tensor([1]).to(
                    c.device
                ),
            }
            infos["tx_emb_core"] = tx_emb_core

        xstarts = diffusion(tx_emb, tx_emb_uncond, infos).cpu() # (num frames,)

        for idx, (xstart, length) in enumerate(zip(xstarts, infos["output_lengths"])):
            xstart = xstart[:length]

            from src.tools.extract_joints import extract_joints

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

            if "vertices" in output:
                path = npy_paths[idx].replace(".npy", "_verts.npy")
                np.save(path, output["vertices"])

            if "smpldata" in output:
                path = npy_paths[idx].replace(".npy", "_smpl.npz")
                np.savez(path, **output["smpldata"])

            print(f"Joints rendering {idx}")
            joints_renderer(
                joints, title="", output=joints_video_paths[idx], canonicalize=False
            )
            print(joints_video_paths[idx])
            print()

            if "vertices" in output and c.animations:
                print(f"SMPL rendering {idx}")
                smpl_renderer(
                    output["vertices"], title="", output=smpl_video_paths[idx], video=False
                )
                print(smpl_video_paths[idx])
                print()

            print("Rendering done")


if __name__ == "__main__":
    generate()