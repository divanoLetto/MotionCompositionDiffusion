import os
import numpy as np 
import json
import orjson
import numpy as np
import torch

from evaluate import T
import src.prepare  # noqa
from TMR.mtt.metrics import calculate_activation_statistics_normalized
from TMR.mtt.load_tmr_model import load_tmr_model_easy
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.guofeats.motion_representation_local import guofeats_to_joints as guofeats_to_joints_local
from TMR.mtt.metrics import calculate_frechet_distance, calculate_activation_statistics_normalized
from TMR.src.model.tmr import get_sim_matrix
from src.stmc import read_submotions
from TMR.mtt.load_mtt_texts_motions import load_mtt_texts_motions
# from TMR.mtt.stmc_metrics import get_gt_metrics, get_exp_metrics, print_latex
# from TMR.mtt.experiments import experiments


def get_gt_metrics(motion_latents_gt, text_latents_gt, motions_guofeats_gt):
    metrics = {"m2m_score": 1.00, "fid": 0.00}  # by definition  # by definition
    sim_matrix_gt = get_sim_matrix(motion_latents_gt, text_latents_gt).numpy()

    # motion-to-text retrieval metrics
    m2t_top_1_lst = []
    m2t_top_3_lst = []
    # TMR motion-to-motion (M2M) score
    m2m_score_lst = []
    for idx in range(len(sim_matrix_gt)):
        # score between 0 and 1
        m2m_score_lst.append((sim_matrix_gt[idx, idx] + 1) / 2)
        asort = np.argsort(sim_matrix_gt[idx])[::-1]
        m2t_top_1_lst.append(1 * (idx in asort[:1]))
        m2t_top_3_lst.append(1 * (idx in asort[:3]))

    metrics["m2t_top_1"] = np.mean(m2t_top_1_lst)
    metrics["m2t_top_3"] = np.mean(m2t_top_3_lst)
    metrics["m2t_score"] = np.mean(m2m_score_lst)

    # Transition distance:
    trans_dist_lst = []
    for motion_guofeats_gt in motions_guofeats_gt:
        # for the text baseline for example
        N = len(motion_guofeats_gt)
        inter_points = np.array([N // 4, 2 * N // 4, 3 * N // 4])

        gt_motion_guofeats = torch.from_numpy(motion_guofeats_gt)
        gt_joints_local = guofeats_to_joints_local(gt_motion_guofeats)
        gt_joints_local = gt_joints_local - gt_joints_local[:, [0]]

        # Same distance as in TEACH
        trans_dist_lst.append(
            torch.linalg.norm(
                (gt_joints_local[inter_points] - gt_joints_local[inter_points - 1]),
                dim=-1,
            )
            .mean(-1)
            .flatten()
        )

    # Transition distance
    metrics["transition"] = torch.concatenate(trans_dist_lst).mean()
    return metrics


def get_exp_metrics(
    exp,
    tmr_forward,
    text_dico,
    timelines_dict,
    gt_mu,
    gt_cov,
    text_latents_gt,
    motion_latents_gt,
    fps,
):
    metrics = {}

    folder = exp["generations_folder"]

    only_text = exp.get("only_text", False)
    y_is_z_axis = exp.get("y_is_z_axis", False)

    # Motion-to-text retrieval metrics
    m2t_top_1_lst = []
    m2t_top_3_lst = []

    # TMR scores
    m2m_score_lst = []
    m2t_score_lst = []

    # Transition distance
    trans_dist_lst = []

    # Store motion latents for FID+
    fid_realism_crop_motion_latents_lst = []

    for key_name, timeline in timelines_dict.items():
        intervals = [(x.start, x.end) for x in timeline]
        texts = [x.text for x in timeline]

        path = os.path.join(folder, key_name + ".npy")
        motions = np.load(path)

        if len(motions.shape) == 4:
            assert len(motions) == 1
            motions = motions[0]
        assert motions.shape[-1] == 3 # should be joints
        assert motions.shape[-2] == 24 or motions.shape[-2] == 22
        if motions.shape[-2] == 24:
            motions = motions[..., :22, :]
        if not y_is_z_axis:
            x, y, z = T(motions)
            motions = T(np.stack((x, z, -y), axis=0))

        gfeats = joints_to_guofeats(motions)
        joints_local = guofeats_to_joints_local(torch.from_numpy(gfeats))
        joints_local = joints_local - joints_local[:, :, [0]]

        ### REALISM FID+
        # Compute latents for 5 random crops of 5 seconds
        m_len = gfeats.shape[1]
        # same random crops for all the methods
        n_realisim_seq = magic_number # 5.0
        n_real_nframes = int(n_realisim_seq * fps)
        nb_samples = 5
        np.random.seed(0)
        realism_idx = np.random.randint(0, m_len - n_real_nframes, nb_samples)

        realism_crop_motions = [gfeats[x : x + n_real_nframes] for x in realism_idx]
        realism_crop_motion_latents = tmr_forward(realism_crop_motions)
        fid_realism_crop_motion_latents_lst.append(realism_crop_motion_latents)

        inter_points = np.array(
            sorted(
                [
                    x
                    for x in list(
                        set(x[0] for x in intervals).union(set(x[1] for x in intervals))
                    )
                    if x < len(joints_local)  # can be sliced
                ]
            )
        )[1:-1]

        # take random points
        if only_text:
            N = len(joints_local)
            inter_points = np.array([N // 4, 2 * N // 4, 3 * N // 4])

        trans_dist_lst.append(
            torch.linalg.norm(
                (joints_local[inter_points] - joints_local[inter_points - 1]),
                dim=-1,
            )
            .mean(-1)
            .flatten()
        )

        ### SEMANTICS
        if only_text:
            # do not use real crops but the entire sequence (less than 10s)
            big_intervals = [(0, len(gfeats[0])) for text in texts]
            gfeats_crops = [gfeats[start:end] for start, end in big_intervals]
        else:
            gfeats_crops = [gfeats[start:end] for start, end in intervals]

        crop_latents = tmr_forward(gfeats_crops)
        sim_matrix_m2t = get_sim_matrix(crop_latents, text_latents_gt).numpy()
        sim_matrix_m2m = get_sim_matrix(crop_latents, motion_latents_gt).numpy()

        for idx_text, text in enumerate(texts):
            text_number = text_dico[text]

            m2t_score_lst.append((sim_matrix_m2t[idx_text, text_number] + 1) / 2)
            m2m_score_lst.append((sim_matrix_m2m[idx_text, text_number] + 1) / 2)

            asort_m2t = np.argsort(sim_matrix_m2t[idx_text])[::-1]
            m2t_top_1_lst.append(1 * (text_number in asort_m2t[:1]))
            m2t_top_3_lst.append(1 * (text_number in asort_m2t[:3]))

    fid_realism_crop_motion_latents = np.concatenate(
        fid_realism_crop_motion_latents_lst
    )

    mu, cov = calculate_activation_statistics_normalized(
        fid_realism_crop_motion_latents
    )

    # FID+ metrics
    metrics["fid"] = calculate_frechet_distance(
        gt_mu.astype(float),
        gt_cov.astype(float),
        mu.astype(float),
        cov.astype(float),
    )

    # Motion-to-text retrieval metrics
    metrics["m2t_top_1"] = np.mean(m2t_top_1_lst)
    metrics["m2t_top_3"] = np.mean(m2t_top_3_lst)

    # TMR scores
    metrics["m2t_score"] = np.mean(m2t_score_lst)
    metrics["m2m_score"] = np.mean(m2m_score_lst)

    # Transition distance
    metrics["transition"] = torch.concatenate(trans_dist_lst).mean()
    return metrics


amass_folder = "./datasets/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas"
base_path_splitcomplex_humanml3d = f"{os.getcwd()}/pretrained_models/mdm-smpl_splitcomplex_humanml3d"

exp_gt = {
        "name": "gt",
        "generations_folder": f"{amass_folder}/"
}
exp_complex_mtt_stmc = {
        "name": "mtt stmc",
        "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/test.txt",
        "generations_folder": f"{base_path_splitcomplex_humanml3d}/generations_mtt_Mlast_Dsmpl_Sstmc/",
        "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/annotations_test.json"
}
exp_complex_mtt_mcd_7 = {
        "name": "mtt mcd 7",
        "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/test.txt",
        "generations_folder": f"{base_path_splitcomplex_humanml3d}/generations_mtt_Mlast_Dsmpl_Smcd_G7/",
        "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/annotations_test.json"
}
exp_complex_mtt_mcd_13 = {
        "name": "mtt mcd 13",
        "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/test.txt",
        "generations_folder": f"{base_path_splitcomplex_humanml3d}/generations_mtt_Mlast_Dsmpl_Smcd_G13/",
        "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/annotations_test.json"
}

ablation_exp_complex_mtt_mcd = [
    {
        "name": f"mtt mcd {i}",
        "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/test.txt",
        "generations_folder": f"{base_path_splitcomplex_humanml3d}/generations_mtt_Mlast_Dsmpl_Smcd_G{i}/",
        "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/annotations_test.json"
    } for i in range(5,20)
]

### SETTINGS 
# input_types = [exp_gt, exp_text_humanml, exp_stmc_humanml, exp_submotions_humanml] 
input_types = [exp_gt, exp_complex_mtt_stmc, exp_complex_mtt_mcd_7] 

# FOLDER to evaluate
np.random.seed(0)
fps = 20.0
device = "cpu"
mtt_timelines = "mtt/MTT.txt"

tmr_forward = load_tmr_model_easy(device)
texts_gt, motions_guofeats_gt = load_mtt_texts_motions(fps)

text_dico = {t: i for i, t in enumerate(texts_gt)}

text_latents_gt = tmr_forward(texts_gt)
motion_latents_gt = tmr_forward(motions_guofeats_gt)
print(f"text_latents_gt shape {text_latents_gt.shape}")
print(f"motion_latents_gt shape {motion_latents_gt.shape}")
gt_mu, gt_cov = calculate_activation_statistics_normalized(motion_latents_gt.numpy())

timelines = read_submotions(mtt_timelines, fps)
assert len(timelines) == 500
timelines_dict = {str(idx).zfill(4): timeline for idx, timeline in enumerate(timelines)}

for azz in range(1,15):
    magic_number = azz
    print("Magic number: ", magic_number)
    for experiment in input_types:

        if experiment["name"] == "gt":
            metrics = get_gt_metrics(
                motion_latents_gt, 
                text_latents_gt, 
                motions_guofeats_gt,
            )

        else:
            metrics = get_exp_metrics(
                experiment,
                tmr_forward,
                text_dico,
                timelines_dict,
                gt_mu,
                gt_cov,
                text_latents_gt,
                motion_latents_gt,
                fps,
            )
        print(experiment["name"],":")
        print(metrics)