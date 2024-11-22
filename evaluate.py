import os
import numpy as np 
import json
import orjson
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

import src.prepare  # noqa

from TMR.mtt.metrics import calculate_activation_statistics_normalized
from TMR.mtt.load_tmr_model import load_tmr_model_easy
from TMR.src.guofeats import joints_to_guofeats
from TMR.src.guofeats.motion_representation_local import guofeats_to_joints as guofeats_to_joints_local
import torch
from TMR.mtt.metrics import calculate_frechet_distance, calculate_activation_statistics_normalized
from TMR.src.model.tmr import get_sim_matrix


def T(x):
    if isinstance(x, torch.Tensor):
        return x.permute(*torch.arange(x.ndim - 1, -1, -1))
    else:
        return x.transpose(*np.arange(x.ndim - 1, -1, -1))


def get_metrics(
    tmr_forward,
    text_dico,
    gt_mu,
    gt_cov,
    text_latents_gt,
    motion_latents_gt,
    generations_folder,
    infos, 
    diversity_times
):
    metrics = {}
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

    for key, path_motion in infos.items():
        
        texts = [path_motion["text"]]
        # path = os.path.join(amass_folder, path_motion["motion_path"] + ".npy") # ground truth 
        path = os.path.join(generations_folder, key + ".npy") 

        motion = np.load(path)
        x, y, z = motion.T
        joint = np.stack((x, z, -y), axis=0).T
        motion_guofeats = joints_to_guofeats(joint)

        # gt_motion_guofeats = path_motion["motion"]
        joints_local = guofeats_to_joints_local(torch.from_numpy(motion_guofeats))
        joints_local = joints_local - joints_local[:, [0]]
        
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

        ### REALISM FID+
        nb_samples = 1
        realism_crop_motions = [motion_guofeats] # [motion_guofeats[x : x + n_real_nframes] for x in realism_idx]
        realism_crop_motion_latents = tmr_forward(realism_crop_motions)
        fid_realism_crop_motion_latents_lst.append(realism_crop_motion_latents)

        ### SEMANTICS
        
        # do not use real crops but the entire sequence (less than 10s)        
        crop_latents = tmr_forward([motion_guofeats])
        sim_matrix_m2t = get_sim_matrix(crop_latents, text_latents_gt).numpy()
        sim_matrix_m2m = get_sim_matrix(crop_latents, motion_latents_gt).numpy()

        for idx_text, text in enumerate(texts):
            text_number = text_dico[text]

            m2t_score_lst.append((sim_matrix_m2t[idx_text, text_number] + 1) / 2)
            m2m_score_lst.append((sim_matrix_m2m[idx_text, text_number] + 1) / 2)

            asort_m2t = np.argsort(sim_matrix_m2t[idx_text])[::-1]
            m2t_top_1_lst.append(1 * (text_number in asort_m2t[:1]))
            m2t_top_3_lst.append(1 * (text_number in asort_m2t[:3]))

    motion_latents = torch.concatenate(fid_realism_crop_motion_latents_lst)
    mu, cov = calculate_activation_statistics_normalized(motion_latents.numpy())

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
    metrics["transition"] = torch.concatenate(trans_dist_lst).mean().cpu().numpy()  

    # metrics["diversity"] = calculate_diversity(motion_latents, diversity_times) # 300

    return metrics


def load_test_texts_motions(path_ids, path_all_texts, path_annotations, DEBUG=0, fps=20):
    test_texts, motions = [], []
    id_to_text_path = {}

    # Load test ids
    with open(path_ids) as f:
        lines = f.readlines()
    
    if DEBUG != 0:
        lines = lines[:DEBUG] 
    
    for l in lines:
        id = l.strip() 
        id_to_text_path[id] = {
            "text": None,
            "motion_path": None,
            "motion": None
        }
    # Load test texts
    with open(path_all_texts) as f:
        lines_t = f.readlines()
    for l in lines_t:
        idx = l.split("-")[0].strip()
        text = l.split("-")[1].strip()
        if idx in id_to_text_path.keys(): # il  test set non ha tutti i testi
            assert id_to_text_path[idx]["text"] is None
            id_to_text_path[idx]["text"] = text    
    # Load motion paths
    with open(path_annotations, "rb") as ff:
        annotations = orjson.loads(ff.read())
    for idx in id_to_text_path.keys():
        path = annotations[idx]["path"]
        id_to_text_path[idx]["motion_path"] = path
    # remove humanact12
    id_to_text_path = {k: v for k, v in id_to_text_path.items() if "humanact12" not in v["motion_path"]}# Giusto o sbagliato?
    # Load motions
    for idx in id_to_text_path.keys():
        amass_path = id_to_text_path[idx]["motion_path"]
        path = os.path.join(amass_folder, amass_path + ".npy")
        motion = np.load(path)
        # Croppo i movimenti maggiori di 10 secondi prendendo gli estremi specificati nel file annotations
        start_frame = int(annotations[idx]["annotations"][0]["start"] * fps)
        end_frame = int(annotations[idx]["annotations"][0]["end"] * fps)
        motion = motion[start_frame:end_frame]

        x, y, z = motion.T
        joint = np.stack((x, z, -y), axis=0).T
        feats = joints_to_guofeats(joint)
        id_to_text_path[idx]["motion"] = feats

    test_texts = [id_to_text_path[i]["text"] for i in id_to_text_path.keys()]
    motions = [id_to_text_path[i]["motion"] for i in id_to_text_path.keys()]
    
    return test_texts, motions, id_to_text_path


amass_folder = "./datasets/motions/AMASS_20.0_fps_nh_smpljoints_neutral_nobetas"
base_path_splitme_humanml3d = f"{os.getcwd()}/pretrained_models/mdm-smpl_splitme_humanml3d"
base_path_splitme_kitml = f"{os.getcwd()}/pretrained_models/mdm-smpl_splitme_kitml"
base_path_humanml_clip = f"{os.getcwd()}/pretrained_models/mdm-smpl_clip_smplrifke_humanml3d"
base_path_kitml_clip = f"{os.getcwd()}/pretrained_models/mdm-smpl_clip_smplrifke_kitml"

def main():    
    exp_gt = {
            "name": "gt",
            "generations_folder": f"{amass_folder}/"
    }
    exp_text_humanml = {
            "name": "text",
            "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/test.txt",
            "generations_folder": f"{base_path_splitme_humanml3d}/generations_text/", 
            "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/annotations_test.json"
    }
    exp_submotions_humanml = {
            "name": "submotions mcd",
            "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/test.txt",
            "generations_folder": f"{base_path_splitme_humanml3d}/generations_submotions/",
            "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/annotations_test.json"
    }
    exp_stmc_humanml ={
            "name": "submotions stmc",
            "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/test.txt",
            "generations_folder":f"{base_path_splitme_humanml3d}/generations_submotions_stmc/", 
            "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/splits/complex/annotations_test.json"
    }
    exp_multitext_text_humanml ={
            "name":"multi_text",
            "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/test.txt", 
            "generations_folder":  f"{base_path_humanml_clip}/generations_multitext_text/",
            "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/annotations.json"
    }
    exp_multitext_submotions_humanml = {
            "name": "multi_t",
            "path_ids":f"{os.getcwd()}/datasets/annotations/humanml3d/splits/test.txt", 
            "generations_folder": f"{base_path_humanml_clip}/generations_multitext_submotions/",
            "path_annotations": f"{os.getcwd()}/datasets/annotations/humanml3d/annotations.json"
    }
    exp_text_kitml = {
            "name": "text",
            "path_ids":f"{os.getcwd()}/datasets/annotations/kitml/splits/complex/test.txt",
            "generations_folder": f"{base_path_splitme_kitml}/generations_text/", 
            "path_annotations": f"{os.getcwd()}/datasets/annotations/kitml/splits/complex/annotations_test.json"
    }
    exp_submotions_kitml = {
            "name": "submotions",
            "path_ids":f"{os.getcwd()}/datasets/annotations/kitml/splits/complex/test.txt",
            "generations_folder": f"{base_path_splitme_kitml}/generations_submotions/",
            "path_annotations": f"{os.getcwd()}/datasets/annotations/kitml/splits/complex/annotations_test.json"
    }
    exp_stmc_kitml ={
            "name": "submotions_stmc",
            "path_ids":f"{os.getcwd()}/datasets/annotations/kitml/splits/complex/test.txt",
            "generations_folder":f"{base_path_splitme_kitml}/generations_submotions_stmc/", 
            "path_annotations": f"{os.getcwd()}/datasets/annotations/kitml/splits/complex/annotations_test.json"
    }
    exp_multitext_text_kitml ={
            "name":"multi_text_k",
            "path_ids":f"{os.getcwd()}/datasets/annotations/kitml/splits/test.txt", 
            "generations_folder":  f"{base_path_kitml_clip}/generations_multitext_text/",
            "path_annotations": f"{os.getcwd()}/datasets/annotations/kitml/annotations.json"
    }
    exp_multitext_kitml_submotions = {
            "name": "multi_k_t",
            "path_ids":f"{os.getcwd()}/datasets/annotations/kitml/splits/test.txt", 
            "generations_folder": f"{base_path_kitml_clip}/generations_multitext_submotions/",
            "path_annotations": f"{os.getcwd()}/datasets/annotations/kitml/annotations.json"
    }
    
    ### SETTINGS 
    # input_types = [exp_gt, exp_text_humanml, exp_stmc_humanml, exp_submotions_humanml] 
    input_types = [exp_gt, exp_text_kitml] 
    
    DEBUG = 0 # to test the evaluation script, only the firsts #{DEBUG} eleemnts with {DEBUG}!=0 are considered
    
    ###
    np.random.seed(0)
    device = "cpu"
    fps = 20.0
    path_ids = input_types[1]["path_ids"]
    assert all([inp["path_ids"]==path_ids for inp in input_types[1:]])
    path_annotations = input_types[1]["path_annotations"]
    assert all([inp["path_annotations"]==path_annotations for inp in input_types[1:]])
    dataset = "humanml3d" if "humanml3d" in input_types[1]["path_ids"] else "kitml"
    
    result = []
    
    names = [i["name"] for i in input_types]
    print(f"\nExperiment on {names} and with debug [{DEBUG}] of the split [{path_ids}]\n")
    
    with open(path_ids) as f:
        ids_gt = f.readlines()
        ids_gt = [i.strip() for i in ids_gt]
    
    annotations = json.load(open(path_annotations))
    ids_gt = [i for i in ids_gt if  "humanact12" not in annotations[i]["path"]]
    # filter annotations
    annotations = {k: v for k, v in annotations.items() if "humanact12" not in v["path"]} 
    
    if DEBUG != 0:
        ids_gt = ids_gt[:DEBUG]
    
    texts_gt, motions_guofeats_gt = [], []
    for idx in ids_gt:
        value = annotations[idx]
        # prendo il testo della prima annotazione testuale
        texts_gt.append(value["annotations"][-1]["text"])
        
        motion_path = exp_gt["generations_folder"] + value["path"]+".npy"
        motion = np.load(motion_path) # [num_frames, 24, 3], con 0 < num_frames   
        start_frame = int(value["annotations"][0]["start"] * fps)
        end_frame = int(value["annotations"][0]["end"] * fps)
        motion = motion[start_frame:end_frame] 
        
        x, y, z = motion.T
        joint = np.stack((x, z, -y), axis=0).T # [num_frames, 24, 3]
        feats = joints_to_guofeats(joint) # [num_frames, 263]
        motions_guofeats_gt.append(feats)
    
    tmr_forward = load_tmr_model_easy(device, dataset)
    
    print(f"texts_gt len {len(texts_gt)}")
    diversity_times = 300 if 300 < len(texts_gt) else len(texts_gt) - 1
    
    # texts_gt: list(str) - N elements for testset
    # motions_guofeats_gt: list: N * tensor(n_frames, 263)) - N elements for testset
    text_latents_gt = tmr_forward(texts_gt) #  tensor(N, 256)
    motion_latents_gt = tmr_forward(motions_guofeats_gt)  # tensor(N, 256)
    print(f"text_latents_gt shape {text_latents_gt.shape}")
    print(f"motion_latents_gt shape {motion_latents_gt.shape}")
    gt_mu, gt_cov = calculate_activation_statistics_normalized(motion_latents_gt.numpy())
    
    result.append(["Experiment", "R1", "R3", "R10", "M2T", "M2M", "FID", "Trans"])
    print_result(result[0])
    
    for experiment in input_types:
        dic_m2t = {}
        dic_m2m = {}
        # Load the motions
        motions_guofeats = []
        for idx in ids_gt:
            if experiment["name"] == "gt":
                motion_path = experiment["generations_folder"] + annotations[idx]["path"]+".npy"
                motion = np.load(motion_path) # [num_frames, 24, 3], with 0 < num_frames   
                start_frame = int(annotations[idx]["annotations"][0]["start"] * fps)
                end_frame = int(annotations[idx]["annotations"][0]["end"] * fps)
                
                motion = motion[start_frame:end_frame] 
            else:
                motion_path = experiment["generations_folder"] + idx +".npy"
                motion = np.load(motion_path)
                if motion.shape[0] > 200:
                    motion = motion[:200]
        
                x, y, z = motion.T
                joint = np.stack((x, z, -y), axis=0).T # [num_frames, 24, 3]
                feats = joints_to_guofeats(joint) # [num_frames, 263]
                motions_guofeats.append(feats)
        
    motion_latents = tmr_forward(motions_guofeats)  # tensor(N, 256)
    sim_matrix_m2t = get_sim_matrix(motion_latents, text_latents_gt).numpy()
    sim_matrix_m2m = get_sim_matrix(motion_latents, motion_latents_gt).numpy()
    
    # motion-to-text retrieval metrics
    m2t_top_1_lst = []
    m2t_top_3_lst = []
    m2t_top_10_lst = []
    # TMR motion-to-motion (M2M) score
    m2t_score_lst = []
    m2m_score_lst = []
    
    for idx in range(len(sim_matrix_m2t)):
    
        # score between 0 and 1
        m2t_score_lst.append((sim_matrix_m2t[idx, idx] + 1) / 2)
        m2m_score_lst.append((sim_matrix_m2m[idx, idx] + 1) / 2)
        
        asort = np.argsort(sim_matrix_m2t[idx])[::-1]
        m2t_top_1_lst.append(1 * (idx in asort[:1]))
        m2t_top_3_lst.append(1 * (idx in asort[:3]))
        m2t_top_10_lst.append(1 * (idx in asort[:10]))

        dic_m2t[ids_gt[idx]] = (sim_matrix_m2t[idx, idx] + 1) / 2
        dic_m2m[ids_gt[idx]] = (sim_matrix_m2m[idx, idx] + 1) / 2

    m2t_top_1 = np.mean(m2t_top_1_lst)
    m2t_top_3 = np.mean(m2t_top_3_lst)
    m2t_top_10 = np.mean(m2t_top_10_lst)
    m2t_score = np.mean(m2t_score_lst)
    m2m_score = np.mean(m2m_score_lst)
    
    # Transition distance:
    trans_dist_lst = []
    for motion_guofeats in motions_guofeats:
        # for the text baseline for example
        N = len(motion_guofeats)
        inter_points = np.array([N // 4, 2 * N // 4, 3 * N // 4]) # tre frames 

        gt_motion_guofeats = torch.from_numpy(motion_guofeats) # (n_frames, 263)
        gt_joints_local = guofeats_to_joints_local(gt_motion_guofeats)
        
        gt_joints_local = gt_joints_local - gt_joints_local[:, [0]] # (n_frames, 22, 3) 

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
    transition = torch.concatenate(trans_dist_lst).mean().numpy().item()
    # diversity = calculate_diversity(motion_latents, diversity_times) 
    
    mu, cov = calculate_activation_statistics_normalized(motion_latents.numpy())
    
    # FID+ metrics
    fid = calculate_frechet_distance(
            gt_mu.astype(float),
            gt_cov.astype(float),
            mu.astype(float),
            cov.astype(float),
    )
    
    result.append( [experiment["name"], m2t_top_1*100, m2t_top_3*100, m2t_top_10*100, m2t_score, m2m_score, fid, transition*100 ] )
    print_result(result[-1])
    
    
if __name__ == "__main__":
    main()