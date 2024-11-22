import os 


def update_cfg(cfg, c):
    if hasattr(c, "fold"):
        fold_id = c.fold
        cfg.run_dir = f"outputs/mdm-smpl_folds/{fold_id}/"
        if not os.path.exists(cfg.run_dir): 
            os.makedirs(cfg.run_dir) 
    if hasattr(c, "diffusion") and hasattr(c["diffusion"], "weight"):
        cfg.diffusion.weight = c.diffusion.weight
    if hasattr(c, "diffusion") and hasattr(c["diffusion"], "mcd"):
        cfg.diffusion.mcd = c.diffusion.mcd
    if hasattr(c, "submotions_dir"):
        cfg.submotions_dir = c.submotions_dir