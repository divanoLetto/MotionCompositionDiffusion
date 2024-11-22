import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="embeddings", version_base="1.3")
def text_embeddings(cfg: DictConfig):
    device = cfg.device

    import src.prepare  # noqa
    from src.data.text_mtt import save_text_embeddings_mtt

    dataname = cfg.dataset
    # Compute sent embeddings
    modelname = cfg.data.text_encoder.modelname
    logger.info(f"Compute text embeddings for {modelname}")
    path = f"mtt_2/"
    save_text_embeddings_mtt(path, modelname=modelname, device=device)


if __name__ == "__main__":
    text_embeddings()
