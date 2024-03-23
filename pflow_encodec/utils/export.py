from pathlib import Path

import hydra
import torch


def export_lightning_ckpt(input_path, output_path):
    ckpt = torch.load(input_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    cfg = ckpt["hyper_parameters"]
    model_config = cfg["net"]
    data_config = {}
    data_config["mean"] = cfg["mean"]
    data_config["std"] = cfg["std"]
    data_config["text2latent_ratio"] = cfg["text2latent_ratio"]
    state_dict = {k[len("net.") :]: v for k, v in state_dict.items()}

    model = hydra.utils.instantiate(model_config)
    model.load_state_dict(state_dict)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "data_config": data_config,
            "model_config": model_config,
        },
        output_path,
    )
