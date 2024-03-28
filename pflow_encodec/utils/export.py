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


def merge_ckpts(ckpt_paths, output_path):
    assert len(ckpt_paths) > 1, "Please provide more than one checkpoint path"
    ckpts = [torch.load(p, map_location="cpu") for p in ckpt_paths]
    state_dicts = [ckpt["state_dict"] for ckpt in ckpts]
    keys = set(state_dicts[0].keys())

    # key check
    for key in keys:
        for state_dict in state_dicts:
            assert key in state_dict, f"{key} not found in state_dict"

    # shape check
    for key in keys:
        tensors = [state_dict[key] for state_dict in state_dicts]
        shapes = [t.shape for t in tensors]
        assert len(set(shapes)) == 1, f"Shapes of {key} are not the same: {shapes}"

    new_state_dict = {}
    for key in keys:
        new_state_dict[key] = torch.stack([state_dict[key] for state_dict in state_dicts]).mean(dim=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data_config = ckpts[0]["data_config"]
    model_config = ckpts[0]["model_config"]
    torch.save(
        {
            "state_dict": new_state_dict,
            "data_config": data_config,
            "model_config": model_config,
        },
        output_path,
    )
