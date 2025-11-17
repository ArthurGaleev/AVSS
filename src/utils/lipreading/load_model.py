import json
import os

import gdown
import torch

from src.utils.io_utils import ROOT_PATH
from src.utils.lipreading.model import Lipreading

PRETRAINED_MODEL_LINKS = {
    "resnet18_dctcn": "https://drive.google.com/uc?id=179NgMsHo9TeZCLLtNWFVgRehDvzteMZE",
    "resnet18_mstcn": "https://drive.google.com/uc?id=1vqMpxZ5LzJjg50HlZdj_QFJGm2gQmDUD",
    "snv1x_dsmstcn3x": "https://drive.google.com/uc?id=1gUsKdcUSni1xxYEbQOXFNXtpfnRHTmk_",
    "snv1x_tcn2x": "https://drive.google.com/uc?id=1cocXeZYtgvX4S3246C0vp0qOBS1eloi6",
    "snv1x_tcn1x": "https://drive.google.com/uc?id=1a3umSDxTBebXl3detdumBYjVCF-UZ-kU",
    "snv05x_tcn2x": "https://drive.google.com/uc?id=110MsJpKreB8fwtGPTHsODSW2D_umWT0E",
    "snv05x_tcn1x": "https://drive.google.com/uc?id=197QXMxZ_fmsDxyvsqbDcV6XD7GUlaKHi",
}


# default config
args = {
    "dataset": "lrw",
    "num_classes": 500,
    "modality": "video",
    # directory
    "data-dir": ROOT_PATH / "data" / "datasets" / "dla_dataset" / "mouths",
    # model config
    "backbone_type": "resnet",
    "relu_type": "relu",
    "width_mult": 1.0,
    # TCN config
    "tcn_kernel_size": "",
    "tcn_num_layers": 4,
    "tcn_dropout": 0.2,
    "tcn_dwpw": False,
    "tcn_width_mult": 1,
    # DenseTCN config
    "densetcn_block_config": "",
    "densetcn_kernel_size_set": "",
    "densetcn_dropout": 0.2,
    "densetcn_reduced_size": 256,
    "densetcn_se": False,
    "densetcn_condense": False,
    # mixup
    "alpha": 0.4,
    # test
    "model_path": None,
    "allow_size_mismatch": False,
    # feature extractor
    "extract_feats": False,
    "mouth_patch_path": None,
    "mouth_embedding_out_path": None,
    # json pathname
    "config_path": None,
    # other vars
    "interval": 50,
    "workers": 8,
    # paths
    "logging_dir": "./train_logs",
    # use boundaries
    "use_boundary": False,
}


def update_args(config_path):
    with open(config_path, "r") as f:
        args.update(json.load(f))

    if args.get("tcn_kernel_size", ""):
        args["tcn_options"] = {
            "num_layers": args["tcn_num_layers"],
            "kernel_size": args["tcn_kernel_size"],
            "dropout": args["tcn_dropout"],
            "dwpw": args["tcn_dwpw"],
            "width_mult": args["tcn_width_mult"],
        }
    else:
        args["tcn_options"] = {}

    if args.get("densetcn_block_config", ""):
        args["densetcn_options"] = {
            "block_config": args["densetcn_block_config"],
            "growth_rate_set": args["densetcn_growth_rate_set"],
            "reduced_size": args["densetcn_reduced_size"],
            "kernel_size_set": args["densetcn_kernel_size_set"],
            "dilation_size_set": args["densetcn_dilation_size_set"],
            "squeeze_excitation": args["densetcn_se"],
            "dropout": args["densetcn_dropout"],
        }
    else:
        args["densetcn_options"] = {}


def load_model(model_name, model, device, optimizer=None, allow_size_mismatch=False):
    """
    Load model from file
    If optimizer is passed, then the loaded dictionary is expected to contain also the states of the optimizer.
    If optimizer not passed, only the model weights will be loaded
    """

    url = PRETRAINED_MODEL_LINKS[model_name]
    output = ROOT_PATH / "src" / "utils" / "lipreading" / "loaded_models"
    output.mkdir(exist_ok=True, parents=True)
    output = str(output / f"{model_name}.pth")
    gdown.download(url, output, quiet=False)

    load_path = output
    # load dictionary
    assert os.path.isfile(
        load_path
    ), "Error when loading the model, provided path not found: {}".format(load_path)
    checkpoint = torch.load(load_path, weights_only=False, map_location=device)
    loaded_state_dict = checkpoint["model_state_dict"]

    if allow_size_mismatch:
        loaded_sizes = {k: v.shape for k, v in loaded_state_dict.items()}
        model_state_dict = model.state_dict()
        model_sizes = {k: v.shape for k, v in model_state_dict.items()}
        mismatched_params = []
        for k in loaded_sizes:
            if loaded_sizes[k] != model_sizes[k]:
                mismatched_params.append(k)
        for k in mismatched_params:
            del loaded_state_dict[k]

    # copy loaded state into current model and, optionally, optimizer
    model.load_state_dict(loaded_state_dict, strict=not allow_size_mismatch)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, optimizer, checkpoint["epoch_idx"], checkpoint
    return model


def load_lipreading_model(model_name, logger, device):
    # update default args with args from selected model
    update_args(
        ROOT_PATH
        / "src"
        / "utils"
        / "lipreading"
        / "configs"
        / f"lrw_{model_name}.json"
    )

    model = Lipreading(
        modality=args["modality"],
        num_classes=args["num_classes"],
        tcn_options=args["tcn_options"],
        densetcn_options=args["densetcn_options"],
        backbone_type=args["backbone_type"],
        relu_type=args["relu_type"],
        width_mult=args["width_mult"],
        use_boundary=args["use_boundary"],
        extract_feats=args["extract_feats"],
    ).to(device)

    model = load_model(
        model_name,
        model,
        device=device,
        allow_size_mismatch=args["allow_size_mismatch"],
    )
    logger.info(f"Lipreading model has been successfully loaded")
    return model
