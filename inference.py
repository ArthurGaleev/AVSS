import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import select_most_suitable_gpu, set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.utils.torch_utils import set_tf32_allowance

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(
        config.inferencer.seed, config.inferencer.get("save_reproducibility", True)
    )
    set_tf32_allowance(config.inferencer.get("tf32_allowance", False))

    device = config.inferencer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device, free_memories = select_most_suitable_gpu()
        print(f"Using GPU: {device} with {free_memories / 1024 ** 3:.2f} GB free")

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        metrics["inference"].append(instantiate(metric_config))

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
