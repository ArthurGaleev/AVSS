import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os


from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import (
    select_most_suitable_gpu,
    set_random_seed,
    setup_saving_and_logging,
)
from src.utils.torch_utils import set_tf32_allowance

warnings.filterwarnings("ignore", category=UserWarning)


def init_process(backend='nccl'):
    "Init the distributed env"
    world_size = int(os.environ["WORLD_SIZE"])
    current_rank = int(os.environ["RANK"])
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = '127.0.0.1'
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = '29500'
    print("MASTER_ADDR: ", os.environ["MASTER_ADDR"])
    print("MASTER_PORT: ", os.environ["MASTER_PORT"])
    print("Starting init process group. Current rank: ", current_rank)
    torch.distributed.init_process_group(backend, rank=current_rank, world_size=world_size)
    print("Finished init process group. Current rank: ", current_rank)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    device = config.trainer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if config.trainer.distributed:
            init_process()
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(f"cuda:{local_rank}")
            device = f"cuda:{local_rank}"
            if local_rank != 0:
                torch.distributed.barrier()
        else:
            device, free_memories = select_most_suitable_gpu()
            logger.info(f"Using GPU: {device} with {free_memories / 1024 ** 3:.2f} GB free")


    set_random_seed(
        config.trainer.seed, config.trainer.get("save_reproducibility", True)
    )
    set_tf32_allowance(config.trainer.get("tf32_allowance", False))

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.distributed and torch.distributed.get_rank() == 0:
        torch.distributed.barrier()
            

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    if config.trainer.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model)
    elif config.trainer.parallel:
        model = torch.nn.DataParallel(model)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")
    trainer = Trainer(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dtype=config.trainer.get("dtype", "float32"),
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
