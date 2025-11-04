import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    assert len(dataset_items)
    return {
        key: (
            torch.stack([item[key] for item in dataset_items])
            if key in ["audio_first", "audio_second", "audio_mix"]
            else [item[key] for item in dataset_items]
        )
        for key in dataset_items[0].keys()
    }
