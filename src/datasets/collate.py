import torch
import torch.nn.functional as F


def pad_tensors(tensors: list[torch.Tensor]):
    """
    pad tensors with same number of dimensiona to the same shape
    """
    max_shape = [max(s) for s in zip(*[t.shape for t in tensors])]
    padded_tensors = [
        F.pad(
            t,
            sum(
                [
                    (0, max_shape[dim_ind] - t.shape[dim_ind])
                    for dim_ind in reversed(range(len(max_shape)))
                ],
                (),
            ),
            "constant",
            0,
        )
        for t in tensors
    ]
    return torch.stack(padded_tensors)


def collate_fn(dataset_items: list[dict]):
    assert len(dataset_items)
    batch_by_column = {
        key: [item[key] for item in dataset_items] for key in dataset_items[0].keys()
    }
    result_batch = {
        "spectrogram_first": pad_tensors(batch_by_column["spectrogram_first"]),
        "spectrogram_second": pad_tensors(batch_by_column["spectrogram_second"]),
        "spectrogram_mix": pad_tensors(batch_by_column["spectrogram_mix"]),
        "audio_path_first": batch_by_column["audio_path_first"],
        "audio_path_second": batch_by_column["audio_path_second"],
        "audio_path_mix": batch_by_column["audio_path_mix"],
        "audio_first": batch_by_column["audio_first"],
        "audio_second": batch_by_column["audio_second"],
        "audio_mix": batch_by_column["audio_mix"],
    }
    return result_batch
