from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(
        self,
        audio_mix_dir,
        audio_first_dir=None,
        audio_second_dir=None,
        mouths_dir=None,
        *args,
        **kwargs,
    ):
        data = []
        for path in tqdm(list(Path(audio_mix_dir).iterdir()), desc="Creating dataset"):
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["audio_path_mix"] = str(path)
                t_info = torchaudio.info(str(path))
                mix_length = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = mix_length
                switch = False
                if audio_first_dir is not None and audio_second_dir is not None:
                    if mouths_dir:
                        assert Path(mouths_dir).exists(), "Mouths dir doesnt exist"
                        first_mouth_path, second_mouth_path = path.stem.split("_")
                        first_mouth_path = mouths_dir / (first_mouth_path + ".npz")
                        second_mouth_path = mouths_dir / (second_mouth_path + ".npz")
                        if first_mouth_path.exists():
                            entry["mouth_path"] = first_mouth_path
                        else:
                            switch = True
                            assert (
                                second_mouth_path.exists()
                            ), "One of the mouths should exist"
                            entry["mouth_path"] = second_mouth_path
                    entry["audio_path_first"] = str(audio_first_dir / path.name)
                    entry["audio_path_second"] = str(audio_second_dir / path.name)
                    if switch:
                        entry["audio_path_first"], entry["audio_path_second"] = (
                            entry["audio_path_second"],
                            entry["audio_path_first"],
                        )
                    t_info = torchaudio.info(str(audio_first_dir / path.name))
                    first_length = t_info.num_frames / t_info.sample_rate
                    t_info = torchaudio.info(str(audio_second_dir / path.name))
                    second_length = t_info.num_frames / t_info.sample_rate
                    assert mix_length == first_length == second_length
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
