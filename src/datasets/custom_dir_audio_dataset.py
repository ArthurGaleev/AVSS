from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset
from tqdm.auto import tqdm


class CustomDirAudioDataset(BaseDataset):
    def __init__(
        self,
        audio_mix_dir,
        audio_first_dir=None,
        audio_second_dir=None,
        *args,
        **kwargs,
    ):
        data = []
        for path in tqdm(list(Path(audio_mix_dir).iterdir()), desc="Creating dataset"):
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["audio_path_mix"] = str(path)
                if audio_first_dir is not None and audio_second_dir is not None:
                    entry["audio_path_first"] = str(audio_first_dir / path.name)
                    entry["audio_path_second"] = str(audio_second_dir / path.name)
                t_info = torchaudio.info(str(path))
                mix_length = t_info.num_frames / t_info.sample_rate
                if audio_first_dir is not None:
                    t_info = torchaudio.info(str(audio_first_dir / path.name))
                    first_length = t_info.num_frames / t_info.sample_rate
                    t_info = torchaudio.info(str(audio_second_dir / path.name))
                    second_length = t_info.num_frames / t_info.sample_rate
                    assert mix_length == first_length == second_length
                entry["audio_len"] = mix_length
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
