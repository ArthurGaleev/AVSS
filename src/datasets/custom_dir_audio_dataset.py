from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


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
        # audio_first_IDS=set([path.stem for path in Path(audio_first_dir).iterdir()])
        # audio_second_IDS=set([path.stem for path in Path(audio_second_dir).iterdir()])
        for path in Path(audio_mix_dir).iterdir():
            # ID1, ID2=path.stem.split("_")
            # if(ID1 not in audio_first_IDS or ID2 not in audio_second_IDS):
            #     continue
            # entry = {}
            # if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
            #     entry["audio_path_mix"] = str(path)
            #     entry["audio_path_first"] = str(audio_first_dir/(ID1+path.suffix))
            #     entry["audio_path_second"] = str(audio_second_dir/(ID2+path.suffix))
            # Need to uncomment for big dataset for now use the dataset we created
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
