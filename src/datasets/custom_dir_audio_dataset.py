from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.lipreading.preprocess import get_preprocessing_pipeline


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

                if audio_first_dir is not None and audio_second_dir is not None:
                    entry["audio_path_first"] = str(audio_first_dir / path.name)
                    entry["audio_path_second"] = str(audio_second_dir / path.name)
                    t_info = torchaudio.info(str(audio_first_dir / path.name))
                    first_length = t_info.num_frames / t_info.sample_rate
                    t_info = torchaudio.info(str(audio_second_dir / path.name))
                    second_length = t_info.num_frames / t_info.sample_rate
                    assert mix_length == first_length == second_length

                    if mouths_dir:
                        assert (
                            self.lipreading_model
                        ), "Need to initialize pre-trained lipreading model for embedds extraction"

                        first_mouth_path, second_mouth_path = path.stem.split("_")

                        first_mouth_path = mouths_dir / (first_mouth_path + ".npz")
                        second_mouth_path = mouths_dir / (second_mouth_path + ".npz")

                        preprocessing_func = get_preprocessing_pipeline()
                        if first_mouth_path.exists():
                            mouth_data = preprocessing_func(
                                np.load(first_mouth_path)["data"]
                            )
                        else:
                            assert (
                                second_mouth_path.exists()
                            ), "One of mouth paths should exist"
                            mouth_data = preprocessing_func(
                                np.load(second_mouth_path)["data"]
                            )

                        with torch.no_grad():
                            entry["mouth_embedds"] = (
                                self.lipreading_model(
                                    torch.FloatTensor(mouth_data)[
                                        None, None, :, :, :
                                    ].to(self.device),
                                    lengths=[mouth_data.shape[0]],
                                )
                                .squeeze(0)
                                .cpu()  # TODO cpu?
                            )  # (T, H*W) shape, e.g. (50, 1024)

            if len(entry) > 0:
                data.append(entry)

        # delete unnecessary model and empty cache
        if self.lipreading_model:
            del self.lipreading_model  # TODO correct?
            torch.cuda.empty_cache()

        super().__init__(data, *args, **kwargs)
