from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from src.utils.lipreading.load_model import load_lipreading_model
from src.utils.lipreading.preprocess import get_preprocessing_pipeline


class CustomDirAudioDataset(BaseDataset):
    def __init__(
        self,
        audio_mix_dir,
        audio_first_dir=None,
        audio_second_dir=None,
        mouths_dir=None,
        lipreading_model_name=None,
        device=None,
        *args,
        **kwargs,
    ):
        assert device is not None, "Device should be specified"
        lipreading_model = None
        preprocessing_func = get_preprocessing_pipeline()
        data = []
        for path in tqdm(list(Path(audio_mix_dir).iterdir()), desc="Creating dataset"):
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["audio_path_mix"] = str(path)
                t_info = torchaudio.info(str(path))
                mix_length = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = mix_length
                if audio_first_dir is not None and audio_second_dir is not None:
                    if mouths_dir:
                        assert lipreading_model_name is not None
                        assert Path(mouths_dir).exists(), "Mouths dir doesnt exist"
                        first_mouth_path, second_mouth_path = path.stem.split("_")
                        entry["mouth_path_first"] = mouths_dir / (
                            first_mouth_path + ".npz"
                        )
                        entry["mouth_path_second"] = mouths_dir / (
                            second_mouth_path + ".npz"
                        )
                        assert (
                            entry["mouth_path_first"].exists()
                            and entry["mouth_path_second"].exists()
                        )
                        with torch.no_grad():
                            load_dir = (
                                ROOT_PATH
                                / "data/saved/mouth_embs"
                                / lipreading_model_name
                            )
                            load_dir.mkdir(exist_ok=True, parents=True)
                            entry["mouth_emb_path_first"] = load_dir / (
                                f"mouth_emb_{entry['mouth_path_first'].stem}.pth"
                            )
                            entry["mouth_emb_path_second"] = load_dir / (
                                f"mouth_emb_{entry['mouth_path_second'].stem}.pth"
                            )
                            for mouth_name, mouth_emb_name in [
                                ("mouth_path_first", "mouth_emb_path_first"),
                                ("mouth_path_second", "mouth_emb_path_second"),
                            ]:
                                if not entry[mouth_emb_name].exists():
                                    if lipreading_model is None:
                                        lipreading_model = load_lipreading_model(
                                            model_name=lipreading_model_name,
                                            device=device,
                                        )
                                    mouth_data = preprocessing_func(
                                        np.load(entry[mouth_name])["data"]
                                    )
                                    mouth_embed = lipreading_model(
                                        torch.FloatTensor(mouth_data)[
                                            None, None, :, :, :
                                        ].to(device),
                                        lengths=[mouth_data.shape[0]],
                                    ).squeeze(
                                        0
                                    )  # (T, H*W) shape, e.g. (50, 1024)
                                    torch.save(mouth_embed, entry[mouth_emb_name])
                        entry["audio_path_first"] = str(audio_first_dir / path.name)
                        entry["audio_path_second"] = str(audio_second_dir / path.name)
                        t_info = torchaudio.info(str(audio_first_dir / path.name))
                        first_length = t_info.num_frames / t_info.sample_rate
                        t_info = torchaudio.info(str(audio_second_dir / path.name))
                        second_length = t_info.num_frames / t_info.sample_rate
                        assert mix_length == first_length == second_length
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, device=device, *args, **kwargs)
