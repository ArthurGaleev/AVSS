import io
import os
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import requests
import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


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


YANDEX_URL = {
    "dla_dataset_small_a": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": os.getenv("YANDEX_DISK_URL"),
    }
}


class YandexDownload(CustomDirAudioDataset):
    def __init__(
        self,
        download_name="dla_dataset_small_a",
        data_dir=None,
        part="train",
        *args,
        **kwargs,
    ):
        assert download_name in YANDEX_URL
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets"
        if not (data_dir / download_name / "audio" / part).exists():
            data_dir.mkdir(exist_ok=True, parents=True)
            download_info = YANDEX_URL[download_name]
            final_url = download_info["base_url"] + urlencode(
                dict(public_key=download_info["public_key"])
            )
            response = requests.get(final_url)
            download_url = response.json()["href"]
            download_response = requests.get(download_url)
            zip = zipfile.ZipFile(io.BytesIO(download_response.content))
            zip.extractall(data_dir)
        data_dir = data_dir / download_name / "audio" / part
        super().__init__(
            data_dir / "mix", data_dir / "s1", data_dir / "s2", *args, **kwargs
        )
