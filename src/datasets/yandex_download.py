import io
import os
import zipfile
from urllib.parse import urlencode

import requests

from src.datasets.custom_dir_audio_dataset import CustomDirAudioDataset
from src.utils.io_utils import ROOT_PATH

YANDEX_URL = {
    "dla_dataset": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": os.getenv("YANDEX_DISK_URL"),
    },
    "dla_dataset_small_a": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": os.getenv("YANDEX_DISK_URL"),
    },
    "dla_dataset_small_av": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": os.getenv("YANDEX_DISK_URL"),
    },
    "dla_dataset_onebatch_a": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": os.getenv("YANDEX_DISK_URL"),
    },
}


class YandexDownload(CustomDirAudioDataset):
    def __init__(
        self,
        download_name="dla_dataset",
        data_dir=None,
        part="train",
        lipreading_model=None,
        device="cpu",
        *args,
        **kwargs,
    ):
        self.lipreading_model = lipreading_model
        self.device = device
        
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
            print("Downloading zip data...")
            download_response = requests.get(download_url)
            print("Successfully downloaded")
            zip = zipfile.ZipFile(io.BytesIO(download_response.content))
            zip.extractall(data_dir)
        data_dir = data_dir / download_name
        super().__init__(
            data_dir / "audio" / part / "mix", 
            data_dir / "audio" / part / "s1", 
            data_dir / "audio" / part / "s2", 
            data_dir / "mouths",
            *args, **kwargs
        )
