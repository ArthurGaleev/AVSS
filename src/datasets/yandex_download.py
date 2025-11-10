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
        "public_key": "https://disk.360.yandex.ru/d/R99_Q19X6ztnVw",
    },
    "dla_dataset_small_av": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": "https://disk.360.yandex.ru/d/5pz96ysIZi33IQ",
    },
    "dla_dataset_onebatch_a": {
        "base_url": "https://cloud-api.yandex.net/v1/disk/public/resources/download?",
        "public_key": "https://disk.360.yandex.ru/d/NFPe8zivc2nvLA",
    },
}


class YandexDownload(CustomDirAudioDataset):
    def __init__(
        self,
        download_name="dla_dataset",
        data_dir=None,
        part="train",
        *args,
        **kwargs,
    ):
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
        data_dir = data_dir / download_name / "audio" / part
        super().__init__(
            data_dir / "mix", data_dir / "s1", data_dir / "s2", *args, **kwargs
        )
