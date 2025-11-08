import zipfile
from pathlib import Path

from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi

from src.datasets.custom_dir_audio_dataset import CustomDirAudioDataset
from src.utils.io_utils import ROOT_PATH

load_dotenv()

KAGGLE_DATASETS = {"dla_dataset_big": "arthurgaleev/audio-visual-source-separation"}


class KaggleDownload(CustomDirAudioDataset):
    def __init__(
        self,
        download_name="dla_dataset_big",
        data_dir=None,
        part="train",
        *args,
        **kwargs,
    ):
        assert download_name in KAGGLE_DATASETS, f"Unknown dataset: {download_name}"
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets"
        dataset_dir = data_dir / download_name
        part_dir = dataset_dir / "audio" / part
        if not part_dir.exists():
            data_dir.mkdir(exist_ok=True, parents=True)
            api = KaggleApi()
            api.authenticate()
            kaggle_dataset = KAGGLE_DATASETS[download_name]
            api.dataset_download_files(
                kaggle_dataset, path=data_dir, unzip=False, quiet=True
            )
            zip_path = data_dir / f"{download_name}.zip"
            assert Path(zip_path).exists(), "It should have downloaded"
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)
        data_dir = dataset_dir / "audio" / part
        super().__init__(
            data_dir / "mix", data_dir / "s1", data_dir / "s2", *args, **kwargs
        )
