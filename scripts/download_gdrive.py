import zipfile
from pathlib import Path

import gdown

GDRIVE_URLS = {
    "models": {
        # rtfs-4-reuse
        "https://drive.google.com/uc?id=1lzO_cyN0WKof1rtnCPZpN-BHcELO-85c": "data/models/rtfs-4-reuse.pth",
        # rtfs-3-no-video
        "https://drive.google.com/uc?id=1FMlXhzC_3-AuLKo30rvCgH0PRwVJo-tM": "data/models/rtfs-3-no-video.pth",
        # rtfs-11-reuse
        "https://drive.google.com/uc?id=13r68YvHAvXJzcTwH5D5M7oHg2lk9js_e": "data/models/rtfs-11-reuse.pth",
        # rtfs-3
        "https://drive.google.com/uc?id=1CGfWvINud5d9tppRq3tnJNqDYgC65uX9": "data/models/rtfs-3.pth",
        # dprnn
        "https://drive.google.com/uc?id=1IU4mFseGPtac6g58jTS48GzlMI1wMJQ8": "data/models/dprnn.pth",
    },
    "dataset": {
        "https://drive.google.com/uc?id=178uAdx9S6lln1z0Yr9ws-tBpmDkSuHqD": "data/datasets/dla_dataset_small_av"
    },
}


def download_models(gdrive_urls):
    if "models" not in gdrive_urls:
        raise ValueError("Cannot upload model files")
    path_gzip = Path("data/models").absolute()
    path_gzip.mkdir(exist_ok=True, parents=True)
    for url, path in gdrive_urls["models"].items():
        gdown.download(url, path, quiet=False)


def download_dataset(gdrive_urls):
    if "dataset" not in gdrive_urls:
        raise ValueError("Cannot upload dataset files")
    path_gzip = Path("data/datasets").absolute()
    path_gzip.mkdir(exist_ok=True, parents=True)
    for url, path in gdrive_urls["dataset"].items():
        zip_path = path + ".zip"
        gdown.download(url, zip_path)
        if zip_path.endswith(".zip"):
            extract_folder = path
            Path(extract_folder).mkdir(exist_ok=True, parents=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_folder)
            Path(zip_path).unlink()


if __name__ == "__main__":
    download_models(GDRIVE_URLS)
    download_dataset(GDRIVE_URLS)
