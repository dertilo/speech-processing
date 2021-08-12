import os
import subprocess

import datasets
from datasets import DownloadConfig

# see https://raw.githubusercontent.com/huggingface/datasets/master/datasets/openslr/openslr.py

if __name__ == "__main__":
    local_cache = f'{os.environ["HOME"]}/.cache/huggingface/datasets/downloads'
    new_data_dir = os.environ["HF_DATASET_DIR"]
    assert new_data_dir.endswith("huggingface_cache/datasets")

    dataset_names = list(
        [
            "SLR32",
            "SLR35",
            "SLR36",
            "SLR41",
            "SLR42",
            "SLR43",
            "SLR44",
            "SLR52",
            "SLR53",
            "SLR54",
            "SLR63",
            "SLR64",
            "SLR65",
            "SLR66",
            "SLR69",
            "SLR70",
            "SLR71",
            "SLR72",
            "SLR73",
            "SLR74",
            "SLR75",
            "SLR76",
            "SLR77",
            "SLR78",
            "SLR79",
            "SLR80",
            "SLR86",
        ]
    )

    for lang in dataset_names:
        datasets.load_dataset(
            "openslr",
            lang,
            keep_in_memory=True,
            download_config=DownloadConfig(extract_compressed_file=False),
        )
        output = subprocess.check_output(
            f"rsync -qaz {local_cache} {new_data_dir}/", shell=True
        )
        print(output)

    print(f"you might want to manually rm -rf {local_cache}")
