import os
from dataclasses import asdict
from dataclasses import dataclass
from itertools import islice
from typing import Optional

import datasets
from tqdm import tqdm
from util import data_io

from speech_processing.asr_corpora import ASRCorpus
from speech_processing.speech_utils import ASRSample
from speech_processing.speech_utils import torchaudio_info

HF_DATASETS = "huggingface/datasets"


def build_asr_sample(d, HF_DATASETS_CACHE) -> Optional[ASRSample]:
    file = d["path"]
    if not file.startswith(HF_DATASETS_CACHE):
        file = f"{HF_DATASETS_CACHE}{file.split(HF_DATASETS)[-1]}"
    if os.path.isfile(file):
        num_frames, sample_rate = torchaudio_info(file)
        duration = (num_frames - 2) / sample_rate

        o = ASRSample(
            file.split("/")[-1], file, sample_rate, end=duration, text=d["sentence"]
        )
    else:
        print(f"WARNING: {file} not fount!!")
        o = None
    return o


@dataclass
class CommonVoiceRawData(ASRCorpus):
    lang: str = "en"
    split_name: str = "train"
    num_samples: int = 100
    HF_DATASETS_CACHE: str = "/tmp/HF_CACHE"

    def _build_manifest(self):
        assert self.HF_DATASETS_CACHE.endswith(HF_DATASETS), self.HF_DATASETS_CACHE
        os.environ["HF_DATASETS_CACHE"] = self.HF_DATASETS_CACHE
        some_could_fail_factor = 2  # reserve more, to compensate possible failures
        try:  # cause some could fail
            ds = datasets.load_dataset(
                "common_voice",
                self.lang,
                keep_in_memory=True,
                split=f"{self.split_name}[:{self.num_samples * some_could_fail_factor}]",
                cache_dir=self.HF_DATASETS_CACHE,
            )
        except Exception:
            ds = datasets.load_dataset(
                "common_voice",
                self.lang,
                keep_in_memory=True,
                split=f"{self.split_name}",
                cache_dir=self.HF_DATASETS_CACHE,
            )
        data = islice(
            filter(
                lambda x: x is not None,
                (
                    build_asr_sample(d, self.HF_DATASETS_CACHE)
                    for d in tqdm(ds, desc=f"building: {asdict(self)}")
                ),
            ),
            self.num_samples,
        )
        data_io.write_jsonl(
            self.get_filepath(self.manifest_name),
            (asdict(d) for d in data),
        )


if __name__ == "__main__":
    dataset = CommonVoiceRawData(
        "/tmp/cache",
        "de",
        "validation",
        HF_DATASETS_CACHE=os.environ["HF_DATASETS_CACHE"],
    ).build_or_get()
