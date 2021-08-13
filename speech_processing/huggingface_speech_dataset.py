import os
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import List
from typing import Optional

import datasets
from tqdm import tqdm
from util import data_io

from speech_processing.asr_corpora import ASRCorpus
from speech_processing.asr_corpora import SoxResampleSegment
from speech_processing.speech_utils import ASRSample
from speech_processing.speech_utils import torchaudio_info

HF_DATASETS = "huggingface_cache/datasets"


def build_asr_sample(d, HF_DATASETS_CACHE) -> Optional[ASRSample]:
    file = d["path"]
    if not file.startswith(HF_DATASETS_CACHE):
        file = f"{HF_DATASETS_CACHE}{file.split(HF_DATASETS)[-1]}"
    if os.path.isfile(file):
        num_frames, sample_rate = torchaudio_info(file)
        duration = (num_frames - 2) / sample_rate
        p = Path(file).stem
        o = ASRSample(p, file, sample_rate, end=duration, text=d["sentence"])
    else:
        print(f"WARNING: {file} not fount!!")
        o = None
    return o


@dataclass
class HuggingfaceSpeechDataset(ASRCorpus):
    lang: str = "en"
    split_name: str = "train"
    num_samples: int = None
    HF_DATASETS_CACHE: str = "/tmp/HF_CACHE"
    hf_dataset_name: str = "common_voice"

    @property
    def name(self):
        num_samples_s = (
            "-" + str(self.num_samples) if self.num_samples is not None else ""
        )
        return f"{self.hf_dataset_name}-{self.lang}-{self.split_name}{num_samples_s}"

    def _build_manifest(self):
        assert self.HF_DATASETS_CACHE.endswith(HF_DATASETS), self.HF_DATASETS_CACHE
        os.environ["HF_DATASETS_CACHE"] = self.HF_DATASETS_CACHE
        some_could_fail_factor = 2  # reserve more, to compensate possible failures
        try:  # cause some could fail
            assert self.num_samples is not None
            ds = datasets.load_dataset(
                self.hf_dataset_name,
                self.lang,
                keep_in_memory=True,
                split=f"{self.split_name}[:{self.num_samples * some_could_fail_factor}]",
                cache_dir=self.HF_DATASETS_CACHE,
            )
        except Exception:
            ds = datasets.load_dataset(
                self.hf_dataset_name,
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


@dataclass
class HFSpeechDatasetProcessDump(SoxResampleSegment, HuggingfaceSpeechDataset):
    def read_raw_data(self) -> List[ASRSample]:
        rawdata = HuggingfaceSpeechDataset(
            self.cache_base,
            lang=self.lang,
            split_name=self.split_name,
            num_samples=self.num_samples,
            HF_DATASETS_CACHE=self.HF_DATASETS_CACHE,
            hf_dataset_name=self.hf_dataset_name,
        ).build_or_get()
        return [ASRSample(**d) for d in data_io.read_jsonl(rawdata)]


if __name__ == "__main__":
    corpora_names = [
        # "SLR61", # argentinian
        "SLR71",
        "SLR72",  # colombian
        "SLR73",
        "SLR74",
        "SLR75",
    ]
    for corpus_name in corpora_names:
        dataset = HFSpeechDatasetProcessDump(
            hf_dataset_name="openslr",
            cache_base="/tmp/cache",
            lang=corpus_name,
            max_duration=20,
            split_name="train",
            # num_samples=100_000,
            mode="dask",
            HF_DATASETS_CACHE=os.environ["HF_DATASETS_CACHE"],
        )
        dataset.build_or_get()
        os.makedirs("/tmp/spanish", exist_ok=True)
        shutil.move(dataset.cache_dir, f"/tmp/spanish/{corpus_name}")
