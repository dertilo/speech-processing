import os
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from itertools import islice
from typing import List
from typing import Optional

import datasets
from tqdm import tqdm
from util import data_io

from speech_processing.asr_corpora import ASRCorpus
from speech_processing.asr_corpora import ProcessedASRCorpus
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
    num_samples: int = None
    HF_DATASETS_CACHE: str = "/tmp/HF_CACHE"

    @property
    def name(self):
        num_samples_s = (
            "-" + str(self.num_samples) if self.num_samples is not None else ""
        )
        return f"{self.lang}-{self.split_name}{num_samples_s}"

    def _build_manifest(self):
        assert self.HF_DATASETS_CACHE.endswith(HF_DATASETS), self.HF_DATASETS_CACHE
        os.environ["HF_DATASETS_CACHE"] = self.HF_DATASETS_CACHE
        some_could_fail_factor = 2  # reserve more, to compensate possible failures
        try:  # cause some could fail
            assert self.num_samples is not None
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


@dataclass
class CommonVoiceCorpusProcessDump(CommonVoiceRawData, ProcessedASRCorpus):
    def read_raw_data(self) -> List[ASRSample]:
        rawdata = CommonVoiceRawData(
            self.cache_base,
            lang=self.lang,
            split_name=self.split_name,
            num_samples=self.num_samples,
            HF_DATASETS_CACHE=self.HF_DATASETS_CACHE,
        ).build_or_get()
        return [ASRSample(**d) for d in data_io.read_jsonl(rawdata)]

    def process(self, sample: ASRSample) -> Optional[ASRSample]:
        duration = sample.end - sample.start

        processed_audio_file = f"{self.cache_dir}/{sample.id}.wav"
        # with NamedTemporaryFile(suffix="_tmp.wav") as tmp_file:
        subprocess.check_output(
            # first channel only
            f"sox '{sample.audio_filepath}' -c 1 -r 16000 {processed_audio_file} trim {sample.start} {duration}",
            shell=True,
        )
        # transcode_perturbation(tmp_file.name, processed_audio_file)

        return ASRSample(
            sample.id,
            processed_audio_file,
            sample.sample_rate,
            text=sample.text,
            start=0.0,
            end=duration,
        )


if __name__ == "__main__":
    dataset = CommonVoiceRawData(
        "/tmp/cache",
        "de",
        "validation",
        HF_DATASETS_CACHE=os.environ["HF_DATASETS_CACHE"],
    ).build_or_get()
