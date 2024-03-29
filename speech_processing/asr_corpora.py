import itertools
import multiprocessing
import os
import shutil
import subprocess
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional

from tqdm import tqdm
from util import data_io
from util.dataclass_utils import CachedData
from util.processing_utils import process_with_threadpool

from speech_processing.speech_utils import ASRSample


@dataclass
class ASRCorpus(CachedData):
    manifest_name = "manifest.jsonl"

    @abstractmethod
    def _build_manifest(self):
        raise NotImplementedError

    def _build(self):
        self._build_manifest()
        self.__manifest_file_exists()

    def _get(self) -> str:
        self.__manifest_file_exists()
        return self.get_filepath(self.manifest_name)

    def __manifest_file_exists(self):
        if not os.path.isfile(self.get_filepath(self.manifest_name)):
            shutil.rmtree(Path(self.get_filepath(self.manifest_name)).parent)
            raise Exception(f"{self.get_filepath(self.manifest_name)} not existing!")


@dataclass
class ProcessedASRCorpus(ASRCorpus):

    min_duration: float = 0.2
    max_duration: float = 120.0
    min_chars: int = 2
    limit: Optional[int] = None
    mode: str = "sequential"

    @abstractmethod
    def read_raw_data(self) -> List[ASRSample]:
        raise NotImplementedError

    @abstractmethod
    def process(self, sample: ASRSample) -> Optional[ASRSample]:
        raise NotImplementedError

    def _build_manifest(self):
        raw_samples = self.read_raw_data()
        asr_samples = self.__process_filter(raw_samples)
        print(f"{asdict(self)}: {len(asr_samples)} of {len(raw_samples)} are good!")
        data_io.write_jsonl(
            self.get_filepath(self.manifest_name), (asdict(s) for s in asr_samples)
        )

    def filter_sample(self, s: ASRSample):
        duration = s.end - s.start
        good_duration = self.min_duration < duration < self.max_duration
        good_transcript = len(s.text) >= self.min_chars
        all_good = all([good_transcript, good_duration])
        return all_good

    def __process_filter(self, raw_samples: List[ASRSample]):
        if self.mode == "sequential":
            return self.__process_filter_sequentially(raw_samples)
        elif self.mode == "threadpool":
            return self.__process_filter_threading(raw_samples)
        elif self.mode == "dask":
            return self.__process_filter_dask(raw_samples)
        else:
            raise NotImplementedError

    def __not_failted(self, s):
        return self.filter_sample(s) if s is not None else False

    def __process_filter_sequentially(self, raw_samples: List[ASRSample]):
        inputs = itertools.islice(raw_samples, self.limit)
        processed_g = (self.process(t) for t in inputs)
        g = tqdm(
            filter(self.__not_failted, processed_g),
            desc=f"sequentially processing of {str(asdict(self))[:9]}",
        )  # noqa
        return list(g)

    def __process_filter_threading(self, raw_samples: List[ASRSample]):
        num_cpus = multiprocessing.cpu_count()
        inputs = itertools.islice(raw_samples, self.limit)
        processed_g = process_with_threadpool(
            inputs,
            self.process,
            max_workers=2 * num_cpus,
        )
        g = tqdm(
            filter(
                self.__not_failted,
                processed_g,
            ),
            desc=f"threadpool-based processing of {str(asdict(self))[:9]}",
        )
        return list(g)

    def __process_filter_dask(self, raw_samples: List[ASRSample]):
        # TODO: not sure whether this is intelligent!
        # cause inputs come from main-process and
        # results are collected in main-process, not really parallel

        num_cpus = multiprocessing.cpu_count()
        n_workers = 2 * num_cpus
        inputs = itertools.islice(raw_samples, self.limit)

        from dask.distributed import Client, LocalCluster
        import dask.bag as db

        with LocalCluster(
            n_workers=n_workers,
            local_directory="/tmp",
        ) as cluster, Client(cluster) as _:

            result = (
                db.from_sequence(inputs).map(self.process).filter(self.__not_failted)
            )
            asr_samples = result.compute()
        return asr_samples


# pylint: disable=abstract-method
@dataclass
class SoxResampleSegment(ProcessedASRCorpus):
    def process(self, sample: ASRSample) -> ASRSample:
        duration = sample.end - sample.start

        processed_audio_file = f"{self.cache_dir}/{sample.id}.wav"
        if not os.path.isfile(processed_audio_file):
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
