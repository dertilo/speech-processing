import itertools
import multiprocessing
import os
import shutil
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

    def _build(self):
        raw_samples = self.read_raw_data()
        asr_samples = self.process_filter(raw_samples)
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

    def process_filter(self, raw_samples: List[ASRSample]):
        if self.mode == "sequential":
            return self.process_filter_sequentially(raw_samples)
        elif self.mode == "threadpool":
            return self.process_filter_threading(raw_samples)
        else:
            raise NotImplementedError

    def __not_failted(self, s):
        return self.filter_sample(s) if s is not None else False

    def process_filter_sequentially(self, raw_samples: List[ASRSample]):
        inputs = itertools.islice(raw_samples, self.limit)
        processed_g = (self.process(t) for t in inputs)
        g = tqdm(
            filter(self.__not_failted, processed_g),
            desc=f"sequentially processing of {str(asdict(self))[:9]}",
        )  # noqa
        return list(g)

    def process_filter_threading(self, raw_samples: List[ASRSample]):
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
