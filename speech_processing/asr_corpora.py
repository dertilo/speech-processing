import itertools
from abc import abstractmethod
from dataclasses import asdict
from dataclasses import dataclass
from typing import List
from typing import Optional

from tqdm import tqdm
from util import data_io
from util.dataclass_utils import CachedData

from speech_processing.speech_utils import ASRSample


@dataclass
class ASRCorpus(CachedData):

    min_duration: float = 0.2
    max_duration: float = 120.0
    min_chars: int = 2
    limit: Optional[int] = None

    manifest_name = "manifest.jsonl"

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

    def _get(self) -> str:
        return self.get_filepath(self.manifest_name)

    def filter_sample(self, s: ASRSample):
        duration = s.end - s.start
        good_duration = self.min_duration < duration < self.max_duration
        good_transcript = len(s.text) >= self.min_chars
        all_good = all([good_transcript, good_duration])
        return all_good

    def process_filter(self, raw_samples: List[ASRSample]):

        asr_samples: List[ASRSample] = list(
            itertools.islice(
                filter(
                    lambda s: self.filter_sample(s) if s is not None else False,
                    (
                        self.process(t)
                        for t in tqdm(raw_samples, desc="asr-samples for manifest")
                    ),
                ),
                self.limit,
            )  # noqa
        )  # noqa
        return asr_samples
