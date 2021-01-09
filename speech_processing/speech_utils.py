from dataclasses import dataclass
from typing import Optional

import torchaudio

MAX_16_BIT_PCM: float = float(2 ** 15)  # 32768.0 for 16 bit, see "format"


def torchaudio_info(audio_file):
    try:
        si, _ = torchaudio.info(audio_file)  # torchaudio==0.7.2
        num_frames = si.length / si.channels
        sample_rate = si.rate
    except Exception:
        info = torchaudio.info(audio_file)  # torchaudio==0.8.0
        num_frames = info.num_frames
        sample_rate = info.sample_rate
    return num_frames, sample_rate


@dataclass
class AudioSample:
    id: str
    audio_filepath: str
    sample_rate: int
    start: float = 0.0  # in sec
    end: Optional[float] = None  # TODO: does anyone rely on start end?

    @property
    def duration(self):
        return self.end - self.start


@dataclass
class ASRSample(AudioSample):
    text: Optional[str] = None  # used only for training,
