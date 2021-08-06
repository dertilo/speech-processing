from typing import Generator
from typing import Optional

import numpy as np
from nemo.collections.asr.parts.preprocessing import AudioSegment

from speech_processing.speech_utils import MAX_16_BIT_PCM


def break_into_chunks(array: np.ndarray, chunk_size):
    buffer = []
    for a in array:
        buffer.append(a)
        if len(buffer) == chunk_size:
            chunk = np.concatenate(buffer)
            yield chunk
            buffer = []

    if len(buffer) > 0:
        yield np.concatenate(buffer)


def resample_stream_file(
    audio_filepath, target_sample_rate, offset=0.0, duration=None, chunk_duration=0.05
):
    array = load_and_resample(audio_filepath, target_sample_rate, offset, duration)
    return break_into_chunks(array, int(target_sample_rate * chunk_duration))


def load_and_resample(audio_filepath, target_sample_rate, offset=0.0, duration=None):
    audio = AudioSegment.from_file(
        audio_filepath,
        target_sr=target_sample_rate,
        offset=offset,
        duration=0
        if duration is None
        else duration,  # cause nemo wants 0 if no duration
        trim=False,
    )
    a = audio.samples.squeeze()
    a = a / np.max(a) * (MAX_16_BIT_PCM - 1)
    a = a.astype(np.int16)
    a = np.expand_dims(a, axis=1)
    return a


def build_buffer_audio_arrays_generator(chunk_size=1600) -> Generator:
    chunk = yield
    buffer = np.zeros(0, dtype=np.int16)
    while chunk is not None:
        valid_chunk: Optional[np.ndarray] = None
        assert buffer.dtype == chunk.dtype
        buffer = np.concatenate([buffer, chunk])
        if len(buffer) >= chunk_size:
            valid_chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]
            assert len(buffer) <= chunk_size, len(buffer)
        chunk = yield valid_chunk

    if len(buffer) > 0:
        assert len(buffer) <= chunk_size, len(buffer)
        yield buffer
        # could be that part of the signal is thrown away
