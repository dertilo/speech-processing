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
