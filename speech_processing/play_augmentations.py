# pylint: skip-file
# flake8: noqa
import os
import shutil
from pathlib import Path

import soundfile as sf
from nemo.collections.asr.parts.perturb import AudioAugmentor
from nemo.collections.asr.parts.perturb import GainPerturbation
from nemo.collections.asr.parts.perturb import process_augmentations
from nemo.collections.asr.parts.perturb import ShiftPerturbation
from nemo.collections.asr.parts.perturb import SpeedPerturbation
from nemo.collections.asr.parts.perturb import TimeStretchPerturbation
from nemo.collections.asr.parts.perturb import TranscodePerturbation
from nemo.collections.asr.parts.perturb import WhiteNoisePerturbation
from nemo.collections.asr.parts.segment import AudioSegment

from speech_processing.nemo_perturbation import SoxPerturbations


def augment_file(audio_filepath, aug_dir, augmentor):
    curr_sample_rate = 16_000
    audio = AudioSegment.from_file(
        audio_filepath,
        target_sr=curr_sample_rate,
        offset=0,
        # duration=duration,
        trim=False,
    )
    augmentor.perturb(audio)
    aug_file = f"{aug_dir}/{audio_filepath.split('/')[-1].split('.')[0]}.wav"
    sf.write(
        aug_file,
        audio.samples,
        curr_sample_rate,
    )
    return aug_file


def get_file(f, aug_dir):
    file1 = f"{aug_dir}/{f.split('/')[-1]}"
    if not os.path.isfile(file1):
        shutil.copy(f, file1)
    return file1


def get_test_files():
    DATA_PATH = "/home/tilo/data/audio/data"
    corpus_name = "dev-other"
    raw_data_path = f"{DATA_PATH}/corpora/LibriSpeech/{corpus_name}"
    file1_nmraid = f"{raw_data_path}/3660/6517/3660-6517-0031.flac"
    file2_nmraid = f"{raw_data_path}/1630/102884/1630-102884-0001.flac"
    aug_dir = "/tmp/aug_dir"
    os.makedirs(aug_dir, exist_ok=True)
    file1 = get_file(file1_nmraid, aug_dir)
    file2 = get_file(file2_nmraid, aug_dir)
    return file1, file2


if __name__ == "__main__":
    # fmt: off
    augmentations = [
        # (0.5, WhiteNoisePerturbation(min_level=-50, max_level=-30)),
        (1.0, TranscodePerturbation()),
        # (1.0, GainPerturbation(min_gain_dbfs=-20,max_gain_dbfs=20)),
        # (1.0, ShiftPerturbation(min_shift_ms=-300, max_shift_ms=300)),
        # (0.1, TimeStretchPerturbation(min_speed_rate=0.9,max_speed_rate=1.1,num_rates=9)),# sounds really bad!!
        # (1.0, SpeedPerturbation(sr=16_000, min_speed_rate=0.9,max_speed_rate=1.1,resample_type="kaiser_fast")),
        (1.0, SoxPerturbations()),
    ]
    # fmt: on
    file1, file2 = get_test_files()
    # os.system(f"play {file1}")
    augmentor = AudioAugmentor(perturbations=augmentations)
    aug_dir = Path(file1).parent

    for k in range(2):
        for f in [file1, file2]:
            aug_file = augment_file(f, aug_dir, augmentor)
            os.system(f"play {aug_file}")
