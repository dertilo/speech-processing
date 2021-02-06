# pylint: skip-file
# flake8: noqa
import os
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
import soundfile as sf
from nemo.collections.asr.parts.perturb import Perturbation
from nemo.collections.asr.parts.segment import AudioSegment

from speech_processing.sox_signal_augmentation import add_signals
from speech_processing.sox_signal_augmentation import build_random_noise
from speech_processing.sox_signal_augmentation import build_random_pert
from speech_processing.sox_signal_augmentation import varying_gain_pert

# TODO(tilo): pass params to init


class SoxPerturbations(Perturbation):
    """"""

    def __init__(self):
        pass

    def perturb(self, data):
        att_factor = 0.8
        max_level = np.max(np.abs(data._samples))
        norm_factor = att_factor / max_level
        norm_samples = norm_factor * data._samples
        with NamedTemporaryFile(suffix=".wav") as orig_f, NamedTemporaryFile(
            suffix="_augmented.wav"
        ) as tmp_file:
            sf.write(orig_f.name, norm_samples.transpose(), 16000)

            original = orig_f.name
            augmented = tmp_file.name

            min_SNR = 20  # normal:20, less:30, evenless:40

            signal_gain = round(np.random.triangular(left=-10, mode=0.0, right=30), 2)
            noise = build_random_noise(min_SNR, original, signal_gain)
            gain_pert_sig = varying_gain_pert(original)

            pert_sig = build_random_pert(gain_pert_sig, signal_gain)

            sox_cmd = add_signals([noise, pert_sig], augmented)
            FNULL = open(os.devnull, "w")
            subprocess.call(
                ["bash", "-c", sox_cmd, "> /dev/null 2>&1"],
                stdout=FNULL,
                stderr=subprocess.STDOUT,
            )

            new_data = AudioSegment.from_file(augmented, target_sr=16000)
        data._samples = new_data._samples
        return
