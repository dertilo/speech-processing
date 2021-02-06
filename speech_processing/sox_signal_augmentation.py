# pylint: skip-file
# flake8: noqa
# fmt:off
import os
import random
import subprocess
from typing import Dict

import numpy as np


MAX_FREQ = 7999


def to_str(v):
    if isinstance(v, tuple):
        s = " ".join(str(x) for x in v)
    elif isinstance(v, float) or isinstance(v, int):
        s = str(v)
    else:
        assert False

    return s


def transcode_perturbation(file, output_file):
    """
    stolen from nvidia/nemo
    """
    _rng = np.random.RandomState()
    _codecs = ["g711", "amr-nb"]

    codec_ind = random.randint(0, len(_codecs) - 1)
    if _codecs[codec_ind] == "amr-nb":
        rates = list(range(0, 8))
        rate = rates[random.randint(0, len(rates) - 1)]
        _ = subprocess.check_output(
            f"sox {file} -V0 -C {rate} -t amr-nb - | sox -t amr-nb - -V0 -b 16 -r 16000 {output_file}",
            shell=True,
        )
    elif _codecs[codec_ind] == "g711":
        _ = subprocess.check_output(
            f"sox {file} -V0  -r 8000 -c 1 -e a-law {output_file}", shell=True
        )


def build_sox_distortions(signal, params):
    param_str = " ".join([k + " " + to_str(v) for k, v in params.items()])
    sox_params = f"sox <({signal}) -p {param_str} "
    return sox_params


def build_dynamic_noise(
    audio_file,
    amod_lowpass_cutoff=0.1,
    lowpass_cutoff=MAX_FREQ,
    highpass_cutoff=1,
    noise_gain=-4,
):
    """
    band-pass-filtered whitenoise multiplied by very-low-freq whitenoise
    gives non-static/dynamically chaning noise
    :param amod_lowpass_cutoff: upper freq for noise-power changes, how "dynamic" noise it is
    play original.wav synth whitenoise lowpass 0.1 synth whitenoise amod gain -n 0 lowpass 100 highpass 1
    """

    sox_params = (
        f"sox {audio_file} -p synth whitenoise lowpass {amod_lowpass_cutoff} "
        f"synth whitenoise amod gain -n {noise_gain} lowpass {lowpass_cutoff} highpass {highpass_cutoff}"
    )
    return sox_params


def build_varying_amplitude_factor(audio_file, lowpass_cutoff=1, ac_gain=-9):
    """
    lowpass_cutoff is upper freq of ac component
    """
    ac = f"sox {audio_file} -p synth whitenoise lowpass {lowpass_cutoff} gain -n {ac_gain}"
    # WTF! dc is made by muting the original-signal and giving it an offset/dcshift!! why 0.5??
    dc = f"sox {audio_file} -p gain -90 dcshift 0.5"
    return f"sox -m <({ac}) <({dc}) -p"


def multiply_signals(signal_a, signal_b):
    return f"sox -T <({signal_a}) <({signal_b}) -p"


def varying_gain_pert(audio_file, upper_freq_for_gain_var=1, ac_gain=-6):
    factor = build_varying_amplitude_factor(
        audio_file, upper_freq_for_gain_var, ac_gain
    )
    signal = f"sox {audio_file} -p "
    return multiply_signals(factor, signal)


def add_signals_trim_to_len(original, signals, augmented):
    signals_to_add = " ".join([f"<({s})" for s in signals])
    sox_cmd = f"sox -m {signals_to_add} -b 16 {augmented} trim 0 $(soxi -D {original})"
    return sox_cmd


def add_signals(signals, outfile):
    signals_to_add = " ".join([f"<({s})" for s in signals])
    sox_cmd = f"sox -m {signals_to_add} -b 16 {outfile}"
    return sox_cmd


def build_random_bandpass(min_low=50, min_band_width=100, max_high=1000) -> Dict:
    d = {}
    max_high_cutoff = MAX_FREQ
    if np.random.choice([True, False], p=[0.5, 0.5]):
        lowpass = int(round(np.random.uniform(low=min_low, high=MAX_FREQ)))
        d["lowpass"] = lowpass
        max_high_cutoff = lowpass - min_band_width

    if np.random.choice([True, False], p=[0.5, 0.5]):
        highpass = int(
            round(np.random.uniform(low=1, high=min(max_high, max_high_cutoff)))
        )
        d["highpass"] = highpass

    return d


def build_random_noise(min_SNR, original_file, signal_gain):
    noise_power = round(np.random.uniform(-60, signal_gain - min_SNR), 2)
    lowpass = int(round(np.random.uniform(low=100, high=MAX_FREQ)))
    highpass = int(round(np.random.uniform(low=1, high=lowpass)))
    noise = build_dynamic_noise(
        original_file, np.random.uniform(0.1, 2), lowpass, highpass, noise_power
    )
    return noise


def build_random_pert(sig, signal_gain=0.0):
    # fmt:off
    pert_params = {
        "tempo": round(np.random.triangular(left=0.8, mode=1.0, right=1.1), 2),
        "pitch": int(round(np.random.triangular(left=-100, mode=0, right=100))),
        # normal 100, less: 50, evenless: 30
        "reverb": (int(round(np.random.uniform(low=0, high=50))), 50, 100, 100, 0, 0,),
        "gain -n": signal_gain,
    }
    # fmt:off
    pert_params.update(build_random_bandpass(50, 100, 1000))
    pert_sig = build_sox_distortions(sig, pert_params)
    return pert_sig



if __name__ == "__main__":
    from data_preparation.play_augmentations import get_test_files

    augmented = "/tmp/augmented.wav"
    min_SNR = 20  # normal:20, less:30, evenless:40

    if os.path.isfile(augmented):
        os.remove(augmented)
    file1, file2 = get_test_files()

    for original in [file1,file2]:
        signal_gain = round(np.random.triangular(left=-10, mode=0.0, right=30), 2)
        noise = build_random_noise(min_SNR, original, signal_gain)
        gain_pert_sig = varying_gain_pert(original)

        pert_sig=build_random_pert(gain_pert_sig,signal_gain)

        sox_cmd = add_signals(
            [noise,pert_sig], augmented
        )
        FNULL = open(os.devnull, "w")
        subprocess.call(["bash", "-c", sox_cmd], stdout=FNULL, stderr=subprocess.STDOUT)
        os.system(f"play {augmented}")
