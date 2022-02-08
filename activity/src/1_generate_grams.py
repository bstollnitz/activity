"""Generates spectrograms or scaleograms and saves them, along with the
corresponding labels."""

import argparse
import logging
import zipfile
from pathlib import Path

import numpy as np
import pywt

import utils_grams
import utils_signals
from common import (DATA_COMPRESSED_DIR, DATA_ORIGINAL_DIR, DATA_PROCESSED_DIR,
                    GramType, TrainOrTest)


def generate_all(data_compressed_dir: str, data_original_dir: str,
                 data_processed_dir: str) -> None:
    """Unzips the data, reads the original signals and labels, creates
    spectrograms and scaleograms from the signals, and saves spectrograms,
    scaleograms and labels.

    This code takes several minutes to run.
    """
    _unzip_original_data(data_compressed_dir, data_original_dir)

    _generate_grams(GramType.SPECTROGRAMS, TrainOrTest.TRAIN, data_original_dir,
                    data_processed_dir)
    _generate_grams(GramType.SPECTROGRAMS, TrainOrTest.TEST, data_original_dir,
                    data_processed_dir)
    _generate_grams(GramType.SCALEOGRAMS, TrainOrTest.TRAIN, data_original_dir,
                    data_processed_dir)
    _generate_grams(GramType.SCALEOGRAMS, TrainOrTest.TEST, data_original_dir,
                    data_processed_dir)

    _save_labels(TrainOrTest.TRAIN, data_original_dir, data_processed_dir)
    _save_labels(TrainOrTest.TEST, data_original_dir, data_processed_dir)

    _save_activity_names(data_original_dir, data_processed_dir)


def _unzip_original_data(data_compressed_dir: str,
                         data_original_dir: str) -> None:
    """Unzips the original data.
    """
    zip_file = next(Path(data_compressed_dir).iterdir())
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(path=data_original_dir)


def _generate_grams(
    gram_type: GramType,
    train_or_test: TrainOrTest,
    data_original_dir: str,
    data_processed_dir: str,
) -> None:
    """Reads original data, computes spectrograms or scaleograms, and saves
    them.
    """
    logging.info("Generating %s %s.", train_or_test.value, gram_type.value)

    signals = utils_signals.read_signals(train_or_test, data_original_dir)
    grams = create_grams(gram_type, signals)
    utils_grams.save_grams(grams, gram_type, train_or_test, data_processed_dir)


def create_grams(gram_type: GramType, signals: np.ndarray) -> np.ndarray:
    """Generates spectrograms or scaleograms from signals, and returns them.
    """
    (num_instances, num_components,
     num_timesteps) = signals.shape  # (_, 9, 128)
    grams = np.zeros((num_instances, num_components, num_timesteps,
                      num_timesteps))  # (_, 9, 128, 128)

    create_gram_func = (_create_spectrogram if gram_type
                        == GramType.SPECTROGRAMS else _create_scaleogram)

    for instance in range(num_instances):
        for component in range(num_components):
            signal = signals[instance, component, :]
            gram = create_gram_func(signal)  # (128, 128)
            grams[instance, component, :, :] = gram

    return grams


def _create_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Creates spectrogram for signal, and returns it.

    The resulting spectrogram represents time on the x axis, frequency
    on the y axis, and the color shows amplitude.
    """
    n = len(signal)  # 128
    sigma = 3
    time_list = np.arange(n)
    spectrogram = np.zeros((n, n))

    for (i, time) in enumerate(time_list):
        # We isolate the original signal at a particular time by multiplying
        # it with a Gaussian filter centered at that time.
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        # Then we calculate the FFT. Some FFT values may be complex, so we take
        # the absolute value to guarantee that they're all real.
        # The FFT is the same size as the original signal.
        ugt = np.abs(np.fft.fftshift(np.fft.fft(ug)))
        # The result becomes a column in the spectrogram.
        spectrogram[:, i] = ugt

    return spectrogram


def _get_gaussian_filter(b: float, b_list: np.ndarray,
                         sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at time value b, for all
    time values in b_list, with standard deviation sigma.
    """
    a = 1 / (2 * sigma**2)
    return np.exp(-a * (b_list - b)**2)


def _create_scaleogram(signal: np.ndarray) -> np.ndarray:
    """Creates scaleogram for signal, and returns it.

    The resulting scaleogram represents scale in the first dimension, time in
    the second dimension, and the color shows amplitude.
    """
    n = len(signal)  # 128

    # In the PyWavelets implementation, scale 1 corresponds to a wavelet with
    # domain [-8, 8], which means that it covers 17 samples (upper - lower + 1).
    # Scale s corresponds to a wavelet with s*17 samples.
    # The scales in scale_list range from 1 to 16.75. The widest wavelet is
    # 17*16.75 = 284.75 wide, which is just over double the size of the signal.
    scale_list = np.arange(start=0, stop=n) / 8 + 1  # 128
    wavelet = "mexh"
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]
    return scaleogram


def _save_labels(train_or_test: TrainOrTest, data_original_dir: str,
                 data_processed_dir: str) -> None:
    """Reads labels and saves them in the same location as spectrograms and
    scaleograms.
    """
    labels = utils_signals.read_labels(train_or_test, data_original_dir)
    utils_grams.save_labels(labels, train_or_test, data_processed_dir)


def _save_activity_names(data_original_dir: str,
                         data_processed_dir: str) -> None:
    """Reads the human activity names and saves them to the same location
    as spectrograms and scaleograms.
    """
    activity_names = utils_signals.read_activity_names(data_original_dir)
    utils_grams.save_activity_names(activity_names, data_processed_dir)


def main() -> None:
    logging.info("Generating spectrograms and scaleograms.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_compressed_dir",
                        dest="data_compressed_dir",
                        default=DATA_COMPRESSED_DIR)
    parser.add_argument("--data_original_dir",
                        dest="data_original_dir",
                        default=DATA_ORIGINAL_DIR)
    parser.add_argument("--data_processed_dir",
                        dest="data_processed_dir",
                        default=DATA_PROCESSED_DIR)
    args = parser.parse_args()

    generate_all(args.data_compressed_dir, args.data_original_dir,
                 args.data_processed_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
