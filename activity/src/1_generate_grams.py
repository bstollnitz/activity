"""Generates spectrograms or scaleograms and saves them, along with the
corresponding labels."""

import argparse
import logging
import math
import zipfile
from pathlib import Path

import numpy as np
import pywt

import utils_grams
import utils_signals
from common import (DATA_COMPRESSED_DIR, DATA_ORIGINAL_DIR, DATA_PROCESSED_DIR,
                    REDUCTION_FACTOR, GramType, TrainOrTest, get_absolute_dir)


def generate_all(data_compressed_dir: str, data_original_dir: str,
                 data_processed_dir: str) -> None:
    """Unzips the data, reads the original signals and labels, creates
    spectrograms and scaleograms from the signals, and saves spectrograms,
    scaleograms and labels.

    This code takes about 10-12 minutes to run on my machine, when using a
    REDUCTION_FACTOR of 2, and about 5-6 minutes with a REDUCTION_FACTOR of 4.
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
    # signals has shape (num_instances, 9, 128).
    (num_instances, num_components, num_timesteps) = signals.shape
    gram_size = math.ceil(num_timesteps / REDUCTION_FACTOR)
    # grams has shape (num_instances, 9, gram_size, gram_size).
    grams = np.zeros((num_instances, num_components, gram_size, gram_size))

    create_gram_func = (_create_spectrogram if gram_type
                        == GramType.SPECTROGRAMS else _create_scaleogram)

    for instance in range(num_instances):
        for component in range(num_components):
            signal = signals[instance, component, :]
            # gram has shape (gram_size, gram_size).
            gram = create_gram_func(signal)
            grams[instance, component, :, :] = gram

    return grams


def _create_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Creates spectrogram for signal, and returns it.
    """
    # Length of the signal: 128.
    n = len(signal)
    # Discrete time points for the signal readings.
    time_list = np.arange(n)
    # The second dimension of the output spectrogram corresponds to the times
    # where we will center the Gabor filter. These times are stored in
    # time_slide.
    time_slide = np.arange(n, step=REDUCTION_FACTOR)
    # The first dimension corresponds to the frequencies of the FFT.
    # The FFT is the same size as the input signal, but we'll reduce it
    # to have the same size as time_slide. So we'll end up with square
    # spectrograms.
    gram_size = len(time_slide)
    spectrogram = np.zeros((gram_size, gram_size), dtype=complex)

    for (i, time) in enumerate(time_slide):
        sigma = 3
        # We isolate the original filter at a particular time by multiplying
        # it with a Gaussian filter centered at that time.
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        # We reduce the fft so that we end up with square spectrograms.
        ugt = _reduce_fft(np.fft.fftshift(np.fft.fft(ug)), gram_size)
        # The resulting spectrogram represents time in the x axis, frequency
        # in the y axis, and the color shows amplitude.
        spectrogram[:, i] = ugt

    # We get real values by taking the absolute value. Then we normalize to get
    # values between 0 and 1.
    spectrogram = _normalize(np.abs(spectrogram))
    return spectrogram


def _get_gaussian_filter(b: float, b_list: np.ndarray,
                         sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at time value b, for all
    time values in b_list, with standard deviation sigma.
    """
    a = 1 / (2 * sigma**2)
    return np.exp(-a * (b_list - b)**2)


def _reduce_fft(fft_signal: np.ndarray, gram_size: int) -> np.ndarray:
    """Reduces fft signal to be of size gram_size, and returns that.

    Ignores high frequencies, positive and negative. Keeps low frequencies.
    """
    begin = (len(fft_signal) - gram_size) // 2
    end = begin + gram_size
    return fft_signal[begin:end]


def _normalize(my_array: np.ndarray) -> np.ndarray:
    """Normalizes an ndarray to values between 0 and 1, and returns that.
    """
    my_array -= my_array.min()
    my_array /= my_array.max()
    return my_array


def _create_scaleogram(signal: np.ndarray) -> np.ndarray:
    """Creates scaleogram for signal, and returns it.
    """
    # Length of the signal: 128.
    n = len(signal)

    # In the PyWavelets implementation, scale 1 corresponds to a wavelet with
    # domain [-8, 8], which means that it covers 17 samples (upper - lower + 1).
    # Scale s corresponds to a wavelet with s*17 samples.
    # We want half of the widest wavelet to cover roughly all samples.
    # The scales in scale_list range from 1 to 16.75. The widest wavelet is
    # 17*16.75 = 284.75 wide, which is just over double the size of the signal.

    # The first dimension of the spectrogram corresponds to the scale list.
    # Therefore, we make sure that scale_list has shape (gram_size,).

    scale_list = np.arange(start=0, stop=n, step=REDUCTION_FACTOR) / 8 + 1

    wavelet = "mexh"

    # The resulting scaleogram represents scale in the first dimension, time in
    # the second dimension, and the color shows amplitude.
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0][:, ::REDUCTION_FACTOR]
    scaleogram = _normalize(scaleogram)

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
                        default=get_absolute_dir(DATA_COMPRESSED_DIR))
    parser.add_argument("--data_original_dir",
                        dest="data_original_dir",
                        default=get_absolute_dir(DATA_ORIGINAL_DIR))
    parser.add_argument("--data_processed_dir",
                        dest="data_processed_dir",
                        default=get_absolute_dir(DATA_PROCESSED_DIR))
    args = parser.parse_args()
    data_compressed_dir = args.data_compressed_dir
    data_original_dir = args.data_original_dir
    data_processed_dir = args.data_processed_dir

    generate_all(data_compressed_dir, data_original_dir, data_processed_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
