"""Creates visualizations for the activity project."""

import logging

import numpy as np
import pywt

from common import DATA_ORIGINAL_DIR, GramType, TrainOrTest, get_absolute_dir
from utils_graph import (graph_fft_signals, graph_gaussian_signals, graph_gram,
                         graph_grams, graph_signal, graph_signals,
                         graph_wavelet_signals)
from utils_signals import read_activity_names, read_labels, read_signals

# Signal graphs


def _display_signal_graphs() -> None:
    """Displays the 9 components that make up a single data point:
    * (x, y, z) representing the total acceleration signal from the
    smartphone accelerometer.
    * (x, y, z) representing the body acceleration signal obtained by
    subtracting the gravity from the total acceleration.
    * (x, y, z) representing the angular velocity vector measured by the
    gyroscope.
    """
    signals = read_signals(TrainOrTest.TRAIN,
                           get_absolute_dir(DATA_ORIGINAL_DIR, False))
    components = signals[0, :, :]
    label = _get_first_label()
    logging.info("Label: %s", label)
    graph_signals(components, label)


def _get_first_label() -> str:
    data_original_dir = get_absolute_dir(DATA_ORIGINAL_DIR, False)
    labels = read_labels(TrainOrTest.TRAIN, data_original_dir)
    activity_names = read_activity_names(data_original_dir)
    label = activity_names[labels[0]]
    return label


# FFT graphs


def _display_fft_graphs() -> None:
    """Displays a set of graphs that helps get intuition for FFT.
    """
    n = 100
    x = np.arange(n)
    y1 = 2 * np.cos(2 * np.pi * 2 * x / n)
    y2 = np.cos(2 * np.pi * 10 * x / n + 2)
    y3 = y1 + y2
    fft = np.abs(np.fft.fftshift(np.fft.fft(y3)))
    fft = fft[len(fft) // 2:]
    frequencies = np.arange(len(fft))
    graph_fft_signals(x, y1, y2, y3, frequencies, fft)


## Spectrogram graphs


def _display_spectrogram_graphs() -> None:
    """Displays graphs that demonstrate how spectrograms work.
    """
    all_signals = read_signals(TrainOrTest.TRAIN,
                               get_absolute_dir(DATA_ORIGINAL_DIR, False))
    signal = all_signals[0, 0, :]
    graph_signal(signal, "Signal", "Time", "Amplitude")
    fft = np.abs(np.fft.fftshift(np.fft.fft(signal)))
    fft = fft[fft.shape[0] // 2:]
    graph_signal(fft, "FFT of Signal", "Frequency", "Amplitude")

    spectrogram = _create_spectrogram(signal, True)
    graph_gram(spectrogram, GramType.SPECTROGRAMS)

    nine_signals = all_signals[0, :, :]
    spectrograms = []
    for signal in nine_signals:
        spectrograms.append(_create_spectrogram(signal))

    graph_grams(spectrograms, _get_first_label(), GramType.SPECTROGRAMS)


def _create_spectrogram(signal: np.ndarray,
                        should_graph: bool = False) -> np.ndarray:
    """Creates spectrogram for signal, and returns it.
    Shows graph for the signal with a Gaussian filter overlayed, the
    filtered signal, and the FFT of the signal.
    """
    n = len(signal)  # 128
    sigma = 3
    time_list = np.arange(n)
    spectrogram = np.zeros((n, n))

    for (i, time) in enumerate(time_list):
        g = _get_gaussian_filter(time, time_list, sigma)
        ug = signal * g
        ugt = np.abs(np.fft.fftshift(np.fft.fft(ug)))
        spectrogram[:, i] = ugt
        if should_graph and i in (20, 100):
            graph_gaussian_signals(signal, g, ug, ugt)

    return spectrogram


def _get_gaussian_filter(b: float, b_list: np.ndarray,
                         sigma: float) -> np.ndarray:
    """Returns the values of a Gaussian filter centered at time value b, for all
    time values in b_list, with standard deviation sigma.
    """
    a = 1 / (2 * sigma**2)
    return np.exp(-a * (b_list - b)**2)


## Scaleogram graphs


def _display_scaleogram_graphs() -> None:
    """Displays graphs that demonstrate how scaleograms work.
    """
    all_signals = read_signals(TrainOrTest.TRAIN,
                               get_absolute_dir(DATA_ORIGINAL_DIR, False))
    signal = all_signals[0, 0, :]

    graph_wavelet_signals(signal, times=[20, 100], scales=[2, 8])

    scaleogram = _create_scaleogram(signal)
    graph_gram(scaleogram, GramType.SCALEOGRAMS)

    nine_signals = all_signals[0, :, :]
    scaleograms = []
    for signal in nine_signals:
        scaleograms.append(_create_scaleogram(signal))
    graph_grams(scaleograms, _get_first_label(), GramType.SCALEOGRAMS)


def _create_scaleogram(signal: np.ndarray) -> np.ndarray:
    """Creates scaleogram for signal and returns it.
    """
    n = len(signal)
    scale_list = np.arange(start=0, stop=n) / 8 + 1
    wavelet = "mexh"
    scaleogram = pywt.cwt(signal, scale_list, wavelet)[0]
    return scaleogram


def main() -> None:
    _display_signal_graphs()
    _display_fft_graphs()
    _display_spectrogram_graphs()
    _display_scaleogram_graphs()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
