"""Utilities to produce visualizations."""

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pywt

from common import GramType

very_light_gray = "#efefef"
light_gray = "#999999"
dark_gray = "#444444"
orange = "#EF6C00"
teal = "#007b96"


def _style_axis2d(ax, xlabel: str, ylabel: str):
    """Styles a 2D graph.
    """
    ax.set_xlabel(xlabel, {"color": dark_gray})
    ax.set_ylabel(ylabel, {"color": dark_gray})
    ax.set_title(ax.get_title(), {"color": dark_gray})
    ax.tick_params(axis="x", colors=light_gray)
    ax.tick_params(axis="y", colors=light_gray)
    ax.set_facecolor(very_light_gray)
    for spine in ax.spines.values():
        spine.set_edgecolor(light_gray)


def graph_signal(y: np.ndarray, title: str, labelx: str, labely: str) -> None:
    """Displays a signal."""
    x = np.arange(len(y))

    plt.figure(figsize=plt.figaspect(0.5))
    ax = plt.axes()

    ax.plot(x, y, color=orange, linewidth=0.4)
    ax.set_title(title)
    _style_axis2d(ax, labelx, labely)

    plt.show()


def graph_signals(components: np.ndarray, label: np.ndarray) -> None:
    """Displays graphs containing the 9 components that make up a data point.
    """
    fig = plt.figure(figsize=plt.figaspect(0.5))
    plt.suptitle(f"9 signals measured while \"{label}\"",
                 color=dark_gray,
                 fontsize=15)
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(3, 3, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(3, 3, 4, sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(3, 3, 5, sharex=ax1, sharey=ax1)
    ax6 = fig.add_subplot(3, 3, 6, sharex=ax1, sharey=ax1)
    ax7 = fig.add_subplot(3, 3, 7, sharex=ax1, sharey=ax1)
    ax8 = fig.add_subplot(3, 3, 8, sharex=ax1, sharey=ax1)
    ax9 = fig.add_subplot(3, 3, 9, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=3)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    titles = [
        "Total Acceleration X", "Total Acceleration Y", "Total Acceleration Z",
        "Body Acceleration X", "Body Acceleration Y", "Body Acceleration Z",
        "Angular Velocity X", "Angular Velocity Y", "Angular Velocity Z"
    ]

    for (i, ax) in enumerate(axes):
        y = components[i, :]
        x = np.arange(len(y))
        ax.plot(x, y, color=orange, linewidth=0.4)
        ax.set_title(titles[i])
        _style_axis2d(ax, "Time", "Amplitude")

    plt.show()


def graph_fft_signals(x: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                      y3: np.ndarray, frequencies: np.ndarray,
                      fft: np.ndarray) -> None:
    """Displays 4 graphs: two containing signals, one containing the sum of
    the previous signals, and another containing the fft of the sum-signal.
    """
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(4, 1, 4)
    fig.tight_layout(pad=4)

    ax1.plot(x, y1, color=orange, linewidth=0.4)
    ax1.set_title("Signal 1")
    _style_axis2d(ax1, "Time (Seconds)", "Amplitude")

    ax2.plot(x, y2, color=orange, linewidth=0.4)
    ax2.set_title("Signal 2")
    _style_axis2d(ax2, "Time (Seconds)", "Amplitude")

    ax3.plot(x, y3, color=orange, linewidth=0.4)
    ax3.set_title("Signal 3 = Signal 1 + Signal 2")
    _style_axis2d(ax3, "Time (Seconds)", "Amplitude")

    ax4.plot(frequencies, fft, color=orange, linewidth=0.4)
    ax4.set_title("FFT of Signal 3")
    _style_axis2d(ax4, "Frequency (Hz)", "Magnitude")

    plt.show()


def graph_gaussian_signals(y1: np.ndarray, y2: np.ndarray, y3: np.ndarray,
                           y4: np.ndarray) -> None:
    """Displays three graphs: one containing the signal and Gaussian filter,
    another containing the filtered signal, and another containing the FFT.
    """
    x1 = np.arange(len(y1))
    x2 = np.arange(len(y2))
    x3 = np.arange(len(y3))
    x4 = np.arange(len(y4)) - len(y4) // 2

    fig = plt.figure(figsize=plt.figaspect(1.5))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(3, 1, 3)
    fig.tight_layout(pad=3)

    ax1.plot(x1, y1, color=orange, linewidth=0.4)
    ax1.plot(x2, y2, color=teal, linewidth=0.4)
    ax1.set_title("Signal and Gaussian Filter")
    _style_axis2d(ax1, "Time", "Amplitude")
    ax1.legend(labels=["Signal", "Gaussian filter"], fontsize=8)

    ax2.plot(x3, y3, color=orange, linewidth=0.4)
    ax2.set_title("Filtered Signal")
    _style_axis2d(ax2, "Time", "Amplitude")

    ax3.plot(x4, y4, color=orange, linewidth=0.4)
    ax3.set_title("FFT of Filtered Signal")
    _style_axis2d(ax3, "Frequency", "Amplitude")

    plt.show()


def graph_gram(gram: np.ndarray, gram_type: GramType) -> None:
    """Displays a spectrogram or scaleogram.
    """
    ax = plt.axes()
    n = gram.shape[0]
    if gram_type == GramType.SPECTROGRAMS:
        extent = [0, n, -n / 2, n / 2]
        plt.imshow(gram, cmap="Oranges", extent=extent)
        ax.set_title("Spectrogram")
        _style_axis2d(ax, "Time", "Frequency")
    else:
        plt.imshow(gram, cmap="Oranges")
        # We'll show a tick every 10 values.
        ticks = np.arange(stop=n, step=10)
        labels = np.arange(stop=n, step=10) / 8 + 1
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_title("Scaleogram")
        _style_axis2d(ax, "Time", "Scale")
    cb = plt.colorbar()
    cb.outline.set_color(light_gray)
    cb.ax.tick_params(axis="y", colors=light_gray)

    plt.show()


def graph_grams(grams: List[np.ndarray], label: str,
                gram_type: GramType) -> None:
    """Displays 9 spectrograms or scaleograms.
    """
    fig = plt.figure(figsize=(7, 7))
    gram = "spectrogram" if gram_type == GramType.SPECTROGRAMS else "scaleogram"
    plt.suptitle(f"9 {gram}s measured while \"{label}\"",
                 color=dark_gray,
                 fontsize=15)
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(3, 3, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(3, 3, 4, sharex=ax1, sharey=ax1)
    ax5 = fig.add_subplot(3, 3, 5, sharex=ax1, sharey=ax1)
    ax6 = fig.add_subplot(3, 3, 6, sharex=ax1, sharey=ax1)
    ax7 = fig.add_subplot(3, 3, 7, sharex=ax1, sharey=ax1)
    ax8 = fig.add_subplot(3, 3, 8, sharex=ax1, sharey=ax1)
    ax9 = fig.add_subplot(3, 3, 9, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=2)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    titles = [
        "Total Acceleration X", "Total Acceleration Y", "Total Acceleration Z",
        "Body Acceleration X", "Body Acceleration Y", "Body Acceleration Z",
        "Angular Velocity X", "Angular Velocity Y", "Angular Velocity Z"
    ]

    n = grams[0].shape[0]
    for (i, gram) in enumerate(grams):
        ax = axes[i]
        if gram_type == GramType.SPECTROGRAMS:
            ax.imshow(gram, cmap="Oranges", extent=[0, n, -n / 2, n / 2])
        else:
            ticks = np.arange(stop=n, step=50)
            labels = np.arange(stop=n, step=50) / 8 + 1
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.imshow(gram, cmap="Oranges")
        ax.set_title(titles[i])
        _style_axis2d(ax, "", "")

    plt.show()


def graph_wavelet_signals(signal: np.ndarray, times: List[int],
                          scales: List[int]) -> None:
    """Graphs signal with wavelets of different scales and in different
    locations overlayed.
    """
    t = np.arange(len(signal))

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(2, 2, 4, sharex=ax1, sharey=ax1)
    fig.tight_layout(pad=4)

    n = len(signal)
    wavelet1 = _get_wavelet(times[0], scales[0], n)
    wavelet2 = _get_wavelet(times[1], scales[0], n)
    wavelet3 = _get_wavelet(times[0], scales[1], n)
    wavelet4 = _get_wavelet(times[1], scales[1], n)

    ax1.plot(t, signal, color=orange, linewidth=0.4)
    ax1.plot(t, wavelet1, color=teal, linewidth=0.4)
    ax1.set_title(f"Wavelet at time {times[0]} with scale {scales[0]}")
    _style_axis2d(ax1, "Time", "Amplitude")
    ax1.legend(labels=["Signal", "Wavelet"], fontsize=8)

    ax2.plot(t, signal, color=orange, linewidth=0.4)
    ax2.plot(t, wavelet2, color=teal, linewidth=0.4)
    ax2.set_title(f"Wavelet at time {times[1]} with scale {scales[0]}")
    _style_axis2d(ax2, "Time", "Amplitude")
    ax2.legend(labels=["Signal", "Wavelet"], fontsize=8)

    ax3.plot(t, signal, color=orange, linewidth=0.4)
    ax3.plot(t, wavelet3, color=teal, linewidth=0.4)
    ax3.set_title(f"Wavelet at time {times[0]} with scale {scales[1]}")
    _style_axis2d(ax3, "Time", "Amplitude")
    ax3.legend(labels=["Signal", "Wavelet"], fontsize=8)

    ax4.plot(t, signal, color=orange, linewidth=0.4)
    ax4.plot(t, wavelet4, color=teal, linewidth=0.4)
    ax4.set_title(f"Wavelet at time {times[1]} with scale {scales[1]}")
    _style_axis2d(ax4, "Time", "Amplitude")
    ax4.legend(labels=["Signal", "Wavelet"], fontsize=8)

    plt.show()


def _get_wavelet(translate: int, scale: float, length: int) -> np.ndarray:
    """Calculates the values of a Mexican hat wavelet with the length specified,
    and at the time translation and scale specified as parameters.
    """
    wavelet_name = "mexh"
    wavelet_width = 17
    [wavelet, _
    ] = pywt.ContinuousWavelet(wavelet_name).wavefun(length=int(wavelet_width *
                                                                scale))
    middle = len(wavelet) // 2
    result = np.zeros(length)
    src_start = 0
    src_end = len(wavelet)
    dest_start = translate - middle
    dest_end = dest_start + len(wavelet)
    if dest_start < 0:
        src_start -= dest_start
        dest_start = 0
    if dest_end > length:
        src_end -= dest_end - length
        dest_end = length
    result[dest_start:dest_end] = wavelet[src_start:src_end]
    return result
