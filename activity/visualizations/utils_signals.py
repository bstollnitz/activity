"""Utilities for reading signals, labels and activity names from the original
data."""

from pathlib import Path
from typing import List

import numpy as np

from common import TrainOrTest


def read_signals(train_or_test: TrainOrTest,
                 data_original_dir: str) -> np.ndarray:
    """Reads the train or test activity signals from the directory passed as
    parameter.

    We have access to 7352 labels for training data, and 2947 for test data.

    We read 9 different signals:
    * (x, y, z) representing the total acceleration signal from the
    smartphone accelerometer.
    * (x, y, z) representing the body acceleration signal obtained by
    subtracting the gravity from the total acceleration.
    * (x, y, z) representing the angular velocity vector measured by the
    gyroscope.

    Each data point contains 128 readings over time.

    Returns an ndarray of shape (7352, 9, 128) for training signals
    or (2947, 9, 128) for test signals, containing ("number of
    samples", "number of data signals", "number of readings over time").
    """
    signals_dir_path = Path(data_original_dir, "UCI HAR Dataset",
                            train_or_test.value, "Inertial Signals")
    file_paths = list(signals_dir_path.iterdir())

    # We sort the file paths alphabetically because iterdir returns paths in
    # arbitrary order.
    file_paths.sort()

    signals_list = []
    for file_path in file_paths:  # 9
        rows = []
        with open(file_path, "rt", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:  # number of samples, 2947 or 7352
            row = [float(reading) for reading in line.split()]  # 128
            row = _standardize(row)
            rows.append(row)
        signals_list.append(rows)
    signals = np.array(signals_list)  # shape (9, 7352 or 2947, 128)
    signals = np.transpose(signals, (1, 0, 2))  # shape (7352 or 2947, 9, 128)
    return signals


def _standardize(my_array: np.ndarray) -> np.ndarray:
    """Standardizes an ndarray by subtracting its average and dividing by its
    standard deviation.
    """
    my_array -= np.average(my_array)
    my_array /= np.std(my_array)
    return my_array


def read_labels(train_or_test: TrainOrTest,
                data_original_dir: str) -> np.ndarray:
    """Reads the train or test labels.

    Returns an ndarray of shape (7352,) for training labels or (2947,)
    for test labels. Each label is an integer between 0 and 5.
    """
    labels_file_path = Path(data_original_dir, "UCI HAR Dataset",
                            train_or_test.value, f"y_{train_or_test.value}.txt")

    with open(labels_file_path, "rt", encoding="utf-8") as f:
        labels_list = f.readlines()
    return np.array([int(label.rstrip()) - 1 for label in labels_list])


def read_activity_names(data_original_dir: str) -> List[str]:
    """Reads the list of names for human activities, and returns a list
    containing those activities.

    Activities are: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING,
    STANDING, LAYING.
    """
    activity_names_path = Path(data_original_dir, "UCI HAR Dataset",
                               "activity_labels.txt")
    activity_names = []
    with open(activity_names_path, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        (_, value) = line.split(" ")
        pretty_value = value.rstrip().lower().replace("_", " ")
        activity_names.append(pretty_value)
    return activity_names
