"""Utilities to read and write spectrograms, scaleograms, labels and
activity names."""

import json
from pathlib import Path
from typing import Callable, List

import h5py
import numpy as np
from tqdm import tqdm

from common import GramType, TrainOrTest


def save_grams(gram_type: GramType, train_or_test: TrainOrTest,
               data_processed_dir: str, signals: np.ndarray,
               create_gram_func: Callable[[np.ndarray], np.ndarray]) -> None:
    """Saves spectrograms or scaleograms.
    """
    (num_instances, num_components,
     num_timesteps) = signals.shape  # (_, 9, 128)
    chunk_shape = (1, num_components, num_timesteps, num_timesteps)
    full_shape = (num_instances, num_components, num_timesteps, num_timesteps)

    gram_path = _get_gram_file_path(gram_type, train_or_test,
                                    data_processed_dir)

    with h5py.File(gram_path, "w") as file:
        dataset = file.create_dataset(name=gram_type.value,
                                      shape=chunk_shape,
                                      maxshape=full_shape,
                                      chunks=chunk_shape)

        for instance in tqdm(range(num_instances)):
            instance_grams = np.zeros(chunk_shape)  # (9, 128, 128)

            for component in range(num_components):
                signal = signals[instance, component, :]
                gram = create_gram_func(signal)  # (128, 128)
                instance_grams[0, component, :, :] = gram

            dataset.resize(instance + 1, axis=0)
            dataset[instance, :, :, :] = instance_grams


def save_labels(
    labels: np.ndarray,
    train_or_test: TrainOrTest,
    data_processed_dir: str,
) -> None:
    """Saves labels in the same location as spectrograms and scaleograms.
    """
    labels_path = _get_label_file_path(train_or_test, data_processed_dir)

    with h5py.File(labels_path, "w") as file:
        file.create_dataset(name="labels", data=labels)


def save_activity_names(activity_names: List[str],
                        data_processed_dir: str) -> None:
    """Saves the activity names in the same location as spectrograms and
    scaleograms.
    """
    activity_names_path = _get_activity_names_path(data_processed_dir)

    with open(activity_names_path, "wt", encoding="utf-8") as f:
        json.dump(activity_names, f)


def read_grams(
    gram_type: GramType,
    train_or_test: TrainOrTest,
    data_processed_dir: str,
) -> h5py.Dataset:
    """Reads test or train spectrograms or scaleograms that represent the
    signals.

    Returns a dataset of spectrograms or scaleograms, of shape
        (7352, 9, gram_size, gram_size) for train or
        (2947, 9, gram_size, gram_size) for test.
    """
    path = _get_gram_file_path(gram_type, train_or_test, data_processed_dir)
    file = h5py.File(path, "r")
    key = list(file.keys())[0]
    return file[key]


def read_labels(train_or_test: TrainOrTest,
                data_processed_dir: str) -> h5py.Dataset:
    """Reads the train or test activity labels.

    Returns a dataset of shape (7352,) for training labels or
    (2947,) for test labels, containing one integer per training sample.
    """
    path = _get_label_file_path(train_or_test, data_processed_dir)
    file = h5py.File(path, "r")
    key = list(file.keys())[0]
    return file[key]


def read_activity_names(data_processed_dir: str) -> List[str]:
    """Reads the file with activity names, and returns a list containing those
    names.
    """
    activity_names_path = _get_activity_names_path(data_processed_dir)
    with open(activity_names_path, "rt", encoding="utf-8") as f:
        activity_names = json.load(f)
    return activity_names


def _get_gram_file_path(
    gram_type: GramType,
    train_or_test: TrainOrTest,
    data_processed_dir: str,
) -> Path:
    """Returns path to a gram signal file.
    """
    folder = Path(data_processed_dir)
    folder.mkdir(exist_ok=True)
    return Path(folder, f"{train_or_test.value}_{gram_type.value}.hdf5")


def _get_label_file_path(train_or_test: TrainOrTest,
                         data_processed_dir: str) -> Path:
    """Returns path to a label file.
    """
    return Path(data_processed_dir, f"{train_or_test.value}_labels.hdf5")


def _get_activity_names_path(data_processed_dir: str) -> Path:
    """Returns path to the file with activity names.
    """
    return Path(data_processed_dir, "activity_names.json")
