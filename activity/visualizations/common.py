"""Constants, enums, and code used by most other files."""

from enum import Enum


class GramType(Enum):
    SPECTROGRAMS = "spectrograms"
    SCALEOGRAMS = "scaleograms"


class TrainOrTest(Enum):
    TRAIN = "train"
    TEST = "test"


DATA_ORIGINAL_DIR = "activity/data_original"
