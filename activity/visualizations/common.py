"""Constants, enums, and code used by most other files."""

from enum import Enum
from pathlib import Path


class GramType(Enum):
    SPECTROGRAMS = "spectrograms"
    SCALEOGRAMS = "scaleograms"


class TrainOrTest(Enum):
    TRAIN = "train"
    TEST = "test"


DATA_ORIGINAL_DIR = str(Path(Path(__file__).parent.parent, "data_original"))
