from numpy.typing import NDArray
from dataclasses import dataclass
from enum import Enum
import numpy as np

@dataclass
class IterationResult:
    """
    Data class to store the iteration results.
    """
    data: NDArray[np.int_]
    num_clusters: int
    labels: NDArray[np.int_]
    noise_points: NDArray[np.bool_]
    db_score: float
    iteration: int
    epsilon: float

class VerboseLevel(Enum):
    """
    Enumeration for controlling the verbosity level of output messages.

    Levels:
        NONE (0): No output
        BASIC (1): Basic progress information
        DETAILED (2): Detailed progress and debug information
    """
    NONE = 0
    BASIC = 1
    DETAILED = 2