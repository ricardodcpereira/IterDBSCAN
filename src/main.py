from src.iterative_dbscan.iter_dbscan import adaptiveDBSCAN, VerboseLevel
from src.optimal_cluster.optimal_cluster import OptimalCluster
from typing import Union, Tuple, Literal
import pandas as pd
import numpy as np

def iter_dbscan(
    dataset: Union[np.ndarray, pd.DataFrame],
    min_samples: int = 3,
    metric: str = 'euclidean',
    stability_threshold: float = 0.1,
    stability_window: int = 5,
    step_growth_ratio: int = 5,
    verbose: Literal['none', 'basic', 'detailed'] = 'detailed'
) -> Tuple[np.ndarray, float]:
    """
    Performs iterative DBSCAN clustering with automatic parameter optimization.

    Parameters
    ----------
    dataset : Union[np.ndarray, pd.DataFrame]
        Input data for clustering. Can be either numpy array or pandas DataFrame.
    min_samples : int, default=3
        Minimum number of samples in a neighborhood for a point to be considered a core point.
    metric : str, default='euclidean'
        Distance metric used for the DBSCAN algorithm.
    stability_threshold : float, default=0.1
        Maximum allowed variation for considering a trend stable.
    stability_window : int, default=5
        Number of iterations to consider for trend analysis.
    verbose : str, default='detailed'
        Controls the level of logging output. Options: 'none', 'basic', 'detailed'

    Returns
    -------
    Tuple[List[IterationResult], OptimizationResult]
        - List of all iteration results
        - Optimization result containing best iteration and selection reasoning"
    """
    verbose_map = {
        'none': VerboseLevel.NONE,
        'basic': VerboseLevel.BASIC,
        'detailed': VerboseLevel.DETAILED
    }
    verbose_level = verbose_map[verbose.lower()]

    dbscan = adaptiveDBSCAN(
        dataset=dataset,
        metric=metric,
        verbose_level=verbose_level
    )
    
    iterations = dbscan.iter_fit(min_samples, step_growth_ratio=step_growth_ratio)
    
    if iterations:
        optimizer = OptimalCluster(
            stability_threshold=stability_threshold,
            stability_window=stability_window,
            verbose_level=verbose_level
        )

        result = optimizer.find_optimal_clustering(iterations)

        return iterations, result
    else:
        return None, None