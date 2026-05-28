"""
Adaptive DBSCAN Implementation with Iterative Refinement

This module implements an adaptive version of the DBSCAN clustering algorithm
that iteratively refines clusters based on local density characteristics.

Classes:
    VerboseLevel: Controls output verbosity levels
    ClusteringResult: Stores clustering operation results
    adaptiveDBSCAN: Main clustering implementation with iterative refinement

Example:
    from iter_dbscan import adaptiveDBSCAN
    
    data = np.random.rand(100, 2)
    clusterer = adaptiveDBSCAN(data)
    results = clusterer.iter_fit(min_samples=5)
"""

from typing import Union, Tuple, Optional
import numpy as np
from sklearn.metrics import pairwise_distances, davies_bouldin_score
from dataclasses import dataclass
from numpy.typing import NDArray
import pandas as pd
from ..configs import IterationResult, VerboseLevel

@dataclass
class ClusteringResult:
    """
    Data class containing the results of a DBSCAN clustering operation.

    Attributes
    ----------
    labels : NDArray[np.int_]
        Cluster labels for each point in the dataset. -1 indicates noise points.
    num_clusters : int
        Total number of clusters found (excluding noise).
    noise_points : NDArray[np.bool_]
        Boolean mask indicating which points are classified as noise.
    """
    labels: NDArray[np.int_]
    num_clusters: int
    noise_points: NDArray[np.bool_]

class adaptiveDBSCAN:
    """
    Adaptive DBSCAN clustering algorithm with iterative refinement capabilities.

    This implementation extends traditional DBSCAN by incorporating adaptive epsilon
    adjustment, local growth statistics, and noise point filtering mechanisms.

    Attributes
    ----------
    SATURATION_THRESHOLD_DEFAULT : float
        Default threshold for determining cluster saturation (0.7)
    GROWTH_THRESHOLD_MULTIPLIER_DEFAULT : float
        Default multiplier for growth threshold calculation (1.0)
    """
    
    SATURATION_THRESHOLD_DEFAULT: float = 0.7
    GROWTH_THRESHOLD_MULTIPLIER_DEFAULT: float = 1.0
    STEP_GROWTH_RATIO: int = 5.0

    def __init__(self, dataset: Union[np.ndarray, pd.DataFrame], 
                 metric: str = 'euclidean', 
                 verbose_level: VerboseLevel = VerboseLevel.NONE) -> None:
        """
        Initialize the adaptive DBSCAN clustering algorithm.

        Parameters
        ----------
        dataset : Union[np.ndarray, pd.DataFrame]
            Input data to cluster
        metric : str, default='euclidean'
            Distance metric for pairwise distance computation
        verbose_level : VerboseLevel, default=VerboseLevel.NONE
            Controls output verbosity

        Raises
        ------
        ValueError
            If dataset is empty or contains NaN values
        """
        self.verbose = verbose_level
        if not hasattr(dataset, 'to_numpy') and not isinstance(dataset, np.ndarray):
            raise ValueError("Dataset must be a DataFrame or NumPy array")
        
        if hasattr(dataset, 'to_numpy'):
            self.data: np.ndarray = dataset.to_numpy()
        else:
            self.data: np.ndarray = dataset
        
        if np.any(np.isnan(self.data)):
            raise ValueError("Input data contains NaN values")
        
        if len(self.data) == 0:
            raise ValueError("Input data is empty")
            
        self.size: int = len(self.data)
        self._log(f"Dataset initialized with {self.size} points and {self.data.shape[1]} dimensions", VerboseLevel.BASIC)
        self._log("Computing pairwise distances between all points...", VerboseLevel.BASIC)
        self.distances: np.ndarray = pairwise_distances(self.data, metric=metric)
        self._log(f"Using {metric} distance metric", VerboseLevel.DETAILED)

        self.neighbors_matrix: Optional[np.ndarray] = None
        self.epsilon: Optional[float] = None
        self.min_samples: Optional[int] = None
        self.prev_neighbors_counts: Optional[np.ndarray] = None
        self.prev_epsilon: Optional[float] = None
        self.iter: int = 0
        self.noise_points: np.ndarray = np.full(self.size, False)
        self.previous_labels: Optional[np.ndarray] = None

    def _log(self, message: str, level: VerboseLevel = VerboseLevel.BASIC) -> None:
        """
        Print message if verbose level is sufficient.

        Parameters
        ----------
        message : str
            Message to print
        level : VerboseLevel
            Minimum verbosity level required to print message
        """
        if self.verbose.value >= level.value:
            print(message)

    def _validate_parameters(self, epsilon: float, min_samples: int) -> None:
        """
        Validate clustering parameters.

        Parameters
        ----------
        epsilon : float
            Maximum distance between neighbors
        min_samples : int
            Minimum number of points to form a dense region

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if min_samples < 1:
            raise ValueError("Min_samples must be at least 1")

    def _calculate_initial_epsilon(self, distances: np.ndarray) -> float:
        """
        Calculate initial epsilon value based on distance matrix.

        Parameters
        ----------
        distances : np.ndarray
            Pairwise distance matrix

        Returns
        -------
        float
            Initial epsilon value
        """
        self._log("[Analysis] Starting initial epsilon calculation", VerboseLevel.DETAILED)
        return np.min(distances[distances > 0]), np.max(distances[distances > 0])

    def _compute_neighbors(self, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute neighbor relationships and counts for all points.

        Parameters
        ----------
        epsilon : float
            Maximum distance between neighbors

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            neighbors_matrix: Boolean matrix of neighbor relationships
            neighbors_counts: Array of neighbor counts for each point
        """
        neighbors_matrix = self.distances <= epsilon
        np.fill_diagonal(neighbors_matrix, False)
        neighbors_counts = np.sum(neighbors_matrix, axis=1)
        return neighbors_matrix, neighbors_counts

    def _compute_potential_neighbors(self, epsilon: float, prev_epsilon: float) -> np.ndarray:
        """
        Compute potential neighbor counts at an extended radius.

        Parameters
        ----------
        epsilon : float
            Current epsilon value
        prev_epsilon : float
            Previous epsilon value

        Returns
        -------
        np.ndarray
            Array of potential neighbor counts
        """
        larger_radius = epsilon + (epsilon - prev_epsilon)
        potential_neighbors_matrix = self.distances <= larger_radius
        np.fill_diagonal(potential_neighbors_matrix, False)
        return np.sum(potential_neighbors_matrix, axis=1)

    def _cluster_points(self, neighbors_matrix: np.ndarray, 
                       neighbors_counts: np.ndarray, 
                       min_samples: int, 
                       noise_mask: Optional[np.ndarray] = None) -> ClusteringResult:
        """
        Perform DBSCAN clustering using pre-computed neighbor relationships.

        Parameters
        ----------
        neighbors_matrix : np.ndarray
            Boolean matrix of neighbor relationships
        neighbors_counts : np.ndarray
            Array of neighbor counts
        min_samples : int
            Minimum points required for core point
        noise_mask : Optional[np.ndarray]
            Pre-identified noise points

        Returns
        -------
        ClusteringResult
            Clustering results including labels and noise points
        """
        if noise_mask is None:
            noise_mask = np.full(self.size, False)

        labels = np.full(self.size, -1)
        cluster_id = 0
        core_points = (neighbors_counts >= min_samples) & ~noise_mask

        for point_idx in np.where(core_points)[0]:
            if labels[point_idx] != -1:
                continue

            labels[point_idx] = cluster_id
            seed_points = np.where(neighbors_matrix[point_idx] & ~noise_mask)[0]

            i = 0
            processed = set()
            while i < len(seed_points):
                current_point = seed_points[i]
                if current_point in processed:
                    i += 1
                    continue

                processed.add(current_point)

                if labels[current_point] == -1:
                    labels[current_point] = cluster_id

                if neighbors_counts[current_point] >= min_samples:
                    new_neighbors = np.where(neighbors_matrix[current_point] & ~noise_mask)[0]
                    for neighbor in new_neighbors:
                        if neighbor not in processed:
                            seed_points = np.append(seed_points, neighbor)

                i += 1
            cluster_id += 1

        return ClusteringResult(
            labels=labels,
            num_clusters=cluster_id,
            noise_points=noise_mask
        )

    def fit(self, epsilon: float, min_samples: int) -> ClusteringResult:
        """
        Perform standard DBSCAN clustering.

        Parameters
        ----------
        epsilon : float
            Maximum distance between neighbors
        min_samples : int
            Minimum points required for core point

        Returns
        -------
        ClusteringResult
            Results of clustering operation
        """
        self._validate_parameters(epsilon, min_samples)
        self.epsilon = epsilon
        self.min_samples = min_samples
        neighbors_matrix, neighbors_counts = self._compute_neighbors(epsilon)
        return self._cluster_points(neighbors_matrix, neighbors_counts, min_samples)

    def _compute_local_growth_statistics(self, neighbors_growth: np.ndarray, 
                                       prev_labels: np.ndarray, 
                                       valid_mask: np.ndarray) -> np.ndarray:
        """
        Compute growth statistics for points using local and global metrics.

        Parameters
        ----------
        neighbors_growth : np.ndarray
            Changes in neighbor counts
        prev_labels : np.ndarray
            Previous iteration's cluster labels
        valid_mask : np.ndarray
            Mask of valid points to consider

        Returns
        -------
        np.ndarray
            Growth thresholds for each point
        """
        growth_thresholds = np.zeros(self.size)
        unique_clusters = np.unique(prev_labels)

        no_cluster_mask = (prev_labels == -1)
        unassigned_valid_mask = no_cluster_mask & valid_mask
        unassigned_growth = neighbors_growth[unassigned_valid_mask]

        if len(unassigned_growth) > 0:
            global_threshold = (
                np.mean(unassigned_growth) - 
                (self.growth_threshold_multiplier * np.std(unassigned_growth))
            )
        else:
            global_threshold = 0.0
        growth_thresholds[no_cluster_mask] = global_threshold

        cluster_thresholds = {}
        for cluster in unique_clusters:
            if cluster == -1:
                continue
                
            cluster_mask = (prev_labels == cluster) & valid_mask
            if np.any(cluster_mask):
                cluster_growth = neighbors_growth[cluster_mask]
                cluster_thresholds[cluster] = (
                    np.mean(cluster_growth) - 
                    (self.growth_threshold_multiplier * np.std(cluster_growth))
                )
                growth_thresholds[cluster_mask] = cluster_thresholds[cluster]

        unassigned_points = np.where(no_cluster_mask & valid_mask)[0]
        for point_idx in unassigned_points:
            nearby_clusters = set()
            
            for cluster in cluster_thresholds:
                cluster_points = prev_labels == cluster
                distances_to_cluster = self.distances[point_idx, cluster_points]
                if np.any(distances_to_cluster <= self.epsilon):
                    nearby_clusters.add(cluster)
            
            if nearby_clusters:
                most_lenient_threshold = max(cluster_thresholds[c] for c in nearby_clusters)
                growth_thresholds[point_idx] = most_lenient_threshold

        return growth_thresholds

    def iter_fit(self, min_samples: int, 
                 growth_threshold_multiplier: float = GROWTH_THRESHOLD_MULTIPLIER_DEFAULT, 
                 saturation_threshold: float = SATURATION_THRESHOLD_DEFAULT,
                 step_growth_ratio: int = STEP_GROWTH_RATIO):
        """
        Perform iterative DBSCAN clustering with adaptive refinement.

        Parameters
        ----------
        min_samples : int
            Minimum points required for core point
        growth_threshold_multiplier : float
            Multiplier for growth threshold calculation
        saturation_threshold : float
            Threshold for cluster saturation

        Returns
        -------
        List[IterationResult]
            Results from each iteration of clustering

        Raises
        ------
        ValueError
            If saturation_threshold is not between 0 and 1
        """
        self._log(f"Starting iterative clustering with min_samples={min_samples}", VerboseLevel.BASIC)

        initial_epsilon, max_epsilon = self._calculate_initial_epsilon(self.distances)
        self._log(f"Initial epsilon value: {initial_epsilon:.6f}", VerboseLevel.DETAILED)

        if not 0 <= saturation_threshold <= 1:
            raise ValueError("saturation_threshold must be between 0 and 1")

        self.min_samples = min_samples
        self.growth_threshold_multiplier = growth_threshold_multiplier
        self.epsilon = initial_epsilon
        self.prev_epsilon = initial_epsilon
        consecutive_clusters = 0
        prev_num_clusters = 0
        iter_results = []

        while True:
            if self.epsilon > max_epsilon:
                self._log("\nClustering algorithm stopped - Could not step away from 1 cluster.", VerboseLevel.BASIC)
                return None
            self._log(f"\nIteration {self.iter}", VerboseLevel.BASIC)
            self._log(f"Current epsilon: {self.epsilon:.6f}", VerboseLevel.DETAILED)

            neighbors_matrix, current_neighbors_counts = self._compute_neighbors(self.epsilon)

            if self.iter == 0:
                result = self._cluster_points(
                    neighbors_matrix, 
                    current_neighbors_counts,
                    min_samples,
                    self.noise_points
                )
            else:
                neighbors_growth = current_neighbors_counts - self.prev_neighbors_counts
                potential_neighbors = self._compute_potential_neighbors(self.epsilon, self.prev_epsilon)

                with np.errstate(divide='ignore', invalid='ignore'):
                    saturation_levels = current_neighbors_counts / potential_neighbors
                    saturation_levels = np.nan_to_num(saturation_levels)

                is_saturated = saturation_levels >= saturation_threshold
                valid_mask = ~self.noise_points & ~is_saturated

                if np.any(valid_mask):
                    growth_thresholds = self._compute_local_growth_statistics(
                        neighbors_growth,
                        self.previous_labels,
                        valid_mask
                    )
                    new_noise_points = (neighbors_growth < growth_thresholds) & ~is_saturated
                    self.noise_points |= new_noise_points

                result = self._cluster_points(
                    neighbors_matrix, 
                    current_neighbors_counts,
                    min_samples,
                    self.noise_points
                )

            valid_points = ~self.noise_points
            self._log(f"Found {result.num_clusters} clusters", VerboseLevel.BASIC)
            self._log(f"Noise points: {np.sum(self.noise_points)} ({np.sum(self.noise_points)/self.size*100:.1f}%)", VerboseLevel.DETAILED)

            if np.sum(valid_points) > 1 and result.num_clusters > 1:
                db_score = davies_bouldin_score(
                    self.data[valid_points],
                    result.labels[valid_points]
                )

                iter_results.append(
                    IterationResult(
                        self.data,
                        result.num_clusters,
                        result.labels,
                        self.noise_points,
                        db_score,
                        self.iter,
                        self.epsilon
                    )
                )

                self.prev_epsilon = self.epsilon
                # growth_factor = 1 + np.log1p(db_score) / 10
                growth_factor = 1 + 0.1 * (1 - np.exp(-db_score/step_growth_ratio))
                self.epsilon = self.prev_epsilon * growth_factor
            else:
                self.prev_epsilon = self.epsilon
                self.epsilon = self.prev_epsilon * 1.05

            self.previous_labels = result.labels.copy()
            self.prev_neighbors_counts = current_neighbors_counts
            self.iter += 1
            
            if result.num_clusters == prev_num_clusters:
                consecutive_clusters += 1
            else:
                consecutive_clusters = 0
            
            if consecutive_clusters >= 100:
                self._log("\nClustering algorithm terminated - number of clusters remained unchanged for 100 consecutive iterations", VerboseLevel.BASIC)
                self._log(f"Total iterations: {self.iter}", VerboseLevel.BASIC)
                return iter_results   
            
            if result.num_clusters == 1 and prev_num_clusters > 1:
                self._log("\nClustering converged - single cluster remaining", VerboseLevel.BASIC)
                self._log(f"Total iterations: {self.iter}", VerboseLevel.BASIC)
                return iter_results
            
            prev_num_clusters = result.num_clusters