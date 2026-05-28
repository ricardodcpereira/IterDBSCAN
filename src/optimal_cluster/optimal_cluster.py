"""
Optimal Clustering Analysis Module

This module provides functionality for analyzing and optimizing clustering results
through iteration analysis and scoring mechanisms. It implements methods for:
- Trend analysis of clustering metrics
- Multi-criteria scoring of clustering results
- Optimal clustering selection

The main class OptimalCluster evaluates clustering iterations based on:
- Davies-Bouldin index scores
- Noise point ratios
- Cluster count stability
- Iteration progress

Classes:
    Trend: Enumeration of possible trend directions
    ClusteringScore: Named tuple for storing scoring results
    OptimizationResult: Named tuple for optimization outcomes
    OptimalCluster: Main implementation for clustering optimization

Example:
    from optimal_cluster import OptimalCluster
    
    optimizer = OptimalCluster()
    result = optimizer.find_optimal_clustering(iteration_results)
    print(f"Best iteration found: {result.reason}")
"""

from typing import List, Tuple, NamedTuple, Dict
from enum import Enum
import numpy as np
import math
from ..configs import IterationResult, VerboseLevel

class Trend(Enum):
    """
    Enumeration of possible trend directions in metric analysis.

    Values
    ------
    INCREASING : str
        Metrics show an upward trend
    DECREASING : str
        Metrics show a downward trend
    STABLE : str
        Metrics remain relatively constant
    """
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"

class ClusteringScore(NamedTuple):
    """
    Named tuple containing clustering evaluation scores.

    Attributes
    ----------
    total_score : float
        Combined score from all evaluation criteria
    breakdown : Dict[str, float]
        Individual scores for each evaluation criterion
    """
    total_score: float
    breakdown: Dict[str, float]

class OptimizationResult(NamedTuple):
    """
    Named tuple containing optimization results.

    Attributes
    ----------
    best_iteration : IterationResult
        Best clustering iteration found
    reason : str
        Explanation for the selection
    all_scores : List[Dict[str, float]]
        Scores for all iterations
    """
    best_iteration: IterationResult
    reason: str
    all_scores: List[Dict[str, float]]

class OptimalCluster:
    """
    Optimization analyzer for clustering results.

    This class implements methods for analyzing clustering iterations
    and determining optimal results based on multiple criteria.

    Parameters
    ----------
    stability_threshold : float, default=0.1
        Maximum allowed variation for considering a trend stable
    stability_window : int, default=5
        Number of iterations to consider for trend analysis
    verbose_level : VerboseLevel, default=VerboseLevel.NONE
        Controls the level of logging output
    """
    
    def __init__(self, stability_threshold: float = 0.1, 
                 stability_window: int = 5,
                 verbose_level: VerboseLevel = VerboseLevel.NONE) -> None:
        
        self.stability_threshold = stability_threshold
        self.stability_window = stability_window
        self.verbose = verbose_level

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

    def analyze_trend(self, window: List[float]) -> Tuple[Trend, float]:
        """
        Analyze the trend in a window of values.
        
        Parameters
        ----------
        window : List[float]
            List of values to analyze
            
        Returns
        -------
        Tuple[Trend, float]
            Trend type and slope
        """
        self._log(f"Analyzing trend for window of size {len(window)}", VerboseLevel.DETAILED)
        
        actual_size = len(window)
        if actual_size < 2:
            self._log("Insufficient data points for trend analysis", VerboseLevel.DETAILED)
            return Trend.STABLE, 0.0
            
        x = np.arange(actual_size)
        y = np.array(window)
        slope, _ = np.polyfit(x, y, 1)
        
        mean = np.mean(window)
        max_deviation = np.max(np.abs(window - mean))
        relative_variation = max_deviation / mean if mean != 0 else float('inf')
        
        self._log(f"Relative variation: {relative_variation:.4f}, Slope: {slope:.4f}", VerboseLevel.DETAILED)
        
        if relative_variation <= self.stability_threshold:
            return Trend.STABLE, slope
        elif slope > 0:
            return Trend.INCREASING, slope
        else:
            return Trend.DECREASING, slope

    def _calculate_normalized_scores(self, iteration: IterationResult, iteration_idx: int, 
                                  total_points: int, total_iterations: int,
                                  normalized_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Calculate normalized scores for a clustering iteration.

        Parameters
        ----------
        iteration : IterationResult
            Current iteration to score
        iteration_idx : int
            Index of current iteration
        total_points : int
            Total number of data points
        total_iterations : int
            Total number of iterations
        normalized_ranges : Dict[str, Tuple[float, float]]
            Min and max values for each metric

        Returns
        -------
        Dict[str, float]
            Dictionary containing normalized scores
        """
        scores = {}

        raw_db_score = max(0, 2.0 - iteration.db_score) / 2.0
        min_db, max_db = normalized_ranges['db_score']
        db_range = max_db - min_db
        normalized_db = (raw_db_score - min_db) / db_range if db_range > 0 else 0.5
        scores['db_score'] = normalized_db * 2.0

        raw_noise = 1.0 - abs((np.sum(iteration.noise_points) / total_points) - 0.1)
        min_noise, max_noise = normalized_ranges['noise_ratio']
        noise_range = max_noise - min_noise
        normalized_noise = (raw_noise - min_noise) / noise_range if noise_range > 0 else 0.5
        scores['noise_ratio'] = normalized_noise * 1.5

        num_clusters = iteration.num_clusters
        if num_clusters < 2:
            raw_cluster = 0.0
        else:
            data_size = len(iteration.data)
            optimal_min = max(2, int(np.log(data_size) / 2))
            optimal_max = max(5, int(np.sqrt(data_size) / 4))
            
            if optimal_min <= num_clusters <= optimal_max:
                raw_cluster = 1.0
            else:
                distance = min(abs(num_clusters - optimal_min), abs(num_clusters - optimal_max))
                decay_factor = max(1, (optimal_max - optimal_min) / 2)
                raw_cluster = 1.0 - (1 - math.exp(-distance/decay_factor))

        min_cluster, max_cluster = normalized_ranges['cluster_count']
        cluster_range = max_cluster - min_cluster
        normalized_cluster = (raw_cluster - min_cluster) / cluster_range if cluster_range > 0 else 0.5
        scores['cluster_count'] = normalized_cluster * 1.5

        raw_iter = np.log1p(iteration_idx) / np.log1p(total_iterations)
        min_iter, max_iter = normalized_ranges['iteration_progress']
        iter_range = max_iter - min_iter
        normalized_iter = (raw_iter - min_iter) / iter_range if iter_range > 0 else 0.5
        scores['iteration_progress'] = normalized_iter * 0.5

        scores['total_score'] = sum(scores.values())
        return scores

    def score_clustering(self, iteration: IterationResult, total_points: int, iteration_idx: int, total_iterations: int) -> ClusteringScore:
        """
        Calculate a comprehensive score for a clustering iteration based on multiple criteria.

        Parameters
        ----------
        iteration : IterationResult
            The clustering iteration results to evaluate
        total_points : int
            Total number of data points in the dataset
        iteration_idx : int
            Index of the current iteration
        total_iterations : int
            Total number of iterations performed

        Returns
        -------
        ClusteringScore
            Named tuple containing the total score and breakdown of individual criteria scores:
            - db_score: Davies-Bouldin index score (weight: 2.0)
            - noise_ratio: Score based on optimal noise point ratio (weight: 1.5)
            - cluster_count: Score based on number of clusters (weight: 1.5)
            - iteration_progress: Score based on iteration progress (weight: 0.5)
        """
        self._log(f"Scoring iteration {iteration_idx + 1}/{total_iterations}", VerboseLevel.BASIC)
        scores = {}

        if total_iterations < 2:
            self._log("Single iteration mode - using DB score only", VerboseLevel.DETAILED)
            db_score = max(0, 2.0 - iteration.db_score) / 2.0
            scores['db_score'] = db_score * 2.0
            scores['total_score'] = scores['db_score']
            return ClusteringScore(scores['total_score'], scores)

        db_score = max(0, 2.0 - iteration.db_score) / 2.0
        scores['db_score'] = db_score * 2.0
        self._log(f"DB Score: {scores['db_score']:.3f}", VerboseLevel.DETAILED)

        noise_ratio = np.sum(iteration.noise_points) / total_points
        optimal_noise_ratio = 0.1
        noise_score = 1.0 - abs(noise_ratio - optimal_noise_ratio)
        scores['noise_ratio'] = noise_score * 1.5
        self._log(f"Noise Ratio Score: {scores['noise_ratio']:.3f}", VerboseLevel.DETAILED)

        num_clusters = iteration.num_clusters
        if 2 <= num_clusters <= 10:
            cluster_count_score = 1.0 - (abs(num_clusters - 5) / 5)
        else:
            cluster_count_score = 0.0
        scores['cluster_count'] = cluster_count_score * 1.5
        self._log(f"Cluster Count Score: {scores['cluster_count']:.3f}", VerboseLevel.DETAILED)

        iteration_score = np.log1p(iteration_idx) / np.log1p(total_iterations)
        scores['iteration_progress'] = iteration_score * 0.5
        self._log(f"Iteration Progress Score: {scores['iteration_progress']:.3f}", VerboseLevel.DETAILED)

        scores['total_score'] = sum(scores.values())
        self._log(f"Total Score: {scores['total_score']:.3f}", VerboseLevel.BASIC)
        
        return ClusteringScore(scores['total_score'], scores)

    def find_optimal_clustering(self, iterations: List[IterationResult]) -> OptimizationResult:
        """
        Determine the optimal clustering result from a series of iterations.

        The function evaluates each iteration using multiple criteria:
        1. Davies-Bouldin index score
        2. Noise point ratio
        3. Number of clusters
        4. Iteration progress
        
        It also considers the stability of these metrics over time using trend analysis.

        Parameters
        ----------
        iterations : List[IterationResult]
            List of clustering iterations to evaluate

        Returns
        -------
        OptimizationResult
            Named tuple containing:
            - best_iteration: The optimal clustering iteration
            - reason: Detailed explanation of the selection
            - all_scores: List of score breakdowns for all iterations

        Notes
        -----
        The function normalizes all metrics before combining them and considers
        both the absolute scores and the stability of the clustering solution
        when making the final selection.
        """
        self._log(f"\nStarting optimization analysis for {len(iterations)} iterations", VerboseLevel.BASIC)

        if len(iterations) < 2:
            self._log("Single iteration analysis mode", VerboseLevel.BASIC)
            current = iterations[-1]
            score = self.score_clustering(
                iteration=current,
                total_points=len(current.data),
                iteration_idx=0,
                total_iterations=len(iterations)
            )
            return OptimizationResult(
                current,
                f"Using only DB score ({score.breakdown['db_score']:.3f}) due to insufficient iterations",
                [score.breakdown]
            )

        total_points = len(iterations[0].data)
        self._log(f"Analyzing dataset with {total_points} points", VerboseLevel.DETAILED)

        raw_metrics = {
            'db_score': [],
            'noise_ratio': [],
            'cluster_count': [],
            'iteration_progress': []
        }

        self._log("Collecting raw metrics", VerboseLevel.DETAILED)
        for i, current in enumerate(iterations):
            raw_metrics['db_score'].append(max(0, 2.0 - current.db_score) / 2.0)
            raw_metrics['noise_ratio'].append(1.0 - abs((np.sum(current.noise_points) / total_points) - 0.1))
            num_clusters = current.num_clusters
            if 2 <= num_clusters <= 10:
                cluster_count_score = 1.0 - (abs(num_clusters - 5) / 5)
            else:
                cluster_count_score = 0.0
            raw_metrics['cluster_count'].append(cluster_count_score)
            raw_metrics['iteration_progress'].append(np.log1p(i) / np.log1p(len(iterations)))

        self._log("Calculating normalization ranges", VerboseLevel.DETAILED)
        
        normalized_ranges = {
            metric: (min(values), max(values)) 
            for metric, values in raw_metrics.items()
        }

        best_candidates = []
        all_scores = []

        self._log("Evaluating iterations with normalized scores", VerboseLevel.DETAILED)
        for i, current in enumerate(iterations):
            window_start = max(0, i - self.stability_window + 1)
            window_end = i + 1
            current_window = iterations[window_start:window_end]

            num_clusters = [iter.num_clusters for iter in current_window]
            db_scores = [iter.db_score for iter in current_window]

            clusters_trend, _ = self.analyze_trend(num_clusters)
            score_trend, _ = self.analyze_trend(db_scores)

            self._log(f"Iteration {i+1}: Cluster trend: {clusters_trend}, Score trend: {score_trend}", 
                     VerboseLevel.DETAILED)

            scores = self._calculate_normalized_scores(
                current, i, total_points, len(iterations), normalized_ranges
            )
            all_scores.append(scores)
            if clusters_trend in [Trend.STABLE, Trend.DECREASING] and \
               score_trend in [Trend.STABLE, Trend.DECREASING]: # A lower dbscore is better
                reason = "Selected because:\n" + \
                        "\n".join(f"- {k}: {v:.3f}" for k, v in scores.items())
                best_candidates.append((current, scores['total_score'], reason))

        if not best_candidates:
            self._log("No stable solutions found", VerboseLevel.BASIC)
            return OptimizationResult(
                iterations[-1], 
                "No stable solutions found",
                all_scores
            )

        best_candidates.sort(key=lambda x: x[1], reverse=True)
        self._log(f"Found optimal solution with score {best_candidates[0][1]:.3f}", VerboseLevel.BASIC)
        
        return OptimizationResult(
            best_candidates[0][0],
            best_candidates[0][2],
            all_scores
        )