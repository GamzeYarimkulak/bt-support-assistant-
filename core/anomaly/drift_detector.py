"""
Distribution drift detection for monitoring semantic shifts in tickets.
"""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import ks_2samp, wasserstein_distance
import structlog

logger = structlog.get_logger()


# ============================================================================
# PHASE 5: Drift Score Structure
# ============================================================================

@dataclass
class DriftScore:
    """
    Drift score for a time window comparing against baseline.
    
    Contains:
    - Time window boundaries
    - Volume z-score (spike detection)
    - Category divergence (distribution shift)
    - Embedding shift (semantic drift)
    - Combined score (weighted aggregate)
    """
    window_start: datetime
    window_end: datetime
    volume_zscore: float
    category_divergence: float
    embedding_shift: float
    combined_score: float


# ============================================================================
# PHASE 5: Simple Window-Based Drift Detector
# ============================================================================

class WindowDriftDetector:
    """
    Simple drift detector for time-windowed ticket data (PHASE 5).
    
    Compares each window against a baseline (reference windows) and computes:
    - Volume z-score (ticket count spikes)
    - Category divergence (distribution changes)
    - Embedding shift (semantic drift)
    """
    
    def __init__(self, min_reference_windows: int = 5):
        """
        Initialize drift detector.
        
        Args:
            min_reference_windows: Minimum number of reference windows needed
        """
        self.min_reference_windows = min_reference_windows
        self.baseline_volume_mean = None
        self.baseline_volume_std = None
        self.baseline_category_dist = None
        self.baseline_centroid = None
        
        logger.info("window_drift_detector_initialized",
                   min_reference_windows=min_reference_windows)
    
    def fit_reference(self, windows: List['WindowStats']) -> None:
        """
        Fit baseline statistics from reference windows.
        
        Computes:
        - Mean and std of ticket counts (for z-score)
        - Average category distribution
        - Average centroid embedding
        
        Args:
            windows: List of WindowStats to use as reference/baseline
            
        Raises:
            ValueError: If insufficient reference windows
        """
        if len(windows) < self.min_reference_windows:
            raise ValueError(
                f"Need at least {self.min_reference_windows} reference windows, got {len(windows)}"
            )
        
        logger.info("fitting_reference_baseline", num_windows=len(windows))
        
        # Volume statistics
        volumes = [w.total_tickets for w in windows]
        self.baseline_volume_mean = np.mean(volumes)
        self.baseline_volume_std = np.std(volumes)
        if self.baseline_volume_std == 0:
            self.baseline_volume_std = 1.0  # Avoid division by zero
        
        # Category distribution (average across windows)
        all_categories = set()
        for w in windows:
            all_categories.update(w.counts_by_category.keys())
        
        category_dists = []
        for w in windows:
            total = w.total_tickets
            dist = {cat: w.counts_by_category.get(cat, 0) / max(total, 1) 
                   for cat in all_categories}
            category_dists.append(dist)
        
        # Average category distribution
        self.baseline_category_dist = {}
        for cat in all_categories:
            probs = [d[cat] for d in category_dists]
            self.baseline_category_dist[cat] = np.mean(probs)
        
        # Centroid embedding (mean of all centroids)
        centroids = np.array([w.centroid_embedding for w in windows])
        self.baseline_centroid = np.mean(centroids, axis=0)
        
        logger.info("reference_baseline_fitted",
                   volume_mean=self.baseline_volume_mean,
                   volume_std=self.baseline_volume_std,
                   num_categories=len(self.baseline_category_dist))
    
    def score_window(self, window: 'WindowStats') -> DriftScore:
        """
        Compute drift score for a single window against the baseline.
        
        Args:
            window: WindowStats to score
            
        Returns:
            DriftScore with volume, category, and embedding drift metrics
            
        Raises:
            ValueError: If baseline not fitted
        """
        if self.baseline_volume_mean is None:
            raise ValueError("Must call fit_reference() before scoring")
        
        # 1. Volume z-score
        volume_zscore = (window.total_tickets - self.baseline_volume_mean) / self.baseline_volume_std
        
        # 2. Category divergence (Jensen-Shannon divergence)
        window_cat_dist = {}
        total = window.total_tickets
        for cat, count in window.counts_by_category.items():
            window_cat_dist[cat] = count / max(total, 1)
        
        # Ensure all categories are present
        all_cats = set(self.baseline_category_dist.keys()) | set(window_cat_dist.keys())
        
        baseline_probs = np.array([self.baseline_category_dist.get(cat, 0.0) for cat in all_cats])
        window_probs = np.array([window_cat_dist.get(cat, 0.0) for cat in all_cats])
        
        # Normalize
        baseline_probs = baseline_probs / (baseline_probs.sum() + 1e-8)
        window_probs = window_probs / (window_probs.sum() + 1e-8)
        
        # JS divergence
        epsilon = 1e-8
        m = (baseline_probs + window_probs) / 2
        js_div = 0.5 * np.sum(baseline_probs * np.log((baseline_probs + epsilon) / (m + epsilon))) + \
                 0.5 * np.sum(window_probs * np.log((window_probs + epsilon) / (m + epsilon)))
        
        category_divergence = float(js_div)
        
        # 3. Embedding shift (cosine distance)
        embedding_shift = float(cosine(self.baseline_centroid, window.centroid_embedding))
        
        # Handle NaN from cosine (if vectors are zero)
        if np.isnan(embedding_shift):
            embedding_shift = 0.0
        
        # 4. Combined score (simple weighted sum)
        # Normalize each component to roughly [0, 1] range
        volume_component = min(abs(volume_zscore) / 3.0, 1.0)  # z > 3 is very anomalous
        category_component = min(category_divergence / 0.5, 1.0)  # JS div > 0.5 is high
        embedding_component = min(embedding_shift / 0.3, 1.0)  # cosine dist > 0.3 is significant
        
        # Weighted combination (equal weights)
        combined_score = (volume_component + category_component + embedding_component) / 3.0
        
        drift_score = DriftScore(
            window_start=window.window_start,
            window_end=window.window_end,
            volume_zscore=float(volume_zscore),
            category_divergence=category_divergence,
            embedding_shift=embedding_shift,
            combined_score=combined_score
        )
        
        logger.debug("window_scored",
                    window_start=window.window_start,
                    volume_zscore=volume_zscore,
                    combined_score=combined_score)
        
        return drift_score


class DriftDetector:
    """
    Detects distribution drift in ticket embeddings and features.
    Useful for monitoring if ticket patterns change significantly over time.
    """
    
    def __init__(self, drift_threshold: float = 0.3):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Threshold for flagging significant drift
        """
        self.drift_threshold = drift_threshold
        logger.info("drift_detector_initialized", threshold=drift_threshold)
    
    def detect_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect drift in embedding distributions.
        
        Args:
            reference_embeddings: Baseline/reference embeddings (n_ref x dim)
            current_embeddings: Current period embeddings (n_curr x dim)
            
        Returns:
            Dictionary with drift metrics
        """
        if len(reference_embeddings) == 0 or len(current_embeddings) == 0:
            logger.warning("empty_embeddings_for_drift_detection")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "method": "none"
            }
        
        # Compute centroids
        ref_centroid = np.mean(reference_embeddings, axis=0)
        curr_centroid = np.mean(current_embeddings, axis=0)
        
        # Cosine distance between centroids
        centroid_cosine_dist = cosine(ref_centroid, curr_centroid)
        
        # Euclidean distance (normalized by embedding dimension)
        centroid_euclidean_dist = euclidean(ref_centroid, curr_centroid) / np.sqrt(len(ref_centroid))
        
        # Mean Maximum Discrepancy (simplified version)
        mmd = self._compute_mmd(reference_embeddings, current_embeddings)
        
        # Combine metrics
        drift_score = (centroid_cosine_dist + centroid_euclidean_dist + mmd) / 3
        
        drift_detected = drift_score > self.drift_threshold
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "centroid_cosine_distance": float(centroid_cosine_dist),
            "centroid_euclidean_distance": float(centroid_euclidean_dist),
            "mmd": float(mmd),
            "method": "embedding_drift",
            "n_reference": len(reference_embeddings),
            "n_current": len(current_embeddings)
        }
        
        logger.info("embedding_drift_detected" if drift_detected else "no_embedding_drift",
                   drift_score=drift_score,
                   threshold=self.drift_threshold)
        
        return result
    
    def detect_distribution_drift(
        self,
        reference_dist: Dict[str, float],
        current_dist: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect drift in categorical distributions (e.g., priority, category).
        
        Args:
            reference_dist: Reference distribution (category -> proportion)
            current_dist: Current distribution (category -> proportion)
            
        Returns:
            Dictionary with drift metrics
        """
        if not reference_dist or not current_dist:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "method": "distribution"
            }
        
        # Get all categories
        all_categories = set(reference_dist.keys()) | set(current_dist.keys())
        
        # Build probability vectors
        ref_probs = np.array([reference_dist.get(cat, 0.0) for cat in all_categories])
        curr_probs = np.array([current_dist.get(cat, 0.0) for cat in all_categories])
        
        # Normalize (in case they don't sum to 1)
        ref_probs = ref_probs / (ref_probs.sum() + 1e-8)
        curr_probs = curr_probs / (curr_probs.sum() + 1e-8)
        
        # KL divergence (with smoothing to avoid log(0))
        epsilon = 1e-8
        kl_div = np.sum(curr_probs * np.log((curr_probs + epsilon) / (ref_probs + epsilon)))
        
        # Jensen-Shannon divergence (symmetric)
        m = (ref_probs + curr_probs) / 2
        js_div = 0.5 * np.sum(ref_probs * np.log((ref_probs + epsilon) / (m + epsilon))) + \
                 0.5 * np.sum(curr_probs * np.log((curr_probs + epsilon) / (m + epsilon)))
        
        # Chi-square distance
        chi_square = np.sum((ref_probs - curr_probs) ** 2 / (ref_probs + curr_probs + epsilon))
        
        # Use JS divergence as main score (bounded and symmetric)
        drift_score = float(js_div)
        drift_detected = drift_score > self.drift_threshold
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "kl_divergence": float(kl_div),
            "js_divergence": float(js_div),
            "chi_square_distance": float(chi_square),
            "method": "categorical_distribution",
            "n_categories": len(all_categories)
        }
        
        logger.info("distribution_drift_detected" if drift_detected else "no_distribution_drift",
                   drift_score=drift_score,
                   threshold=self.drift_threshold)
        
        return result
    
    def detect_count_drift(
        self,
        reference_counts: np.ndarray,
        current_counts: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect drift in count/volume patterns using time series.
        
        Args:
            reference_counts: Historical count time series
            current_counts: Current period count time series
            
        Returns:
            Dictionary with drift metrics
        """
        if len(reference_counts) == 0 or len(current_counts) == 0:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "method": "count"
            }
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = ks_2samp(reference_counts, current_counts)
        
        # Wasserstein distance (Earth Mover's Distance)
        wd = wasserstein_distance(reference_counts, current_counts)
        
        # Normalize wasserstein distance by reference mean
        ref_mean = np.mean(reference_counts)
        normalized_wd = wd / (ref_mean + 1e-8)
        
        # Mean and variance comparison
        ref_mean = np.mean(reference_counts)
        curr_mean = np.mean(current_counts)
        ref_std = np.std(reference_counts)
        curr_std = np.std(current_counts)
        
        mean_change_ratio = abs(curr_mean - ref_mean) / (ref_mean + 1e-8)
        std_change_ratio = abs(curr_std - ref_std) / (ref_std + 1e-8)
        
        # Combine metrics
        drift_score = (normalized_wd + mean_change_ratio + std_change_ratio) / 3
        drift_detected = drift_score > self.drift_threshold or ks_pvalue < 0.05
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": float(drift_score),
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "wasserstein_distance": float(wd),
            "normalized_wasserstein": float(normalized_wd),
            "mean_change_ratio": float(mean_change_ratio),
            "std_change_ratio": float(std_change_ratio),
            "method": "count_drift"
        }
        
        logger.info("count_drift_detected" if drift_detected else "no_count_drift",
                   drift_score=drift_score,
                   ks_pvalue=ks_pvalue)
        
        return result
    
    def _compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = "rbf",
        gamma: float = 1.0
    ) -> float:
        """
        Compute Maximum Mean Discrepancy between two samples.
        
        Args:
            X: First sample (n_samples_X x n_features)
            Y: Second sample (n_samples_Y x n_features)
            kernel: Kernel type ('rbf' or 'linear')
            gamma: Kernel bandwidth parameter
            
        Returns:
            MMD value
        """
        if kernel == "rbf":
            # RBF kernel MMD
            XX = self._rbf_kernel(X, X, gamma)
            YY = self._rbf_kernel(Y, Y, gamma)
            XY = self._rbf_kernel(X, Y, gamma)
            
            mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            # Linear kernel (simpler, faster)
            mean_X = np.mean(X, axis=0)
            mean_Y = np.mean(Y, axis=0)
            mmd = np.sum((mean_X - mean_Y) ** 2)
        
        return float(max(0, mmd))  # MMD should be non-negative
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """
        Compute RBF (Gaussian) kernel between X and Y.
        
        Args:
            X: First sample
            Y: Second sample
            gamma: Kernel bandwidth
            
        Returns:
            Kernel matrix
        """
        # Compute pairwise squared distances
        XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
        YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
        distances = XX + YY - 2 * np.dot(X, Y.T)
        
        # Apply RBF kernel
        K = np.exp(-gamma * distances)
        
        return K

