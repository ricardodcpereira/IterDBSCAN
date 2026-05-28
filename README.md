# IterDBSCAN

A novel clustering-based approach for identifying and mitigating small disjuncts in imbalanced datasets. IterDBSCAN extends the traditional DBSCAN algorithm to handle varying densities and eliminate bridging points.

## Installation

Using [uv](https://docs.astral.sh/uv/):

```bash
# Install uv (if not installed)
pip install uv

# Create virtual environment and install dependencies
uv venv
uv pip install -e .
```

Using pip:

```bash
pip install -r requirements.txt
```

## Overview

Small disjuncts are small, underrepresented clusters within a class (within-class imbalance) that pose significant challenges for machine learning algorithms, particularly under class imbalance conditions.

### Research Contributions

1. **IterDBSCAN Algorithm**: A novel clustering algorithm that extends DBSCAN to:
   - Handle varying densities within data
   - Eliminate bridging points (links between distinct sub-concepts)
   - Automatically optimize clustering through multi-criteria evaluation

2. **IterDBSCAN-SMOTE**: A cluster-aware oversampling method that:
   - Preserves the relative representation of identified clusters
   - Oversamples each small disjunct independently while maintaining structure
   - Balances classes without distorting the minority class's internal distribution

## Modules

### `src/iterative_dbscan/`

**`iter_dbscan.py`**: Core IterDBSCAN algorithm  
- Iterative refinement of DBSCAN with adaptive epsilon
- Growth-based filtering to remove bridging points
- Multi-iteration clustering with automatic optimization

**`iterdbscan_smote.py`**: IterDBSCAN-SMOTE oversampling  
- Cluster-aware SMOTE oversampling
- Preserves cluster size proportions after oversampling
- Handles noise points separately

### `src/optimal_cluster/`

**`optimal_cluster.py`**: Clustering optimization module  
- Multi-criteria scoring (Davies-Bouldin index, noise ratio, stability)
- Trend analysis for metric convergence
- Automatic selection of optimal clustering iteration

## Usage Examples

### Identifying Small Disjuncts with IterDBSCAN

```python
import numpy as np
from src.main import iter_dbscan

# Load minority class data
X_minority = np.random.rand(100, 5)  # Your minority class features

# Run IterDBSCAN to identify small disjuncts
_, results = iter_dbscan(
    X_minority, 
    min_samples=2,           # Minimum points to form a disjunct
    step_growth_ratio=20,    # Controls epsilon growth rate
    verbose='none'           # Options: 'none', 'basic', 'detailed'
)

if results:
    # Get cluster labels from the optimal iteration
    cluster_labels = results.best_iteration.labels
    num_clusters = results.best_iteration.num_clusters
    
    print(f"Found {num_clusters} clusters")
    print(f"Davies-Bouldin score: {results.best_iteration.db_score:.3f}")
    
    # Noise points are labeled as -1
    noise_mask = cluster_labels == -1
    print(f"Noise points: {noise_mask.sum()}")
```

### IterDBSCAN-SMOTE Oversampling

The oversampling strategy preserves the relative proportion between clusters while balancing the classes:

```python
import numpy as np
import pandas as pd
from src.main import iter_dbscan
from src.iterative_dbscan import iterdbscan_SMOTE

# Load your imbalanced dataset
data = pd.read_csv('your_dataset.csv')
X = data.drop('class', axis=1).values
y = data['class'].values

minority_class = 1
X_minority = X[y == minority_class]
X_majority = X[y != minority_class]
y_minority = y[y == minority_class]
y_majority = y[y != minority_class]

# Step 1: Identify small disjuncts in minority class
_, results = iter_dbscan(X_minority, min_samples=2, step_growth_ratio=20, verbose='none')

if results:
    cluster_labels = results.best_iteration.labels
    
    # Step 2: Prepare dataset with cluster labels
    # Majority class gets the highest cluster number
    num_minority_clusters = len(np.unique(cluster_labels[cluster_labels != -1]))
    majority_cluster_label = num_minority_clusters
    
    # Combine data
    X_combined = np.vstack([X_minority, X_majority])
    y_clusters = np.hstack([
        cluster_labels, 
        np.full(len(X_majority), majority_cluster_label)
    ])
    y_original = np.hstack([y_minority, y_majority])
    
    # Step 3: Apply IterDBSCAN-SMOTE
    # Distributes oversampling budget equally among all clusters
    X_resampled, y_resampled = iterdbscan_SMOTE(
        X=X_combined,
        y_iterdbscan=y_clusters,
        y=y_original,
        smote_method='SMOTE'  # Choose SMOTE variant
    )
    
    print(f"Original dataset: {len(X)} samples")
    print(f"Resampled dataset: {len(X_resampled)} samples")
    print(f"Class distribution: {np.bincount(y_resampled.astype(int))}")
```

### Available SMOTE Methods

```python
from src.iterative_dbscan import SMOTE_METHODS

# Available SMOTE variants:
print(list(SMOTE_METHODS.keys()))
# ['SMOTE', 'SMOTE_TomekLinks', 'MWMOTE', 'DBSMOTE', 'Cluster_SMOTE']
```

## API Reference

### `iter_dbscan(data, min_samples, step_growth_ratio, verbose)`

Performs iterative DBSCAN clustering with growth-based bridge point removal.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `np.ndarray` | Feature matrix of minority class samples |
| `min_samples` | `int` | Minimum samples for core points (typically 2 or 3) |
| `step_growth_ratio` | `int` | Controls epsilon growth rate between iterations |
| `verbose` | `str` | Verbosity level: `'none'`, `'basic'`, or `'detailed'` |

**Returns:** 
- `clusterer`: adaptiveDBSCAN object
- `results`: Contains `best_iteration` with:
  - `labels`: Cluster assignments (-1 for noise)
  - `num_clusters`: Number of identified clusters
  - `db_score`: Davies-Bouldin score
  - `epsilon`: Epsilon value used

---

### `iterdbscan_SMOTE(X, y_iterdbscan, y, smote_method)`

Applies cluster-aware SMOTE oversampling while preserving cluster proportions.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray` | Feature matrix (minority + majority classes) |
| `y_iterdbscan` | `np.ndarray` | Cluster labels from IterDBSCAN (-1=noise, 0 to n-1=clusters) |
| `y` | `np.ndarray` or `pd.Series` | Original class labels |
| `smote_method` | `str` | SMOTE variant: `'SMOTE'`, `'SMOTE_TomekLinks'`, `'MWMOTE'`, `'DBSMOTE'`, `'Cluster_SMOTE'` |

**Returns:** 
- `X_resampled`: Resampled feature matrix
- `y_resampled`: Resampled class labels

**How it works:**
1. Separates noise points (labeled -1)
2. Computes oversampling budget: Δ = N_maj - N_min
3. Distributes budget equally among all minority clusters: Δ_c = Δ/(k+1)
4. Applies SMOTE to each cluster independently
5. Reintegrates noise points into final dataset

---

## Key Concepts

### Small Disjuncts
Small disjuncts are small clusters within a class (within-class imbalance). They represent rare sub-concepts that are challenging for classifiers, especially under class imbalance. Traditional approaches treat the entire minority class uniformly, but small disjuncts require specialized handling.

### Bridging Points
Points in low-density regions that create spurious connections between distinct sub-concepts. IterDBSCAN identifies and removes these points using growth-based analysis: points whose neighborhood growth between iterations falls below a threshold are excluded from clustering.

### Cluster Preservation
Unlike standard oversampling methods that inflate all clusters equally, IterDBSCAN-SMOTE maintains the relative size proportions between clusters after resampling.

---

## Experimental Results

Based on empirical evaluation across 78 real-world imbalanced datasets from the KEEL repository, IterDBSCAN-SMOTE demonstrated:

- **Best F1-score** (0.567) among all tested methods
- **Highest Disjunct Score** (0.685), indicating superior handling of small disjuncts
- **Lower Error Coverage** (12.4% vs 19.5% baseline), showing fewer errors originate from small disjuncts
- **Balanced precision-recall trade-off** (precision: 0.548, recall: 0.762)

Tested against state-of-the-art methods: SMOTE, SMOTE+TomekLinks, MWMOTE, DBSMOTE, Cluster-SMOTE, and CBO across 7 classifier families.
