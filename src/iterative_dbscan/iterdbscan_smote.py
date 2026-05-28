import numpy as np
from collections import Counter
import smote_variants


SMOTE_METHODS = {
    'SMOTE': smote_variants.SMOTE,
    'SMOTE_TomekLinks': smote_variants.SMOTE_TomekLinks,
    'MWMOTE': smote_variants.MWMOTE,
    'DBSMOTE': smote_variants.DBSMOTE,
    'Cluster_SMOTE': smote_variants.cluster_SMOTE
}


def iterdbscan_SMOTE(X, y_iterdbscan, y, smote_method='SMOTE'):
    """
    Apply SMOTE oversampling to each cluster identified by iterative DBSCAN.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features)
    y_iterdbscan : numpy.ndarray
        Cluster labels from iterDBSCAN (-1 for noise, 0 to n-1 for clusters)
    y : numpy.ndarray or pandas.Series
        Original class labels
    smote_method : str
        Name of the SMOTE method to use. Options: 'SMOTE', 
        'SMOTE_TomekLinks', 'MWMOTE', 'DBSMOTE', 'Cluster_SMOTE'
        
    Returns:
    --------
    final_X : numpy.ndarray
        Resampled feature matrix
    final_y : numpy.ndarray
        Resampled class labels
    """
    if smote_method not in SMOTE_METHODS:
        raise ValueError(f"Unknown SMOTE method: {smote_method}. Available methods: {list(SMOTE_METHODS.keys())}")
    
    smote = SMOTE_METHODS[smote_method]()
    classes = np.unique(y_iterdbscan)
    nr_classes = len(classes)
    print(f"Starting iterdbscan_SMOTE with {len(X)} samples and {nr_classes} classes: {classes}")

    noise_mask = y_iterdbscan == -1
    noise_X = X[noise_mask] if np.any(noise_mask) else None
    noise_y = y[noise_mask] if np.any(noise_mask) else None
    noise_count = len(noise_X) if noise_X is not None else 0
    print(f"Noise samples found: {noise_count}")

    non_noise_mask = y_iterdbscan != -1
    X = X[non_noise_mask].copy()
    y_iterdbscan = y_iterdbscan[non_noise_mask].copy()
    print(f"Non-noise samples: {len(X)}")

    classes_non_noise = np.unique(y_iterdbscan)
    nr_classes = len(classes_non_noise)

    maj_mask = y_iterdbscan == nr_classes - 1
    X_maj = X[maj_mask]
    y_maj = y_iterdbscan[maj_mask]
    y_maj_original = y[maj_mask]
    maj_size = len(y_maj)
    print(f"Majority class size: {maj_size}")

    final_X = X_maj.copy()
    final_y = y_maj_original.copy()
    
    all_min_X = []
    all_min_y = []

    min_size = len(X[y_iterdbscan != nr_classes - 1])
    oversampling_size = int((maj_size - min_size)/(nr_classes - 1))
    print(f"Oversampling: {oversampling_size}")
    for cluster in range(nr_classes - 1):
        cluster_mask = y_iterdbscan == cluster
        cluster_X = X[cluster_mask]
        cluster_y = y_iterdbscan[cluster_mask]
        cluster_y_original = y[cluster_mask]
        original_class_value = cluster_y_original.iloc[0]

        original_cluster_size = len(cluster_X)

        maj_indices = np.random.choice(len(X_maj), oversampling_size + original_cluster_size, replace=False)
        X_maj_dummy = X_maj[maj_indices]
        y_maj_dummy = y_maj[maj_indices]

        print(f"Processing cluster {cluster} with original size: {original_cluster_size}")

        try:
            minmax_X = np.vstack((cluster_X, X_maj_dummy))
            minmax_y = np.hstack((cluster_y, y_maj_dummy))
            print(f"  Combined data for cluster {cluster}: {len(minmax_X)} samples")

            X_resampled, y_resampled = smote.sample(minmax_X, minmax_y)
            min_mask = y_resampled == cluster
            min_X_res = X_resampled[min_mask]
            min_y_res = y_resampled[min_mask]
            min_y_res = np.full(len(min_X_res), original_class_value)
            
            resampled_cluster_size = len(min_X_res)
            print(f"  Cluster {cluster} after SMOTE: {original_cluster_size} -> {resampled_cluster_size}")
            
            all_min_X.append(min_X_res)
            all_min_y.extend(min_y_res)
        except Exception as e:
            print(f"  ERROR: Iterdbscan-SMOTE failed for cluster {cluster}: {str(e)}")
            continue

    if all_min_X:
        all_min_X = np.vstack(all_min_X)
        all_min_y = np.array(all_min_y)
        print(f"Total minority samples after SMOTE: {len(all_min_X)}")
        
        final_X = np.vstack((final_X, all_min_X))
        final_y = np.hstack((final_y, all_min_y))

    if noise_X is not None and len(noise_X) > 0:
        final_X = np.vstack((final_X, noise_X))
        final_y = np.hstack((final_y, noise_y))
        print(f"Added {noise_count} noise samples back to final dataset")

    final_size = len(final_X)
    print(f"Final dataset size: {final_size}")
    for class_label, count in sorted(Counter(final_y).items()):
        percentage = (count / len(final_y)) * 100
        print(f"  Class {class_label}: {count} samples ({percentage:.1f}%)")

    return final_X, final_y
