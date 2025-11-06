#!/usr/bin/env python3
"""
Generate centroids for MyRocks LSM vector index using FAISS k-means clustering.

This script uses FAISS k-means to generate centroids efficiently, with support for
GPU acceleration and memory-efficient batch processing.

Usage:
    python generate_centroids_balanced.py --dataset-path /path/to/train.parquet \
                                           --num-centroids 256 \
                                           --output centroids.csv
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
import faiss


def load_vectors_from_parquet(parquet_path: Path, max_samples: int | None = None) -> np.ndarray:
    """Load vectors from parquet file.

    Args:
        parquet_path: Path to the parquet file containing vectors
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        numpy array of shape (n_samples, dimension)
    """
    print(f"Loading vectors from {parquet_path}...")

    # Read parquet file using polars
    df = pl.read_parquet(parquet_path)

    # Get the vector column (usually named 'emb' or 'vector')
    vector_column = None
    for col in df.columns:
        if col in ['emb', 'vector', 'embedding']:
            vector_column = col
            break

    if vector_column is None:
        # Try to find a column with list type
        for col in df.columns:
            if df[col].dtype == pl.List:
                vector_column = col
                break

    if vector_column is None:
        raise ValueError(f"Could not find vector column in {parquet_path}. Columns: {df.columns}")

    print(f"Found vector column: '{vector_column}'")

    # Limit samples if requested
    if max_samples is not None and len(df) > max_samples:
        print(f"Sampling {max_samples} vectors from {len(df)} total vectors")
        df = df.sample(n=max_samples, seed=42)

    # Convert to numpy array
    vectors = np.array(df[vector_column].to_list())

    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")
    return vectors


def balanced_kmeans_iterative(
    vectors: np.ndarray,
    num_centroids: int,
    max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    tolerance: float = 0.2,
    max_rebalance_ratio: float = 0.3,
    convergence_threshold: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform balanced k-means clustering with iterative rebalancing.

    This algorithm uses an iterative rebalancing approach:
    1. Assign points to nearest centroids (standard k-means)
    2. Identify clusters with size disparities
    3. Reassign points from oversized to undersized clusters
    4. Update centroids and repeat

    Args:
        vectors: Input vectors of shape (n_samples, dimension)
        num_centroids: Number of centroids to generate
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        verbose: Print progress
        tolerance: Allowed deviation from ideal cluster size (default 0.2 = ±20%)
        max_rebalance_ratio: Maximum fraction of cluster to rebalance per iteration
        convergence_threshold: Threshold for centroid movement to consider converged

    Returns:
        Tuple of (centroids, assignments)
    """
    np.random.seed(random_state)
    n_samples, n_features = vectors.shape

    # Ensure vectors are float32
    vectors = vectors.astype(np.float32)

    # Calculate target cluster size and bounds
    target_size = n_samples / num_centroids
    min_size = int(target_size * (1 - tolerance))
    max_size = int(target_size * (1 + tolerance))

    if verbose:
        print(f"\n=== Balanced K-Means with Iterative Rebalancing ===")
        print(f"Samples: {n_samples}, Centroids: {num_centroids}")
        print(f"Target cluster size: {target_size:.1f}")
        print(f"Allowed range: [{min_size}, {max_size}] (±{tolerance*100:.0f}%)")
        print(f"Max rebalance per iteration: {max_rebalance_ratio*100:.0f}%")

    # Initialize centroids using k-means++ (via FAISS)
    if verbose:
        print(f"\nInitializing centroids with k-means++...")

    # Use a few iterations of standard k-means for initialization
    # Disable FAISS automatic sampling by setting max_points_per_centroid very high
    kmeans_init = faiss.Kmeans(
        d=n_features,
        k=num_centroids,
        niter=10,
        verbose=False,
        seed=random_state,
        gpu=False,
        max_points_per_centroid=10000000  # Disable automatic sampling
    )
    kmeans_init.train(vectors)
    centroids = kmeans_init.centroids.copy()

    if verbose:
        print(f"  Used all {n_samples} vectors for initialization (FAISS sampling disabled)")

    if verbose:
        print(f"Starting iterative rebalancing (max {max_iter} iterations)...")

    # Main iteration loop
    for iteration in range(max_iter):
        iteration_start = time.time()

        # Step 1: Assign each point to nearest centroid
        index = faiss.IndexFlatL2(n_features)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)
        assignments = assignments.flatten()
        distances = distances.flatten()

        # Calculate cluster sizes
        cluster_sizes = np.bincount(assignments, minlength=num_centroids)

        # Step 2: Identify oversized and undersized clusters
        oversized = np.where(cluster_sizes > max_size)[0]
        undersized = np.where(cluster_sizes < min_size)[0]

        num_rebalanced = 0

        # Step 3: Rebalance if needed
        if len(oversized) > 0 and len(undersized) > 0:
            # Compute full distance matrix for rebalancing
            # (only for points in oversized clusters to save memory)
            oversized_mask = np.isin(assignments, oversized)
            oversized_indices = np.where(oversized_mask)[0]

            if len(oversized_indices) > 0:
                # Get distances to all centroids for oversized points
                all_distances, all_nearest = index.search(vectors[oversized_indices], num_centroids)

                # Create a rebalancing plan
                rebalance_plan = []

                for cluster_id in oversized:
                    # How many points to remove from this cluster
                    excess = cluster_sizes[cluster_id] - max_size
                    max_remove = int(cluster_sizes[cluster_id] * max_rebalance_ratio)
                    num_to_remove = min(excess, max_remove)

                    if num_to_remove <= 0:
                        continue

                    # Find points in this cluster
                    cluster_mask = (assignments == cluster_id)
                    cluster_points = np.where(cluster_mask)[0]

                    # Find points furthest from own centroid (candidates for reassignment)
                    point_distances = distances[cluster_points]
                    furthest_indices = np.argsort(point_distances)[-num_to_remove:]
                    candidates = cluster_points[furthest_indices]

                    # For each candidate, find best alternative cluster
                    for point_idx in candidates:
                        # Find this point in the oversized_indices array
                        oversized_pos = np.where(oversized_indices == point_idx)[0]
                        if len(oversized_pos) == 0:
                            continue
                        oversized_pos = oversized_pos[0]

                        # Get distances to all centroids for this point
                        point_all_dist = all_distances[oversized_pos]
                        point_all_nearest = all_nearest[oversized_pos]

                        # Find best alternative cluster (prefer undersized)
                        for k in range(1, num_centroids):  # Skip first (current cluster)
                            alternative_cluster = point_all_nearest[k]

                            # Prefer undersized clusters
                            if alternative_cluster in undersized:
                                # Check if alternative distance is reasonable
                                # (within 2x of current distance to avoid breaking quality)
                                if point_all_dist[k] < point_all_dist[0] * 2.0:
                                    rebalance_plan.append((point_idx, alternative_cluster))
                                    break

                # Apply rebalancing plan
                for point_idx, new_cluster in rebalance_plan:
                    old_cluster = assignments[point_idx]
                    assignments[point_idx] = new_cluster
                    cluster_sizes[old_cluster] -= 1
                    cluster_sizes[new_cluster] += 1
                    num_rebalanced += 1

        # Step 4: Update centroids
        centroids_old = centroids.copy()
        for k in range(num_centroids):
            cluster_points = vectors[assignments == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)
            # If cluster is empty, keep old centroid (will be reassigned next iteration)

        # Check convergence
        centroid_shift = np.max(np.sqrt(np.sum((centroids - centroids_old) ** 2, axis=1)))

        # Calculate balance metrics
        imbalance = (cluster_sizes.max() - cluster_sizes.min()) / target_size * 100

        iteration_time = time.time() - iteration_start

        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iter {iteration:3d}: shift={centroid_shift:.6f}, "
                  f"imbalance={imbalance:.1f}%, rebalanced={num_rebalanced:5d}, "
                  f"time={iteration_time:.2f}s")
            print(f"         sizes: min={cluster_sizes.min():4d}, "
                  f"max={cluster_sizes.max():5d}, mean={cluster_sizes.mean():.1f}")

        # Check convergence
        if centroid_shift < convergence_threshold and len(oversized) == 0 and len(undersized) == 0:
            if verbose:
                print(f"\n✓ Converged at iteration {iteration}")
            break

    if verbose:
        print(f"\n=== Final Balance Statistics ===")
        print(f"  Min size: {cluster_sizes.min()}")
        print(f"  Max size: {cluster_sizes.max()}")
        print(f"  Mean size: {cluster_sizes.mean():.1f}")
        print(f"  Std dev: {cluster_sizes.std():.1f}")
        print(f"  Imbalance: {imbalance:.1f}%")
        print(f"  Clusters in target range [{min_size}, {max_size}]: "
              f"{np.sum((cluster_sizes >= min_size) & (cluster_sizes <= max_size))}/{num_centroids}")

    return centroids, assignments


def balanced_kmeans_with_verification(
    vectors: np.ndarray,
    num_centroids: int,
    max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    training_sample_size: int = None,
    tolerance: float = 0.2,
    max_rebalance_ratio: float = 0.3
) -> np.ndarray:
    """Wrapper for balanced k-means with training sampling and full dataset verification.

    Args:
        vectors: Input vectors of shape (n_samples, dimension)
        num_centroids: Number of centroids to generate
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        verbose: Print progress
        training_sample_size: Number of samples to use for training (None = use all)
        tolerance: Allowed deviation from ideal cluster size
        max_rebalance_ratio: Maximum fraction to rebalance per iteration

    Returns:
        Centroids array of shape (num_centroids, dimension)
    """
    np.random.seed(random_state)
    n_samples, n_features = vectors.shape

    # Determine training sample
    if training_sample_size is not None and training_sample_size < n_samples:
        if verbose:
            print(f"\nUsing {training_sample_size}/{n_samples} vectors for training "
                  f"({training_sample_size/n_samples*100:.1f}%)")
        training_indices = np.random.choice(n_samples, training_sample_size, replace=False)
        training_vectors = vectors[training_indices]
    else:
        training_vectors = vectors

    n_training = len(training_vectors)

    if verbose:
        print(f"\nGenerating {num_centroids} centroids using Balanced K-Means...")
        print(f"Training vectors: {n_training}")
        print(f"Dimension: {n_features}")

    start_time = time.time()

    # Train balanced k-means on training data
    centroids, training_assignments = balanced_kmeans_iterative(
        training_vectors,
        num_centroids=num_centroids,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
        tolerance=tolerance,
        max_rebalance_ratio=max_rebalance_ratio
    )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n✓ Balanced k-means completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

        # Always verify on FULL dataset
        print(f"\n=== Verifying on FULL Dataset ({n_samples} vectors) ===")
        print(f"  Computing cluster assignments for all vectors...")

        # Assign all vectors to nearest centroid
        index = faiss.IndexFlatL2(n_features)
        index.add(centroids.astype(np.float32))

        batch_size_verify = 50000
        labels_full = np.zeros(n_samples, dtype=np.int32)

        vectors_float32 = vectors.astype(np.float32)
        for i in range(0, n_samples, batch_size_verify):
            end_i = min(i + batch_size_verify, n_samples)
            _, labels_batch = index.search(vectors_float32[i:end_i], 1)
            labels_full[i:end_i] = labels_batch.flatten()
            if verbose and (i + batch_size_verify) % (batch_size_verify * 5) == 0:
                print(f"    Processed {i + batch_size_verify}/{n_samples} vectors...")

        cluster_counts_full = np.bincount(labels_full, minlength=num_centroids)

        print(f"\n  Cluster size statistics:")
        print(f"  Min size: {cluster_counts_full.min()}")
        print(f"  Max size: {cluster_counts_full.max()}")
        print(f"  Mean size: {cluster_counts_full.mean():.1f}")
        print(f"  Std dev: {cluster_counts_full.std():.1f}")
        print(f"  Target size: {n_samples // num_centroids}")

        ideal_size_full = n_samples / num_centroids
        imbalance_full = (cluster_counts_full.max() - cluster_counts_full.min()) / ideal_size_full * 100
        print(f"  Imbalance: {imbalance_full:.1f}% (lower is better)")

        # Show distribution
        print(f"\n  Cluster size distribution:")
        target_range_min = int(ideal_size_full * (1 - tolerance))
        target_range_max = int(ideal_size_full * (1 + tolerance))
        bins = [0, int(ideal_size_full*0.8), int(ideal_size_full*0.9),
                int(ideal_size_full*1.1), int(ideal_size_full*1.2), n_samples]
        bin_labels = ["<80%", "80-90%", "90-110%", "110-120%", ">120%"]
        hist, _ = np.histogram(cluster_counts_full, bins=bins)
        for label, count in zip(bin_labels, hist):
            if count > 0:
                print(f"    {label} of target: {count} clusters")

        # Clusters within tolerance
        in_range = np.sum((cluster_counts_full >= target_range_min) &
                         (cluster_counts_full <= target_range_max))
        print(f"\n  Clusters within tolerance (±{tolerance*100:.0f}%): {in_range}/{num_centroids}")

        # Quality assessment
        print(f"\n  Quality Assessment:")
        if imbalance_full < tolerance * 100:
            print(f"    ✓ EXCELLENT balance ({imbalance_full:.1f}% imbalance)")
        elif imbalance_full < 50:
            print(f"    ✓ GOOD balance ({imbalance_full:.1f}% imbalance)")
        elif imbalance_full < 100:
            print(f"    ⚠ MODERATE balance ({imbalance_full:.1f}% imbalance)")
        else:
            print(f"    ✗ POOR balance ({imbalance_full:.1f}% imbalance)")

    return centroids


def faiss_kmeans(
    vectors: np.ndarray,
    num_centroids: int,
    max_iter: int = 300,
    random_state: int = 42,
    verbose: bool = True,
    init_sample_size: int = 50000,
    training_sample_size: int = None,
    batch_size: int = 1000,
    use_gpu: bool = True
) -> np.ndarray:
    """Perform k-means clustering using FAISS library.

    Uses GPU acceleration if available for fast, memory-efficient clustering.

    Args:
        vectors: Input vectors of shape (n_samples, dimension)
        num_centroids: Number of centroids to generate
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
        verbose: Print progress
        init_sample_size: Not used (kept for compatibility)
        training_sample_size: Number of samples to use (None = use all)
        batch_size: Not used (kept for compatibility)
        use_gpu: Use GPU acceleration if available

    Returns:
        Centroids array of shape (num_centroids, dimension)
    """
    np.random.seed(random_state)
    n_samples, n_features = vectors.shape

    # Determine training sample
    if training_sample_size is not None and training_sample_size < n_samples:
        if verbose:
            print(f"Using {training_sample_size}/{n_samples} vectors for training ({training_sample_size/n_samples*100:.1f}%)")
        training_indices = np.random.choice(n_samples, training_sample_size, replace=False)
        training_vectors = vectors[training_indices]
    else:
        training_vectors = vectors

    n_training = len(training_vectors)

    # Ensure vectors are float32 (required by FAISS)
    training_vectors = training_vectors.astype(np.float32)

    if verbose:
        print(f"\nGenerating {num_centroids} centroids using FAISS K-Means...")
        print(f"Training vectors: {n_training}")
        print(f"Dimension: {n_features}")
        print(f"Target cluster size: {n_training // num_centroids}")

    # Check GPU availability
    gpu_available = faiss.get_num_gpus() > 0 and use_gpu
    if verbose:
        if gpu_available:
            print(f"Device: GPU (found {faiss.get_num_gpus()} GPU(s))")
        else:
            print(f"Device: CPU")

    start_time = time.time()

    # Create FAISS k-means object
    kmeans = faiss.Kmeans(
        d=n_features,
        k=num_centroids,
        niter=max_iter,
        verbose=verbose,
        seed=random_state,
        gpu=gpu_available
    )

    if verbose:
        print(f"Running FAISS k-means...")

    # Train k-means
    kmeans.train(training_vectors)

    # Get centroids
    centroids = kmeans.centroids

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n✓ K-means clustering completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

        # Compute cluster statistics on training data
        print(f"  Computing cluster assignments for training data...")
        _, assignments_train = kmeans.index.search(training_vectors, 1)
        assignments_train = assignments_train.flatten()

        cluster_counts = np.bincount(assignments_train, minlength=num_centroids)

        print(f"\n=== Cluster Statistics (Training Data: {n_training} vectors) ===")
        print(f"  Min size: {cluster_counts.min()}")
        print(f"  Max size: {cluster_counts.max()}")
        print(f"  Mean size: {cluster_counts.mean():.1f}")
        print(f"  Std dev: {cluster_counts.std():.1f}")
        print(f"  Target size: {n_training // num_centroids}")

        ideal_size = n_training / num_centroids
        imbalance = (cluster_counts.max() - cluster_counts.min()) / ideal_size * 100
        print(f"  Imbalance: {imbalance:.1f}% (lower is better)")

        # Always verify on FULL dataset (even if no sampling was used)
        print(f"\n=== Verifying on FULL Dataset ({n_samples} vectors) ===")
        print(f"  Computing cluster assignments for all vectors...")

        # Assign all vectors to nearest centroid (in batches to save memory)
        batch_size_verify = 50000
        labels_full = np.zeros(n_samples, dtype=np.int32)

        vectors_float32 = vectors.astype(np.float32)
        for i in range(0, n_samples, batch_size_verify):
            end_i = min(i + batch_size_verify, n_samples)
            _, labels_batch = kmeans.index.search(vectors_float32[i:end_i], 1)
            labels_full[i:end_i] = labels_batch.flatten()
            if verbose and (i + batch_size_verify) % (batch_size_verify * 5) == 0:
                print(f"    Processed {i + batch_size_verify}/{n_samples} vectors...")

        cluster_counts_full = np.bincount(labels_full, minlength=num_centroids)

        print(f"\n  Cluster size statistics:")
        print(f"  Min size: {cluster_counts_full.min()}")
        print(f"  Max size: {cluster_counts_full.max()}")
        print(f"  Mean size: {cluster_counts_full.mean():.1f}")
        print(f"  Std dev: {cluster_counts_full.std():.1f}")
        print(f"  Target size: {n_samples // num_centroids}")

        ideal_size_full = n_samples / num_centroids
        imbalance_full = (cluster_counts_full.max() - cluster_counts_full.min()) / ideal_size_full * 100
        print(f"  Imbalance: {imbalance_full:.1f}% (lower is better)")

        # Show distribution
        print(f"\n  Cluster size distribution:")
        bins = [0, int(ideal_size_full*0.8), int(ideal_size_full*0.9),
                int(ideal_size_full*1.1), int(ideal_size_full*1.2), n_samples]
        bin_labels = ["<80%", "80-90%", "90-110%", "110-120%", ">120%"]
        hist, _ = np.histogram(cluster_counts_full, bins=bins)
        for label, count in zip(bin_labels, hist):
            if count > 0:
                print(f"    {label} of target: {count} clusters")

        # Quality assessment
        print(f"\n  Quality Assessment:")
        if imbalance_full < 20:
            print(f"    ✓ EXCELLENT balance ({imbalance_full:.1f}% imbalance)")
        elif imbalance_full < 50:
            print(f"    ✓ GOOD balance ({imbalance_full:.1f}% imbalance)")
        elif imbalance_full < 100:
            print(f"    ⚠ MODERATE balance ({imbalance_full:.1f}% imbalance)")
        else:
            print(f"    ✗ POOR balance ({imbalance_full:.1f}% imbalance)")

    return centroids


def save_centroids_to_csv(centroids: np.ndarray, output_path: Path, rocksdb_format: bool = True) -> None:
    """Save centroids to CSV file in the format expected by MyRocks.

    Args:
        centroids: Centroids array of shape (num_centroids, dimension)
        output_path: Path to output CSV file
        rocksdb_format: If True, use RocksDB C++ format (no header, quoted arrays only).
    """
    print(f"\nSaving {len(centroids)} centroids to {output_path}...")

    with output_path.open('w', newline='') as f:
        if rocksdb_format:
            # RocksDB C++ format: one quoted JSON array per line, no header, no ID
            for centroid in centroids:
                centroid_json = json.dumps(centroid.tolist())
                f.write(f'"{centroid_json}"\n')
        else:
            # SQL format: CSV with header and ID column
            import csv
            writer = csv.writer(f)
            writer.writerow(['id', 'centroid'])
            for i, centroid in enumerate(centroids):
                centroid_json = json.dumps(centroid.tolist())
                writer.writerow([i, centroid_json])

    print(f"✓ Successfully saved centroids to {output_path}")

    # Print file size
    file_size = output_path.stat().st_size
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Generate balanced centroids for MyRocks LSM vector index')
    parser.add_argument(
        '--dataset-path',
        type=Path,
        required=True,
        help='Path to the training dataset parquet file'
    )
    parser.add_argument(
        '--num-centroids',
        type=int,
        default=256,
        help='Number of centroids to generate (default: 256)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to use from dataset (None = use all)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=100,
        help='Maximum number of k-means iterations (default: 100)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--rocksdb-format',
        action='store_true',
        default=True,
        help='Output in RocksDB C++ format (default: True)'
    )
    parser.add_argument(
        '--init-sample-size',
        type=int,
        default=50000,
        help='Number of samples for k-means++ initialization (default: 50000)'
    )
    parser.add_argument(
        '--training-sample-size',
        type=int,
        default=None,
        help='Number of samples for training iterations (None = use all, recommended: 200000-300000 for speed)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='MiniBatch size for k-means (default: 1000)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MyRocks Balanced K-Means Centroid Generation")
    print("(Iterative Rebalancing Algorithm)")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output}")
    print(f"Number of centroids: {args.num_centroids}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all (1M vectors)'}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Random state: {args.random_state}")
    print("=" * 70)

    # Check if dataset exists
    if not args.dataset_path.exists():
        print(f"\n✗ Error: Dataset file not found: {args.dataset_path}")
        return 1

    try:
        # Load vectors
        vectors = load_vectors_from_parquet(args.dataset_path, args.max_samples)

        print(f"\nStarting Balanced K-Means clustering...")
        start_time = time.time()

        # Generate centroids using balanced k-means with iterative rebalancing
        centroids = balanced_kmeans_with_verification(
            vectors,
            num_centroids=args.num_centroids,
            max_iter=args.max_iter,
            random_state=args.random_state,
            verbose=True,
            training_sample_size=args.training_sample_size,
            tolerance=0.2,  # Allow ±20% deviation from ideal cluster size
            max_rebalance_ratio=0.3  # Rebalance up to 30% of cluster per iteration
        )

        total_time = time.time() - start_time
        print(f"\nTotal clustering time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

        # Create output directory if it doesn't exist
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        save_centroids_to_csv(centroids, args.output, rocksdb_format=args.rocksdb_format)

        print("\n" + "=" * 70)
        print("✓ Balanced centroid generation completed successfully!")
        print("=" * 70)
        print(f"\nCentroids saved to: {args.output}")
        print(f"Total time: {total_time / 60:.2f} minutes")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
