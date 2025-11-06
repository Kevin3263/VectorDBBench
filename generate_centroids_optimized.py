#!/usr/bin/env python3
"""
Optimized centroid generation for MyRocks LSM vector index.

This script implements multiple balanced k-means algorithms optimized for:
1. Minimal reconstruction loss (quantization error)
2. Balanced cluster sizes
3. Fast convergence

Algorithms available:
- optimal_transport: Uses Sinkhorn algorithm for balanced soft assignment
- hungarian: Uses Hungarian algorithm for perfectly balanced hard assignment
- faiss_pq: Uses FAISS Product Quantization for optimal loss
- hybrid: Combines FAISS initialization with optimal transport balancing

Usage:
    python generate_centroids_optimized.py --dataset-path /path/to/train.parquet \
                                            --num-centroids 256 \
                                            --output centroids.csv \
                                            --algorithm optimal_transport
"""

import argparse
import json
import time
from pathlib import Path
from typing import Tuple, Optional
import warnings

import numpy as np
import polars as pl
import faiss
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax


def load_vectors_from_parquet(parquet_path: Path, max_samples: Optional[int] = None) -> np.ndarray:
    """Load vectors from parquet file.

    Args:
        parquet_path: Path to the parquet file containing vectors
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        numpy array of shape (n_samples, dimension)
    """
    print(f"Loading vectors from {parquet_path}...")

    df = pl.read_parquet(parquet_path)

    # Get the vector column
    vector_column = None
    for col in ['emb', 'vector', 'embedding']:
        if col in df.columns:
            vector_column = col
            break

    if vector_column is None:
        for col in df.columns:
            if df[col].dtype == pl.List:
                vector_column = col
                break

    if vector_column is None:
        raise ValueError(f"Could not find vector column in {parquet_path}. Columns: {df.columns}")

    print(f"Found vector column: '{vector_column}'")

    if max_samples is not None and len(df) > max_samples:
        print(f"Sampling {max_samples} vectors from {len(df)} total vectors")
        df = df.sample(n=max_samples, seed=42)

    vectors = np.array(df[vector_column].to_list(), dtype=np.float32)

    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")
    return vectors


def compute_reconstruction_loss(vectors: np.ndarray, centroids: np.ndarray,
                                assignments: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Compute reconstruction loss (quantization error).

    Args:
        vectors: Input vectors (n_samples, dimension)
        centroids: Cluster centroids (num_centroids, dimension)
        assignments: Optional pre-computed assignments

    Returns:
        Tuple of (average_loss, assignments)
    """
    n_features = vectors.shape[1]

    if assignments is None:
        # Compute assignments
        index = faiss.IndexFlatL2(n_features)
        index.add(centroids.astype(np.float32))
        distances, assignments = index.search(vectors.astype(np.float32), 1)
        distances = distances.flatten()
        assignments = assignments.flatten()
    else:
        # Compute distances for given assignments
        distances = np.sum((vectors - centroids[assignments]) ** 2, axis=1)

    avg_loss = np.mean(distances)
    return avg_loss, assignments


def balanced_kmeans_hungarian(
    vectors: np.ndarray,
    num_centroids: int,
    max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Balanced k-means using Hungarian algorithm for optimal assignment.

    This algorithm ensures perfectly balanced clusters by solving the assignment
    as a linear sum assignment problem. Each iteration:
    1. Compute distance matrix between all points and centroids
    2. Use Hungarian algorithm to find optimal balanced assignment
    3. Update centroids based on assignments
    4. Repeat until convergence

    Args:
        vectors: Input vectors (n_samples, dimension)
        num_centroids: Number of centroids
        max_iter: Maximum iterations
        random_state: Random seed
        verbose: Print progress

    Returns:
        Tuple of (centroids, assignments, metrics)
    """
    np.random.seed(random_state)
    n_samples, n_features = vectors.shape
    vectors = vectors.astype(np.float32)

    target_size = n_samples // num_centroids

    if verbose:
        print(f"\n=== Balanced K-Means (Hungarian Algorithm) ===")
        print(f"Samples: {n_samples}, Centroids: {num_centroids}")
        print(f"Target cluster size: {target_size}")

    # Initialize with k-means++
    if verbose:
        print(f"Initializing centroids with k-means++...")

    kmeans_init = faiss.Kmeans(
        d=n_features,
        k=num_centroids,
        niter=20,
        verbose=False,
        seed=random_state,
        gpu=False
    )
    kmeans_init.train(vectors)
    centroids = kmeans_init.centroids.copy()

    metrics = {'losses': [], 'imbalances': [], 'times': []}

    for iteration in range(max_iter):
        iter_start = time.time()

        # For Hungarian algorithm, we need to create a balanced bipartite graph
        # Since we may have n_samples != num_centroids * target_size exactly,
        # we'll use a chunk-based approach

        assignments = np.zeros(n_samples, dtype=np.int32)

        # Process in chunks for perfect balance
        chunk_size = num_centroids * target_size
        num_chunks = (n_samples + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_vectors = vectors[start_idx:end_idx]
            chunk_size_actual = end_idx - start_idx

            # Compute distance matrix
            index = faiss.IndexFlatL2(n_features)
            index.add(centroids)
            distances, _ = index.search(chunk_vectors, num_centroids)

            # Create cost matrix for Hungarian algorithm
            # We need to replicate centroids to match chunk size
            num_replicas = (chunk_size_actual + num_centroids - 1) // num_centroids

            cost_matrix = np.zeros((chunk_size_actual, num_centroids * num_replicas))
            for i in range(num_replicas):
                cost_matrix[:, i * num_centroids:(i + 1) * num_centroids] = distances

            # Pad if necessary
            cost_matrix = cost_matrix[:, :chunk_size_actual]

            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Map back to centroid indices
            chunk_assignments = col_ind % num_centroids
            assignments[start_idx:end_idx] = chunk_assignments

        # Update centroids
        centroids_old = centroids.copy()
        for k in range(num_centroids):
            cluster_points = vectors[assignments == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)

        # Compute metrics
        cluster_sizes = np.bincount(assignments, minlength=num_centroids)
        imbalance = (cluster_sizes.max() - cluster_sizes.min()) / target_size * 100

        loss, _ = compute_reconstruction_loss(vectors, centroids, assignments)
        centroid_shift = np.max(np.sqrt(np.sum((centroids - centroids_old) ** 2, axis=1)))

        iter_time = time.time() - iter_start

        metrics['losses'].append(loss)
        metrics['imbalances'].append(imbalance)
        metrics['times'].append(iter_time)

        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iter {iteration:3d}: loss={loss:.4f}, imbalance={imbalance:.1f}%, "
                  f"shift={centroid_shift:.6f}, time={iter_time:.2f}s")

        # Check convergence
        if centroid_shift < 1e-4:
            if verbose:
                print(f"\n✓ Converged at iteration {iteration}")
            break

    if verbose:
        print(f"\n=== Final Statistics ===")
        print(f"  Loss: {metrics['losses'][-1]:.4f}")
        print(f"  Imbalance: {metrics['imbalances'][-1]:.1f}%")
        print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")

    return centroids, assignments, metrics


def balanced_kmeans_optimal_transport(
    vectors: np.ndarray,
    num_centroids: int,
    max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    temperature: float = 0.05,
    sinkhorn_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Balanced k-means using Sinkhorn optimal transport.

    This algorithm uses the Sinkhorn algorithm to find a soft balanced assignment
    that minimizes reconstruction loss while maintaining cluster balance.

    Args:
        vectors: Input vectors (n_samples, dimension)
        num_centroids: Number of centroids
        max_iter: Maximum k-means iterations
        random_state: Random seed
        verbose: Print progress
        temperature: Temperature for Sinkhorn algorithm (lower = harder assignment)
        sinkhorn_iterations: Number of Sinkhorn iterations

    Returns:
        Tuple of (centroids, assignments, metrics)
    """
    np.random.seed(random_state)
    n_samples, n_features = vectors.shape
    vectors = vectors.astype(np.float32)

    target_size = n_samples / num_centroids

    if verbose:
        print(f"\n=== Balanced K-Means (Optimal Transport / Sinkhorn) ===")
        print(f"Samples: {n_samples}, Centroids: {num_centroids}")
        print(f"Target cluster size: {target_size:.1f}")
        print(f"Temperature: {temperature}, Sinkhorn iterations: {sinkhorn_iterations}")

    # Initialize with k-means++
    if verbose:
        print(f"Initializing centroids with k-means++...")

    kmeans_init = faiss.Kmeans(
        d=n_features,
        k=num_centroids,
        niter=20,
        verbose=False,
        seed=random_state,
        gpu=False
    )
    kmeans_init.train(vectors)
    centroids = kmeans_init.centroids.copy()

    metrics = {'losses': [], 'imbalances': [], 'times': []}

    # Uniform distribution over centroids
    centroid_weights = np.ones(num_centroids) / num_centroids

    for iteration in range(max_iter):
        iter_start = time.time()

        # Compute distance matrix
        index = faiss.IndexFlatL2(n_features)
        index.add(centroids)
        distances_sq, _ = index.search(vectors, num_centroids)

        # Sinkhorn algorithm for optimal transport
        # Cost matrix: C[i,j] = distance from point i to centroid j
        C = distances_sq

        # Convert to probabilities with temperature
        K = np.exp(-C / temperature)

        # Marginals
        a = np.ones(n_samples) / n_samples  # Uniform over samples
        b = centroid_weights  # Uniform over centroids

        # Sinkhorn iterations
        u = np.ones(n_samples)
        v = np.ones(num_centroids)

        for _ in range(sinkhorn_iterations):
            u = a / (K @ v + 1e-10)
            v = b / (K.T @ u + 1e-10)

        # Transport plan
        P = u[:, None] * K * v[None, :]

        # Hard assignment (argmax of transport plan)
        assignments = np.argmax(P, axis=1).astype(np.int32)

        # Update centroids using soft assignment (weighted by transport plan)
        centroids_old = centroids.copy()
        for k in range(num_centroids):
            weights = P[:, k]
            if weights.sum() > 1e-10:
                centroids[k] = np.average(vectors, axis=0, weights=weights)
            else:
                # If cluster gets no points, keep old centroid
                centroids[k] = centroids_old[k]

        # Compute metrics
        cluster_sizes = np.bincount(assignments, minlength=num_centroids)
        imbalance = (cluster_sizes.max() - cluster_sizes.min()) / target_size * 100

        loss, _ = compute_reconstruction_loss(vectors, centroids, assignments)
        centroid_shift = np.max(np.sqrt(np.sum((centroids - centroids_old) ** 2, axis=1)))

        iter_time = time.time() - iter_start

        metrics['losses'].append(loss)
        metrics['imbalances'].append(imbalance)
        metrics['times'].append(iter_time)

        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iter {iteration:3d}: loss={loss:.4f}, imbalance={imbalance:.1f}%, "
                  f"shift={centroid_shift:.6f}, time={iter_time:.2f}s")

        # Check convergence
        if centroid_shift < 1e-4:
            if verbose:
                print(f"\n✓ Converged at iteration {iteration}")
            break

    if verbose:
        print(f"\n=== Final Statistics ===")
        print(f"  Loss: {metrics['losses'][-1]:.4f}")
        print(f"  Imbalance: {metrics['imbalances'][-1]:.1f}%")
        print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")

    return centroids, assignments, metrics


def balanced_kmeans_faiss_pq(
    vectors: np.ndarray,
    num_centroids: int,
    max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """K-means using FAISS with Product Quantization for optimal loss.

    This uses FAISS's optimized k-means which is very fast and produces
    good quality centroids, though cluster balance is not guaranteed.

    Args:
        vectors: Input vectors (n_samples, dimension)
        num_centroids: Number of centroids
        max_iter: Maximum iterations
        random_state: Random seed
        verbose: Print progress
        use_gpu: Use GPU if available

    Returns:
        Tuple of (centroids, assignments, metrics)
    """
    np.random.seed(random_state)
    n_samples, n_features = vectors.shape
    vectors = vectors.astype(np.float32)

    if verbose:
        print(f"\n=== FAISS K-Means (Optimized for Loss) ===")
        print(f"Samples: {n_samples}, Centroids: {num_centroids}")

    gpu_available = faiss.get_num_gpus() > 0 and use_gpu
    if verbose:
        print(f"Device: {'GPU' if gpu_available else 'CPU'}")

    start_time = time.time()

    # Use FAISS k-means with more iterations for better quality
    kmeans = faiss.Kmeans(
        d=n_features,
        k=num_centroids,
        niter=max_iter,
        verbose=verbose,
        seed=random_state,
        gpu=gpu_available,
        spherical=False,
        nredo=5,  # Multiple runs to get better centroids
        update_index=True
    )

    kmeans.train(vectors)
    centroids = kmeans.centroids

    # Get assignments
    _, assignments = kmeans.index.search(vectors, 1)
    assignments = assignments.flatten()

    elapsed = time.time() - start_time

    # Compute metrics
    cluster_sizes = np.bincount(assignments, minlength=num_centroids)
    target_size = n_samples / num_centroids
    imbalance = (cluster_sizes.max() - cluster_sizes.min()) / target_size * 100

    loss, _ = compute_reconstruction_loss(vectors, centroids, assignments)

    metrics = {
        'losses': [loss],
        'imbalances': [imbalance],
        'times': [elapsed]
    }

    if verbose:
        print(f"\n=== Final Statistics ===")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Loss: {loss:.4f}")
        print(f"  Imbalance: {imbalance:.1f}%")
        print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")

    return centroids, assignments, metrics


def balanced_kmeans_hybrid(
    vectors: np.ndarray,
    num_centroids: int,
    max_iter: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    balance_weight: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Hybrid approach: FAISS initialization + iterative balancing with loss optimization.

    This combines:
    1. Fast FAISS initialization for good initial centroids
    2. Iterative rebalancing that optimizes both loss and balance
    3. Adaptive rebalancing based on imbalance level

    Args:
        vectors: Input vectors (n_samples, dimension)
        num_centroids: Number of centroids
        max_iter: Maximum iterations
        random_state: Random seed
        verbose: Print progress
        balance_weight: Weight for balance vs loss (0=pure loss, 1=pure balance)

    Returns:
        Tuple of (centroids, assignments, metrics)
    """
    np.random.seed(random_state)
    n_samples, n_features = vectors.shape
    vectors = vectors.astype(np.float32)

    target_size = n_samples / num_centroids

    if verbose:
        print(f"\n=== Hybrid Balanced K-Means ===")
        print(f"Samples: {n_samples}, Centroids: {num_centroids}")
        print(f"Target cluster size: {target_size:.1f}")
        print(f"Balance weight: {balance_weight}")

    # Initialize with FAISS for good quality
    if verbose:
        print(f"Initializing with FAISS k-means...")

    kmeans_init = faiss.Kmeans(
        d=n_features,
        k=num_centroids,
        niter=30,
        verbose=False,
        seed=random_state,
        gpu=False,
        nredo=3
    )
    kmeans_init.train(vectors)
    centroids = kmeans_init.centroids.copy()

    metrics = {'losses': [], 'imbalances': [], 'times': []}

    for iteration in range(max_iter):
        iter_start = time.time()

        # Compute distances to all centroids
        index = faiss.IndexFlatL2(n_features)
        index.add(centroids)
        all_distances, all_nearest = index.search(vectors, num_centroids)

        # Initial assignment (nearest centroid)
        assignments = all_nearest[:, 0].copy()

        # Calculate current cluster sizes
        cluster_sizes = np.bincount(assignments, minlength=num_centroids)

        # Adaptive rebalancing
        max_size = int(target_size * 1.2)  # Allow 20% deviation
        min_size = int(target_size * 0.8)

        oversized = np.where(cluster_sizes > max_size)[0]
        undersized = np.where(cluster_sizes < min_size)[0]

        # Rebalance by reassigning points
        if len(oversized) > 0 and len(undersized) > 0:
            # Calculate how aggressive to be based on current imbalance
            current_imbalance = (cluster_sizes.max() - cluster_sizes.min()) / target_size
            aggressiveness = min(1.0, current_imbalance / 0.5)  # More aggressive if more imbalanced

            for cluster_id in oversized:
                excess = cluster_sizes[cluster_id] - max_size
                num_to_move = int(excess * aggressiveness)

                if num_to_move <= 0:
                    continue

                # Find points in this cluster
                cluster_mask = (assignments == cluster_id)
                cluster_points = np.where(cluster_mask)[0]

                # Calculate penalty for reassignment (distance increase)
                point_distances = all_distances[cluster_points, :]
                current_distances = point_distances[:, 0]

                # For each point, find best alternative considering both distance and balance
                penalties = np.zeros((len(cluster_points), num_centroids))
                for k in range(num_centroids):
                    distance_penalty = point_distances[:, k] - current_distances

                    # Add balance incentive for undersized clusters
                    balance_bonus = 0
                    if k in undersized:
                        balance_bonus = -current_imbalance * balance_weight * target_size

                    penalties[:, k] = distance_penalty - balance_bonus

                # Select points with minimum penalty for reassignment
                min_penalties = np.min(penalties[:, 1:], axis=1)  # Exclude current cluster
                candidates_idx = np.argsort(min_penalties)[:num_to_move]

                # Reassign
                for idx in candidates_idx:
                    point_idx = cluster_points[idx]
                    best_alternative = np.argmin(penalties[idx, 1:]) + 1  # +1 because we excluded column 0
                    new_cluster = all_nearest[point_idx, best_alternative]

                    assignments[point_idx] = new_cluster
                    cluster_sizes[cluster_id] -= 1
                    cluster_sizes[new_cluster] += 1

        # Update centroids
        centroids_old = centroids.copy()
        for k in range(num_centroids):
            cluster_points = vectors[assignments == k]
            if len(cluster_points) > 0:
                centroids[k] = cluster_points.mean(axis=0)

        # Compute metrics
        imbalance = (cluster_sizes.max() - cluster_sizes.min()) / target_size * 100
        loss, _ = compute_reconstruction_loss(vectors, centroids, assignments)
        centroid_shift = np.max(np.sqrt(np.sum((centroids - centroids_old) ** 2, axis=1)))

        iter_time = time.time() - iter_start

        metrics['losses'].append(loss)
        metrics['imbalances'].append(imbalance)
        metrics['times'].append(iter_time)

        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iter {iteration:3d}: loss={loss:.4f}, imbalance={imbalance:.1f}%, "
                  f"shift={centroid_shift:.6f}, time={iter_time:.2f}s")

        # Check convergence
        if centroid_shift < 1e-4 and imbalance < 20:
            if verbose:
                print(f"\n✓ Converged at iteration {iteration}")
            break

    if verbose:
        print(f"\n=== Final Statistics ===")
        print(f"  Loss: {metrics['losses'][-1]:.4f}")
        print(f"  Imbalance: {metrics['imbalances'][-1]:.1f}%")
        print(f"  Cluster sizes: min={cluster_sizes.min()}, max={cluster_sizes.max()}, mean={cluster_sizes.mean():.1f}")

    return centroids, assignments, metrics


def verify_on_full_dataset(
    vectors: np.ndarray,
    centroids: np.ndarray,
    num_centroids: int,
    verbose: bool = True
) -> dict:
    """Verify centroids on full dataset and compute comprehensive metrics.

    Args:
        vectors: Full dataset vectors
        centroids: Computed centroids
        num_centroids: Number of centroids
        verbose: Print detailed statistics

    Returns:
        Dictionary of metrics
    """
    n_samples, n_features = vectors.shape

    if verbose:
        print(f"\n=== Verification on Full Dataset ({n_samples} vectors) ===")

    # Compute assignments in batches
    batch_size = 50000
    assignments = np.zeros(n_samples, dtype=np.int32)
    total_loss = 0.0

    index = faiss.IndexFlatL2(n_features)
    index.add(centroids.astype(np.float32))

    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        distances, labels = index.search(vectors[i:end_i].astype(np.float32), 1)
        assignments[i:end_i] = labels.flatten()
        total_loss += distances.sum()

    avg_loss = total_loss / n_samples

    # Compute cluster statistics
    cluster_sizes = np.bincount(assignments, minlength=num_centroids)
    target_size = n_samples / num_centroids

    imbalance = (cluster_sizes.max() - cluster_sizes.min()) / target_size * 100

    # Distribution
    bins = [0, int(target_size*0.8), int(target_size*0.9),
            int(target_size*1.1), int(target_size*1.2), n_samples]
    bin_labels = ["<80%", "80-90%", "90-110%", "110-120%", ">120%"]
    hist, _ = np.histogram(cluster_sizes, bins=bins)

    metrics = {
        'loss': avg_loss,
        'imbalance': imbalance,
        'cluster_sizes': cluster_sizes,
        'min_size': cluster_sizes.min(),
        'max_size': cluster_sizes.max(),
        'mean_size': cluster_sizes.mean(),
        'std_size': cluster_sizes.std(),
        'distribution': dict(zip(bin_labels, hist))
    }

    if verbose:
        print(f"  Average reconstruction loss: {avg_loss:.4f}")
        print(f"  Cluster size range: [{cluster_sizes.min()}, {cluster_sizes.max()}]")
        print(f"  Mean cluster size: {cluster_sizes.mean():.1f} (target: {target_size:.1f})")
        print(f"  Std dev: {cluster_sizes.std():.1f}")
        print(f"  Imbalance: {imbalance:.1f}%")

        print(f"\n  Cluster size distribution:")
        for label, count in zip(bin_labels, hist):
            if count > 0:
                print(f"    {label} of target: {count} clusters")

        print(f"\n  Quality Assessment:")
        if imbalance < 20 and avg_loss < 50:
            print(f"    ✓ EXCELLENT (low imbalance, low loss)")
        elif imbalance < 50 and avg_loss < 100:
            print(f"    ✓ GOOD (moderate balance, acceptable loss)")
        elif imbalance < 100:
            print(f"    ⚠ MODERATE (needs improvement)")
        else:
            print(f"    ✗ POOR (high imbalance)")

    return metrics


def save_centroids_to_csv(centroids: np.ndarray, output_path: Path, rocksdb_format: bool = True) -> None:
    """Save centroids to CSV file.

    Args:
        centroids: Centroids array
        output_path: Output file path
        rocksdb_format: Use RocksDB C++ format
    """
    print(f"\nSaving {len(centroids)} centroids to {output_path}...")

    with output_path.open('w', newline='') as f:
        if rocksdb_format:
            for centroid in centroids:
                centroid_json = json.dumps(centroid.tolist())
                f.write(f'"{centroid_json}"\n')
        else:
            import csv
            writer = csv.writer(f)
            writer.writerow(['id', 'centroid'])
            for i, centroid in enumerate(centroids):
                centroid_json = json.dumps(centroid.tolist())
                writer.writerow([i, centroid_json])

    print(f"✓ Successfully saved centroids to {output_path}")
    file_size = output_path.stat().st_size
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description='Generate optimized balanced centroids for MyRocks LSM vector index'
    )
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
        '--algorithm',
        type=str,
        choices=['optimal_transport', 'hungarian', 'faiss', 'hybrid', 'compare'],
        default='optimal_transport',
        help='Algorithm to use (default: optimal_transport). Use "compare" to run all and compare.'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=100,
        help='Maximum number of iterations (default: 100)'
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
        '--training-sample-size',
        type=int,
        default=None,
        help='Number of samples for training (None = use all)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MyRocks Optimized Centroid Generation")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Number of centroids: {args.num_centroids}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Random state: {args.random_state}")
    print("=" * 70)

    if not args.dataset_path.exists():
        print(f"\n✗ Error: Dataset file not found: {args.dataset_path}")
        return 1

    try:
        # Load vectors
        vectors = load_vectors_from_parquet(args.dataset_path, args.max_samples)

        # Sample for training if specified
        if args.training_sample_size and args.training_sample_size < len(vectors):
            print(f"\nUsing {args.training_sample_size}/{len(vectors)} vectors for training")
            training_indices = np.random.choice(len(vectors), args.training_sample_size, replace=False)
            training_vectors = vectors[training_indices]
        else:
            training_vectors = vectors

        start_time = time.time()

        if args.algorithm == 'compare':
            # Run all algorithms and compare
            print("\n" + "=" * 70)
            print("COMPARISON MODE: Running all algorithms")
            print("=" * 70)

            results = {}

            algorithms = [
                ('FAISS (Loss-optimized)', balanced_kmeans_faiss_pq),
                ('Hybrid', balanced_kmeans_hybrid),
                ('Optimal Transport', balanced_kmeans_optimal_transport),
            ]

            for name, algo_func in algorithms:
                print(f"\n{'='*70}")
                print(f"Running: {name}")
                print(f"{'='*70}")

                if algo_func == balanced_kmeans_faiss_pq:
                    centroids, assignments, metrics = algo_func(
                        training_vectors,
                        args.num_centroids,
                        args.max_iter,
                        args.random_state,
                        verbose=True
                    )
                else:
                    centroids, assignments, metrics = algo_func(
                        training_vectors,
                        args.num_centroids,
                        args.max_iter,
                        args.random_state,
                        verbose=True
                    )

                # Verify on full dataset
                full_metrics = verify_on_full_dataset(
                    vectors, centroids, args.num_centroids, verbose=True
                )

                results[name] = {
                    'centroids': centroids,
                    'metrics': full_metrics,
                    'training_time': sum(metrics['times'])
                }

            # Print comparison
            print("\n" + "=" * 70)
            print("COMPARISON RESULTS")
            print("=" * 70)
            print(f"{'Algorithm':<25} {'Loss':>10} {'Imbalance':>12} {'Time (s)':>10}")
            print("-" * 70)

            for name, result in results.items():
                print(f"{name:<25} {result['metrics']['loss']:>10.4f} "
                      f"{result['metrics']['imbalance']:>11.1f}% "
                      f"{result['training_time']:>10.2f}")

            # Find best by different criteria
            best_loss = min(results.items(), key=lambda x: x[1]['metrics']['loss'])
            best_balance = min(results.items(), key=lambda x: x[1]['metrics']['imbalance'])

            # Combined score (normalized loss + imbalance)
            for name, result in results.items():
                result['score'] = result['metrics']['loss'] / 100 + result['metrics']['imbalance'] / 100
            best_overall = min(results.items(), key=lambda x: x[1]['score'])

            print("\n" + "=" * 70)
            print(f"Best for loss: {best_loss[0]} (loss={best_loss[1]['metrics']['loss']:.4f})")
            print(f"Best for balance: {best_balance[0]} (imbalance={best_balance[1]['metrics']['imbalance']:.1f}%)")
            print(f"Best overall: {best_overall[0]} (score={best_overall[1]['score']:.2f})")
            print("=" * 70)

            # Save the best overall
            centroids = best_overall[1]['centroids']
            print(f"\nSaving best centroids from: {best_overall[0]}")

        else:
            # Run single algorithm
            if args.algorithm == 'optimal_transport':
                centroids, assignments, metrics = balanced_kmeans_optimal_transport(
                    training_vectors,
                    args.num_centroids,
                    args.max_iter,
                    args.random_state,
                    verbose=True
                )
            elif args.algorithm == 'hungarian':
                centroids, assignments, metrics = balanced_kmeans_hungarian(
                    training_vectors,
                    args.num_centroids,
                    args.max_iter,
                    args.random_state,
                    verbose=True
                )
            elif args.algorithm == 'faiss':
                centroids, assignments, metrics = balanced_kmeans_faiss_pq(
                    training_vectors,
                    args.num_centroids,
                    args.max_iter,
                    args.random_state,
                    verbose=True
                )
            elif args.algorithm == 'hybrid':
                centroids, assignments, metrics = balanced_kmeans_hybrid(
                    training_vectors,
                    args.num_centroids,
                    args.max_iter,
                    args.random_state,
                    verbose=True
                )

            # Verify on full dataset
            verify_on_full_dataset(vectors, centroids, args.num_centroids, verbose=True)

        total_time = time.time() - start_time

        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Save centroids
        save_centroids_to_csv(centroids, args.output, rocksdb_format=args.rocksdb_format)

        print("\n" + "=" * 70)
        print("✓ Optimized centroid generation completed successfully!")
        print("=" * 70)
        print(f"Total time: {total_time / 60:.2f} minutes")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
