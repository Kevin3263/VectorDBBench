#!/usr/bin/env python3
"""
Quick test script to compare centroid generation algorithms on a small dataset.

This creates a synthetic dataset and tests all algorithms to demonstrate
the tradeoff between balance and reconstruction loss.
"""

import numpy as np
import time
from generate_centroids_optimized import (
    balanced_kmeans_optimal_transport,
    balanced_kmeans_hungarian,
    balanced_kmeans_faiss_pq,
    balanced_kmeans_hybrid,
    verify_on_full_dataset
)


def generate_synthetic_data(n_samples=10000, n_features=128, n_true_clusters=32, random_state=42):
    """Generate synthetic clustered data for testing.

    Args:
        n_samples: Number of samples
        n_features: Dimension
        n_true_clusters: Number of true underlying clusters
        random_state: Random seed

    Returns:
        numpy array of shape (n_samples, n_features)
    """
    np.random.seed(random_state)

    # Generate cluster centers
    true_centroids = np.random.randn(n_true_clusters, n_features).astype(np.float32) * 10

    # Assign samples to clusters
    assignments = np.random.randint(0, n_true_clusters, n_samples)

    # Generate samples around cluster centers with noise
    vectors = true_centroids[assignments] + np.random.randn(n_samples, n_features).astype(np.float32) * 2

    return vectors


def main():
    print("=" * 80)
    print("Centroid Generation Algorithm Comparison")
    print("=" * 80)

    # Generate test data
    print("\nGenerating synthetic test data...")
    n_samples = 50000
    n_features = 256
    num_centroids = 128

    vectors = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        n_true_clusters=64,
        random_state=42
    )

    print(f"Dataset: {n_samples} vectors, {n_features} dimensions")
    print(f"Target: {num_centroids} centroids")
    print(f"Target cluster size: {n_samples // num_centroids}")

    # Test algorithms
    algorithms = [
        ('FAISS (loss-optimized)', balanced_kmeans_faiss_pq),
        ('Hybrid (balanced + loss)', balanced_kmeans_hybrid),
        ('Optimal Transport (Sinkhorn)', balanced_kmeans_optimal_transport),
    ]

    results = {}

    for name, algo_func in algorithms:
        print("\n" + "=" * 80)
        print(f"Testing: {name}")
        print("=" * 80)

        start = time.time()

        if algo_func == balanced_kmeans_faiss_pq:
            centroids, assignments, metrics = algo_func(
                vectors,
                num_centroids,
                max_iter=50,
                random_state=42,
                verbose=True,
                use_gpu=False  # Use CPU for fair comparison
            )
        else:
            centroids, assignments, metrics = algo_func(
                vectors,
                num_centroids,
                max_iter=50,
                random_state=42,
                verbose=True
            )

        elapsed = time.time() - start

        # Get final metrics
        final_metrics = verify_on_full_dataset(
            vectors, centroids, num_centroids, verbose=False
        )

        results[name] = {
            'loss': final_metrics['loss'],
            'imbalance': final_metrics['imbalance'],
            'min_size': final_metrics['min_size'],
            'max_size': final_metrics['max_size'],
            'std_size': final_metrics['std_size'],
            'time': elapsed
        }

        print(f"✓ Completed in {elapsed:.2f}s")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Algorithm':<30} {'Loss':>10} {'Imbal%':>8} {'Min':>6} {'Max':>6} {'StdDev':>8} {'Time(s)':>8}")
    print("-" * 80)

    for name, r in results.items():
        print(f"{name:<30} {r['loss']:>10.3f} {r['imbalance']:>7.1f}% "
              f"{r['min_size']:>6} {r['max_size']:>6} {r['std_size']:>8.1f} {r['time']:>8.2f}")

    print("=" * 80)

    # Analysis
    print("\nANALYSIS:")
    print("-" * 80)

    best_loss = min(results.items(), key=lambda x: x[1]['loss'])
    best_balance = min(results.items(), key=lambda x: x[1]['imbalance'])
    fastest = min(results.items(), key=lambda x: x[1]['time'])

    print(f"Best Loss: {best_loss[0]} ({best_loss[1]['loss']:.3f})")
    print(f"Best Balance: {best_balance[0]} ({best_balance[1]['imbalance']:.1f}% imbalance)")
    print(f"Fastest: {fastest[0]} ({fastest[1]['time']:.2f}s)")

    # Compute combined score (normalized)
    max_loss = max(r['loss'] for r in results.values())
    max_imbal = max(r['imbalance'] for r in results.values())

    for name, r in results.items():
        # Normalized score: lower is better
        r['combined_score'] = (r['loss'] / max_loss) * 0.5 + (r['imbalance'] / max_imbal) * 0.5

    best_combined = min(results.items(), key=lambda x: x[1]['combined_score'])
    print(f"Best Combined: {best_combined[0]} (score={best_combined[1]['combined_score']:.3f})")

    print("\nRECOMMENDATION:")
    print("-" * 80)
    if best_combined[1]['imbalance'] < 20 and best_combined[1]['loss'] < max_loss * 1.1:
        print(f"✓ Use '{best_combined[0]}' for best balance of loss and cluster size distribution")
    else:
        print(f"⚠ Results vary significantly. Consider:")
        print(f"  - For lowest loss: {best_loss[0]}")
        print(f"  - For best balance: {best_balance[0]}")

    print("=" * 80)


if __name__ == '__main__':
    main()
