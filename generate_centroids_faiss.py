#!/usr/bin/env python3
"""
Production centroid generation for IVF indexes using FAISS k-means.

This script prioritizes IVF search quality (low reconstruction loss) over
forced cluster balance. It uses FAISS k-means with configurable sampling
and reports balance metrics for monitoring.

Key principles:
1. Natural clustering (no forced balancing) = best IVF recall
2. Use more samples = better centroid quality
3. Monitor imbalance but don't constrain it
4. Optimize for reconstruction loss (quantization error)

Usage:
    python generate_centroids_faiss.py \
        --dataset-path /path/to/train.parquet \
        --num-centroids 256 \
        --output centroids.csv \
        --training-sample-size 500000 \
        --nredo 5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import polars as pl
import faiss


def load_vectors_from_parquet(parquet_path: Path, max_samples: int = None) -> np.ndarray:
    """Load vectors from parquet file.

    Args:
        parquet_path: Path to the parquet file containing vectors
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        numpy array of shape (n_samples, dimension)
    """
    print(f"Loading vectors from {parquet_path}...")

    df = pl.read_parquet(parquet_path)

    # Find vector column
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
        print(f"Limiting to {max_samples} vectors from {len(df)} total")
        df = df.head(max_samples)

    vectors = np.array(df[vector_column].to_list(), dtype=np.float32)

    print(f"Loaded {len(vectors)} vectors with dimension {vectors.shape[1]}")
    return vectors


def compute_cluster_metrics(
    vectors: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    num_centroids: int
) -> Dict:
    """Compute comprehensive cluster quality metrics.

    Args:
        vectors: Input vectors (n_samples, dimension)
        centroids: Cluster centroids (num_centroids, dimension)
        assignments: Cluster assignments for each vector
        num_centroids: Number of centroids

    Returns:
        Dictionary of metrics
    """
    n_samples = len(vectors)
    target_size = n_samples / num_centroids

    # Cluster sizes
    cluster_sizes = np.bincount(assignments, minlength=num_centroids)

    # Reconstruction loss (quantization error)
    distances = np.sum((vectors - centroids[assignments]) ** 2, axis=1)
    avg_loss = np.mean(distances)
    total_loss = np.sum(distances)

    # Balance metrics
    min_size = cluster_sizes.min()
    max_size = cluster_sizes.max()
    mean_size = cluster_sizes.mean()
    std_size = cluster_sizes.std()

    imbalance_pct = (max_size - min_size) / target_size * 100 if target_size > 0 else 0

    # Distribution
    bins = [0, int(target_size*0.8), int(target_size*0.9),
            int(target_size*1.1), int(target_size*1.2), n_samples]
    hist, _ = np.histogram(cluster_sizes, bins=bins)

    # Empty clusters
    num_empty = np.sum(cluster_sizes == 0)

    metrics = {
        # Loss metrics
        'avg_reconstruction_loss': float(avg_loss),
        'total_reconstruction_loss': float(total_loss),

        # Size metrics
        'min_cluster_size': int(min_size),
        'max_cluster_size': int(max_size),
        'mean_cluster_size': float(mean_size),
        'std_cluster_size': float(std_size),
        'target_cluster_size': float(target_size),

        # Balance metrics
        'imbalance_percent': float(imbalance_pct),
        'num_empty_clusters': int(num_empty),

        # Distribution
        'distribution': {
            'below_80pct': int(hist[0]),
            '80_to_90pct': int(hist[1]),
            '90_to_110pct': int(hist[2]),  # Target range
            '110_to_120pct': int(hist[3]),
            'above_120pct': int(hist[4])
        },

        # Cluster sizes array (for detailed analysis)
        'cluster_sizes': cluster_sizes.tolist()
    }

    return metrics


def print_metrics(metrics: Dict, dataset_name: str = "Dataset") -> None:
    """Print metrics in a readable format.

    Args:
        metrics: Metrics dictionary from compute_cluster_metrics
        dataset_name: Name of the dataset (for display)
    """
    print(f"\n{'='*70}")
    print(f"Metrics for {dataset_name}")
    print(f"{'='*70}")

    print(f"\n📊 Cluster Size Statistics:")
    print(f"  Target size:     {metrics['target_cluster_size']:.1f}")
    print(f"  Mean size:       {metrics['mean_cluster_size']:.1f}")
    print(f"  Std deviation:   {metrics['std_cluster_size']:.1f}")
    print(f"  Min size:        {metrics['min_cluster_size']}")
    print(f"  Max size:        {metrics['max_cluster_size']}")
    print(f"  Range:           [{metrics['min_cluster_size']}, {metrics['max_cluster_size']}]")

    print(f"\n⚖️  Balance Metrics:")
    print(f"  Imbalance:       {metrics['imbalance_percent']:.1f}%")
    if metrics['num_empty_clusters'] > 0:
        print(f"  Empty clusters:  {metrics['num_empty_clusters']} ⚠️")
    else:
        print(f"  Empty clusters:  0 ✓")

    print(f"\n📈 Distribution (% of target):")
    dist = metrics['distribution']
    total_clusters = sum(dist.values())
    print(f"  <80%:       {dist['below_80pct']:4d} clusters ({dist['below_80pct']/total_clusters*100:5.1f}%)")
    print(f"  80-90%:     {dist['80_to_90pct']:4d} clusters ({dist['80_to_90pct']/total_clusters*100:5.1f}%)")
    print(f"  90-110%:    {dist['90_to_110pct']:4d} clusters ({dist['90_to_110pct']/total_clusters*100:5.1f}%) ← Target")
    print(f"  110-120%:   {dist['110_to_120pct']:4d} clusters ({dist['110_to_120pct']/total_clusters*100:5.1f}%)")
    print(f"  >120%:      {dist['above_120pct']:4d} clusters ({dist['above_120pct']/total_clusters*100:5.1f}%)")

    print(f"\n🎯 Reconstruction Loss (Quantization Error):")
    print(f"  Average:         {metrics['avg_reconstruction_loss']:.4f}")
    print(f"  Total:           {metrics['total_reconstruction_loss']:.2f}")

    print(f"\n💡 IVF Search Quality Assessment:")
    loss = metrics['avg_reconstruction_loss']
    imbal = metrics['imbalance_percent']

    if loss < 15:
        loss_quality = "EXCELLENT"
    elif loss < 25:
        loss_quality = "GOOD"
    elif loss < 35:
        loss_quality = "FAIR"
    else:
        loss_quality = "POOR"

    if imbal < 20:
        balance_note = "(imbalance is fine for IVF)"
    elif imbal < 40:
        balance_note = "(moderate imbalance, monitor if issues)"
    elif imbal < 60:
        balance_note = "(high imbalance, check for empty clusters)"
    else:
        balance_note = "(very high imbalance, investigate data distribution)"

    print(f"  Recall quality:  {loss_quality} (loss={loss:.2f})")
    print(f"  Balance:         {imbal:.1f}% {balance_note}")

    print(f"{'='*70}\n")


def train_faiss_kmeans(
    vectors: np.ndarray,
    num_centroids: int,
    niter: int = 100,
    nredo: int = 5,
    random_state: int = 42,
    use_gpu: bool = True,
    verbose: bool = True,
    max_points_per_centroid: int = None
) -> Tuple[np.ndarray, Dict]:
    """Train FAISS k-means clustering.

    Args:
        vectors: Input vectors (n_samples, dimension)
        num_centroids: Number of centroids to generate
        niter: Number of k-means iterations
        nredo: Number of random initializations (best one selected)
        random_state: Random seed
        use_gpu: Use GPU if available
        verbose: Print progress
        max_points_per_centroid: Maximum samples per centroid (None = use all)

    Returns:
        Tuple of (centroids, metrics_dict)
    """
    n_samples, n_features = vectors.shape
    vectors = vectors.astype(np.float32)

    # Determine actual training sample size
    if max_points_per_centroid is None:
        # Use all data
        training_size = n_samples
        max_points = 10**9  # Very large number to disable FAISS sampling
    else:
        training_size = min(n_samples, max_points_per_centroid * num_centroids)
        max_points = max_points_per_centroid

    if verbose:
        print(f"\n{'='*70}")
        print(f"FAISS K-Means Training")
        print(f"{'='*70}")
        print(f"Vectors:              {n_samples:,}")
        print(f"Dimension:            {n_features}")
        print(f"Target centroids:     {num_centroids}")
        print(f"Training samples:     {training_size:,} ({training_size/n_samples*100:.1f}%)")
        print(f"K-means iterations:   {niter}")
        print(f"Random restarts:      {nredo}")

    # Check GPU
    gpu_available = faiss.get_num_gpus() > 0 and use_gpu
    if verbose:
        if gpu_available:
            print(f"Device:               GPU ({faiss.get_num_gpus()} available)")
        else:
            print(f"Device:               CPU")
        print(f"{'='*70}")

    # Sample if needed
    if training_size < n_samples:
        if verbose:
            print(f"\nSampling {training_size:,} vectors for training...")
        np.random.seed(random_state)
        indices = np.random.choice(n_samples, training_size, replace=False)
        training_vectors = vectors[indices]
    else:
        training_vectors = vectors

    start_time = time.time()

    # Create FAISS k-means
    kmeans = faiss.Kmeans(
        d=n_features,
        k=num_centroids,
        niter=niter,
        nredo=nredo,
        verbose=verbose,
        seed=random_state,
        gpu=gpu_available,
        spherical=False,
        max_points_per_centroid=max_points,
        update_index=True
    )

    if verbose:
        print(f"\nTraining k-means...")

    # Train
    kmeans.train(training_vectors)

    elapsed = time.time() - start_time

    # Get centroids
    centroids = kmeans.centroids

    if verbose:
        print(f"\n✓ K-means training completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")

    # Compute metrics on training data
    _, assignments_train = kmeans.index.search(training_vectors, 1)
    assignments_train = assignments_train.flatten()

    metrics_train = compute_cluster_metrics(
        training_vectors, centroids, assignments_train, num_centroids
    )
    metrics_train['training_time_seconds'] = elapsed
    metrics_train['training_samples'] = training_size

    return centroids, metrics_train


def verify_on_full_dataset(
    vectors: np.ndarray,
    centroids: np.ndarray,
    num_centroids: int,
    batch_size: int = 100000,
    verbose: bool = True
) -> Dict:
    """Verify centroids on full dataset and compute metrics.

    Args:
        vectors: Full dataset vectors
        centroids: Computed centroids
        num_centroids: Number of centroids
        batch_size: Batch size for processing
        verbose: Print progress

    Returns:
        Dictionary of metrics
    """
    n_samples, n_features = vectors.shape

    if verbose:
        print(f"\nVerifying on full dataset ({n_samples:,} vectors)...")

    # Create index
    index = faiss.IndexFlatL2(n_features)
    index.add(centroids.astype(np.float32))

    # Assign in batches
    assignments = np.zeros(n_samples, dtype=np.int32)

    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        _, labels = index.search(vectors[i:end_i].astype(np.float32), 1)
        assignments[i:end_i] = labels.flatten()

        if verbose and end_i % (batch_size * 5) == 0:
            print(f"  Processed {end_i:,}/{n_samples:,} vectors...")

    # Compute metrics
    metrics = compute_cluster_metrics(vectors, centroids, assignments, num_centroids)

    return metrics


def save_centroids_to_csv(
    centroids: np.ndarray,
    output_path: Path,
    rocksdb_format: bool = True
) -> None:
    """Save centroids to CSV file.

    Args:
        centroids: Centroids array (num_centroids, dimension)
        output_path: Output file path
        rocksdb_format: Use RocksDB C++ format (quoted JSON arrays)
    """
    print(f"\nSaving {len(centroids)} centroids to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', newline='') as f:
        if rocksdb_format:
            # RocksDB format: one quoted JSON array per line
            for centroid in centroids:
                centroid_json = json.dumps(centroid.tolist())
                f.write(f'"{centroid_json}"\n')
        else:
            # Standard CSV with header
            import csv
            writer = csv.writer(f)
            writer.writerow(['id', 'centroid'])
            for i, centroid in enumerate(centroids):
                centroid_json = json.dumps(centroid.tolist())
                writer.writerow([i, centroid_json])

    print(f"✓ Centroids saved to {output_path}")

    file_size = output_path.stat().st_size
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")


def save_metrics_to_json(
    metrics: Dict,
    output_path: Path
) -> None:
    """Save metrics to JSON file for monitoring.

    Args:
        metrics: Metrics dictionary
        output_path: Output JSON file path
    """
    print(f"\nSaving metrics to {output_path}...")

    with output_path.open('w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved for monitoring")


def main():
    parser = argparse.ArgumentParser(
        description='Generate centroids for IVF indexes using FAISS k-means'
    )
    parser.add_argument(
        '--dataset-path',
        type=Path,
        required=True,
        help='Path to training dataset parquet file'
    )
    parser.add_argument(
        '--num-centroids',
        type=int,
        default=256,
        help='Number of centroids to generate (default: 256)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output CSV file path for centroids'
    )
    parser.add_argument(
        '--metrics-output',
        type=Path,
        default=None,
        help='Output JSON file for metrics (default: same as output with .json extension)'
    )
    parser.add_argument(
        '--niter',
        type=int,
        default=100,
        help='Number of k-means iterations (default: 100)'
    )
    parser.add_argument(
        '--nredo',
        type=int,
        default=5,
        help='Number of random restarts - higher = better quality (default: 5, recommend: 5-10)'
    )
    parser.add_argument(
        '--training-sample-size',
        type=int,
        default=None,
        help='Number of vectors to use for training (default: all, recommend: 500k-1M for speed)'
    )
    parser.add_argument(
        '--max-points-per-centroid',
        type=int,
        default=None,
        help='Max training points per centroid (default: None = use all, alternative to training-sample-size)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Use GPU if available (default: True)'
    )
    parser.add_argument(
        '--rocksdb-format',
        action='store_true',
        default=True,
        help='Output in RocksDB C++ format (default: True)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Production IVF Centroid Generation (FAISS)")
    print("=" * 70)
    print(f"Dataset:          {args.dataset_path}")
    print(f"Output:           {args.output}")
    print(f"Centroids:        {args.num_centroids}")
    print(f"Iterations:       {args.niter}")
    print(f"Restarts (nredo): {args.nredo}")
    print(f"Random seed:      {args.random_state}")
    print("=" * 70)

    # Check dataset exists
    if not args.dataset_path.exists():
        print(f"\n✗ Error: Dataset not found: {args.dataset_path}")
        return 1

    try:
        # Load vectors
        vectors = load_vectors_from_parquet(args.dataset_path)
        total_vectors = len(vectors)

        # Determine training sample size
        if args.training_sample_size is not None:
            training_sample_size = min(args.training_sample_size, total_vectors)
        elif args.max_points_per_centroid is not None:
            training_sample_size = min(
                args.max_points_per_centroid * args.num_centroids,
                total_vectors
            )
        else:
            training_sample_size = total_vectors

        # Train k-means
        start_time = time.time()

        centroids, metrics_train = train_faiss_kmeans(
            vectors,
            num_centroids=args.num_centroids,
            niter=args.niter,
            nredo=args.nredo,
            random_state=args.random_state,
            use_gpu=args.use_gpu,
            verbose=True,
            max_points_per_centroid=args.max_points_per_centroid
        )

        # Print training metrics
        print_metrics(metrics_train, f"Training Data ({training_sample_size:,} vectors)")

        # Verify on full dataset if we used sampling
        if training_sample_size < total_vectors:
            metrics_full = verify_on_full_dataset(
                vectors,
                centroids,
                args.num_centroids,
                verbose=True
            )
            print_metrics(metrics_full, f"Full Dataset ({total_vectors:,} vectors)")
        else:
            metrics_full = metrics_train

        total_time = time.time() - start_time

        # Save centroids
        save_centroids_to_csv(
            centroids,
            args.output,
            rocksdb_format=args.rocksdb_format
        )

        # Save metrics for monitoring
        metrics_output = args.metrics_output or args.output.with_suffix('.metrics.json')

        monitoring_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'num_centroids': args.num_centroids,
                'niter': args.niter,
                'nredo': args.nredo,
                'training_samples': training_sample_size,
                'total_samples': total_vectors,
                'random_state': args.random_state
            },
            'training_metrics': metrics_train,
            'full_dataset_metrics': metrics_full,
            'total_time_seconds': total_time
        }

        save_metrics_to_json(monitoring_data, metrics_output)

        print("\n" + "=" * 70)
        print("✓ Centroid generation completed successfully!")
        print("=" * 70)
        print(f"Total time:       {total_time/60:.2f} minutes")
        print(f"Centroids:        {args.output}")
        print(f"Metrics:          {metrics_output}")
        print("\n💡 For IVF search: Low reconstruction loss = better recall")
        print("   Imbalance is monitored but not constrained (natural clustering)")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
