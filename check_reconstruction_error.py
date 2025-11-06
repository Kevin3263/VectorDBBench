#!/usr/bin/env python3
"""
Check reconstruction error for generated centroids.
Reconstruction error = average distance from each vector to its nearest centroid.
"""

import argparse
import json
import numpy as np
import polars as pl
import faiss
from pathlib import Path


def load_centroids_from_csv(csv_path: Path) -> np.ndarray:
    """Load centroids from RocksDB format CSV."""
    centroids = []
    with open(csv_path, 'r') as f:
        for line in f:
            # Remove quotes and parse JSON array
            line = line.strip().strip('"')
            centroid = json.loads(line)
            centroids.append(centroid)
    return np.array(centroids, dtype=np.float32)


def load_vectors_from_parquet(parquet_path: Path) -> np.ndarray:
    """Load vectors from parquet file."""
    df = pl.read_parquet(parquet_path)

    # Find vector column
    vector_column = None
    for col in ['emb', 'vector', 'embedding']:
        if col in df.columns:
            vector_column = col
            break

    if vector_column is None:
        raise ValueError(f"Could not find vector column")

    vectors = np.array(df[vector_column].to_list(), dtype=np.float32)
    return vectors


def calculate_reconstruction_error(vectors: np.ndarray, centroids: np.ndarray,
                                   batch_size: int = 50000) -> dict:
    """Calculate reconstruction error statistics."""
    n_samples = len(vectors)
    n_centroids = len(centroids)

    print(f"Calculating reconstruction error...")
    print(f"  Vectors: {n_samples}")
    print(f"  Centroids: {n_centroids}")

    # Build FAISS index
    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids)

    # Calculate distances in batches
    all_distances = []
    for i in range(0, n_samples, batch_size):
        end_i = min(i + batch_size, n_samples)
        distances, _ = index.search(vectors[i:end_i], 1)
        all_distances.append(distances.flatten())

        if (i + batch_size) % (batch_size * 5) == 0:
            print(f"  Processed {i + batch_size}/{n_samples} vectors...")

    all_distances = np.concatenate(all_distances)

    # Calculate statistics
    stats = {
        'mean': float(np.mean(all_distances)),
        'median': float(np.median(all_distances)),
        'std': float(np.std(all_distances)),
        'min': float(np.min(all_distances)),
        'max': float(np.max(all_distances)),
        'rmse': float(np.sqrt(np.mean(all_distances)))  # Root Mean Squared Error
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Check reconstruction error for centroids')
    parser.add_argument('--centroids', type=Path, required=True,
                       help='Path to centroids CSV file')
    parser.add_argument('--dataset', type=Path, required=True,
                       help='Path to dataset parquet file')

    args = parser.parse_args()

    print("=" * 70)
    print("Reconstruction Error Analysis")
    print("=" * 70)
    print(f"Centroids: {args.centroids}")
    print(f"Dataset: {args.dataset}")
    print()

    # Load data
    print("Loading centroids...")
    centroids = load_centroids_from_csv(args.centroids)
    print(f"  Loaded {len(centroids)} centroids, dimension {centroids.shape[1]}")

    print("\nLoading vectors...")
    vectors = load_vectors_from_parquet(args.dataset)
    print(f"  Loaded {len(vectors)} vectors, dimension {vectors.shape[1]}")

    # Calculate reconstruction error
    print()
    stats = calculate_reconstruction_error(vectors, centroids)

    # Print results
    print("\n" + "=" * 70)
    print("Reconstruction Error Statistics")
    print("=" * 70)
    print(f"Mean squared distance:    {stats['mean']:.6f}")
    print(f"Median squared distance:  {stats['median']:.6f}")
    print(f"Std dev:                  {stats['std']:.6f}")
    print(f"Min distance:             {stats['min']:.6f}")
    print(f"Max distance:             {stats['max']:.6f}")
    print(f"RMSE:                     {stats['rmse']:.6f}")
    print("=" * 70)

    # Interpretation
    print("\nInterpretation:")
    print(f"  - Lower values = better representation")
    print(f"  - RMSE of {stats['rmse']:.4f} means vectors are on average")
    print(f"    {stats['rmse']:.4f} distance units from their nearest centroid")


if __name__ == '__main__':
    main()
