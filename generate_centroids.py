#!/usr/bin/env python3
"""
Generate centroids for MyRocks LSM vector index using k-means clustering.

This script reads the training dataset and generates centroids that can be used
for the MyRocks LSM vector index. The centroids are saved to a CSV file that
can be loaded by the MyRocks implementation.

Usage:
    python generate_centroids.py --dataset-path /path/to/shuffle_train.parquet \
                                  --num-centroids 256 \
                                  --output centroids.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans


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


def generate_centroids(vectors: np.ndarray, num_centroids: int, random_state: int = 42) -> np.ndarray:
    """Generate centroids using k-means clustering.

    Args:
        vectors: Input vectors of shape (n_samples, dimension)
        num_centroids: Number of centroids to generate
        random_state: Random seed for reproducibility

    Returns:
        Centroids array of shape (num_centroids, dimension)
    """
    print(f"\nGenerating {num_centroids} centroids using k-means clustering...")
    print(f"Input: {len(vectors)} vectors, dimension: {vectors.shape[1]}")

    start_time = time.time()

    # Use mini-batch k-means for large datasets (faster)
    if len(vectors) > 10000:
        from sklearn.cluster import MiniBatchKMeans
        print("Using MiniBatchKMeans for large dataset...")
        kmeans = MiniBatchKMeans(
            n_clusters=num_centroids,
            random_state=random_state,
            batch_size=1000,
            max_iter=100,
            verbose=1
        )
    else:
        print("Using standard KMeans...")
        kmeans = KMeans(
            n_clusters=num_centroids,
            random_state=random_state,
            max_iter=100,
            verbose=1
        )

    # Fit k-means
    kmeans.fit(vectors)

    elapsed = time.time() - start_time
    print(f"\nK-means clustering completed in {elapsed:.2f} seconds")
    print(f"Inertia (sum of squared distances): {kmeans.inertia_:.2f}")

    return kmeans.cluster_centers_


def save_centroids_to_csv(centroids: np.ndarray, output_path: Path, rocksdb_format: bool = False) -> None:
    """Save centroids to CSV file in the format expected by MyRocks.

    Args:
        centroids: Centroids array of shape (num_centroids, dimension)
        output_path: Path to output CSV file
        rocksdb_format: If True, use RocksDB C++ format (no header, quoted arrays only).
                       If False, use SQL format (with id,centroid header).
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
            writer = csv.writer(f)
            writer.writerow(['id', 'centroid'])
            for i, centroid in enumerate(centroids):
                centroid_json = json.dumps(centroid.tolist())
                writer.writerow([i, centroid_json])

    print(f"✓ Successfully saved centroids to {output_path}")

    # Print file size
    file_size = output_path.stat().st_size
    print(f"  File size: {file_size / 1024:.2f} KB")


def validate_centroids(csv_path: Path) -> None:
    """Validate the generated centroids CSV file.

    Args:
        csv_path: Path to the CSV file to validate
    """
    print(f"\nValidating centroids file...")

    with csv_path.open('r') as f:
        reader = csv.DictReader(f)
        count = 0
        dimensions = set()

        for row in reader:
            count += 1
            centroid_id = int(row['id'])
            centroid_json = row['centroid']

            # Validate JSON
            try:
                centroid = json.loads(centroid_json)
                dimensions.add(len(centroid))
            except json.JSONDecodeError as e:
                print(f"  ✗ Invalid JSON in row {count}: {e}")
                return

            # Check ID matches row number
            if centroid_id != count - 1:
                print(f"  ✗ ID mismatch at row {count}: expected {count-1}, got {centroid_id}")
                return

    if len(dimensions) != 1:
        print(f"  ✗ Inconsistent dimensions: {dimensions}")
        return

    dimension = dimensions.pop()
    print(f"  ✓ Validation passed!")
    print(f"  ✓ Total centroids: {count}")
    print(f"  ✓ Dimension: {dimension}")


def main():
    parser = argparse.ArgumentParser(description='Generate centroids for MyRocks LSM vector index')
    parser.add_argument(
        '--dataset-path',
        type=Path,
        default='/tmp/vectordb_bench/dataset/openai/openai_small_50k/shuffle_train.parquet',
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
        default='/tmp/myrocks_centroids.csv',
        help='Output CSV file path'
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
        help='Output in RocksDB C++ format (no header, quoted arrays only) instead of SQL format'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MyRocks Centroid Generation")
    print("=" * 70)
    print(f"Dataset: {args.dataset_path}")
    print(f"Output: {args.output}")
    print(f"Number of centroids: {args.num_centroids}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print(f"Random state: {args.random_state}")
    print("=" * 70)

    # Check if dataset exists
    if not args.dataset_path.exists():
        print(f"\n✗ Error: Dataset file not found: {args.dataset_path}")
        return 1

    try:
        # Load vectors
        vectors = load_vectors_from_parquet(args.dataset_path, args.max_samples)

        # Generate centroids
        centroids = generate_centroids(vectors, args.num_centroids, args.random_state)

        # Create output directory if it doesn't exist
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        save_centroids_to_csv(centroids, args.output, rocksdb_format=args.rocksdb_format)

        # Validate the output (only for SQL format)
        if not args.rocksdb_format:
            validate_centroids(args.output)

        print("\n" + "=" * 70)
        print("✓ Centroid generation completed successfully!")
        print("=" * 70)
        print(f"\nNext steps:")
        print(f"1. Use the generated centroids file: {args.output}")
        print(f"2. Update myrocks.py optimize() method to load centroids:")
        print(f"   self._load_centroids_from_csv('{args.output}')")
        print(f"3. Uncomment the LSM index creation code in optimize()")
        print(f"4. Run the benchmark again to test with LSM index enabled")

        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
