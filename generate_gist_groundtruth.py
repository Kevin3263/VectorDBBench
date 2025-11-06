#!/usr/bin/env python3
"""
Generate ground truth neighbors for GIST dataset using L2 metric.
This script computes k-nearest neighbors for test queries against the train dataset.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import time

def l2_distance_batch(queries, data_batch):
    """
    Compute L2 distances between queries and a batch of data vectors.

    Args:
        queries: shape (num_queries, dim)
        data_batch: shape (batch_size, dim)

    Returns:
        distances: shape (num_queries, batch_size)
    """
    # Using broadcasting: (q - d)^2 = q^2 - 2*q*d + d^2
    q_squared = np.sum(queries ** 2, axis=1, keepdims=True)  # (num_queries, 1)
    d_squared = np.sum(data_batch ** 2, axis=1)  # (batch_size,)
    qd = queries @ data_batch.T  # (num_queries, batch_size)

    distances = q_squared - 2 * qd + d_squared
    return np.sqrt(np.maximum(distances, 0))  # Avoid negative due to numerical errors


def generate_ground_truth(
    train_file,
    test_file,
    output_file,
    k=100,
    train_batch_size=10000,
):
    """
    Generate ground truth k-nearest neighbors using L2 distance.

    Args:
        train_file: Path to train.parquet (1M vectors)
        test_file: Path to test.parquet (query vectors)
        output_file: Path to output neighbors.parquet
        k: Number of nearest neighbors to find
        train_batch_size: Number of train vectors to process at once
    """
    print(f"Loading test queries from {test_file}...")
    test_pf = pq.ParquetFile(test_file, memory_map=True)
    test_df = test_pf.read().to_pandas()
    test_vectors = np.array(test_df['emb'].tolist(), dtype=np.float32)
    test_ids = test_df['id'].values
    num_queries = len(test_vectors)
    dim = test_vectors.shape[1]

    print(f"Loaded {num_queries} test queries, dimension={dim}")

    print(f"Loading train data from {train_file}...")
    train_pf = pq.ParquetFile(train_file, memory_map=True)
    train_df = train_pf.read().to_pandas()
    train_vectors = np.array(train_df['emb'].tolist(), dtype=np.float32)
    train_ids = train_df['id'].values
    num_train = len(train_vectors)

    print(f"Loaded {num_train} train vectors")

    # Initialize arrays to store top-k neighbors for each query
    top_k_distances = np.full((num_queries, k), np.inf, dtype=np.float32)
    top_k_indices = np.full((num_queries, k), -1, dtype=np.int32)

    print(f"\nComputing L2 distances and finding top-{k} neighbors...")
    print(f"Processing {num_train} train vectors in batches of {train_batch_size}")

    start_time = time.time()

    # Process train data in batches
    num_batches = (num_train + train_batch_size - 1) // train_batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * train_batch_size
        end_idx = min(start_idx + train_batch_size, num_train)

        # Get current batch of train vectors
        data_batch = train_vectors[start_idx:end_idx]
        batch_ids = train_ids[start_idx:end_idx]

        # Compute distances for all queries to this batch
        distances = l2_distance_batch(test_vectors, data_batch)  # (num_queries, batch_size)

        # For each query, update top-k if needed
        for q_idx in range(num_queries):
            # Combine current top-k with new candidates
            combined_distances = np.concatenate([top_k_distances[q_idx], distances[q_idx]])
            combined_indices = np.concatenate([
                top_k_indices[q_idx],
                batch_ids
            ])

            # Get indices of top-k smallest distances
            top_k_idx = np.argpartition(combined_distances, k)[:k]

            # Sort the top-k
            sorted_idx = top_k_idx[np.argsort(combined_distances[top_k_idx])]

            top_k_distances[q_idx] = combined_distances[sorted_idx]
            top_k_indices[q_idx] = combined_indices[sorted_idx]

    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Average time per query: {elapsed_time / num_queries:.4f} seconds")

    # Convert to list format for VectorDBBench
    neighbors_list = [indices.tolist() for indices in top_k_indices]

    # Create output dataframe
    result_df = pd.DataFrame({
        'id': test_ids,
        'neighbors_id': neighbors_list
    })

    print(f"\nSaving ground truth to {output_file}...")
    pq.write_table(pa.Table.from_pandas(result_df), output_file)

    print(f"✓ Successfully generated ground truth for {num_queries} queries")
    print(f"  Each query has {k} nearest neighbors")
    print(f"  Output file: {output_file}")

    # Print sample results
    print(f"\nSample ground truth (first 3 queries):")
    for i in range(min(3, num_queries)):
        print(f"  Query {test_ids[i]}: top-5 neighbors = {top_k_indices[i][:5].tolist()}")
        print(f"                      distances = {top_k_distances[i][:5].tolist()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground truth for GIST dataset")
    parser.add_argument(
        "--train-file",
        default="/tmp/vectordb_bench/dataset/gist/gist_medium_1m/train.parquet",
        help="Path to train.parquet"
    )
    parser.add_argument(
        "--test-file",
        default="/tmp/vectordb_bench/dataset/gist/gist_medium_1m/test.parquet",
        help="Path to test.parquet"
    )
    parser.add_argument(
        "--output",
        default="/tmp/vectordb_bench/dataset/gist/gist_medium_1m/neighbors.parquet",
        help="Path to output neighbors.parquet"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of nearest neighbors to find (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Number of train vectors to process at once (default: 10000)"
    )

    args = parser.parse_args()

    generate_ground_truth(
        train_file=args.train_file,
        test_file=args.test_file,
        output_file=args.output,
        k=args.k,
        train_batch_size=args.batch_size,
    )
