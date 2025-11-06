import logging
import pathlib
import numpy as np
import polars as pl
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def compute_l2_neighbors(train_path: str, test_path: str, output_path: str, k: int = 1000):
    """Compute L2-based ground truth neighbors"""
    log.info(f"Reading train data from {train_path}")
    train_df = pl.read_parquet(train_path)
    train_vectors = np.array(train_df['emb'].to_list(), dtype=np.float32)
    train_ids = train_df['id'].to_numpy()
    
    log.info(f"Reading test data from {test_path}")
    test_df = pl.read_parquet(test_path)
    test_vectors = np.array(test_df['emb'].to_list(), dtype=np.float32)
    test_ids = test_df['id'].to_numpy()
    
    log.info(f"Computing L2 distances for {len(test_vectors)} queries against {len(train_vectors)} vectors")
    
    neighbors_list = []
    distances_list = []
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(test_vectors), batch_size), desc="Computing neighbors"):
        batch_end = min(i + batch_size, len(test_vectors))
        batch_queries = test_vectors[i:batch_end]
        
        # Compute L2 distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        query_norms = np.sum(batch_queries ** 2, axis=1, keepdims=True)
        train_norms = np.sum(train_vectors ** 2, axis=1, keepdims=True).T
        distances = query_norms + train_norms - 2 * np.dot(batch_queries, train_vectors.T)
        
        # Get top-k neighbors for each query
        for dist_row in distances:
            # Get indices of k smallest distances
            top_k_indices = np.argpartition(dist_row, k)[:k]
            # Sort the top-k by distance
            top_k_indices = top_k_indices[np.argsort(dist_row[top_k_indices])]
            
            neighbors_list.append(train_ids[top_k_indices].tolist())
            distances_list.append(dist_row[top_k_indices].tolist())
    
    # Create ground truth dataframe
    gt_df = pl.DataFrame({
        'id': test_ids,
        'neighbors_id': neighbors_list,
        'neighbors_dist': distances_list
    })
    
    log.info(f"Writing ground truth to {output_path}")
    gt_df.write_parquet(output_path)
    
    log.info(f"✅ L2 ground truth generated successfully")
    log.info(f"   Queries: {len(test_ids)}")
    log.info(f"   Neighbors per query: {k}")
    log.info(f"   File size: {pathlib.Path(output_path).stat().st_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    dataset_dir = "/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m"
    
    compute_l2_neighbors(
        train_path=f"{dataset_dir}/train.parquet",
        test_path=f"{dataset_dir}/test.parquet",
        output_path=f"{dataset_dir}/neighbors.parquet",
        k=1000
    )