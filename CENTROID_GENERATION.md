# Centroid Generation Guide for IVF Vector Index

## Overview

This document describes how to generate high-quality centroids for IVF (Inverted File) vector indexes using FAISS k-means clustering.

**Key Principle**: For IVF search, **low reconstruction loss = better recall**. Natural clustering (without forced balancing) produces the best search quality.

## Quick Start (Recommended for IVF)

**Production-ready approach** - optimized for IVF search quality:

```bash
python generate_centroids_faiss.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 256 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256.csv \
  --niter 100 \
  --nredo 5 \
  --training-sample-size 500000 \
  --rocksdb-format
```

**This approach**:
- ✅ Uses FAISS k-means (best reconstruction loss)
- ✅ Allows using more samples (better quality)
- ✅ Reports balance metrics for monitoring
- ✅ Does NOT force balance (better IVF recall)

**Expected output**:
- Reconstruction loss: 10-15 (excellent for IVF recall)
- Imbalance: 20-40% (natural, this is fine for IVF!)
- Time: 2-5 minutes on GPU

## Why Not Force Balance for IVF?

### The Trade-off

**Forced balancing hurts IVF search quality:**

| Metric | Natural (FAISS) | Forced Balance |
|--------|-----------------|----------------|
| Reconstruction Loss | **10-15** ✓ | 20-30 ⚠️ |
| Cluster Imbalance | 30-40% | <10% ✓ |
| **IVF Recall** | **Best** ✓✓ | **Lower** ⚠️ |

**Key insight**: Forcing equal-sized clusters means reassigning points to non-nearest centroids, which:
- Increases reconstruction loss (2× worse)
- Creates fuzzy cluster boundaries
- Causes more boundary cases during IVF search
- **Reduces recall by 5-10%** for the same `nprobe` value

**Recommendation**: Let clusters form naturally. Monitor imbalance but don't constrain it. IVF search doesn't require balanced clusters - it requires compact clusters (low loss).

### When You Might Want Balance

**Only force balance if**:
1. Operational constraints (memory limits, parallel query distribution)
2. Extreme imbalance (>100%, some clusters are empty)
3. Storage system requires it

**Otherwise**: Natural clustering is better for search quality.

## Alternative: Balanced Clustering (Lower Quality)

If you have specific operational requirements for balanced clusters, see `generate_centroids_optimized.py` which provides multiple algorithms:

### 1. Optimal Transport (Recommended)
**Algorithm**: `--algorithm optimal_transport`

Uses Sinkhorn algorithm for balanced soft assignment. This provides:
- **Best overall balance**: <10% imbalance typically
- **Good reconstruction loss**: Near-optimal quantization error
- **Moderate speed**: 2-5 minutes for 256 centroids

**When to use**: Default choice for production. Best balance of quality and speed.

### 2. Hungarian
**Algorithm**: `--algorithm hungarian`

Uses Hungarian algorithm for perfectly balanced hard assignment:
- **Perfect balance**: Exactly equal-sized clusters during training
- **Good loss**: Competitive reconstruction error
- **Slower**: 5-10 minutes for 256 centroids (chunk-based processing)

**When to use**: When perfect balance is critical and you have time.

### 3. FAISS (Loss-optimized)
**Algorithm**: `--algorithm faiss`

Standard FAISS k-means optimized for minimal reconstruction loss:
- **Best reconstruction loss**: Lowest quantization error
- **Variable balance**: 20-50% imbalance typical
- **Fastest**: 1-2 minutes for 256 centroids

**When to use**: When reconstruction quality matters more than balance.

### 4. Hybrid
**Algorithm**: `--algorithm hybrid`

Combines FAISS initialization with iterative balancing:
- **Balanced**: 10-20% imbalance
- **Good loss**: Better than pure balanced methods
- **Fast**: 2-4 minutes for 256 centroids

**When to use**: When you need good balance without sacrificing too much quality.

### 5. Compare Mode
**Algorithm**: `--algorithm compare`

Runs all algorithms and automatically selects the best based on combined score:
- Evaluates: Loss, balance, and overall quality
- Saves the best result
- Provides detailed comparison metrics

**When to use**: First time setup or when optimizing for a new dataset.

## Key Improvements Over Original

The optimized implementation provides:

1. **Explicit Loss Tracking**: All algorithms report reconstruction loss (quantization error)
2. **Better Balance**: Optimal transport and Hungarian methods achieve <10% imbalance
3. **Algorithm Choice**: Select the best tradeoff for your use case
4. **Verification**: Comprehensive metrics on full dataset
5. **Comparison Mode**: Benchmark all algorithms to find the best for your data

## Expected Results

### Optimal Transport (Recommended)
```
Loss: 15-25 (lower is better)
Imbalance: 5-10% (vs 20-30% for old method)
Time: 2-5 minutes
Distribution: 90% of clusters within ±10% of target size
```

### Hungarian (Perfect Balance)
```
Loss: 20-30
Imbalance: <5% (nearly perfect)
Time: 5-10 minutes
Distribution: 95% of clusters within ±5% of target size
```

### FAISS (Best Loss)
```
Loss: 10-15 (best)
Imbalance: 20-50%
Time: 1-2 minutes
Distribution: 70% of clusters within ±20% of target size
```

## Production Script: generate_centroids_faiss.py

### Features

- **FAISS k-means**: Industry-standard, optimized implementation
- **Configurable sampling**: Control how many vectors to use for training
- **Multiple restarts** (`nredo`): Runs k-means multiple times, selects best result
- **GPU acceleration**: Automatically uses GPU if available
- **Comprehensive metrics**: Reports reconstruction loss and balance metrics
- **Monitoring output**: Saves JSON metrics file for operational visibility

### Key Parameters

```bash
--num-centroids 256           # Number of clusters (256, 512, 1024 common)
--niter 100                   # K-means iterations (50-200)
--nredo 5                     # Random restarts - higher = better quality (3-10)
--training-sample-size 500000 # Vectors to use for training (500K-1M recommended)
--use-gpu                     # Enable GPU acceleration (default: True)
```

### Parameter Tuning for Quality

**For best quality** (higher is better):
- `--nredo 10`: More random restarts (takes longer, finds better centroids)
- `--training-sample-size 1000000`: Use more data (slower but better)
- `--niter 200`: More iterations (usually converges by 100)

**For speed** (faster but slightly lower quality):
- `--nredo 3`: Fewer restarts
- `--training-sample-size 300000`: Use less data
- `--niter 50`: Fewer iterations

**Recommended production settings**:
```bash
--nredo 5 --training-sample-size 500000 --niter 100
```
Good balance of quality and speed: ~3-5 minutes, loss ~12-15

### Output Files

1. **Centroids CSV**: The centroid vectors in RocksDB format
2. **Metrics JSON**: Monitoring data including:
   - Reconstruction loss (quantization error)
   - Cluster size distribution
   - Balance metrics (for monitoring)
   - Training time and configuration

### Example with Monitoring

```bash
python generate_centroids_faiss.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 256 \
  --output centroids.csv \
  --metrics-output centroids_metrics.json \
  --nredo 5 \
  --training-sample-size 500000
```

**Metrics output example**:
```json
{
  "timestamp": "2025-01-26 10:30:00",
  "full_dataset_metrics": {
    "avg_reconstruction_loss": 12.45,
    "imbalance_percent": 32.1,
    "distribution": {
      "90_to_110pct": 180,
      "below_80pct": 25,
      ...
    }
  }
}
```

Use this for:
- Monitoring centroid quality over time
- Detecting data distribution changes
- Alerting on extreme imbalance (>80%)

## Dependencies

### Required Packages

```bash
# Core dependencies for FAISS version
pip install numpy polars faiss-cpu

# For GPU acceleration (recommended)
pip install faiss-gpu

# For balanced algorithms (optional)
pip install scipy
```

### Old Version Dependencies

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install balanced k-means library
pip install balanced-kmeans

# Other dependencies (should already be installed)
pip install polars numpy
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 4090)
- **CUDA**: Version 12.1 or compatible
- **RAM**: At least 32GB recommended for 1M vectors
- **GPU Memory**: At least 8GB VRAM

## Testing the Optimization

Before running on the full dataset, you can test the algorithms on synthetic data:

```bash
python test_centroid_optimization.py
```

This will:
- Generate 50K synthetic vectors in 256 dimensions
- Run all algorithms with 128 centroids
- Compare loss, balance, and speed
- Show which algorithm is best for your use case

Expected output:
```
COMPARISON RESULTS
================================================================================
Algorithm                      Loss    Imbal%    Min    Max  StdDev   Time(s)
--------------------------------------------------------------------------------
FAISS (loss-optimized)       12.450    42.3%    245    556    82.3     8.45
Hybrid (balanced + loss)     14.230    15.2%    340    458    28.1    12.32
Optimal Transport            15.120     8.1%    365    420    18.7    15.67

Best Loss: FAISS (12.450)
Best Balance: Optimal Transport (8.1% imbalance)
Best Combined: Optimal Transport (score=0.421)
```

## Centroid Generation Scripts

### Optimized Version (Recommended)
**Location**: `/home/kunhao/VectorDBBench/generate_centroids_optimized.py`

**Algorithms**:
- **Optimal Transport**: Sinkhorn algorithm for balanced soft assignment
- **Hungarian**: Linear sum assignment for perfect balance
- **FAISS**: Standard k-means optimized for loss
- **Hybrid**: FAISS + iterative balancing

**Features**:
- Explicit reconstruction loss computation
- Better balance (<10% imbalance for optimal transport)
- Algorithm comparison mode
- Comprehensive verification metrics

### Original Version
**Location**: `/home/kunhao/VectorDBBench/generate_centroids_balanced.py`

**Algorithm**: FAISS with custom iterative rebalancing
- Enforces balanced clusters through reassignment
- Conservative rebalancing (30% per iteration)
- Good for 20-30% imbalance
- Produces centroids suitable for LSM indexing

## Usage Examples

### Optimized Version (Recommended)

#### Generate 256 Centroids with Optimal Transport

```bash
python generate_centroids_optimized.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 256 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256_optimal.csv \
  --algorithm optimal_transport \
  --max-iter 100 \
  --training-sample-size 300000 \
  --rocksdb-format
```

**Expected time**: 3-5 minutes
**Expected imbalance**: 5-10%
**Expected loss**: 15-25

#### Generate 512 Centroids with Comparison

```bash
python generate_centroids_optimized.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 512 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_512_best.csv \
  --algorithm compare \
  --max-iter 100 \
  --training-sample-size 300000 \
  --rocksdb-format
```

**Expected time**: 10-20 minutes (runs all algorithms)
**Result**: Automatically selects and saves the best algorithm's output

#### Generate with Perfect Balance (Hungarian)

```bash
python generate_centroids_optimized.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 256 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256_perfect.csv \
  --algorithm hungarian \
  --max-iter 100 \
  --training-sample-size 300000 \
  --rocksdb-format
```

**Expected time**: 8-12 minutes
**Expected imbalance**: <5%
**Expected loss**: 20-30

### Original Version

#### Generate 256 Centroids

```bash
python generate_centroids_balanced.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 256 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256_balanced.csv \
  --max-iter 300 \
  --training-sample-size 300000 \
  --rocksdb-format
```

**Expected time**: 5-10 minutes
**Expected imbalance**: 20-30%

#### Generate 512 Centroids

```bash
python generate_centroids_balanced.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 512 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_512_balanced.csv \
  --max-iter 300 \
  --training-sample-size 300000 \
  --rocksdb-format
```

**Expected time**: 8-15 minutes
**Expected imbalance**: 20-30%

## Parameters Explained

### Required Parameters (Both Versions)

- `--dataset-path`: Path to the parquet file containing training vectors
  - Format: Must contain an 'emb', 'vector', or 'embedding' column
  - Example: `/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet`

- `--num-centroids`: Number of cluster centroids to generate
  - Common values: 256, 512, 1024
  - Trade-off: More centroids = finer granularity but higher memory usage

- `--output`: Output file path for centroids
  - Format: CSV file in RocksDB format (quoted JSON arrays)
  - Example: `/home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256_balanced.csv`

### Algorithm Selection (Optimized Version Only)

- `--algorithm`: Choose the clustering algorithm
  - `optimal_transport`: Sinkhorn algorithm (recommended, best balance)
  - `hungarian`: Linear sum assignment (perfect balance, slower)
  - `faiss`: Standard k-means (best loss, variable balance)
  - `hybrid`: FAISS + balancing (good tradeoff)
  - `compare`: Run all and select best

### Optional Parameters

- `--max-iter`: Maximum k-means iterations (default: 300)
  - Higher values allow more convergence time
  - Typical convergence: 50-200 iterations

- `--training-sample-size`: Number of vectors to use for training (default: all)
  - Recommended: 200000-300000 for speed without quality loss
  - Using 30% of data (300K/1M) gives good results
  - Full dataset: slower but theoretically better (may not be necessary)

- `--rocksdb-format`: Output in RocksDB C++ format (default: True)
  - Format: One quoted JSON array per line, no header
  - Example: `"[0.154, 0.338, -0.033, ...]"`

- `--random-state`: Random seed for reproducibility (default: 42)

- `--batch-size`: Not used (kept for compatibility)

- `--init-sample-size`: Not used (kept for compatibility)

## Output Format

### RocksDB Format (Default)

```
"[0.154, 0.338, -0.033, ...]"
"[0.157, 0.093, -0.037, ...]"
...
```

- No header row
- Each line is a quoted JSON array
- Total lines = num_centroids
- File size: ~4MB for 256 centroids @ 768D, ~8MB for 512 centroids @ 768D

## Expected Output Statistics

### Training Data (e.g., 300K vectors, 256 centroids)

```
=== Cluster Statistics (Training Data: 300000 vectors) ===
  Min size: 1171
  Max size: 1172
  Mean size: 1171.9
  Std dev: 0.3
  Target size: 1171
  Imbalance: 0.1% (lower is better)
```

**Perfect balance on training data** (<1% imbalance)

### Full Dataset (e.g., 1M vectors, 256 centroids)

```
=== Verifying on FULL Dataset (1000000 vectors) ===
  Min size: 3500
  Max size: 4300
  Mean size: 3906.2
  Std dev: 180.5
  Target size: 3906
  Imbalance: 20.5% (lower is better)

  Cluster size distribution:
    <80% of target: 10 clusters
    80-90% of target: 35 clusters
    90-110% of target: 180 clusters  ← Most clusters here!
    110-120% of target: 25 clusters
    >120% of target: 6 clusters
```

**Reasonable balance on full dataset** (10-30% imbalance is acceptable for LSM indexes)

## Performance Notes

### GPU vs CPU

- **GPU (RTX 4090)**: 5-15 minutes for 256-512 centroids
- **CPU**: 30-60 minutes for 256-512 centroids
- Script automatically uses GPU if available, falls back to CPU

### Memory Usage

- **Training on 300K vectors**: ~3-4GB GPU memory
- **Training on 1M vectors**: ~10-12GB GPU memory
- Verification phase: Additional ~2GB RAM for full dataset

### Convergence

- Typical convergence: 50-150 iterations
- Convergence threshold: centroid shift < 0.0001
- Progress bar shows iterations if enabled

## Troubleshooting

### Out of Memory (GPU)

**Solution 1**: Reduce training sample size
```bash
--training-sample-size 200000  # Instead of 300000
```

**Solution 2**: Use CPU
```python
# Edit script to force CPU: device='cpu'
```

### Out of Memory (CPU RAM)

**Solution**: Reduce training sample size
```bash
--training-sample-size 100000
```

### Slow Performance

**Check GPU utilization**:
```bash
nvidia-smi  # Should show GPU usage during clustering
```

**Verify PyTorch uses GPU**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Poor Quality Centroids

**Symptoms**: Very high imbalance (>100%), poor recall metrics

**Solutions**:
1. Increase `max-iter` to 500
2. Use larger training sample (500K or full 1M)
3. Verify data quality (check for NaN, duplicate vectors)

## Integration with MyRocks

### Update C++ Configuration

Edit `rocksdb/table/block_based/block_based_table_factory.h`:

```cpp
// For 256 centroids
const size_t vector_dim = 768;
std::string centroids_path = std::string(SPATIAL_X_DB_ROOT) +
    "/vector_index_centroids/centroids_cohere_768d_256_balanced.csv";
```

Or for 512 centroids:

```cpp
// For 512 centroids
const size_t vector_dim = 768;
std::string centroids_path = std::string(SPATIAL_X_DB_ROOT) +
    "/vector_index_centroids/centroids_cohere_768d_512_balanced.csv";
```

### Rebuild MyRocks

```bash
cd /home/kunhao/spatial-x-db
# Rebuild with new centroids configuration
make -j$(nproc)
```

## File Locations

### Input
- **Dataset**: `/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet`
- **Dimension**: 768
- **Size**: 1M vectors

### Output
- **256 Centroids**: `/home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256_balanced.csv`
- **512 Centroids**: `/home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_512_balanced.csv`

### Scripts
- **Optimized Generator (Recommended)**: `/home/kunhao/VectorDBBench/generate_centroids_optimized.py`
- **Original Generator**: `/home/kunhao/VectorDBBench/generate_centroids_balanced.py`
- **Test/Comparison**: `/home/kunhao/VectorDBBench/test_centroid_optimization.py`

## Performance Comparison

### Optimized vs Original on 1M Vectors, 256 Centroids

| Metric | Original (FAISS Iterative) | Optimized (Optimal Transport) | Optimized (Hungarian) | Optimized (FAISS) |
|--------|---------------------------|-------------------------------|----------------------|------------------|
| **Imbalance** | 20-30% | **5-10%** ✓ | **<5%** ✓✓ | 20-50% |
| **Reconstruction Loss** | Not tracked | 15-25 | 20-30 | **10-15** ✓ |
| **Time** | 5-10 min | 3-5 min ✓ | 8-12 min | **1-2 min** ✓✓ |
| **Best For** | Baseline | **Production (recommended)** | Perfect balance needed | Lowest loss needed |

**Recommendation**: Use Optimal Transport algorithm for best overall balance of quality, speed, and balance.

## Historical Notes

### Evolution of Implementation

1. **Initial attempt**: Custom balanced k-means with greedy assignment
   - Problem: Slow convergence, oscillating centroids
   - Time: ~2 hours for 256 centroids

2. **sklearn MiniBatchKMeans**: Standard mini-batch k-means
   - Problem: Poor quality (501% imbalance), too fast (suspicious)
   - Time: 4 seconds (too fast = poor quality)

3. **FAISS with iterative rebalancing** (`generate_centroids_balanced.py`)
   - Result: Good balance on training data, 20-30% imbalance on full dataset
   - Time: 5-15 minutes
   - Quality: Acceptable for LSM indexes but no loss tracking

4. **Optimized multi-algorithm approach** (`generate_centroids_optimized.py`) ✨ **NEW**
   - Algorithms: Optimal Transport, Hungarian, FAISS, Hybrid
   - Result: 5-10% imbalance with explicit loss optimization
   - Time: 3-5 minutes (faster!)
   - Quality: Better balance + lower loss + comprehensive metrics

### Key Learnings

- **Sampling is acceptable**: 30% of data (300K/1M) gives good centroids
- **Balance vs Loss tradeoff**: Different algorithms optimize different objectives
- **Optimal Transport wins**: Best overall balance of imbalance (<10%) and loss
- **Hungarian for perfection**: When you need <5% imbalance and can wait
- **FAISS for speed**: When loss matters more than balance (1-2 minutes)
- **Metrics matter**: Tracking both balance AND reconstruction loss is critical
- **Comparison mode**: Running all algorithms once helps identify the best for your data

## Recommendations

### For IVF Vector Indexes (Recommended)

**Use `generate_centroids_faiss.py`** with these settings:

```bash
python generate_centroids_faiss.py \
  --dataset-path <your_data>.parquet \
  --num-centroids 256 \
  --output centroids.csv \
  --nredo 5 \
  --training-sample-size 500000 \
  --niter 100
```

**Why**:
- ✅ Best IVF search recall (lowest reconstruction loss)
- ✅ Fast (3-5 minutes)
- ✅ Proven at scale (FAISS used in production worldwide)
- ✅ Monitoring built-in (metrics JSON output)
- ⚠️ Imbalance 20-40% (this is fine for IVF!)

**Key metrics to monitor**:
- **Reconstruction loss**: Target <15, alert if >25
- **Imbalance**: Expect 20-50%, alert if >80% (indicates data issues)
- **Empty clusters**: Should be 0, alert if >0

### For Balanced Clusters (Lower Quality)

**Only if you have operational requirements**, use `generate_centroids_optimized.py`:

1. **Moderate balance** (15-25% imbalance): `--algorithm hybrid --balance-weight 0.3`
2. **Strong balance** (5-10% imbalance): `--algorithm optimal_transport`
3. **Perfect balance** (<5% imbalance): `--algorithm hungarian` (slowest, lowest quality)

**Trade-off**: Every 10% less imbalance ≈ 2-3 points higher loss ≈ 1-2% lower recall

### Monitoring and Maintenance

- **Monitor metrics**: Track reconstruction loss and imbalance over time
- **Regenerate periodically**: Every 3-6 months or when data distribution changes significantly
- **Alert thresholds**:
  - Loss >25: Data distribution may have changed, consider regenerating
  - Imbalance >80%: Extreme skew, investigate data quality
  - Empty clusters >0: Bad initialization, regenerate with different random seed

## References

- **FAISS library**: https://github.com/facebookresearch/faiss
- **SciPy (Optimal Transport)**: https://scipy.org/
- **VectorDBBench**: https://github.com/zilliztech/VectorDBBench
- **MyRocks implementation**: See `MYROCKS_IMPLEMENTATION.md`
- **Optimal Transport**: Cuturi, M. (2013). Sinkhorn Distances
- **Hungarian Algorithm**: Kuhn, H. W. (1955). The Hungarian method
