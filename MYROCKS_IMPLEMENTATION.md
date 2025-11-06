# MyRocks Client Implementation for VectorDBBench

## Overview
This document describes the implementation of a MyRocks database client for VectorDBBench, enabling benchmarking of MyRocks' vector search capabilities using Facebook's vector extensions.

**Quick Summary:**
- ✅ Fully functional MyRocks client with LSM vector index support
- ✅ Configured for Cohere 768D dataset
- ✅ L2-based ground truth generated for valid recall metrics
- ✅ Centroids generated (256 and 512 clusters)
- ✅ Error 505 bug fixed (field index mismatch resolved)
- ✅ nprobe parameter support added (configurable search radius)
- ⚠️ Only L2 distance metric currently supported

## Current Environment (October 24-25, 2025)

### System Configuration
- **Location**: `/home/kunhao/`
- **Python**: 3.11.14
- **VectorDBBench**: `/home/kunhao/VectorDBBench`
- **VectorDBBench Conda Env**: `vectordbbench` (clean install as of Oct 24)
- **spatial-x-db**: `/home/kunhao/spatial-x-db`
- **MyRocks Runtime**: `/home/kunhao/myrocks-runtime`
- **MySQL Socket**: `/home/kunhao/myrocks-runtime/mysql.sock`
- **MySQL Password**: `150131`

### Python Environment
- **NumPy**: 2.3.4
- **Pandas**: 2.3.3
- **PyArrow**: 21.0.0 (clean install, compatible)
- **MySQL Connector**: 9.5.0

### Supported Datasets

#### Cohere 768D (L2 Ground Truth)
- **Dimension**: 768D
- **Size**: 1M vectors
- **Metric**: L2 (converted from COSINE)
- **Location**: `/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/`
- **Ground Truth**: ✅ L2-based (4.7 MB, 1000 queries × 1000 neighbors)

#### GIST 960D (L2 Metric)
- **Dimension**: 960D
- **Size**: 1M vectors
- **Metric**: L2
- **Location**: `/tmp/vectordb_bench/dataset/gist/gist_medium_1m/`
- **Downloaded**: ✅ Oct 26, 2025 (2.5 GB)

#### Available Centroids

**Standard k-means (sklearn)**:
- **256 clusters**: `centroids_cohere_768d_256.csv` (4.0 MB, Oct 22)
- **512 clusters**: `centroids_cohere_768d_512.csv` (8.0 MB, Oct 23)

**Balanced k-means (GPU-accelerated, PyTorch)**:
- **256 clusters**: `centroids_cohere_768d_256_balanced.csv` (4.0 MB, Oct 24 16:17)
  - Perfect balance on training data (<1% imbalance)
  - 10-30% imbalance on full dataset (acceptable for LSM)
- **512 clusters**: `centroids_cohere_768d_512_balanced.csv` (8.0 MB, Oct 24 16:28)
  - Generated using balanced-kmeans library
  - Trained on 300K sample vectors
  - Format: RocksDB (quoted JSON arrays)

**GIST 960D (FAISS k-means)**:
- **256 clusters**: `centroids_gist_960d_256.csv` (5.0 MB, Oct 26 15:28)
  - Generated using FAISS k-means (optimized for IVF)
  - Reconstruction loss: 1.19 (excellent for IVF recall)
  - Trained on full 1M vectors
  - Format: RocksDB (quoted JSON arrays)

**Current Active**: Cohere 512 balanced (`centroids_cohere_768d_512_balanced.csv`)

## VectorDBBench Modifications

### Files Modified

#### 1. `vectordb_bench/backend/dataset.py`
Updated dataset configurations:
- Line 159: Changed Cohere `metric_type: MetricType = MetricType.L2`

#### 2. `vectordb_bench/backend/clients/myrocks/config.py`
- Changed connection from TCP (host:port) to Unix socket
- Hardcoded socket path: `/home/kunhao/myrocks-runtime/mysql.sock`
- Added `nprobe: int = 16` parameter to `MyRocksLSMConfig`
- Added nprobe to `search_param()` method

#### 3. `vectordb_bench/backend/clients/myrocks/cli.py`
- Removed `--host` and `--port` CLI options
- Made `--username` optional (default: `root`)
- Added `--nprobe` CLI option (default: 16)
- CLI now requires only `--password`

#### 4. `vectordb_bench/backend/clients/myrocks/myrocks.py`
- Added code in `init()` method to set `fb_vector_search_nprobe` session variable
- Reads nprobe value from `case_config.search_param()`

## MyRocks C++ Configuration

### Critical Files

#### File 1: `storage/rocksdb/rdb_vector_db.cc` (Lines 1255, 1520)
**Purpose**: LSM vector index search implementation

**Configuration**:
```cpp
std::vector<size_t> field_indexes_to_extract = {0};  // First non-PK column
```

**Why Critical**: Must match table schema `(id INT PRIMARY KEY, v JSON)`
- Field index `{0}` = first non-primary-key column (vector column 'v')
- Mismatch causes error 505 "Found data corruption"

#### File 2: `rocksdb/table/block_based/block_based_table_factory.h` (Lines 103, 108)
**Purpose**: RocksDB configuration, loads centroids at C++ level

**Current Configuration** (Cohere 768D, 512 Balanced Centroids):
```cpp
const size_t vector_dim = 768;  // Vector dimension

std::string centroids_path = std::string(SPATIAL_X_DB_ROOT) +
    "/vector_index_centroids/centroids_cohere_768d_512_balanced.csv";
```

**Previous Configurations**:
- 256 standard: `centroids_cohere_768d_256.csv`
- 512 standard: `centroids_cohere_768d_512.csv`
- 256 balanced: `centroids_cohere_768d_256_balanced.csv`

**Important**: After changing centroids path, MyRocks must be rebuilt:
```bash
cd /home/kunhao/spatial-x-db
make -j$(nproc)
# Then restart MySQL server
```

## Centroid Generation

### Standard k-means: `generate_centroids.py`

**Location**: `/home/kunhao/VectorDBBench/generate_centroids.py`

**Usage Example**:

```bash
# Cohere 768D (256 clusters)
python generate_centroids.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 256 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256.csv \
  --max-samples 100000 \
  --rocksdb-format
```

### Balanced k-means: `generate_centroids_balanced.py`

**Location**: `/home/kunhao/VectorDBBench/generate_centroids_balanced.py`

**Documentation**: See `CENTROID_GENERATION.md` for detailed guide

**Usage Example**:

```bash
# Cohere 768D (256 balanced clusters)
python generate_centroids_balanced.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 256 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_256_balanced.csv \
  --max-iter 300 \
  --training-sample-size 300000 \
  --rocksdb-format

# Cohere 768D (512 balanced clusters)
python generate_centroids_balanced.py \
  --dataset-path /tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/train.parquet \
  --num-centroids 512 \
  --output /home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_512_balanced.csv \
  --max-iter 300 \
  --training-sample-size 300000 \
  --rocksdb-format
```

**Benefits of Balanced k-means**:
- Enforces equal-sized clusters (±1 vector on training data)
- GPU-accelerated (5-15 minutes vs 30-60 minutes on CPU)
- Better cluster balance leads to more consistent search performance
- Uses PyTorch balanced-kmeans library

**Output Format** (RocksDB):
```
"[0.154, 0.338, -0.033, ...]"
"[0.157, 0.093, -0.037, ...]"
...
```

## Running Benchmarks

### Cohere 768D - Basic
```bash
python -m vectordb_bench.cli.vectordbbench myrocks \
  --password 150131 \
  --case-type Performance768D1M \
  --db-label "myrocks-cohere-768d-l2"
```

### Cohere 768D - With nprobe Configuration
```bash
# nprobe=16 (default, high recall)
python -m vectordb_bench.cli.vectordbbench myrocks \
  --password 150131 \
  --case-type Performance768D1M \
  --db-label "myrocks-512c-nprobe16" \
  --nprobe 16

# nprobe=8 (balanced)
python -m vectordb_bench.cli.vectordbbench myrocks \
  --password 150131 \
  --case-type Performance768D1M \
  --db-label "myrocks-512c-nprobe8" \
  --nprobe 8

# nprobe=4 (high throughput)
python -m vectordb_bench.cli.vectordbbench myrocks \
  --password 150131 \
  --case-type Performance768D1M \
  --db-label "myrocks-512c-nprobe4" \
  --nprobe 4
```

### Common Options
- `--drop-old`: Drop existing database
- `--load`: Run data loading phase
- `--skip-search-serial`: Skip serial search
- `--skip-search-concurrent`: Skip concurrent search
- `--skip-load`: Skip loading dataset (necessary for search benchmark)
- `--num-concurrency "1,5,10"`: Limit concurrency levels (recommended for OOM avoidance)
- `--nprobe <N>`: Number of nearest centroids to search (1-10000, default: 16)

## Database Schema

```sql
-- Vector table
CREATE TABLE vec_collection (
    id INT PRIMARY KEY,
    v JSON NOT NULL FB_VECTOR_DIMENSION 768
) ENGINE=ROCKSDB;

-- LSM vector index (created during optimize phase)
ALTER TABLE vec_collection
ADD INDEX v_idx(v) FB_VECTOR_INDEX_TYPE 'lsmidx';
```

## Benchmark Results Summary

### Load Phase (Cohere 768D, 1M vectors)

#### 256 Centroids
- **Insert Duration**: 1244.21s (~20.7 min)
- **Insert Rate**: ~804 vectors/second
- **Optimize Duration**: 0.0018s

#### 512 Centroids
- **Insert Duration**: 1199.11s (~20.0 min)
- **Insert Rate**: ~834 vectors/second
- **Optimize Duration**: 0.0016s

### Search Phase Results (Cohere 768D, 1M vectors)

#### 256 Centroids, nprobe=16 (default)
- **Serial QPS**: 5.91
- **Serial P99 Latency**: 1.42s
- **Serial P95 Latency**: 1.38s
- **Recall@100**: 96.57%
- **NDCG@100**: 96.91%
- **Concurrent (c=1)**: QPS=0.82, P99=1.35s, Avg=1.22s
- **Concurrent (c=5)**: QPS=3.69, P99=1.54s, Avg=1.35s
- **Concurrent (c=10)**: QPS=5.91, P99=2.34s, Avg=1.65s

#### 512 Centroids, nprobe=16 (default)
- **Serial QPS**: 7.76
- **Serial P99 Latency**: 1.11s
- **Serial P95 Latency**: 1.06s
- **Recall@100**: 94.60%
- **NDCG@100**: 95.07%
- **Concurrent (c=1)**: QPS=1.04, P99=1.24s, Avg=0.96s
- **Concurrent (c=5)**: QPS=4.76, P99=1.30s, Avg=1.04s
- **Concurrent (c=10)**: QPS=7.76, P99=2.02s, Avg=1.27s

#### 512 Centroids, nprobe=8
- **Serial QPS**: 11.25
- **Serial P99 Latency**: 0.96s
- **Serial P95 Latency**: 0.91s
- **Recall@100**: 87.21%
- **NDCG@100**: 88.18%
- **Concurrent (c=1)**: QPS=0.13, P99=10.72s, Avg=7.84s
- **Concurrent (c=5)**: QPS=4.70, P99=4.31s, Avg=1.05s
- **Concurrent (c=10)**: QPS=11.25, P99=1.48s, Avg=0.88s

#### 512 Centroids, nprobe=4
- **Serial QPS**: 19.76
- **Serial P99 Latency**: 0.55s
- **Serial P95 Latency**: 0.52s
- **Recall@100**: 75.90%
- **NDCG@100**: 77.42%
- **Concurrent (c=1)**: QPS=2.50, P99=0.54s, Avg=0.40s
- **Concurrent (c=5)**: QPS=11.77, P99=0.60s, Avg=0.42s
- **Concurrent (c=10)**: QPS=19.76, P99=0.83s, Avg=0.50s

## Dataset Status

| Dataset | Dimension | Metric | Ground Truth | Centroids | Status |
|---------|-----------|--------|--------------|-----------|--------|
| Cohere  | 768D      | L2     | ✅ Generated  | ✅ 256, 512 (std & balanced) | ✅ Ready |
| GIST    | 960D      | L2     | ⚠️ TBD       | ✅ 256 (FAISS) | ✅ Ready |

**Current Active Configuration**: Cohere 512 balanced centroids

**Notes**:
- Cohere dataset has L2-based ground truth, providing valid recall/NDCG metrics for MyRocks benchmarking
- GIST dataset downloaded (Oct 26, 2025), centroids generated using FAISS k-means optimized for IVF
- Both standard k-means and balanced k-means centroids available for Cohere
- See `CENTROID_GENERATION.md` for centroid generation guide

## Important Notes

### LSM Index Requirements
1. **Proper centroids**: Generated via k-means clustering on training data
2. **C++ level loading**: Centroids loaded at RocksDB initialization
3. **Field index alignment**: Must match table schema (currently `{0}`)
4. **Dimension match**: `vector_dim` must match dataset dimension
5. **Rebuild required**: Any C++ changes require recompilation

### nprobe Parameter
- **Purpose**: Controls number of nearest centroids to search
- **Range**: 1-10000 (default: 16)
- **Trade-off**: Lower nprobe = faster search, lower recall; Higher nprobe = slower search, higher recall
- **Session variable**: `fb_vector_search_nprobe`
- **CLI option**: `--nprobe <N>` in VectorDBBench

### Limitations
- **Distance metric**: L2 only (COSINE/IP not supported)
- **Connection**: Unix socket only in this environment
- **Hardcoded configuration**: Dimension and centroid path require C++ changes

### Storage Requirements
**1M Vectors (Cohere 768D)**:
- ~12GB RocksDB data
- ~25GB total disk usage

**Peak Memory**: 50GB+ during concurrent search (concurrency 20)

## Quick Reference

### File Locations
- **VectorDBBench**: `/home/kunhao/VectorDBBench`
- **spatial-x-db**: `/home/kunhao/spatial-x-db`
- **MySQL Socket**: `/home/kunhao/myrocks-runtime/mysql.sock`
- **Cohere Dataset**: `/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/`
- **Active Centroids**: `/home/kunhao/spatial-x-db/vector_index_centroids/centroids_cohere_768d_512_balanced.csv`
- **All Centroids**: `/home/kunhao/spatial-x-db/vector_index_centroids/`

### Key Configuration Values
- **Vector Dimension**: 768
- **Active Centroids**: 512 (balanced k-means)
- **Field Index**: 0
- **MySQL Password**: 150131
- **Table Structure**: `(id INT PRIMARY KEY, v JSON NOT NULL FB_VECTOR_DIMENSION 768)`
- **nprobe Range**: 1-10000 (recommended: 4-16)

### Quick Commands

**Rebuild MyRocks after changing centroids**:
```bash
cd /home/kunhao/spatial-x-db
make -j$(nproc)
# Kill and restart MySQL server
```

**Load data**:
```bash
python -m vectordb_bench.cli.vectordbbench myrocks \
  --password 150131 \
  --case-type Performance768D1M \
  --db-label "myrocks-512bal" \
  --drop-old --load \
  --skip-search-serial --skip-search-concurrent
```

**Search benchmark (nprobe sweep)**:
```bash
# Load once, then search with different nprobe values
for nprobe in 4 8 16; do
  python -m vectordb_bench.cli.vectordbbench myrocks \
    --password 150131 \
    --case-type Performance768D1M \
    --db-label "myrocks-512bal-nprobe${nprobe}" \
    --nprobe $nprobe \
    --skip-load \
    --num-concurrency "1,5,10"
done
```

## Troubleshooting

### Issue 1: Zero Recall After Loading Data (Oct 24, 2025)

**Symptoms**:
- Benchmark runs successfully
- Search queries execute
- Recall = 0.0, NDCG = 0.0
- FB_VECTOR_L2 returns 0 results

**Root Cause**:
MyRocks was not rebuilt after updating the centroids path in C++ configuration.

**Solution**:
1. Verify centroids file exists at the configured path
2. Rebuild MyRocks: `cd /home/kunhao/spatial-x-db && make -j$(nproc)`
3. Restart MySQL server
4. Drop and reload the database
5. Verify with manual query:
```bash
python -c "
import mysql.connector
import json

conn = mysql.connector.connect(
    user='root',
    password='150131',
    unix_socket='/home/kunhao/myrocks-runtime/mysql.sock'
)

cursor = conn.cursor()
cursor.execute('USE vectordbbench')
cursor.execute('SET SESSION fb_vector_search_nprobe = 16')

# Get first vector
cursor.execute('SELECT v FROM vec_collection WHERE id = 0')
vec = cursor.fetchone()[0]

# Test search
cursor.execute(f\"SELECT id FROM vec_collection ORDER BY FB_VECTOR_L2(v, '{vec}') ASC LIMIT 10\")
results = cursor.fetchall()
print(f'Results: {len(results)} (should be 10)')
"
```

**Expected**: 10 results with ID 0 first

### Issue 2: PyArrow Compatibility Error (Oct 24, 2025)

**Symptoms**:
```
TypeError: Cannot convert numpy.ndarray to numpy.ndarray
```

**Root Cause**:
PyArrow 21.0.0 was initially incompatible with the environment due to conflicting packages from PyTorch/balanced-kmeans installation.

**Solution**:
1. Remove conda environment: `conda env remove -n vectordbbench -y`
2. Create fresh environment: `conda create -n vectordbbench python=3.11 -y`
3. Install VectorDBBench: `pip install -e .`
4. Install MySQL connector: `pip install mysql-connector-python`
5. Verify PyArrow/Pandas compatibility:
```python
from pyarrow.parquet import ParquetFile
pf = ParquetFile('test.parquet', memory_map=True, pre_buffer=True)
batch = next(pf.iter_batches(100))
df = batch.to_pandas()  # Should succeed
```

**Key Packages (Working)**:
- NumPy: 2.3.4
- Pandas: 2.3.3
- PyArrow: 21.0.0
- MySQL Connector: 9.5.0

### Issue 3: Slow Query Performance with High nprobe

**Symptoms**:
- nprobe=32: ~1.1 seconds per query
- Serial search takes ~18 minutes for 1000 queries

**Explanation**:
This is expected behavior. Higher nprobe values search more centroids:
- nprobe=4: ~0.55s/query, 75.90% recall
- nprobe=8: ~0.96s/query, 87.21% recall
- nprobe=16: ~1.11s/query, 94.60% recall (512c)
- nprobe=32: ~1.10s/query, higher recall expected

**Recommendation**: Use nprobe=4,8,16 for most benchmarks. nprobe=32 is rarely needed.

## References
- VectorDBBench: https://github.com/zilliztech/VectorDBBench
- Cohere Dataset: 768D, 1M vectors, L2 ground truth
- Balanced k-means: https://pypi.org/project/balanced-kmeans/
- CENTROID_GENERATION.md: Detailed balanced centroid generation guide
- PGVECTOR_LOCAL_SETUP.md: PostgreSQL with pgvector setup guide

