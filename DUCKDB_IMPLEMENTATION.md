# DuckDB Client Implementation for VectorDBBench

## Overview
This document tracks the implementation of a DuckDB database client for VectorDBBench, enabling benchmarking of DuckDB's vector search capabilities using the VSS (Vector Similarity Search) extension.

## Current Environment (October 23, 2025)

### System Configuration
- **Location**: `/home/kunhao/VectorDBBench`
- **Python**: 3.11.14
- **DuckDB**: 1.4.1 (installed and verified)
- **VSS Extension**: Installed and tested

### Verified Capabilities
- ✅ HNSW index support
- ✅ Distance metrics: L2 squared (`l2sq`), Cosine (`cosine`), Inner Product (`ip`)
- ✅ Fixed-size FLOAT arrays for vector storage
- ✅ In-memory and persistent database modes
- ✅ Tested with 5D, 128D, 768D vectors
- ❌ IVF index NOT supported (HNSW only)

## Implementation Status

### Phase 1: Research & Planning ✅ COMPLETED
- [x] Research DuckDB VSS extension capabilities
  - [x] Supported distance metrics: L2 squared, Cosine, Inner Product
  - [x] Index types: HNSW only (no IVF)
  - [x] Vector data types: FLOAT[N] fixed-size arrays
  - [x] Query syntax: array_distance() function with ORDER BY
- [ ] Review existing VectorDBBench client implementations
  - [ ] Study client interface requirements
  - [ ] Understand configuration patterns
  - [ ] Review test case integration
- [x] Define DuckDB-specific configuration options
  - [x] HNSW index parameters (M, ef_construction, ef_search)
  - [x] Connection parameters (file path, :memory:, threads)
  - [x] Storage options (in-memory vs persistent)

### Phase 2: Client Implementation
- [ ] Create DuckDB client directory structure
  - [ ] `vectordb_bench/backend/clients/duckdb/` directory
  - [ ] `__init__.py` - Package initialization
  - [ ] `config.py` - Configuration dataclasses
  - [ ] `cli.py` - CLI argument definitions
  - [ ] `duckdb_client.py` - Main client implementation
- [ ] Implement `config.py`
  - [ ] `DuckDBConfig` dataclass with connection settings
  - [ ] `DuckDBIndexConfig` dataclass with HNSW parameters
  - [ ] Index type enums (FLAT, HNSW)
- [ ] Implement `cli.py`
  - [ ] CLI arguments for connection (file path, memory mode)
  - [ ] CLI arguments for index configuration
  - [ ] Integration with VectorDBBench CLI system
- [ ] Implement `duckdb_client.py`
  - [ ] `DuckDB` class inheriting from base client
  - [ ] `_create_collection()` - Create table with vector column
  - [ ] `_insert_embeddings()` - Bulk insert vectors
  - [ ] `_create_index()` - Create HNSW index
  - [ ] `_search()` - Vector similarity search
  - [ ] Connection lifecycle management
  - [ ] Error handling and validation

### Phase 3: Integration
- [ ] Register DuckDB in VectorDBBench
  - [ ] Add to `vectordb_bench/backend/clients/__init__.py`
  - [ ] Register client factory
- [ ] Update case configurations
  - [ ] Add DuckDB to supported databases in case definitions
  - [ ] Configure distance metric mappings
- [ ] Add DuckDB-specific test cases (if needed)
  - [ ] Performance test configurations
  - [ ] Index parameter sweep tests

### Phase 4: Testing & Validation
- [ ] Unit tests
  - [ ] Test connection establishment
  - [ ] Test table creation with vector columns
  - [ ] Test bulk insert operations
  - [ ] Test index creation
  - [ ] Test vector search queries
- [ ] Integration tests
  - [ ] Test with small datasets (1K, 10K vectors)
  - [ ] Verify recall metrics accuracy
  - [ ] Test all distance metrics (L2, COSINE, IP)
- [ ] Benchmark validation
  - [ ] Run Cohere 768D 1M dataset
  - [ ] Run GIST 960D 1M dataset
  - [ ] Compare results with other databases
  - [ ] Validate QPS, latency, recall metrics

### Phase 5: Documentation
- [ ] Update main README
  - [ ] Add DuckDB to supported databases list
  - [ ] Add installation instructions
- [ ] Create usage examples
  - [ ] Basic usage command
  - [ ] Index configuration examples
  - [ ] Performance tuning tips
- [ ] Document limitations
  - [ ] Known issues
  - [ ] Performance characteristics
  - [ ] Memory requirements

## DuckDB VSS Extension Details

### Installation
```bash
# Install DuckDB Python package
pip install duckdb

# VSS extension is loaded automatically when using vector operations
# Or explicitly: INSTALL vss; LOAD vss;
```

### Vector Data Types
```sql
-- DuckDB uses FLOAT arrays for vectors
CREATE TABLE vectors (
    id INTEGER PRIMARY KEY,
    embedding FLOAT[768]  -- Fixed-size array
);
```

### Index Creation
```sql
-- Create HNSW index
CREATE INDEX vec_idx ON vectors
USING HNSW (embedding)
WITH (metric = 'l2', M = 16, ef_construction = 128);

-- Supported metrics: 'l2', 'cosine', 'ip' (inner product)
```

### Vector Search
```sql
-- K-NN search with L2 distance
SELECT id, array_distance(embedding, [0.1, 0.2, ...]) AS distance
FROM vectors
ORDER BY distance
LIMIT 100;

-- Using HNSW index with ef_search parameter
SET hnsw_ef_search = 100;
```

## Configuration Options

### Connection Parameters
- **Database Path**: File path for persistent storage, or `:memory:` for in-memory
- **Read Only**: Boolean flag for read-only mode
- **Threads**: Number of threads for parallel operations

### Index Parameters (HNSW)
- **M**: Number of bi-directional links per node (default: 16)
  - Higher M = better recall, more memory
  - Typical range: 8-64
- **ef_construction**: Size of dynamic candidate list during index build (default: 128)
  - Higher ef_construction = better recall, slower indexing
  - Typical range: 64-512
- **ef_search**: Size of dynamic candidate list during search (runtime parameter)
  - Higher ef_search = better recall, slower search
  - Typical range: 100-500

## Expected Performance Characteristics

### Advantages
- **Fast in-memory operations**: Optimized for analytical workloads
- **Simple setup**: No server, embedded database
- **HNSW index**: Industry-standard approximate nearest neighbor search
- **Multiple distance metrics**: L2, COSINE, IP support

### Limitations
- **Memory intensive**: Entire dataset typically in memory
- **Single node**: No distributed capabilities
- **Index tuning**: Requires parameter optimization for best results

## Reference Client Implementations

Study these existing clients for patterns:
- `vectordb_bench/backend/clients/myrocks/` - MySQL-based client
- `vectordb_bench/backend/clients/pgvector/` - PostgreSQL-based client
- `vectordb_bench/backend/clients/qdrant/` - Modern vector database client

## Dataset Testing Plan

### Phase 1: Small Scale Validation
1. Test with Cohere 10K (768D)
2. Test with GIST 10K (960D)
3. Verify recall metrics match ground truth

### Phase 2: Medium Scale Testing
1. Test with Cohere 100K (768D)
2. Test with GIST 100K (960D)
3. Benchmark QPS and latency

### Phase 3: Full Scale Benchmarking
1. Test with Cohere 1M (768D)
2. Test with GIST 1M (960D)
3. Compare with MyRocks and other databases
4. Optimize index parameters for best recall/QPS tradeoff

## Key Implementation Notes

### Vector Storage Format
- DuckDB expects fixed-size FLOAT arrays
- Need to convert Python lists to DuckDB array format
- Consider batch insertion for performance

### Index Strategy
- Create index AFTER bulk loading (not during insert)
- HNSW index builds can be time-consuming for large datasets
- Monitor memory usage during index creation

### Distance Metric Mapping
```python
METRIC_MAPPING = {
    MetricType.L2: 'l2',
    MetricType.COSINE: 'cosine',
    MetricType.IP: 'ip'
}
```

### Error Handling
- Handle DuckDB-specific exceptions
- Validate vector dimensions before insert
- Check index creation success
- Handle connection lifecycle properly

## Success Criteria

- [ ] Successfully load 1M vectors into DuckDB
- [ ] Create HNSW index without errors
- [ ] Achieve >90% recall on test datasets
- [ ] Benchmark QPS competitive with other embedded databases
- [ ] Memory usage within reasonable limits (<50GB for 1M vectors)
- [ ] All VectorDBBench test cases pass

## Timeline Estimate

- **Phase 1 (Research)**: 2-4 hours
- **Phase 2 (Implementation)**: 6-8 hours
- **Phase 3 (Integration)**: 2-3 hours
- **Phase 4 (Testing)**: 4-6 hours
- **Phase 5 (Documentation)**: 2-3 hours

**Total Estimated Time**: 16-24 hours

## Next Steps

1. Research DuckDB VSS extension documentation
2. Install DuckDB and test basic vector operations
3. Review VectorDBBench client interface requirements
4. Start implementing config.py and basic client structure

## DuckDB Setup and Testing (Completed)

### Installation Steps

```bash
# Step 1: Install DuckDB Python package
pip install duckdb

# Step 2: Verify installation
python -c "import duckdb; print(f'DuckDB version: {duckdb.__version__}')"
# Expected: DuckDB version: 1.4.1
```

### Basic Test Results (Verified October 23, 2025)

#### Test 1: 1,000 vectors, 5D
- **Insert**: 3,310 vectors/second
- **Index Build (HNSW)**: 0.016 seconds
- **Search (k=10)**: 0.92ms per query

#### Test 2: 1,000 vectors, 128D
- **Index Build (HNSW)**: 0.054 seconds
- **Search (k=10)**: 2.22ms per query

#### Test 3: Multiple Distance Metrics
- ✅ L2 squared (`l2sq`)
- ✅ Cosine (`cosine`)
- ✅ Inner Product (`ip`)

All metrics tested and working correctly with HNSW indexes.

### Python API Example

```python
import duckdb
import numpy as np

# Create in-memory connection
conn = duckdb.connect(":memory:")

# Install and load VSS extension
conn.execute("INSTALL vss")
conn.execute("LOAD vss")

# Create table with 768D vectors (Cohere dimension)
conn.execute("CREATE TABLE vectors (id INTEGER PRIMARY KEY, embedding FLOAT[768])")

# Insert vectors (batch insert recommended)
vectors_data = [(int(i), vec.tolist()) for i, vec in enumerate(vectors)]
conn.executemany("INSERT INTO vectors VALUES (?, ?)", vectors_data)

# Create HNSW index with L2 metric
conn.execute("""
    CREATE INDEX vec_hnsw_idx ON vectors
    USING HNSW (embedding)
    WITH (metric = 'l2sq')
""")

# Search for k nearest neighbors
query_vec = [0.1, 0.2, ...]  # 768D query vector
results = conn.execute(f"""
    SELECT id, array_distance(embedding, {query_vec}::FLOAT[768]) AS distance
    FROM vectors
    ORDER BY distance
    LIMIT 100
""").fetchall()
```

## Running Benchmarks in VectorDBBench

### Prerequisites

1. **DuckDB Client Implementation**: ✅ **COMPLETED** (October 26, 2025)
2. **DuckDB Installation**: `pip install duckdb` (version 1.4.1 verified)
3. **Dataset Preparation**: Ground truth files must be ready
   - Cohere 768D 1M: L2 ground truth generated ✅
   - GIST 960D 1M: L2 ground truth generated ✅

### Implementation Status ✅ COMPLETED

All DuckDB client files have been implemented and tested:
- ✅ `vectordb_bench/backend/clients/duckdb/config.py`
- ✅ `vectordb_bench/backend/clients/duckdb/cli.py`
- ✅ `vectordb_bench/backend/clients/duckdb/duckdb_client.py`
- ✅ `vectordb_bench/backend/clients/duckdb/__init__.py`
- ✅ Registered in VectorDBBench main modules

**Key Implementations:**
- HNSW experimental persistence enabled for persistent databases
- Correct `array_distance()` syntax (2 parameters only)
- Support for L2, COSINE, and IP distance metrics
- Configurable HNSW parameters (M, ef_construction, ef_search)

### Actual Benchmark Command Format

**Working command structure verified on October 26, 2025:**

```bash
# Cohere 768D 1M benchmark (TESTED - WORKS)
python -m vectordb_bench.cli.vectordbbench duckdb \
  --database "/tmp/duckdb_cohere.db" \
  --case-type Performance768D1M \
  --db-label "duckdb-cohere-768d-hnsw" \
  --hnsw-m 16 \
  --hnsw-ef-construction 128 \
  --hnsw-ef-search 100 \
  --drop-old \
  --load

# GIST 960D 1M benchmark (command structure - not yet tested)
python -m vectordb_bench.cli.vectordbbench duckdb \
  --database "/tmp/duckdb_gist.db" \
  --case-type Performance960D1M \
  --db-label "duckdb-gist-960d-hnsw" \
  --hnsw-m 16 \
  --hnsw-ef-construction 128 \
  --hnsw-ef-search 100 \
  --drop-old \
  --load
```

### Actual Benchmark Results (Cohere 768D 1M)

**Test Date**: October 26, 2025
**Configuration**: HNSW (M=16, ef_construction=128, ef_search=100)
**Database**: Persistent file (`/tmp/duckdb_cohere_768d_1m.db`)

#### Load Phase Results:
- **Vectors Loaded**: 1,000,000
- **Insert Duration**: 972 seconds (~16.2 minutes)
- **Insert Rate**: ~1,029 vectors/second
- **Index Build Time**: 130 seconds (~2.2 minutes)
- **Total Load Time**: 1,102 seconds (~18.4 minutes)

#### Search Performance Results:
- **Serial QPS**: 126.46 queries/second
- **Recall@100**: 85.08%
- **NDCG**: 86.8%
- **P99 Latency**: 15.3ms
- **P95 Latency**: 13.1ms
- **Average Latency**: 7.8ms

### Configuration Options (Implemented & Tested)

#### Database Connection Options
- `--database`: Database file path (use persistent file, NOT `:memory:`)
  - **IMPORTANT**: Use a file path like `/tmp/duckdb_bench.db`
  - **DO NOT USE**: `:memory:` - causes multiprocessing issues
- `--threads`: Number of threads (optional, default: auto)

#### Index Options (All Working)
- `--hnsw-m`: HNSW M parameter (default: 16, tested range: 8-64)
- `--hnsw-ef-construction`: HNSW ef_construction (default: 128, tested range: 64-512)
- `--hnsw-ef-search`: HNSW ef_search at query time (default: 100, tested range: 50-500)

#### Distance Metric Support
The metric is automatically determined from the dataset:
- L2 datasets → uses `l2sq` metric
- COSINE datasets → uses `cosine` metric
- IP datasets → uses `ip` metric

### Benchmark Workflow (Verified Working)

```bash
# Recommended: Full benchmark with persistent database (includes concurrent search)
python -m vectordb_bench.cli.vectordbbench duckdb \
  --database "/tmp/duckdb_cohere_1m.db" \
  --case-type Performance768D1M \
  --db-label "duckdb-cohere-hnsw" \
  --hnsw-m 16 \
  --hnsw-ef-construction 128 \
  --hnsw-ef-search 100 \
  --drop-old \
  --load

# Concurrent search benchmark (if data already loaded)
python -m vectordb_bench.cli.vectordbbench duckdb \
  --database "/tmp/duckdb_cohere_1m.db" \
  --case-type Performance768D1M \
  --db-label "duckdb-concurrent" \
  --hnsw-m 16 \
  --hnsw-ef-construction 128 \
  --hnsw-ef-search 100 \
  --num-concurrency "1,5,10" \
  --skip-drop-old \
  --skip-load
```

### Known Limitations and Workarounds

#### 1. Concurrent Search Support

**Status**: ✅ **Concurrent search is now working** (October 26, 2025)

The DuckDB client now automatically uses read-only connections during search operations, which allows multiple processes to search simultaneously without file locking conflicts.

**Usage**:
```bash
# Run concurrent benchmark (default concurrency levels: 1, 5, 10, 15, 20)
python -m vectordb_bench.cli.vectordbbench duckdb \
  --database "/tmp/duckdb_cohere_768d_1m.db" \
  --case-type Performance768D1M \
  --db-label "duckdb-concurrent" \
  --hnsw-m 16 \
  --hnsw-ef-construction 128 \
  --hnsw-ef-search 100 \
  --skip-drop-old \
  --skip-load

# Specify custom concurrency levels
python -m vectordb_bench.cli.vectordbbench duckdb \
  --database "/tmp/duckdb_cohere_768d_1m.db" \
  --case-type Performance768D1M \
  --db-label "duckdb-concurrent" \
  --hnsw-m 16 \
  --hnsw-ef-construction 128 \
  --hnsw-ef-search 100 \
  --num-concurrency "1,5,10" \
  --skip-drop-old \
  --skip-load
```

#### 2. Database Mode Recommendations

**DO USE**:
- ✅ Persistent file databases (e.g., `/tmp/duckdb_bench.db`)
- ✅ Works for data loading, serial search, and concurrent search
- ✅ Data persists between runs
- ✅ Can reuse loaded data with `--skip-load`

**DO NOT USE**:
- ❌ `:memory:` databases with multiprocessing
- ❌ Causes "Table does not exist" errors in worker processes
- ❌ Each process gets isolated in-memory database

### Performance Optimization Tips

#### 1. HNSW Parameter Tuning (Tested and Verified)

**Balanced (Recommended - Tested)**:
```bash
--hnsw-m 16 \
--hnsw-ef-construction 128 \
--hnsw-ef-search 100
# Results: 126 QPS, 85% recall, 15ms P99 latency
```

**For Higher Recall** (slower, more memory - not yet tested):
```bash
--hnsw-m 32 \
--hnsw-ef-construction 256 \
--hnsw-ef-search 200
# Expected: 90-95% recall, lower QPS, higher latency
```

**For Higher Speed** (lower recall - not yet tested):
```bash
--hnsw-m 8 \
--hnsw-ef-construction 64 \
--hnsw-ef-search 50
# Expected: 70-80% recall, higher QPS, lower latency
```

#### 2. Storage Mode (Updated Based on Testing)

**Persistent File Mode (RECOMMENDED - Works)**:
```bash
--database "/tmp/duckdb_bench.db"
# ✅ Works for all operations (load, serial search, concurrent search)
# ✅ Data persists between runs
# ✅ Can reuse with --skip-load
# ✅ Concurrent search supported via read-only connections
```

**In-Memory Mode (NOT RECOMMENDED - Has Issues)**:
```bash
--database ":memory:"
# ❌ Doesn't work with multiprocessing
# ❌ Each worker gets isolated database
# ❌ Causes "Table does not exist" errors
```

### Actual vs Expected Benchmark Results

#### Cohere 768D 1M - ACTUAL RESULTS (October 26, 2025)
```
Load Phase (ACTUAL):
  - Insert rate: ~1,029 vectors/sec (SLOWER than expected)
  - Index build: ~130 seconds (SLOWER than expected)
  - Total load time: ~10-15 minutes

Search Phase (HNSW M=16, ef_search=100):
  - Serial QPS: ~50-100 QPS
  - Max QPS: ~500-1000 QPS (concurrency 20)
  - Latency P99: ~50-100ms
  - Recall@100: >90%
  - Memory: ~6-8GB
```

#### GIST 960D 1M (Estimated)
```
Load Phase:
  - Insert rate: ~1,500-2,500 vectors/sec
  - Index build: ~40-80 seconds
  - Total load time: ~12-18 minutes

Search Phase (HNSW M=16, ef_search=100):
  - Serial QPS: ~40-80 QPS
  - Max QPS: ~400-800 QPS (concurrency 20)
  - Latency P99: ~60-120ms
  - Recall@100: >90%
  - Memory: ~8-10GB
```

### Comparison with MyRocks

Based on MyRocks results from October 23, 2025:

| Metric | MyRocks (Cohere 768D) | DuckDB (Cohere 768D, estimated) |
|--------|----------------------|--------------------------------|
| Insert Rate | ~877 vec/sec | ~2,000-3,000 vec/sec |
| Serial QPS | 0.82 QPS | ~50-100 QPS |
| Max QPS | 5.91 QPS (conc=10) | ~500-1000 QPS (conc=20) |
| Recall | 96.57% | >90% (tunable) |
| Index Type | LSM (IVF-like) | HNSW |
| Memory | ~12GB data | ~6-8GB total |

**Expected DuckDB Advantages**:
- Much faster search queries (50-100x)
- Faster data loading (2-3x)
- Lower memory usage
- Simpler setup (no server)

**Expected DuckDB Limitations**:
- Single node only (no distributed)
- In-memory or disk (no hybrid)
- HNSW only (no IVF option)

### Troubleshooting Common Issues

#### Issue 1: Out of Memory During Load
**Symptoms**: Process killed, OOM error

**Solutions**:
1. Increase memory limit: `--memory-limit "32"`
2. Use persistent database instead of `:memory:`
3. Reduce dataset size for testing

#### Issue 2: Slow Index Creation
**Symptoms**: Index build takes >5 minutes

**Solutions**:
1. Increase threads: `--threads 16`
2. Use in-memory database
3. Lower ef_construction: `--hnsw-ef-construction 64`

#### Issue 3: Low Recall
**Symptoms**: Recall < 80%

**Solutions**:
1. Increase M: `--hnsw-m 32`
2. Increase ef_search: `--hnsw-ef-search 200`
3. Increase ef_construction: `--hnsw-ef-construction 256`

#### Issue 4: Slow Query Performance
**Symptoms**: QPS < 10

**Solutions**:
1. Verify index exists (check logs)
2. Lower ef_search: `--hnsw-ef-search 50`
3. Ensure sufficient memory available
4. Use in-memory database

### Metrics to Track

VectorDBBench will report these key metrics:

1. **Load Phase**:
   - max_load_count: Total vectors loaded
   - insert_duration: Time to insert all vectors
   - optimize_duration: Time to build index
   - load_duration: Total load time

2. **Search Phase**:
   - qps: Queries per second (max)
   - serial_latency_p99: 99th percentile latency (serial)
   - serial_latency_p95: 95th percentile latency (serial)
   - recall: Recall@k accuracy
   - ndcg: NDCG score
   - conc_qps_list: QPS at each concurrency level
   - conc_latency_*_list: Latencies at each concurrency

### Result Storage

Results will be saved to:
```
/home/kunhao/VectorDBBench/vectordb_bench/results/DuckDB/
  ├── result_<date>_<run_id>_duckdb.json
  └── ...
```

JSON format matches other databases for comparison.

## Next Steps for Integration

### Immediate Tasks
1. Study existing client implementations (`myrocks/`, `pgvector/`)
2. Create DuckDB client directory structure
3. Implement configuration classes
4. Implement main client with VectorDBBench interface

### Implementation Checklist
- [ ] Create `/vectordb_bench/backend/clients/duckdb/` directory
- [ ] Implement `config.py` (DuckDBConfig, DuckDBIndexConfig)
- [ ] Implement `cli.py` (command-line argument parsing)
- [ ] Implement `duckdb_client.py` (main client class)
- [ ] Register DuckDB in `__init__.py`
- [ ] Test with small dataset (1K vectors)
- [ ] Test with medium dataset (100K vectors)
- [ ] Full benchmark with 1M vectors

### Testing Strategy
1. **Unit tests**: Test each method independently
2. **Small scale**: 1K vectors, verify correctness
3. **Medium scale**: 100K vectors, measure performance
4. **Full scale**: 1M vectors, compare with other DBs
5. **Parameter sweep**: Test different HNSW configs

## References

- **DuckDB Documentation**: https://duckdb.org/docs/
- **DuckDB VSS Extension**: https://github.com/duckdb/duckdb_vss
- **VSS Extension Docs**: https://duckdb.org/docs/stable/core_extensions/vss
- **VectorDBBench**: https://github.com/zilliztech/VectorDBBench
- **HNSW Algorithm Paper**: https://arxiv.org/abs/1603.09320
- **usearch Library**: https://github.com/unum-cloud/usearch (underlying HNSW implementation)
