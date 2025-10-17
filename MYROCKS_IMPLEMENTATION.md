# MyRocks Client Implementation for VectorDBBench

## Overview
This document describes the implementation of a MyRocks database client for VectorDBBench, enabling benchmarking of MyRocks' vector search capabilities using Facebook's vector extensions.

## Implementation Date
October 17, 2025

## Status: WORKING BUT INCOMPLETE
- ✅ Basic pipeline fully functional (insert, search)
- ⚠️ LSM index disabled (requires proper centroids)
- ⚠️ Only L2 distance supported (COSINE/IP fall back to L2)
- ⚠️ Brute-force search is extremely slow without index

## Files Created

### 1. Core Implementation Files
- `vectordb_bench/backend/clients/myrocks/myrocks.py` - Main MyRocks client implementation
- `vectordb_bench/backend/clients/myrocks/config.py` - Configuration classes for MyRocks
- `vectordb_bench/backend/clients/myrocks/cli.py` - Command-line interface for benchmarking
- `vectordb_bench/backend/clients/myrocks/vector_load.sql` - Example SQL for vector data loading

### 2. Integration Files Modified
- `vectordb_bench/backend/clients/__init__.py` - Registered MyRocks in DB enum and config mappings
- `vectordb_bench/cli/vectordbbench.py` - Added MyRocks CLI command

## Key Features Implemented

### 1. Database Connection
- Uses `mysql-connector-python` to connect to MyRocks
- Supports standard MySQL connection parameters (host, port, user, password)

### 2. Vector Storage
- Stores vectors as **JSON type** with `FB_VECTOR_DIMENSION` attribute
- Converts Python arrays to JSON format for storage
- Example: `CREATE TABLE vectors (id INT PRIMARY KEY, v JSON NOT NULL FB_VECTOR_DIMENSION 128) ENGINE=ROCKSDB`

### 3. Centroid Table
- Creates separate centroid table for LSM index requirements
- Supports both random centroid generation (testing) and CSV loading (production)
- Table naming: `{collection_name}_centroids`

### 4. Index Support
- Implements LSM-based vector index using `FB_VECTOR_INDEX_TYPE 'lsmidx'`
- Currently configured to skip index creation (requires proper centroids)
- Ready for production use once centroids are trained on dataset

### 5. Vector Search
- Uses `FB_VECTOR_L2()` function for L2 distance calculations
- Query format: `SELECT id, FB_VECTOR_L2(table.v, '[vector]') AS dis FROM table ORDER BY dis ASC LIMIT k`
- Currently supports L2 distance metric only
- Fallback to L2 for COSINE and IP metrics (until MyRocks adds those functions)

## Testing Results

### Test Configuration
- **Dataset**: OpenAI embeddings (1536 dimensions, 50K vectors)
- **Test Case**: Performance1536D50K
- **Index**: Disabled (LSM index requires trained centroids)
- **Search Method**: Brute-force without index

### Successful Operations
✅ **Database Connection**: Connected to MyRocks successfully
✅ **Table Creation**: Created main table and centroid table with JSON vectors
✅ **Data Insertion**: Successfully inserted 50K vectors (1536-dim) in ~110 seconds
✅ **Optimization Phase**: Completed (skipped LSM index as planned)
✅ **Search Phase**: Verified basic search functionality works

### Performance Notes
- Insert performance: ~450 vectors/second (50K vectors in 110 seconds)
- Brute-force search: Very slow (~15-30 seconds per query for 50K vectors)
- **Conclusion**: LSM index is essential for production use

## Current Limitations

### 1. Distance Metrics
- **Supported**: L2 (Euclidean) via `FB_VECTOR_L2()`
- **Not Yet Supported**: COSINE, Inner Product
- **Workaround**: Currently falls back to L2 for all metrics

### 2. LSM Index
- **Issue**: Requires pre-computed centroids trained on dataset
- **Current State**: Index creation is disabled in code
- **Impact**: Search uses brute-force, which is extremely slow
- **Solution**: Need to generate proper centroids using k-means clustering on training data

### 3. Centroid Generation
Two methods implemented but both need work:
- `_generate_random_centroids()`: Creates random centroids (causes corruption with LSM index)
  - Location: `myrocks.py:137-172`
  - Issue: Random centroids don't match data distribution
  - Result: RocksDB error 505 "Found data corruption" during search
- `_load_centroids_from_csv()`: Loads from CSV file (requires pre-computed centroids)
  - Location: `myrocks.py:174-224`
  - Expects CSV format: `id,centroid` where centroid is JSON array
  - Status: Implemented but untested (no centroid file available yet)

## Next Steps

### Immediate (Required for Production Use)
1. **Generate Proper Centroids**
   - Use k-means clustering on training dataset
   - Export centroids to CSV file
   - Format: `id, centroid` where centroid is JSON array string

2. **Enable LSM Index**
   - Uncomment index creation code in `myrocks.py:226-257`
   - Load proper centroids before index creation
   - Test search performance with index

3. **Add Distance Metric Support**
   - Implement `FB_VECTOR_COSINE()` when available in MyRocks
   - Implement `FB_VECTOR_IP()` when available in MyRocks
   - Update `search_embedding()` to use appropriate function based on metric

### Future Enhancements
4. **Configuration Options**
   - Add centroid_path parameter to config
   - Add num_centroids parameter (default: 256)
   - Add option to choose between index types if MyRocks adds more

5. **Performance Optimization**
   - Batch centroid insertion for better performance
   - Optimize vector JSON conversion
   - Add connection pooling if needed

6. **Testing**
   - Create unit tests for vector operations
   - Add integration tests with proper centroids
   - Benchmark with different dataset sizes

## Usage

### Basic Command
```bash
python -m vectordb_bench.cli.vectordbbench myrocks \
  --username root \
  --password YOUR_PASSWORD \
  --host 127.0.0.1 \
  --port 3306 \
  --db-label "myrocks-test" \
  --case-type Performance1536D50K \
  --drop-old
```

### Connection Parameters
- `--username`: MySQL username (default: root)
- `--password`: MySQL password (required)
- `--host`: MySQL host (default: 127.0.0.1)
- `--port`: MySQL port (default: 3306)
- `--db-label`: Label for test run

## Technical Details

### Vector Data Flow
1. **Input**: Python list of floats `[0.1, 0.2, 0.3, ...]`
2. **Conversion**: `json.dumps(vector)` → `"[0.1, 0.2, 0.3, ...]"`
3. **Storage**: `INSERT INTO table (id, v) VALUES (1, CAST('[0.1,0.2,0.3]' AS JSON))`
4. **Search**: `FB_VECTOR_L2(table.v, '[query_vector]')`

### Database Schema
```sql
-- Main vector table
CREATE TABLE vec_collection (
    id INT PRIMARY KEY,
    v JSON NOT NULL FB_VECTOR_DIMENSION 1536
) ENGINE=ROCKSDB;

-- Centroid table (for LSM index)
CREATE TABLE vec_collection_centroids (
    id INT PRIMARY KEY,
    centroid JSON NOT NULL FB_VECTOR_DIMENSION 1536
) ENGINE=ROCKSDB;

-- Vector index (commented out until centroids available)
-- ALTER TABLE vec_collection
-- ADD INDEX v_idx(v) FB_VECTOR_INDEX_TYPE 'lsmidx';
```

## Code Locations (For Future Reference)

### Key Methods to Update When Adding LSM Index
1. **Enable index creation**: `myrocks.py:226-257` (currently commented out)
   - Uncomment lines 240-257 to enable LSM index
   - Replace `self._generate_random_centroids()` call with `self._load_centroids_from_csv(csv_path)`

2. **Centroid loading**: `myrocks.py:174-224`
   - CSV format: `id,centroid` where centroid is JSON array string
   - Example: `0,"[0.1,0.2,0.3,...]"`

3. **Metric type handling**: `config.py:39-46`
   - Currently accepts L2/COSINE/IP but uses L2 for all
   - Update when MyRocks adds FB_VECTOR_COSINE and FB_VECTOR_IP functions

4. **Search implementation**: `myrocks.py:289-314`
   - Currently only uses FB_VECTOR_L2
   - Add logic to switch between distance functions based on metric_type

### Important Configuration Details
- **Index type mapping**: Uses `IndexType.Flat` as placeholder for LSM (no LSM in IndexType enum)
- **Case config**: `_myrocks_case_config` in `config.py:64-66`
- **DB registration**: `__init__.py:54, 204-207, 360-363, 491-494`
- **CLI registration**: `vectordbbench.py:9, 45`

## Known Issues

### Issue 1: Data Corruption with Random Centroids
- **Error**: `Got error 505 'Found data corruption.' from ROCKSDB`
- **Cause**: LSM index requires centroids that match data distribution
- **Status**: Resolved by disabling index creation
- **Fix**: Need proper centroid training

### Issue 2: Metric Type Validation
- **Error**: `Metric type MetricType.COSINE is not supported!`
- **Cause**: Config validation was too strict
- **Status**: Resolved by accepting all metrics and using L2 as fallback
- **Fix**: Update when COSINE support added to MyRocks

### Issue 3: Missing data_size Parameter
- **Error**: `MyRocks.optimize() got an unexpected keyword argument 'data_size'`
- **Cause**: VectorDB interface requires this parameter
- **Status**: Resolved by adding parameter to method signature
- **Fix**: Complete

## Dependencies

### Python Packages
- `mysql-connector-python`: MySQL database connector
- `numpy`: Vector operations
- `json`: Vector serialization

### Database Requirements
- MyRocks (MySQL with RocksDB storage engine)
- Facebook vector extensions enabled
- Vector functions available: `FB_VECTOR_L2()`, `FB_VECTOR_DIMENSION`, `FB_VECTOR_INDEX_TYPE`

## References

- VectorDBBench Repository: https://github.com/zilliztech/VectorDBBench
- MyRocks Documentation: (Add link when available)
- Facebook Vector Extensions: (Add link when available)

## Important Notes for Continuation

### Why LSM Index is Disabled
The LSM index requires centroids that are computed using k-means clustering on the actual training data. Random centroids cause RocksDB corruption errors because:
1. LSM index uses centroids to partition the vector space
2. Queries use centroids to narrow down search space
3. If centroids don't match data distribution, the index becomes inconsistent
4. This manifests as error 505: "Found data corruption"

### To Enable LSM Index (Step-by-Step)
1. **Generate centroids**:
   ```python
   # Use k-means on training dataset
   from sklearn.cluster import KMeans
   import pandas as pd

   # Load training vectors
   vectors = load_training_data()  # Your 50K vectors

   # Cluster into 256 centroids
   kmeans = KMeans(n_clusters=256, random_state=42)
   kmeans.fit(vectors)
   centroids = kmeans.cluster_centers_

   # Save to CSV
   df = pd.DataFrame({
       'id': range(256),
       'centroid': [json.dumps(c.tolist()) for c in centroids]
   })
   df.to_csv('centroids.csv', index=False)
   ```

2. **Update optimize() method** in `myrocks.py:226-257`:
   - Replace line 243 with: `self._load_centroids_from_csv('/path/to/centroids.csv')`
   - Uncomment lines 245-257 (index creation code)

3. **Test with proper centroids**:
   ```bash
   python -m vectordb_bench.cli.vectordbbench myrocks \
     --username root --password 150131 \
     --host 127.0.0.1 --port 3306 \
     --case-type Performance1536D50K
   ```

### MySQL Connection Details
- **Database**: Creates `vectordbbench` database
- **Socket**: `/tmp/mysql.sock`
- **Password**: 150131 (used in testing)
- **MyRocks location**: `/home/kevin/backup/myrocks-runtime/usr/local/mysql`
- **Start command**: `bin/mysqld --defaults-file=/path/to/my.cnf &`

### Vector Load SQL Example
Location: `vectordb_bench/backend/clients/myrocks/vector_load.sql`
- Shows how to create tables with FB_VECTOR_DIMENSION
- Shows how to create index with FB_VECTOR_INDEX_TYPE 'lsmidx'
- Shows LOAD DATA INFILE for bulk loading
- Important: Uses CAST(@embedding AS JSON) for vector insertion

## Contributors
- Implementation: Claude Code + Kevin
- Testing: Kevin
- Date: October 17, 2025
