# Local PostgreSQL + pgvector Setup Guide (No Sudo Required)

## Overview

This document describes how to install and configure PostgreSQL with the pgvector extension locally in your home directory without requiring sudo/root privileges.

## System Information

- **User**: kunhao
- **Home Directory**: `/home/kunhao/`
- **PostgreSQL Version**: 16.3
- **pgvector Version**: 0.7.0
- **Installation Directory**: `/home/kunhao/postgres-local/`
- **Database Port**: 5433 (non-default to avoid conflicts)

## Installation Steps

### 1. Download and Compile PostgreSQL

```bash
# Create installation directory
cd ~
mkdir -p postgres-local
cd postgres-local

# Download PostgreSQL 16.3 source
wget https://ftp.postgresql.org/pub/source/v16.3/postgresql-16.3.tar.gz
tar -xzf postgresql-16.3.tar.gz
cd postgresql-16.3

# Configure PostgreSQL for local installation
./configure --prefix=/home/kunhao/postgres-local/pg16 --without-readline --without-zlib

# Compile and install
make -j$(nproc)
make install
```

### 2. Set Up Environment Variables

Add these to your `~/.bashrc`:

```bash
# PostgreSQL environment variables
export PGHOME=/home/kunhao/postgres-local/pg16
export PATH=$PGHOME/bin:$PATH
export PGDATA=/home/kunhao/postgres-local/pgdata
export PGPORT=5433
export PGDATABASE=vectordbbench
export PGUSER=kunhao
```

Apply the changes:
```bash
source ~/.bashrc
```

### 3. Initialize Database Cluster

```bash
$PGHOME/bin/initdb -D $PGDATA -U kunhao --auth=trust
```

### 4. Configure PostgreSQL

Choose one of the following configurations based on your needs:

#### Option A: Default Configuration (Minimal)
```bash
cat >> $PGDATA/postgresql.conf <<'EOF'

# Minimal Settings
shared_buffers = 128MB                # PostgreSQL default
maintenance_work_mem = 256MB          # Minimum for IVF index creation (default is 64MB)

# Connection Settings
port = 5433
unix_socket_directories = '/home/kunhao/postgres-local/pgdata'

# Logging
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
EOF
```

#### Option B: Moderate Configuration (512MB buffers)
```bash
cat >> $PGDATA/postgresql.conf <<'EOF'

# Moderate Settings
shared_buffers = 512MB                # 4x default
maintenance_work_mem = 256MB          # For index creation

# Connection Settings
port = 5433
unix_socket_directories = '/home/kunhao/postgres-local/pgdata'

# Logging
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
EOF
```

#### Option C: Optimized Configuration (High Memory)
```bash
cat >> $PGDATA/postgresql.conf <<'EOF'

# Optimized Settings (for 32GB RAM system)
shared_buffers = 8GB                      # 25% of RAM
maintenance_work_mem = 4GB                # For index creation
work_mem = 256MB                          # Per-operation memory
effective_cache_size = 16GB               # OS cache estimate

# Parallel Query Settings
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 8

# I/O Settings
random_page_cost = 1.1                    # For SSD storage
effective_io_concurrency = 200

# Connection Settings
port = 5433
unix_socket_directories = '/home/kunhao/postgres-local/pgdata'
max_connections = 100

# Logging
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
EOF
```

**Note**: PostgreSQL default `maintenance_work_mem=64MB` is insufficient for IVF-Flat index creation on 1M vectors. Minimum 256MB is required.

### 5. Download and Compile pgvector Extension

```bash
cd ~/postgres-local

# Download pgvector 0.7.0
wget https://github.com/pgvector/pgvector/archive/refs/tags/v0.7.0.tar.gz
tar -xzf v0.7.0.tar.gz
cd pgvector-0.7.0

# Compile and install pgvector
make PG_CONFIG=$PGHOME/bin/pg_config
make install PG_CONFIG=$PGHOME/bin/pg_config
```

Verify installation:
```bash
ls $PGHOME/share/extension/vector*
# Should show: vector--0.7.0.sql, vector.control
```

### 6. Start PostgreSQL Server

```bash
$PGHOME/bin/pg_ctl -D $PGDATA -l $PGDATA/logfile start
sleep 3
$PGHOME/bin/pg_ctl -D $PGDATA status
```

### 7. Create Database and Enable pgvector

```bash
# Create database
$PGHOME/bin/createdb -p 5433 -U kunhao vectordbbench

# Enable pgvector extension
$PGHOME/bin/psql -p 5433 -U kunhao -d vectordbbench -c "CREATE EXTENSION vector;"

# Verify
$PGHOME/bin/psql -p 5433 -U kunhao -d vectordbbench -c "\dx"
```

## Running VectorDBBench with pgvector

### Prerequisites

Install Python dependencies:
```bash
conda activate vectordbbench
pip install psycopg psycopg-binary pgvector
```

### Benchmark Commands

**Note**: The `--password` flag cannot accept empty string. Use environment variable instead:
```bash
export POSTGRES_PASSWORD="dummy"
```

#### IVF-Flat Index Benchmarks

**256 lists (centroids), probes=16:**
```bash
POSTGRES_PASSWORD="dummy" python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
  --user-name kunhao \
  --host localhost \
  --port 5433 \
  --db-name vectordbbench \
  --case-type Performance768D1M \
  --db-label "pgvector-ivfflat-256-probes16" \
  --lists 256 \
  --probes 16 \
  --drop-old \
  --load \
  --num-concurrency "1,5,10"
```

**512 lists (centroids), probes=16:**
```bash
POSTGRES_PASSWORD="dummy" python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
  --user-name kunhao \
  --host localhost \
  --port 5433 \
  --db-name vectordbbench \
  --case-type Performance768D1M \
  --db-label "pgvector-ivfflat-512-probes16" \
  --lists 512 \
  --probes 16 \
  --drop-old \
  --load \
  --num-concurrency "1,5,10"
```

**Search-only benchmark (skip loading):**
```bash
POSTGRES_PASSWORD="dummy" python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
  --user-name kunhao \
  --host localhost \
  --port 5433 \
  --db-name vectordbbench \
  --case-type Performance768D1M \
  --db-label "pgvector-ivfflat-512-probes8" \
  --lists 512 \
  --probes 8 \
  --skip-load \
  --num-concurrency "1,5,10"
```

#### HNSW Index Benchmarks

```bash
POSTGRES_PASSWORD="dummy" python -m vectordb_bench.cli.vectordbbench pgvectorhnsw \
  --user-name kunhao \
  --host localhost \
  --port 5433 \
  --db-name vectordbbench \
  --case-type Performance768D1M \
  --db-label "pgvector-hnsw-m16-ef64" \
  --m 16 \
  --ef-construction 64 \
  --ef-search 40 \
  --drop-old \
  --load \
  --num-concurrency "1,5,10"
```

### Key Parameters

**IVF-Flat:**
- `--lists`: Number of clusters/centroids
  - Recommended: sqrt(rows) for >1M vectors, rows/1000 for <1M
  - Examples: 256, 512, 1024
  - More lists = faster search but lower recall (with same probes)
  - Memory requirement: ~(lists × dimensions × 4 bytes) for centroids
- `--probes`: Number of clusters to search at query time
  - Higher probes = better recall but slower search
  - Coverage: probes/lists (e.g., 16/512 = 3.125%)
  - Examples: 8, 16, 32
  - Trade-off: Doubling probes roughly halves QPS but improves recall

**HNSW:**
- `--m`: Connections per node (typical: 12-48)
  - Higher = better recall but more memory and slower index build
- `--ef-construction`: Index build quality (typical: 64-200)
  - Higher = better index quality but slower build
- `--ef-search`: Search quality at query time (typical: 10-400)
  - Higher = better recall but slower search

**Common:**
- `--drop-old`: Drop existing table before benchmark
- `--load`: Run data loading phase
- `--skip-load`: Skip loading (for search-only tests)
- `--num-concurrency`: Comma-separated concurrency levels (e.g., "1,5,10")
- `--case-type`: Dataset to use
  - `Performance768D1M`: Cohere 1M vectors, 768 dimensions
  - `Performance768D10M`: Cohere 10M vectors, 768 dimensions

## Database Management

### Start/Stop Commands
```bash
# Start
$PGHOME/bin/pg_ctl -D $PGDATA -l $PGDATA/logfile start

# Stop
$PGHOME/bin/pg_ctl -D $PGDATA stop

# Restart
$PGHOME/bin/pg_ctl -D $PGDATA restart

# Status
$PGHOME/bin/pg_ctl -D $PGDATA status
```

### Connect to Database
```bash
$PGHOME/bin/psql -p 5433 -U kunhao -d vectordbbench
```

### Useful psql Commands
```sql
-- List all databases
\l

-- List all extensions
\dx

-- List all tables
\dt

-- Show table structure
\d table_name

-- Show database size
SELECT pg_size_pretty(pg_database_size('vectordbbench'));

-- Show table size
SELECT pg_size_pretty(pg_total_relation_size('table_name'));

-- Quit psql
\q
```

### Changing Configuration

To modify PostgreSQL settings:

```bash
# Edit configuration
nano /home/kunhao/postgres-local/pgdata/postgresql.conf

# Restart PostgreSQL to apply changes
$PGHOME/bin/pg_ctl -D $PGDATA restart

# Verify new settings
$PGHOME/bin/psql -p 5433 -U kunhao -d vectordbbench -c "SHOW shared_buffers;"
```

## Memory Configuration Best Practices

### IVF-Flat Index Memory Requirements

The `maintenance_work_mem` parameter controls memory available for index creation:

| Number of Lists | Minimum maintenance_work_mem | Recommended |
|----------------|------------------------------|-------------|
| 256 | 128MB | 256MB |
| 512 | 256MB | 512MB |
| 1024 | 364MB | 512MB |
| 2048 | 700MB+ | 1GB |

**Formula**: Memory ≈ (lists × dimensions × 4 bytes) + overhead

### Optimal Settings for 1M Vector Dataset (768 dim)

```bash
# Minimal configuration (sufficient for most cases)
shared_buffers = 128MB
maintenance_work_mem = 256MB  # 512MB for 1024+ lists
work_mem = 4MB
max_parallel_workers = 8
max_parallel_maintenance_workers = 2
```

**Key Points:**
- `shared_buffers`: 128-256MB is sufficient for most workloads
  - OS page cache handles the bulk of caching (7-8GB for 1M vectors)
  - Increasing beyond 256MB provides minimal benefit
- `maintenance_work_mem`: Must match index size requirements
  - Too low = index creation fails
  - Set to 512MB for safety with large indexes
- `work_mem`: 4MB default is adequate
  - Per-operation memory for sorting/hashing
  - pgvector searches don't heavily use this
- `max_parallel_workers`: Doesn't affect IVF-Flat searches
  - IVF-Flat queries run single-threaded per query
  - Performance scales with concurrency, not parallelism

## Monitoring and Performance Analysis

### Check OS Page Cache Usage

Monitor which PostgreSQL files are cached in RAM:

```bash
# Check cache residency for specific table/index
fincore $PGDATA/base/$(psql -p 5433 -U kunhao -d vectordbbench -tAc "SELECT oid FROM pg_database WHERE datname='vectordbbench'")/16790*

# Summary of total cached data
fincore $PGDATA/base/*/16790* 2>/dev/null | awk '{sum+=$1} END {print "Total cached: " sum/1024/1024 " MB"}'
```

### Monitor Query Performance

```sql
-- Check current database size
SELECT pg_size_pretty(pg_database_size('vectordbbench'));

-- Check table and index sizes separately
SELECT
  relname,
  pg_size_pretty(pg_relation_size(oid)) as table_size,
  pg_size_pretty(pg_indexes_size(oid)) as index_size
FROM pg_class
WHERE relname = 'vdbbench_table_test';

-- View all indexes
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
FROM pg_indexes
WHERE tablename = 'vdbbench_table_test';

-- Check memory settings
SELECT name, setting, unit
FROM pg_settings
WHERE name IN ('shared_buffers', 'work_mem', 'maintenance_work_mem',
               'max_parallel_workers', 'max_parallel_maintenance_workers');
```

### Analyze Query Plans

```sql
-- Set search parameters
SET ivfflat.probes = 16;

-- Explain a vector search query
EXPLAIN (ANALYZE, BUFFERS)
SELECT id FROM vdbbench_table_test
ORDER BY embedding <-> (SELECT embedding FROM vdbbench_table_test LIMIT 1)
LIMIT 100;
```

Look for:
- `Index Scan using pgvector_index`: Confirms index usage
- `Buffers: shared hit=X read=Y`: Cache hits vs disk reads
- No "Parallel" nodes: IVF-Flat doesn't use parallel workers

## Troubleshooting

### Index Creation Fails: "memory required is X MB, maintenance_work_mem is Y MB"

**Common errors and solutions:**

```
Error: memory required is 131 MB, maintenance_work_mem is 64 MB
→ Building index with 256 lists requires ~131 MB

Error: memory required is 364 MB, maintenance_work_mem is 256 MB
→ Building index with 1024 lists requires ~364 MB
```

**Solution**: Increase `maintenance_work_mem` based on number of lists:

```bash
# For 256-512 lists
echo "maintenance_work_mem = 256MB" >> $PGDATA/postgresql.conf

# For 1024+ lists
echo "maintenance_work_mem = 512MB" >> $PGDATA/postgresql.conf

# Restart PostgreSQL
$PGHOME/bin/pg_ctl -D $PGDATA restart

# Verify new setting
$PGHOME/bin/psql -p 5433 -U kunhao -d vectordbbench -c "SHOW maintenance_work_mem;"
```

**Note**: PostgreSQL default `maintenance_work_mem=64MB` is insufficient for IVF-Flat index creation on large datasets.

### Port Already in Use

```bash
# Change port in postgresql.conf
echo "port = 5434" >> $PGDATA/postgresql.conf
export PGPORT=5434

# Restart
$PGHOME/bin/pg_ctl -D $PGDATA restart
```

### Connection Issues

```bash
# Check if server is running
$PGHOME/bin/pg_ctl -D $PGDATA status

# Check log file
tail -f $PGDATA/logfile
```

## Directory Structure

```
/home/kunhao/postgres-local/
├── pg16/                          # PostgreSQL installation
│   ├── bin/                       # Executables (psql, pg_ctl, etc.)
│   ├── lib/                       # Libraries
│   └── share/extension/           # Extensions
│       ├── vector--0.7.0.sql
│       └── vector.control
├── pgdata/                        # Database cluster
│   ├── postgresql.conf            # Main configuration
│   ├── pg_hba.conf               # Authentication config
│   ├── base/                      # Database files
│   ├── log/                       # Log files
│   └── logfile                    # Server log
├── postgresql-16.3/              # Source code (can delete after install)
└── pgvector-0.7.0/               # pgvector source (can delete after install)
```

## Index Structure Comparison

| Feature | pgvector IVF-Flat | pgvector HNSW | MyRocks LSM |
|---------|-------------------|---------------|-------------|
| Index Type | Clustering-based | Graph-based | Clustering-based |
| Build Time | Fast | Slower | Fast |
| Search Speed | Medium-Fast | Fast | Medium |
| Memory Usage | Lower | Higher | Lower |
| Comparable Param | probes | ef_search | nprobe |
| Clustering | lists | - | centroids |

**IVF-Flat vs MyRocks LSM**:
- Both use k-means clustering for partitioning
- `lists` (pgvector) ≈ `centroids` (MyRocks)
- `probes` (pgvector) ≈ `nprobe` (MyRocks)
- pgvector uses inverted lists for direct bucket access
- MyRocks embeds vectors in LSM tree structure

## File Locations

- **PostgreSQL Home**: `/home/kunhao/postgres-local/pg16`
- **Data Directory**: `/home/kunhao/postgres-local/pgdata`
- **Configuration**: `/home/kunhao/postgres-local/pgdata/postgresql.conf`
- **Log File**: `/home/kunhao/postgres-local/pgdata/logfile`
- **Unix Socket**: `/home/kunhao/postgres-local/pgdata/.s.PGSQL.5433`
- **Results**: `/home/kunhao/VectorDBBench/vectordb_bench/results/PgVector/`

## Resources

- PostgreSQL Documentation: https://www.postgresql.org/docs/16/
- pgvector GitHub: https://github.com/pgvector/pgvector
- pgvector Documentation: https://github.com/pgvector/pgvector#readme
- VectorDBBench: https://github.com/zilliztech/VectorDBBench
