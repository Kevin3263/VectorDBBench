# Memory-Limited Benchmark Suite for PostgreSQL + pgvector

## ⚠️ CRITICAL: Page Cache Issue Discovered

**IMPORTANT**: We discovered that systemd memory limits alone are **NOT sufficient** for realistic benchmarks!

**The Problem:**
- systemd `MemoryMax` only limits process RSS (Resident Set Size)
- It does **NOT** limit the Linux page cache
- The OS can still cache the entire 8.7GB database in page cache
- Result: Benchmarks measure RAM performance, not disk I/O

**The Solution:**
- **You MUST drop page cache before each benchmark run** (requires sudo)
- See `ADMIN_BENCHMARK_GUIDE.md` for the complete correct procedure

**Without page cache dropping:**
- QPS: 21-22 (unrealistically high)
- P99 Latency: 60-70ms (too fast)
- Measuring: RAM performance ❌

**With proper page cache dropping:**
- QPS: 15-18 (realistic)
- P99 Latency: 80-100ms (realistic disk I/O)
- Measuring: Actual database performance ✅

**If you have sudo access**, please read `ADMIN_BENCHMARK_GUIDE.md` instead of this file.

---

## Overview

This suite runs PostgreSQL and the benchmark client with **strict memory constraints** using systemd cgroups. However, this alone is not sufficient - you also need to drop the page cache (see warning above).

## TL;DR - Quick Command Reference

```bash
# 1. Cleanup
cd /home/kunhao/VectorDBBench && ./cleanup_benchmark.sh

# 2. Start PostgreSQL with memory limit (example: 4GB)
systemd-run --user --scope --unit=postgres-limited \
  -p MemoryMax=4G -p MemoryHigh=3.6G -p MemorySwapMax=0 \
  /home/kunhao/postgres-local/pg16/bin/postgres -D /home/kunhao/postgres-local/pgdata

# 3. Rebuild index (example: 512 lists)
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 512);"

# 4. Run benchmark (example: 1.5GB client, 512 lists, 16 probes)
export POSTGRES_PASSWORD="dummy"
systemd-run --user --scope --unit=benchmark-client \
  -p MemoryMax=1536M -p MemoryHigh=1400M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 --db-name vectordbbench \
    --case-type Performance768D1M \
    --db-label "pgvector-4GB-1.5GB-lists512-probes16-cohere1M" \
    --lists 512 --probes 16 --skip-load --num-concurrency "1"
```

See "Verified Working Example" section below for detailed explanation.

## Why Memory Limits?

**Problem**: With 32GB RAM and 6-8GB database:
- Entire database gets cached in OS page cache
- Subsequent queries are served from RAM
- Benchmark measures RAM performance, not disk I/O
- Results are unrealistic for production workloads

**Solution**: Limit both PostgreSQL and benchmark client memory:
- PostgreSQL limited to 2-8GB (forces disk I/O)
- Benchmark client limited to 1-2GB (prevents query caching)
- OS page cache naturally limited by total process memory
- Results reflect real disk I/O performance

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `test_memory_limits.sh` | Verify memory limit setup works |
| `benchmark_memory_limited.sh` | Run single benchmark with memory limits |
| `benchmark_suite_memory_limited.sh` | Run multiple configurations automatically |
| `monitor_memory.sh` | Real-time memory usage monitoring |
| `cleanup_benchmark.sh` | Stop all benchmark processes |

## ⚠️ CRITICAL: How to Correctly Run Memory-Limited Benchmarks

### The `--skip-load` Problem

**IMPORTANT**: When using `--skip-load`, the benchmark does **NOT** rebuild the index with your specified parameters. It only uses the existing index and changes the `probes` setting at query time.

This means:
- ✅ You can test different `--probes` values with `--skip-load`
- ❌ You CANNOT test different `--lists` values with `--skip-load`
- ❌ The `--lists` parameter is IGNORED when using `--skip-load`

### Correct Procedure for Testing Different Index Configurations

#### Method 1: Manual Index Rebuild (Recommended for speed)

```bash
# 1. Start PostgreSQL with memory limits
systemd-run --user --scope --unit=postgres-limited \
  -p MemoryMax=4G -p MemoryHigh=3.5G -p MemorySwapMax=0 \
  /home/kunhao/postgres-local/pg16/bin/postgres -D /home/kunhao/postgres-local/pgdata

# 2. Rebuild index with desired lists parameter
psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 512);"

# 3. Verify index configuration
psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "SELECT indexname, indexdef FROM pg_indexes
   WHERE tablename = 'vdbbench_table_test' AND indexname = 'pgvector_index';"

# 4. Run benchmark with --skip-load (fast, only tests search)
export POSTGRES_PASSWORD="dummy"
systemd-run --user --scope --unit=benchmark-client \
  -p MemoryMax=1536M -p MemoryHigh=1400M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 \
    --db-name vectordbbench --case-type Performance768D1M \
    --db-label "pgvector-4GB-lists512-probes16" \
    --lists 512 --probes 16 --skip-load --num-concurrency "1"
```

#### Method 2: Full Reload (Slower but ensures clean state)

```bash
# Use --load instead of --skip-load (rebuilds index from scratch)
export POSTGRES_PASSWORD="dummy"
systemd-run --user --scope --unit=benchmark-client \
  -p MemoryMax=1536M -p MemoryHigh=1400M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 \
    --db-name vectordbbench --case-type Performance768D1M \
    --db-label "pgvector-4GB-lists512-probes16" \
    --lists 512 --probes 16 \
    --drop-old --load --num-concurrency "1"
```

**Note**: `--load` requires sufficient `maintenance_work_mem`:
- 256 lists: 128MB minimum, 256MB recommended
- 512 lists: 256MB minimum, 512MB recommended
- 1024 lists: 512MB minimum, 1GB recommended

### Always Verify Your Index Configuration

Before running benchmarks, **always verify** the actual index configuration:

```bash
psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "SELECT indexname, indexdef FROM pg_indexes
   WHERE tablename = 'vdbbench_table_test';"
```

Expected output:
```
indexname    | indexdef
-------------+----------------------------------------------------------
pgvector_index | CREATE INDEX ... WITH (lists='512')
```

### Example: Testing Multiple Configurations Correctly

**Important**: PostgreSQL must already be running with memory limits before executing these commands. See the "Verified Working Example" section below for the complete workflow.

```bash
# Prerequisite: PostgreSQL already running with memory limits
# (See "Verified Working Example" section for how to start PostgreSQL)

# Test 1: 256 lists, 16 probes
psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 256);"

export POSTGRES_PASSWORD="dummy"
systemd-run --user --scope --unit=benchmark-256-16 \
  -p MemoryMax=1536M -p MemoryHigh=1400M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 \
    --db-name vectordbbench --case-type Performance768D1M \
    --db-label "pgvector-lists256-probes16" \
    --lists 256 --probes 16 --skip-load --num-concurrency "1"

# Wait for completion, then test 2: 512 lists, 16 probes
psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 512);"

systemd-run --user --scope --unit=benchmark-512-16 \
  -p MemoryMax=1536M -p MemoryHigh=1400M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 \
    --db-name vectordbbench --case-type Performance768D1M \
    --db-label "pgvector-lists512-probes16" \
    --lists 512 --probes 16 --skip-load --num-concurrency "1"
```

## Verified Working Example (Recommended)

This is the **tested and verified** procedure for running memory-limited benchmarks:

### Step 1: Cleanup Any Existing Processes

```bash
cd /home/kunhao/VectorDBBench
./cleanup_benchmark.sh
```

### Step 2: Start PostgreSQL with Memory Restrictions

```bash
# Example: 4GB PostgreSQL memory limit
systemd-run --user --scope \
  --unit=postgres-limited \
  -p MemoryMax=4G \
  -p MemoryHigh=3.6G \
  -p MemorySwapMax=0 \
  /home/kunhao/postgres-local/pg16/bin/postgres -D /home/kunhao/postgres-local/pgdata

# Wait for PostgreSQL to start
sleep 5

# Verify PostgreSQL is running
/home/kunhao/postgres-local/pg16/bin/pg_ctl -D /home/kunhao/postgres-local/pgdata status

# Verify memory limits are applied
systemctl --user show postgres-limited.scope | grep -E "Memory(Max|High|Swap)" | head -3
```

### Step 3: Rebuild Index with Desired Lists Parameter

**CRITICAL**: You MUST manually rebuild the index if you want to test different `lists` values. The `--lists` parameter is IGNORED when using `--skip-load`.

```bash
# Example: Rebuild with 512 lists
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 512);"

# ALWAYS verify the index configuration
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "SELECT indexname, indexdef FROM pg_indexes
   WHERE tablename = 'vdbbench_table_test' AND indexname = 'pgvector_index';"
```

Expected output should show: `WITH (lists='512')`

### Step 4: Run Benchmark with Memory Restrictions

```bash
# Example: 1.5GB client memory, 512 lists, 16 probes, Cohere 1M dataset
export POSTGRES_PASSWORD="dummy"
systemd-run --user --scope \
  --unit=benchmark-client \
  -p MemoryMax=1536M \
  -p MemoryHigh=1400M \
  -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao \
    --host localhost \
    --port 5433 \
    --db-name vectordbbench \
    --case-type Performance768D1M \
    --db-label "pgvector-4GB-1.5GB-lists512-probes16-cohere1M" \
    --lists 512 \
    --probes 16 \
    --skip-load \
    --num-concurrency "1"
```

**Note**: When using `--skip-load`:
- ✅ The `--probes` parameter WILL take effect (changes query-time behavior)
- ❌ The `--lists` parameter is IGNORED (must rebuild index manually as shown in Step 3)

### Verified Results Example

Using the procedure above with 4GB PostgreSQL + 1.5GB client on Cohere 1M dataset:

| Configuration | QPS | Recall | P99 Latency | P95 Latency |
|---------------|-----|--------|-------------|-------------|
| 256 lists, 16 probes | 10.92 | 94.38% | 119.2ms | 112.5ms |
| 512 lists, 16 probes | 21.79 | 89.96% | 63.9ms | 59.5ms |

**Key Insight**: 512 lists = 2x faster QPS but -4.4% recall (speed vs accuracy tradeoff)

## Quick Start (Alternative Using Scripts)

### 1. Verify Setup

```bash
cd /home/kunhao/VectorDBBench
./test_memory_limits.sh
```

This checks:
- systemd user instance is running
- cgroup support is available
- Memory limits can be applied
- PostgreSQL starts with limits
- Python/vectordbbench is available

### 2. Run Single Benchmark

```bash
# Basic usage with defaults (PG: 2GB, Client: 1GB, Lists: 256, Probes: 32)
./benchmark_memory_limited.sh

# Custom configuration
MEMORY_LIMIT_PG="4G" \
MEMORY_HIGH_PG="3.5G" \
MEMORY_LIMIT_CLIENT="1G" \
MEMORY_HIGH_CLIENT="900M" \
LISTS=512 \
PROBES=16 \
./benchmark_memory_limited.sh
```

### 3. Monitor Memory Usage

In a separate terminal:

```bash
./monitor_memory.sh
```

This shows real-time memory usage:
```
Time                 | PG Memory  | Client Memory | Total System
---------------------+------------+---------------+-------------
2025-10-27 15:30:45 |    1850 MB |        750 MB | 12500/32000 MB
2025-10-27 15:30:47 |    1900 MB |        800 MB | 12800/32000 MB
```

### 4. Run Full Benchmark Suite

```bash
./benchmark_suite_memory_limited.sh
```

Runs 6 configurations:
1. **Tight memory, medium lists**: 2GB PG, 1GB client, 256 lists, 32/16/8 probes
2. **Medium memory, high lists**: 4GB PG, 1GB client, 512 lists, 32/16 probes
3. **High memory, very high lists**: 8GB PG, 2GB client, 1024 lists, 32 probes

Results saved to:
- Results: `/home/kunhao/VectorDBBench/vectordb_bench/results/PgVector/memory_limited/`
- Logs: `/home/kunhao/benchmark_logs/`

### 5. Cleanup

```bash
./cleanup_benchmark.sh
```

## Configuration Parameters

### Memory Limits

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `MEMORY_LIMIT_PG` | PostgreSQL hard limit (OOM if exceeded) | 2G, 4G, 8G |
| `MEMORY_HIGH_PG` | PostgreSQL soft limit (throttles) | ~90% of max |
| `MEMORY_LIMIT_CLIENT` | Benchmark client hard limit | 1G, 2G |
| `MEMORY_HIGH_CLIENT` | Client soft limit | ~90% of max |

**Memory Swap**: Disabled (`MemorySwapMax=0`) to ensure predictable performance.

### Index Parameters

| Parameter | Description | Effect | Recommended |
|-----------|-------------|--------|-------------|
| `LISTS` | Number of IVF-Flat clusters | Higher = faster search, lower recall | 256, 512, 1024 |
| `PROBES` | Clusters searched per query | Higher = better recall, slower search | 8, 16, 32 |

### Recommended Configurations

#### For Realistic Disk I/O Testing (Tight Memory)
```bash
MEMORY_LIMIT_PG="2G"
MEMORY_LIMIT_CLIENT="1G"
LISTS=256
PROBES=32
```
**Effect**: Forces frequent disk I/O, measures worst-case performance.

#### For Moderate Caching (Medium Memory)
```bash
MEMORY_LIMIT_PG="4G"
MEMORY_LIMIT_CLIENT="1G"
LISTS=512
PROBES=16
```
**Effect**: ~50% of database can be cached, balanced performance.

#### For Minimal Cache Pressure (High Memory)
```bash
MEMORY_LIMIT_PG="8G"
MEMORY_LIMIT_CLIENT="2G"
LISTS=1024
PROBES=32
```
**Effect**: Most hot data cached, measures near-optimal performance.

## Understanding Results

### Metrics to Compare

**With Memory Limits** (realistic):
- Lower QPS (more disk I/O)
- Higher latency variance (cache misses)
- Consistent across runs
- Representative of production

**Without Memory Limits** (optimistic):
- Higher QPS (all in RAM)
- Lower latency variance (cache hits)
- First run slow, subsequent runs fast
- Not representative of production

### Example Results

| Configuration | QPS (1 thread) | P99 Latency | Recall |
|---------------|----------------|-------------|--------|
| No limit (warm cache) | ~60 QPS | 100ms | 98% |
| 2GB PG + 1GB client | ~25 QPS | 250ms | 98% |
| 4GB PG + 1GB client | ~40 QPS | 150ms | 98% |
| 8GB PG + 2GB client | ~55 QPS | 110ms | 98% |

## Troubleshooting

### PostgreSQL OOM Killed

**Symptom**: PostgreSQL stops, log shows "out of memory"

**Solutions**:
1. Increase `MEMORY_LIMIT_PG`
2. Reduce PostgreSQL `shared_buffers` (edit `$PGDATA/postgresql.conf`)
3. Reduce `maintenance_work_mem` if running with `--load`

### Benchmark Client OOM Killed

**Symptom**: Python process crashes, no results

**Solutions**:
1. Increase `MEMORY_LIMIT_CLIENT`
2. Ensure concurrency is 1 (multiple threads multiply memory usage)
3. Check if query dataset is pre-loaded in memory

### Memory Limits Not Applied

**Symptom**: Process uses more memory than limit

**Check**:
```bash
# View memory limits
systemctl --user show postgres-limited.scope | grep Memory

# View actual usage
systemctl --user status postgres-limited.scope
```

**Solutions**:
1. Verify systemd user instance: `systemctl --user status`
2. Check cgroup v2 support: `ls /sys/fs/cgroup/user.slice`
3. Run test script: `./test_memory_limits.sh`

### Benchmark Hangs

**Symptom**: Benchmark runs for hours without completing

**Possible causes**:
1. Memory limit too tight (constant swapping/throttling)
2. PostgreSQL `maintenance_work_mem` too low for index creation
3. Disk I/O is genuinely slow

**Check**:
```bash
# Monitor memory usage
./monitor_memory.sh

# Check PostgreSQL logs
tail -f $PGDATA/logfile

# Check if processes are throttled
systemd-cgtop --user
```

## Comparing with MyRocks

To fairly compare pgvector with MyRocks:

1. **Use same memory limits** for both databases
2. **Use comparable index parameters**:
   - pgvector `lists` ≈ MyRocks `centroids`
   - pgvector `probes` ≈ MyRocks `nprobe`
3. **Use single-threaded** queries (concurrency=1)
4. **Document memory configuration** in results

### Example: Fair Comparison

**pgvector**:
```bash
MEMORY_LIMIT_PG=4G LISTS=512 PROBES=16 ./benchmark_memory_limited.sh
```

**MyRocks** (equivalent):
```bash
# Run MyRocks with 4GB memory limit, 512 centroids, nprobe=16
# (requires similar memory limit setup for MyRocks)
```

## Advanced Usage

### Custom Benchmark Script

```bash
#!/bin/bash
# custom_benchmark.sh

# Stop PostgreSQL
$PGHOME/bin/pg_ctl -D $PGDATA stop

# Start with custom memory limit
systemd-run --user --scope \
  --unit=postgres-custom \
  -p MemoryMax=3G \
  -p MemoryHigh=2.7G \
  $PGHOME/bin/postgres -D $PGDATA

sleep 5

# Run custom benchmark
systemd-run --user --scope --wait \
  --unit=benchmark-custom \
  -p MemoryMax=1.5G \
  -p MemoryHigh=1.3G \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao \
    --host localhost \
    --port 5433 \
    --db-name vectordbbench \
    --case-type Performance768D1M \
    --db-label "custom-test" \
    --lists 384 \
    --probes 24 \
    --skip-load \
    --num-concurrency "1"

# Cleanup
systemctl --user stop postgres-custom.scope
```

### Verify Memory Limits Are Working

```bash
# Before benchmark: clear OS cache (requires sudo)
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# Run benchmark
./benchmark_memory_limited.sh

# Check that memory usage stays within limits
systemctl --user status postgres-limited.scope | grep Memory
```

## File Locations

- **Scripts**: `/home/kunhao/VectorDBBench/*.sh`
- **Results**: `/home/kunhao/VectorDBBench/vectordb_bench/results/PgVector/`
- **Logs**: `/home/kunhao/benchmark_logs/` (if configured)
- **PostgreSQL config**: `/home/kunhao/postgres-local/pgdata/postgresql.conf`
- **PostgreSQL logs**: `/home/kunhao/postgres-local/pgdata/logfile`

## References

- **ADMIN_BENCHMARK_GUIDE.md**: **READ THIS IF YOU HAVE SUDO ACCESS** - Complete guide for running realistic benchmarks with proper page cache dropping
- **systemd Resource Control**: https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html
- **cgroup v2**: https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html
- **pgvector Documentation**: https://github.com/pgvector/pgvector
- **PostgreSQL Memory Settings**: See `PGVECTOR_LOCAL_SETUP.md`

## Summary

These scripts enable **realistic, reproducible benchmarks** by:
1. ✅ Limiting both database and client memory
2. ✅ Preventing OS page cache from skewing results
3. ✅ Forcing realistic disk I/O patterns
4. ✅ Providing fair comparison basis across databases
5. ✅ Running single-threaded to avoid OOM issues

**Key Insight**: Without memory limits, you're benchmarking RAM, not your database.

## Correct Workflow Summary

**Always follow this order**:
1. **Cleanup** → Stop all existing processes
2. **Drop page cache** → **REQUIRES SUDO** (see ADMIN_BENCHMARK_GUIDE.md)
3. **Start PostgreSQL with memory limits** → Use systemd-run with MemoryMax
4. **Rebuild index** → Manually via psql if testing different lists values
5. **Verify index** → Always check the actual index configuration
6. **Run benchmark** → Use systemd-run for client with --skip-load

**Critical reminders**:
- ⚠️ **Must drop page cache before each run** (requires sudo, see ADMIN_BENCHMARK_GUIDE.md)
- ⚠️ systemd memory limits alone do NOT prevent page cache from caching database
- ⚠️ `--skip-load` does NOT rebuild the index
- ⚠️ `--lists` parameter is IGNORED when using `--skip-load`
- ✅ Only `--probes` can be changed with `--skip-load`
- ✅ Always verify index configuration before benchmarking
