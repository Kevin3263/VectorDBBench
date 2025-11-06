# Admin Guide: Running Realistic Memory-Limited Benchmarks (Requires sudo)

## For Server Administrators

This guide is for someone with **sudo access** who wants to run **realistic** vector database benchmarks. This guide assumes you may be using AI assistants like Claude Code to help run these commands.

---

## The Problem We Discovered

When running benchmarks with systemd memory limits alone, we discovered a critical issue:

### What We Expected
- PostgreSQL limited to 2GB RAM
- Database size: 8.7GB
- Result: Heavy disk I/O, realistic performance testing

### What Actually Happened
- PostgreSQL process: Only using 130MB RAM
- OS Page Cache: Caching the entire 8.7GB database
- Result: All queries served from RAM cache (unrealistic performance)

**Root Cause**: systemd's `MemoryMax` only limits **process RSS (Resident Set Size)**, NOT the **Linux page cache**. The OS still caches database files in the page cache, which is shared system-wide and not counted against the cgroup memory limits.

## Why This Matters

Without dropping the page cache, you're benchmarking:
- ❌ RAM performance (entire database cached)
- ❌ Unrealistic QPS numbers (too high)
- ❌ Unrealistic latencies (too low)

With proper page cache clearing, you benchmark:
- ✅ Actual disk I/O performance
- ✅ Realistic production-like behavior
- ✅ True memory pressure scenarios

---

## Prerequisites

1. **sudo access** (required to drop page cache)
2. **systemd with cgroup support** (check: `systemctl --version`)
3. **PostgreSQL installed** at `/home/kunhao/postgres-local/pg16/`
4. **VectorDBBench repository** at `/home/kunhao/VectorDBBench/`
5. **Dataset loaded** (Cohere 1M - see main docs for loading instructions)

---

## Complete Benchmark Workflow (The Correct Way)

### Step 0: Initial Setup (One-time)

First, understand your system's memory:

```bash
# Check total system memory
free -h

# Check database size
du -sh /home/kunhao/postgres-local/pgdata
```

**Example Output:**
```
Total memory: 62GB
Database size: 8.7GB
```

Choose memory limits that force disk I/O:
- **Tight**: PostgreSQL 2GB + Client 1GB (for 8GB database = heavy disk I/O)
- **Moderate**: PostgreSQL 4GB + Client 1.5GB (for 8GB database = some disk I/O)
- **Goal**: Database size should be **larger** than PostgreSQL memory limit

---

### Step 1: Cleanup Previous Processes

Always start fresh:

```bash
cd /home/kunhao/VectorDBBench
./cleanup_benchmark.sh
```

**Expected Output:**
```
Cleaning up benchmark processes...
PostgreSQL is not running
Cleanup complete!
```

---

### Step 2: Drop Page Cache (CRITICAL - Requires sudo)

This is the most important step that we were missing:

```bash
# Sync filesystem to ensure all data is written to disk
sync

# Drop page cache (requires sudo)
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
```

**What this does:**
- `echo 1`: Drop page cache only
- `echo 2`: Drop dentries and inodes
- `echo 3`: Drop both (recommended for benchmarking)

**Verify page cache was cleared:**

```bash
# Before dropping cache, you might see:
free -h
# buff/cache: 24GB  ← Database is cached here

# After dropping cache:
free -h
# buff/cache: ~500MB  ← Much less cached
```

**IMPORTANT**: The page cache will start filling up again as soon as PostgreSQL reads data. That's expected and realistic - the benchmark will measure how performance changes as the cache fills.

---

### Step 3: Start PostgreSQL with Memory Limits

Start PostgreSQL **immediately after dropping cache** (before cache refills):

```bash
# Example: 2GB PostgreSQL memory limit
systemd-run --user --scope \
  --unit=postgres-limited \
  -p MemoryMax=2G \
  -p MemoryHigh=1.8G \
  -p MemorySwapMax=0 \
  /home/kunhao/postgres-local/pg16/bin/postgres -D /home/kunhao/postgres-local/pgdata
```

**Parameters Explained:**
- `MemoryMax=2G`: Hard limit (process killed if exceeded)
- `MemoryHigh=1.8G`: Soft limit (process throttled if exceeded)
- `MemorySwapMax=0`: No swap (prevents slow swap performance)

**Wait for startup:**

```bash
sleep 5
```

**Verify PostgreSQL is running:**

```bash
/home/kunhao/postgres-local/pg16/bin/pg_ctl -D /home/kunhao/postgres-local/pgdata status
```

Expected: `pg_ctl: server is running (PID: XXXXX)`

**Verify memory limits are applied:**

```bash
systemctl --user show postgres-limited.scope | grep -E "Memory(Max|High|Swap)" | head -3
```

Expected output:
```
MemoryHigh=1932735283  (1.8GB in bytes)
MemoryMax=2147483648   (2GB in bytes)
MemorySwapMax=0
```

---

### Step 4: Rebuild Index (If Testing Different Lists Values)

**CRITICAL UNDERSTANDING**: When using `--skip-load`, the `--lists` parameter is **IGNORED**. You must manually rebuild the index to test different `lists` values.

```bash
# Example: Rebuild with 512 lists (centroids)
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 512);"
```

**Common list values to test:**
- `lists = 256`: Fewer clusters, higher recall, slower search
- `lists = 512`: Balanced performance
- `lists = 1024`: More clusters, lower recall, faster search

**ALWAYS verify the index was created correctly:**

```bash
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "SELECT indexname, indexdef FROM pg_indexes
   WHERE tablename = 'vdbbench_table_test' AND indexname = 'pgvector_index';"
```

**Expected output:**
```
   indexname    |                          indexdef
----------------+--------------------------------------------------------------------------------------------------------
 pgvector_index | CREATE INDEX pgvector_index ON public.vdbbench_table_test USING ivfflat (embedding) WITH (lists='512')
```

Confirm the `lists='512'` matches what you intended!

---

### Step 5: Run Benchmark with Memory Limits

Now run the actual benchmark:

```bash
# Set password environment variable
export POSTGRES_PASSWORD="dummy"

# Example: 1GB client memory, 512 lists, 16 probes, Cohere 1M dataset
systemd-run --user --scope \
  --unit=benchmark-client \
  -p MemoryMax=1G \
  -p MemoryHigh=900M \
  -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao \
    --host localhost \
    --port 5433 \
    --db-name vectordbbench \
    --case-type Performance768D1M \
    --db-label "pgvector-2GB-1GB-lists512-probes16-cohere1M" \
    --lists 512 \
    --probes 16 \
    --skip-load \
    --num-concurrency "1"
```

**Parameters Explained:**

**Memory Limits:**
- `MemoryMax=1G`: Client memory limit (should be enough for query processing)
- `MemoryHigh=900M`: Soft limit

**Benchmark Parameters:**
- `--case-type Performance768D1M`: Cohere 1M dataset (768 dimensions)
- `--lists 512`: **IGNORED when using --skip-load** (only for display)
- `--probes 16`: Number of clusters to search (DOES take effect)
- `--skip-load`: Don't reload data (faster, only runs search benchmark)
- `--num-concurrency "1"`: Single-threaded (avoid memory multiplier issues)

**Important Notes:**
- When using `--skip-load`, only `--probes` can be changed
- To test different `--lists` values, you MUST rebuild the index manually (Step 4)
- The benchmark will take ~2-3 minutes to complete

---

### Step 6: Monitor Progress (Optional)

While the benchmark is running, you can monitor in another terminal:

```bash
# Watch memory usage
watch -n 2 'free -h'

# Watch PostgreSQL processes
watch -n 2 'ps aux | grep postgres | grep -v grep'

# Check page cache growth
watch -n 5 'free -h | grep Mem'
```

You should see the `buff/cache` value gradually increase as PostgreSQL reads data - this is normal and realistic!

---

### Step 7: Review Results

The benchmark will print results and save to:

```
/home/kunhao/VectorDBBench/vectordb_bench/results/PgVector/result_YYYYMMDD_<UUID>_pgvector.json
```

**Key Metrics to Look For:**

```
QPS: X.XX            # Queries per second
Recall: 0.XXXX       # Accuracy (0.90 = 90%)
P99 Latency: X.XX ms # 99th percentile latency
P95 Latency: X.XX ms # 95th percentile latency
```

**Example Results (WITH proper cache dropping):**

Expected realistic results:
- QPS: 15-25 (much lower than uncached 60+)
- Recall: 85-95% depending on lists/probes
- P99 Latency: 60-120ms (higher due to disk I/O)

**Red Flag Results (without cache dropping):**
- QPS: 50-60+ (suspiciously high)
- P99 Latency: <30ms (too fast, likely cached)

---

## Testing Multiple Configurations

To test different configurations, **always drop cache between runs**:

```bash
# Configuration 1: 256 lists, 16 probes
./cleanup_benchmark.sh
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
# ... start postgres, rebuild index with 256 lists, run benchmark

# Configuration 2: 512 lists, 16 probes
./cleanup_benchmark.sh
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
# ... start postgres, rebuild index with 512 lists, run benchmark

# Configuration 3: 512 lists, 32 probes (only change probes)
# NO need to rebuild index, just run benchmark again with --probes 32
```

---

## Verification Checklist

Before trusting your benchmark results, verify:

- [ ] Page cache was dropped (check `free -h` before starting)
- [ ] PostgreSQL memory limits are set (`systemctl --user show postgres-limited.scope`)
- [ ] Index configuration is correct (`SELECT indexdef FROM pg_indexes`)
- [ ] Database size > PostgreSQL memory limit (forces disk I/O)
- [ ] QPS is realistic (not suspiciously high like >50 for cold cache)
- [ ] Results saved to disk successfully

---

## Understanding Memory Behavior

### What systemd Limits Control:
- ✅ Process RSS (Resident Set Size)
- ✅ PostgreSQL shared_buffers
- ✅ Work memory allocations

### What systemd Limits DON'T Control:
- ❌ Linux page cache (kernel-managed)
- ❌ Filesystem buffers
- ❌ Memory-mapped I/O cache

### Why Page Cache Matters:
```
Without dropping cache:
Query → Check page cache → HIT → Return from RAM (fast)

With cache dropped:
Query → Check page cache → MISS → Read from disk (realistic)
       → Cache result → Next query might hit cache (realistic)
```

---

## Complete Example: Running 3 Configurations

Here's a complete example of testing 3 different configurations correctly:

```bash
# ============================================================
# Test 1: 256 lists, 16 probes, 2GB PostgreSQL
# ============================================================

cd /home/kunhao/VectorDBBench
./cleanup_benchmark.sh

# Drop cache (CRITICAL)
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Start PostgreSQL
systemd-run --user --scope --unit=postgres-limited \
  -p MemoryMax=2G -p MemoryHigh=1.8G -p MemorySwapMax=0 \
  /home/kunhao/postgres-local/pg16/bin/postgres -D /home/kunhao/postgres-local/pgdata

sleep 5

# Rebuild index with 256 lists
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 256);"

# Run benchmark
export POSTGRES_PASSWORD="dummy"
systemd-run --user --scope --unit=benchmark-client \
  -p MemoryMax=1G -p MemoryHigh=900M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 \
    --db-name vectordbbench --case-type Performance768D1M \
    --db-label "pgvector-2GB-1GB-lists256-probes16-cohere1M-COLD" \
    --lists 256 --probes 16 --skip-load --num-concurrency "1"

# Wait for completion...

# ============================================================
# Test 2: 512 lists, 16 probes, 2GB PostgreSQL
# ============================================================

./cleanup_benchmark.sh

# Drop cache again (CRITICAL for each test)
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Start PostgreSQL
systemd-run --user --scope --unit=postgres-limited \
  -p MemoryMax=2G -p MemoryHigh=1.8G -p MemorySwapMax=0 \
  /home/kunhao/postgres-local/pg16/bin/postgres -D /home/kunhao/postgres-local/pgdata

sleep 5

# Rebuild index with 512 lists
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -d vectordbbench -c \
  "DROP INDEX IF EXISTS pgvector_index;
   CREATE INDEX pgvector_index ON vdbbench_table_test
   USING ivfflat (embedding vector_l2_ops) WITH (lists = 512);"

# Run benchmark
systemd-run --user --scope --unit=benchmark-client \
  -p MemoryMax=1G -p MemoryHigh=900M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 \
    --db-name vectordbbench --case-type Performance768D1M \
    --db-label "pgvector-2GB-1GB-lists512-probes16-cohere1M-COLD" \
    --lists 512 --probes 16 --skip-load --num-concurrency "1"

# ============================================================
# Test 3: 512 lists, 32 probes, 2GB PostgreSQL
# ============================================================

./cleanup_benchmark.sh

# Drop cache
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Start PostgreSQL
systemd-run --user --scope --unit=postgres-limited \
  -p MemoryMax=2G -p MemoryHigh=1.8G -p MemorySwapMax=0 \
  /home/kunhao/postgres-local/pg16/bin/postgres -D /home/kunhao/postgres-local/pgdata

sleep 5

# Index already has 512 lists, just need to change probes
# Run benchmark with different probes
systemd-run --user --scope --unit=benchmark-client \
  -p MemoryMax=1G -p MemoryHigh=900M -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name kunhao --host localhost --port 5433 \
    --db-name vectordbbench --case-type Performance768D1M \
    --db-label "pgvector-2GB-1GB-lists512-probes32-cohere1M-COLD" \
    --lists 512 --probes 32 --skip-load --num-concurrency "1"
```

---

## Expected Results Comparison

Based on proper testing with cache dropping:

| Configuration | Cache | QPS | Recall | P99 Latency |
|---------------|-------|-----|--------|-------------|
| 512 lists, 16 probes | Warm (wrong) | 21.79 | 89.96% | 63.9ms |
| 512 lists, 16 probes | Cold (correct) | ~15-18 | 89-91% | ~80-100ms |

The cold cache results should show:
- **Lower QPS** (more disk I/O)
- **Higher latency** (disk reads)
- **More realistic** production behavior

---

## For AI Assistants (Claude Code, etc.)

If you're using an AI assistant to help run these benchmarks:

### Tell your AI assistant:

1. **Always drop page cache between benchmark runs:**
   ```bash
   sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
   ```

2. **The correct workflow is:**
   - Cleanup → Drop cache → Start PostgreSQL → Rebuild index → Run benchmark

3. **When testing different `lists` values:**
   - Must rebuild index manually via psql
   - Cannot use `--skip-load` to change lists

4. **When testing different `probes` values:**
   - Can keep same index
   - Just change `--probes` parameter in benchmark command

5. **Verify results make sense:**
   - QPS should be 15-25 for cold cache with 2GB memory
   - QPS >50 indicates warm cache (benchmark invalid)

---

## Troubleshooting

### Problem: Results too fast (QPS >50)

**Cause**: Page cache not dropped

**Solution**:
```bash
# Check cache size
free -h  # Look at buff/cache column

# Drop cache
sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Verify it dropped
free -h  # buff/cache should be much smaller
```

### Problem: PostgreSQL won't start

**Cause**: Previous instance still running

**Solution**:
```bash
./cleanup_benchmark.sh
# Check if really stopped
ps aux | grep postgres
```

### Problem: Benchmark hangs or OOMs

**Cause**: Memory limit too tight

**Solution**:
- Increase PostgreSQL memory limit
- Increase client memory limit
- Check `maintenance_work_mem` setting

### Problem: Can't verify index configuration

**Cause**: PostgreSQL not running or wrong database

**Solution**:
```bash
# Check PostgreSQL status
/home/kunhao/postgres-local/pg16/bin/pg_ctl -D /home/kunhao/postgres-local/pgdata status

# List all databases
/home/kunhao/postgres-local/pg16/bin/psql -p 5433 -h localhost -U kunhao -l
```

---

## Summary: The Correct Workflow

```
1. Cleanup previous processes
2. Drop page cache (REQUIRES SUDO) ← CRITICAL STEP
3. Start PostgreSQL with memory limits
4. Rebuild index if testing different lists values
5. Run benchmark
6. Review results (should show realistic disk I/O impact)
```

**The most important lesson**: systemd memory limits alone are NOT enough. You MUST drop the page cache to get realistic benchmark results.

---

## Additional Resources

- Main documentation: `MEMORY_LIMITED_BENCHMARKS.md`
- pgvector setup: `PGVECTOR_LOCAL_SETUP.md`
- MyRocks implementation: `MYROCKS_IMPLEMENTATION.md`
- systemd resource control: https://www.freedesktop.org/software/systemd/man/systemd.resource-control.html

---

**Questions?** This guide was created based on actual testing that discovered the page cache issue. If results still seem unrealistic, verify the cache was actually dropped using `free -h` before and after.
