#!/bin/bash
#
# Memory-Limited PostgreSQL + Benchmark Client
# This script runs both PostgreSQL and the benchmark client with strict memory limits
#

set -e

# Configuration
MEMORY_LIMIT_PG="${MEMORY_LIMIT_PG:-2G}"        # PostgreSQL memory limit
MEMORY_HIGH_PG="${MEMORY_HIGH_PG:-1.8G}"       # PostgreSQL soft limit (throttle)
MEMORY_LIMIT_CLIENT="${MEMORY_LIMIT_CLIENT:-1G}"    # Benchmark client memory limit
MEMORY_HIGH_CLIENT="${MEMORY_HIGH_CLIENT:-900M}"   # Client soft limit

# PostgreSQL paths
PGHOME=/home/kunhao/postgres-local/pg16
PGDATA=/home/kunhao/postgres-local/pgdata
PGPORT=5433

# Benchmark configuration
DB_USER="kunhao"
DB_NAME="vectordbbench"
CASE_TYPE="Performance768D1M"
LISTS="${LISTS:-256}"
PROBES="${PROBES:-32}"
DB_LABEL="pgvector-limited-pg${MEMORY_LIMIT_PG}-client${MEMORY_LIMIT_CLIENT}-lists${LISTS}-probes${PROBES}"

echo "=========================================="
echo "Memory-Limited Benchmark Configuration"
echo "=========================================="
echo "PostgreSQL memory limit: $MEMORY_LIMIT_PG (soft: $MEMORY_HIGH_PG)"
echo "Client memory limit: $MEMORY_LIMIT_CLIENT (soft: $MEMORY_HIGH_CLIENT)"
echo "Concurrency: 1 (single-threaded)"
echo "Lists: $LISTS, Probes: $PROBES"
echo "Label: $DB_LABEL"
echo "=========================================="
echo ""

# Step 1: Stop any existing PostgreSQL
echo "[1/5] Stopping existing PostgreSQL..."
$PGHOME/bin/pg_ctl -D $PGDATA stop -m fast 2>/dev/null || true
systemctl --user stop postgres-limited.scope 2>/dev/null || true
sleep 3

# Step 2: Start PostgreSQL with memory limit
echo "[2/5] Starting PostgreSQL with memory limit: $MEMORY_LIMIT_PG..."
systemd-run --user --scope \
  --unit=postgres-limited \
  -p MemoryMax=$MEMORY_LIMIT_PG \
  -p MemoryHigh=$MEMORY_HIGH_PG \
  -p MemorySwapMax=0 \
  $PGHOME/bin/postgres -D $PGDATA

sleep 5

# Verify PostgreSQL is running
if ! $PGHOME/bin/pg_ctl -D $PGDATA status > /dev/null 2>&1; then
    echo "ERROR: PostgreSQL failed to start"
    exit 1
fi

echo "✓ PostgreSQL started successfully"

# Step 3: Display memory configuration
echo ""
echo "[3/5] Verifying configuration..."
echo "PostgreSQL shared_buffers: $($PGHOME/bin/psql -p $PGPORT -U $DB_USER -d $DB_NAME -tAc 'SHOW shared_buffers;')"
echo "PostgreSQL memory limits:"
systemctl --user show postgres-limited.scope | grep -E "Memory(Max|High|Swap)" | head -3

# Step 4: Check database size
echo ""
echo "Database statistics:"
$PGHOME/bin/psql -p $PGPORT -U $DB_USER -d $DB_NAME -c "
SELECT
    pg_size_pretty(pg_database_size('$DB_NAME')) as db_size,
    (SELECT COUNT(*) FROM vdbbench_table_test) as row_count;
"

# Step 5: Run benchmark with memory limit
echo ""
echo "[4/5] Running memory-limited benchmark..."
echo "Starting at: $(date)"
echo ""

# Set password environment variable
export POSTGRES_PASSWORD="dummy"

# Run benchmark client with memory limit
systemd-run --user --scope --wait \
  --unit=benchmark-client \
  -p MemoryMax=$MEMORY_LIMIT_CLIENT \
  -p MemoryHigh=$MEMORY_HIGH_CLIENT \
  -p MemorySwapMax=0 \
  python -m vectordb_bench.cli.vectordbbench pgvectorivfflat \
    --user-name "$DB_USER" \
    --host localhost \
    --port $PGPORT \
    --db-name "$DB_NAME" \
    --case-type "$CASE_TYPE" \
    --db-label "$DB_LABEL" \
    --lists $LISTS \
    --probes $PROBES \
    --skip-load \
    --num-concurrency "1"

BENCHMARK_EXIT_CODE=$?

echo ""
echo "Finished at: $(date)"

# Step 6: Cleanup
echo ""
echo "[5/5] Cleanup..."
systemctl --user stop postgres-limited.scope
sleep 2

echo ""
echo "=========================================="
if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo "✓ Benchmark completed successfully"
else
    echo "✗ Benchmark failed with exit code: $BENCHMARK_EXIT_CODE"
fi
echo "=========================================="

exit $BENCHMARK_EXIT_CODE
