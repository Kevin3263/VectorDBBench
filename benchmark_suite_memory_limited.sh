#!/bin/bash
#
# Comprehensive Memory-Limited Benchmark Suite
# Tests different memory configurations and index parameters
#

set -e

# Configuration
RESULTS_DIR="/home/kunhao/VectorDBBench/vectordb_bench/results/PgVector/memory_limited"
LOG_DIR="/home/kunhao/benchmark_logs"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Test configurations
# Format: "PG_MEMORY|CLIENT_MEMORY|LISTS|PROBES|DESCRIPTION"
CONFIGS=(
    "2G|1G|256|32|Tight memory, medium lists, high probes"
    "2G|1G|256|16|Tight memory, medium lists, medium probes"
    "2G|1G|256|8|Tight memory, medium lists, low probes"
    "4G|1G|512|32|Medium memory, high lists, high probes"
    "4G|1G|512|16|Medium memory, high lists, medium probes"
    "8G|2G|1024|32|High memory, very high lists, high probes"
)

echo "=========================================="
echo "Memory-Limited Benchmark Suite"
echo "=========================================="
echo "Total configurations: ${#CONFIGS[@]}"
echo "Results directory: $RESULTS_DIR"
echo "Log directory: $LOG_DIR"
echo "=========================================="
echo ""

# Run each configuration
CONFIG_NUM=1
TOTAL_CONFIGS=${#CONFIGS[@]}

for config in "${CONFIGS[@]}"; do
    IFS='|' read -r PG_MEM CLIENT_MEM LISTS PROBES DESC <<< "$config"

    echo ""
    echo "=========================================="
    echo "Configuration $CONFIG_NUM/$TOTAL_CONFIGS"
    echo "=========================================="
    echo "Description: $DESC"
    echo "PostgreSQL memory: $PG_MEM"
    echo "Client memory: $CLIENT_MEM"
    echo "Lists: $LISTS, Probes: $PROBES"
    echo "=========================================="
    echo ""

    # Generate log filename
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/benchmark_${TIMESTAMP}_pg${PG_MEM}_client${CLIENT_MEM}_lists${LISTS}_probes${PROBES}.log"

    # Export configuration for the benchmark script
    export MEMORY_LIMIT_PG="$PG_MEM"
    export MEMORY_HIGH_PG=$(echo "$PG_MEM" | sed 's/G$//')
    export MEMORY_HIGH_PG="${MEMORY_HIGH_PG}00M"  # e.g., 2G -> 1800M
    export MEMORY_LIMIT_CLIENT="$CLIENT_MEM"
    export MEMORY_HIGH_CLIENT=$(echo "$CLIENT_MEM" | sed 's/G$//')
    export MEMORY_HIGH_CLIENT="${MEMORY_HIGH_CLIENT}00M"
    export LISTS="$LISTS"
    export PROBES="$PROBES"

    # Run benchmark
    if /home/kunhao/benchmark_memory_limited.sh 2>&1 | tee "$LOG_FILE"; then
        echo "✓ Configuration $CONFIG_NUM completed successfully"
    else
        echo "✗ Configuration $CONFIG_NUM failed"
    fi

    # Wait between runs
    echo ""
    echo "Waiting 10 seconds before next configuration..."
    sleep 10

    ((CONFIG_NUM++))
done

echo ""
echo "=========================================="
echo "All benchmarks completed!"
echo "=========================================="
echo "Results: $RESULTS_DIR"
echo "Logs: $LOG_DIR"
echo ""

# Generate summary
echo "Generating summary..."
ls -lh "$LOG_DIR"
