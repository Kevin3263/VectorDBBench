#!/bin/bash
#
# Test Memory Limits - Verify systemd cgroup constraints work
#

echo "=========================================="
echo "Testing Memory Limit Configuration"
echo "=========================================="
echo ""

# Test 1: Check systemd user instance
echo "[Test 1] Checking systemd user instance..."
if systemctl --user status > /dev/null 2>&1; then
    echo "✓ systemd user instance is running"
else
    echo "✗ systemd user instance is NOT running"
    echo "  Run: systemctl --user status"
    exit 1
fi
echo ""

# Test 2: Check cgroup support
echo "[Test 2] Checking cgroup support..."
USER_ID=$(id -u)
if [ -d "/sys/fs/cgroup/memory/user.slice/user-${USER_ID}.slice" ] || [ -d "/sys/fs/cgroup/user.slice" ]; then
    echo "✓ cgroup user slice exists for user $USER_ID"
    CGROUP_VERSION="v1"
    if [ -d "/sys/fs/cgroup/unified" ]; then
        echo "  (cgroup v1 with v2 available)"
    fi
else
    echo "✗ cgroup user slice not found"
    exit 1
fi
echo ""

# Test 3: Test memory limit on simple process
echo "[Test 3] Testing memory limit on simple process..."
TEST_OUTPUT=$(systemd-run --user --scope -p MemoryMax=100M echo "Memory limit test" 2>&1)
if echo "$TEST_OUTPUT" | grep -q "Memory limit test"; then
    echo "✓ Memory limits can be applied to processes"
    echo "  $TEST_OUTPUT"
else
    echo "✗ Failed to apply memory limits"
    echo "  $TEST_OUTPUT"
    exit 1
fi
echo ""

# Test 4: Start PostgreSQL with memory limit (brief test)
echo "[Test 4] Testing PostgreSQL with memory limit..."
PGHOME=/home/kunhao/postgres-local/pg16
PGDATA=/home/kunhao/postgres-local/pgdata

# Stop any existing PostgreSQL
$PGHOME/bin/pg_ctl -D $PGDATA stop -m fast 2>/dev/null || true
systemctl --user stop postgres-limited.scope 2>/dev/null || true
sleep 2

# Start with memory limit
systemd-run --user --scope \
  --unit=postgres-test \
  -p MemoryMax=2G \
  -p MemoryHigh=1.8G \
  $PGHOME/bin/postgres -D $PGDATA > /dev/null 2>&1

sleep 5

# Check if running
if $PGHOME/bin/pg_ctl -D $PGDATA status > /dev/null 2>&1; then
    echo "✓ PostgreSQL started with memory limit"

    # Check memory limit is applied
    MEM_MAX=$(systemctl --user show postgres-test.scope -p MemoryMax | cut -d= -f2)
    if [ "$MEM_MAX" != "infinity" ] && [ -n "$MEM_MAX" ]; then
        echo "✓ Memory limit verified: $((MEM_MAX / 1024 / 1024)) MB"
    else
        echo "✗ Memory limit not applied correctly"
    fi

    # Check current memory usage
    MEM_CURRENT=$(systemctl --user show postgres-test.scope -p MemoryCurrent | cut -d= -f2)
    echo "  Current memory usage: $((MEM_CURRENT / 1024 / 1024)) MB"
else
    echo "✗ PostgreSQL failed to start with memory limit"
    exit 1
fi

# Cleanup
systemctl --user stop postgres-test.scope
sleep 2

echo ""

# Test 5: Test Python with memory limit
echo "[Test 5] Testing Python with memory limit..."
TEST_PYTHON=$(systemd-run --user --scope --wait \
  -p MemoryMax=500M \
  python3 -c "import sys; print('Python version:', sys.version.split()[0])" 2>&1)

if echo "$TEST_PYTHON" | grep -q "Python version:"; then
    echo "✓ Python can run with memory limit"
    echo "  $(echo "$TEST_PYTHON" | grep "Python version:")"
else
    echo "✗ Python failed with memory limit"
    exit 1
fi
echo ""

# Test 6: Check if vectordbbench module exists
echo "[Test 6] Checking vectordbbench module..."
if python3 -c "import vectordb_bench" 2>/dev/null; then
    echo "✓ vectordbbench module is available"
else
    echo "⚠ vectordbbench module not found"
    echo "  Make sure to activate conda environment:"
    echo "  conda activate vectordbbench"
fi
echo ""

echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "You can now run memory-limited benchmarks:"
echo "  ./benchmark_memory_limited.sh"
echo ""
echo "Or run the full suite:"
echo "  ./benchmark_suite_memory_limited.sh"
echo ""
echo "Monitor memory usage in a separate terminal:"
echo "  ./monitor_memory.sh"
echo ""
