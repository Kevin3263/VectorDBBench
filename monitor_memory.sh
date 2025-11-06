#!/bin/bash
#
# Monitor memory usage of PostgreSQL and benchmark client
# Run this in a separate terminal while benchmark is running
#

echo "Monitoring memory usage (Ctrl+C to stop)..."
echo ""
echo "Time                 | PG Memory  | Client Memory | Total System"
echo "---------------------+------------+---------------+-------------"

while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Get PostgreSQL memory usage (if running)
    if systemctl --user is-active postgres-limited.scope > /dev/null 2>&1; then
        PG_MEM=$(systemctl --user show postgres-limited.scope -p MemoryCurrent | cut -d= -f2)
        PG_MEM_MB=$((PG_MEM / 1024 / 1024))
    else
        PG_MEM_MB="N/A"
    fi

    # Get benchmark client memory usage (if running)
    if systemctl --user is-active benchmark-client.scope > /dev/null 2>&1; then
        CLIENT_MEM=$(systemctl --user show benchmark-client.scope -p MemoryCurrent | cut -d= -f2)
        CLIENT_MEM_MB=$((CLIENT_MEM / 1024 / 1024))
    else
        CLIENT_MEM_MB="N/A"
    fi

    # Get total system memory usage
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%s/%s MB", $3,$2}')

    printf "%s | %10s | %13s | %s\n" \
        "$TIMESTAMP" \
        "${PG_MEM_MB} MB" \
        "${CLIENT_MEM_MB} MB" \
        "$TOTAL_MEM"

    sleep 2
done
