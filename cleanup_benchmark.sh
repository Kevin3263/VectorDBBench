#!/bin/bash
#
# Cleanup all benchmark processes and restart PostgreSQL normally
#

echo "Cleaning up benchmark processes..."

# Stop all benchmark-related systemd units
systemctl --user stop postgres-limited.scope 2>/dev/null || true
systemctl --user stop postgres-test.scope 2>/dev/null || true
systemctl --user stop benchmark-client.scope 2>/dev/null || true

# Wait for cleanup
sleep 2

# Show status
echo ""
echo "Active benchmark units:"
systemctl --user list-units 'postgres-*' 'benchmark-*' --all

echo ""
echo "PostgreSQL status:"
/home/kunhao/postgres-local/pg16/bin/pg_ctl -D /home/kunhao/postgres-local/pgdata status || echo "PostgreSQL is not running"

echo ""
echo "Cleanup complete!"
echo ""
echo "To start PostgreSQL normally (without memory limits):"
echo "  \$PGHOME/bin/pg_ctl -D \$PGDATA -l \$PGDATA/logfile start"
echo ""
