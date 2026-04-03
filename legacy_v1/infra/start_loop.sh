#!/bin/bash
# Usage: ./start_loop.sh [--hours H] [--max-experiments N]
# Defaults: --hours 8 --max-experiments 200
# ANTHROPIC_API_KEY is injected by deploy.sh at deploy time — do not hardcode
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?ERROR: ANTHROPIC_API_KEY not set}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

LOOP_ARGS="$@"

pkill -f run_loop.py 2>/dev/null || true
sleep 2

cd /root

# With tini as PID 1 (proper init), nohup is sufficient.
# tini reaps zombie children and forwards signals — no need for screen.
nohup /opt/conda/bin/python -u run_loop.py $LOOP_ARGS > /root/loop.log 2>&1 &
LOOP_PID=$!
echo "Loop PID: $LOOP_PID"

sleep 4
if kill -0 $LOOP_PID 2>/dev/null; then
    echo "Loop running OK (PID $LOOP_PID)"
    tail -10 /root/loop.log 2>/dev/null
else
    echo "ERROR: Loop died immediately"
    cat /root/loop.log
    exit 1
fi

# Start watchdog if available
if [[ -f /root/watchdog.sh ]]; then
    nohup bash /root/watchdog.sh > /root/watchdog.log 2>&1 &
    echo "Watchdog PID: $!"
fi
