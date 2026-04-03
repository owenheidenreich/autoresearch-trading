#!/bin/bash
# Watchdog: monitors container health every 5 seconds.
# Runs INDEPENDENTLY of the main loop to catch exactly when/why things die.
# Usage: nohup bash watchdog.sh > /root/watchdog.log 2>&1 &

LOG=/root/watchdog.log
DIAG=/root/diagnostics.log

echo "$(date -u '+%H:%M:%S') WATCHDOG START (PID $$)" | tee -a "$LOG"

while true; do
    TS=$(date -u '+%H:%M:%S')

    # 1. Is PID 1 alive? (if this dies, container restarts)
    PID1_CMD=$(cat /proc/1/cmdline 2>/dev/null | tr '\0' ' ' | head -c 60)
    PID1_OK="YES"
    [[ -z "$PID1_CMD" ]] && PID1_OK="DEAD"

    # 2. Is run_loop.py running?
    LOOP_PID=$(pgrep -f "run_loop.py" 2>/dev/null | head -1)
    LOOP_OK="YES(pid=$LOOP_PID)"
    [[ -z "$LOOP_PID" ]] && LOOP_OK="DEAD"

    # 3. Is any training subprocess running?
    TRAIN_PID=$(pgrep -f "train.py" 2>/dev/null | head -1)
    TRAIN_OK=""
    [[ -n "$TRAIN_PID" ]] && TRAIN_OK=" train=YES(pid=$TRAIN_PID)"

    # 4. Cgroup memory
    CGROUP_MB="?"
    if [[ -f /sys/fs/cgroup/memory.current ]]; then
        CGROUP_BYTES=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
        CGROUP_MB=$(echo "$CGROUP_BYTES / 1048576" | bc 2>/dev/null || echo "?")
    elif [[ -f /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
        CGROUP_BYTES=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null)
        CGROUP_MB=$(echo "$CGROUP_BYTES / 1048576" | bc 2>/dev/null || echo "?")
    fi

    # 5. Total process count
    NUM_PIDS=$(ls -1 /proc/[0-9]* -d 2>/dev/null | wc -l)

    # 6. OOM events (from dmesg, if accessible)
    OOM=""
    DMESG_OOM=$(dmesg 2>/dev/null | grep -ci "out of memory\|oom-kill\|killed process" 2>/dev/null)
    [[ "$DMESG_OOM" -gt 0 ]] 2>/dev/null && OOM=" OOM_EVENTS=$DMESG_OOM"

    # 7. Run_loop RSS (if alive)
    LOOP_RSS=""
    if [[ -n "$LOOP_PID" ]] && [[ -f "/proc/$LOOP_PID/status" ]]; then
        LOOP_RSS_KB=$(grep VmRSS "/proc/$LOOP_PID/status" 2>/dev/null | awk '{print $2}')
        [[ -n "$LOOP_RSS_KB" ]] && LOOP_RSS=" loop_rss_mb=$((LOOP_RSS_KB / 1024))"
    fi

    # 8. GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    GPU_INFO=""
    [[ -n "$GPU_MEM" ]] && GPU_INFO=" gpu_mem=${GPU_MEM}MiB"

    echo "$TS pid1=$PID1_OK loop=$LOOP_OK cgroup=${CGROUP_MB}MB pids=$NUM_PIDS$LOOP_RSS$GPU_INFO$TRAIN_OK$OOM" | tee -a "$LOG"

    # If the loop process died, log details and keep watching
    if [[ -z "$LOOP_PID" ]]; then
        echo "$TS >>> LOOP PROCESS NOT FOUND — checking why..." | tee -a "$LOG"
        echo "$TS >>> Last 5 lines of loop.log:" | tee -a "$LOG"
        tail -5 /root/loop.log 2>/dev/null | tee -a "$LOG"
        echo "$TS >>> Last 5 lines of diagnostics.log:" | tee -a "$LOG"
        tail -5 /root/diagnostics.log 2>/dev/null | tee -a "$LOG"
    fi

    sleep 5
done
