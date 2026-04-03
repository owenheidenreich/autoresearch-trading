#!/bin/bash
# v15 overnight training loop
# Runs sequential experiments indefinitely until GPU is torn down or script is killed
# Each experiment: ~6 min on H100. Warm-starts from last kept model.
# Monitor at http://localhost:8420

set -e
cd "$(dirname "$0")"

LOG="results/overnight_$(date +%Y%m%d_%H%M%S).log"
echo "=== v15 Overnight Training ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "Kill with: kill $$" | tee -a "$LOG"
echo "---" | tee -a "$LOG"

COUNTER=0
while true; do
    COUNTER=$((COUNTER + 1))
    echo "" | tee -a "$LOG"
    echo "=== Experiment $COUNTER ($(date +%H:%M:%S)) ===" | tee -a "$LOG"

    # Check if GPU is still reachable
    if ! ./infra/deploy.sh status > /dev/null 2>&1; then
        echo "GPU unreachable. Stopping." | tee -a "$LOG"
        break
    fi

    # Check for stop sentinel
    if [ -f "results/STOP_OVERNIGHT" ]; then
        echo "STOP sentinel found. Stopping gracefully." | tee -a "$LOG"
        break
    fi

    # Reset consecutive revert counter if stuck (prevents loop detection halt)
    python3 -c "
import json
state = json.load(open('training/.inner_loop_state.json'))
if state.get('_consecutive_same_revert', 0) >= 4:
    state['_consecutive_same_revert'] = 0
    state['_last_revert_key'] = None
    json.dump(state, open('training/.inner_loop_state.json', 'w'), indent=2)
    print('Reset loop detection counter')
" 2>&1 | tee -a "$LOG"

    # Run experiment
    python3 tools/inner_loop.py experiment \
        --summary "v15 overnight #$COUNTER: lunch suppression + MIN_HOLD_BARS=2" \
        2>&1 | tee -a "$LOG" | grep -E "Score:|KEPT|REVERTED|profit_factor|win_rate|trades_per_day|model_exit_rate"

    echo "---" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Overnight training complete ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
echo "Total experiments: $COUNTER" | tee -a "$LOG"
