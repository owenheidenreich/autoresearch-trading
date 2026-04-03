#!/bin/bash
# Run all tournament hypotheses sequentially
set -euo pipefail

FEATURES=(overnight_gap event_day option_spread_width pc_volume_ratio iv_percentile rsi_15min)
SSH_PORT="$(grep SSH_PORT .deploy-state | cut -d= -f2)"
SSH_HOST="$(grep SSH_HOST .deploy-state | cut -d= -f2)"
RESULTS_FILE="results/tournament/all_results.jsonl"
mkdir -p results/tournament

SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o PubkeyAuthentication=no -o ServerAliveInterval=30"

for feat in "${FEATURES[@]}"; do
    echo ""
    echo "================================================================"
    echo "TOURNAMENT HYPOTHESIS: $feat"
    echo "================================================================"

    # Step 1: Copy pre-augmented data.pt on Akash
    echo "  Copying data_${feat}.pt -> data.pt on Akash..."
    SSHPASS=autoresearch2026 sshpass -e ssh $SSH_OPTS -p "$SSH_PORT" "root@$SSH_HOST" \
        "cp /root/.cache/autoresearch-trading/features/data_${feat}.pt /root/.cache/autoresearch-trading/features/data.pt"

    # Step 2: Fresh start locally
    rm -f training/best_model.pt
    echo "-5.0" > training/.best_score
    rm -f training/.inner_loop_state.json
    cp training/train.py training/best_train.py

    # Step 3: Run 5 experiments
    for i in 1 2 3 4 5; do
        echo "  --- $feat exp $i/5 ---"
        OUTPUT=$(EXTRA_FEATURES="$feat" python3 tools/inner_loop.py experiment --summary "TOURNAMENT ${feat} exp ${i}/5" 2>&1)

        SCORE=$(echo "$OUTPUT" | grep "Score:" | head -1 | awk '{print $NF}' | tr -d '()')
        if echo "$OUTPUT" | grep -q "KEPT:"; then
            PF=$(echo "$OUTPUT" | grep '"profit_factor"' | head -1 | awk -F': ' '{print $2}' | tr -d ',')
            WR=$(echo "$OUTPUT" | grep '"win_rate"' | head -1 | awk -F': ' '{print $2}' | tr -d ',')
            TPD=$(echo "$OUTPUT" | grep '"trades_per_day"' | head -1 | awk -F': ' '{print $2}' | tr -d ',')
            echo "    KEPT: score=$SCORE PF=$PF WR=$WR TPD=$TPD"
        else
            echo "    REVERTED: score=$SCORE"
        fi
    done

    FINAL_SCORE=$(cat training/.best_score)
    echo "  FINAL: $feat best_score=$FINAL_SCORE"
    echo "{\"hypothesis\":\"$feat\",\"best_score\":$FINAL_SCORE}" >> "$RESULTS_FILE"
done

echo ""
echo "================================================================"
echo "TOURNAMENT COMPLETE"
echo "================================================================"
cat "$RESULTS_FILE"
