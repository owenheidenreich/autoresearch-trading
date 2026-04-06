#!/bin/bash
# Autoresearch experiment loop -- runs on the GPU node
cd /root
export PYTHONPATH=/root

BEST_SCORE=-5.0
BEST_EXP=""
NO_IMPROVE=0
MAX_NO_IMPROVE=8
MAX_EXPERIMENTS=50

# Load best score from previous run if available
if [ -f v2/.best_score ]; then
    BEST_SCORE=$(cat v2/.best_score | tr -d '[:space:]')
    echo "Loaded best score: $BEST_SCORE"
fi

for i in $(seq 1 $MAX_EXPERIMENTS); do
    EXP_ID=$(printf 'exp_%03d' $i)
    echo ""
    echo "========================================"
    echo "EXPERIMENT $i/$MAX_EXPERIMENTS: $EXP_ID"
    echo "========================================"

    # Run experiment
    RESULT=$(python3 -m v2.ops.run_experiment --id $EXP_ID --data v2/data.pt 2>&1)
    echo "$RESULT"

    # Extract score
    SCORE=$(echo "$RESULT" | grep '^score:' | awk '{print $2}')
    BEATS_ALL=true
    for bl in beats_random beats_atm beats_rules beats_trailing; do
        VAL=$(echo "$RESULT" | grep "^${bl}:" | awk '{print $2}')
        if [ "$VAL" != "true" ]; then
            BEATS_ALL=false
        fi
    done

    if [ -z "$SCORE" ]; then
        echo "CRASH: no score extracted"
        NO_IMPROVE=$((NO_IMPROVE + 1))
        if [ -n "$BEST_EXP" ] && [ -f "v2/artifacts/artifacts/$BEST_EXP/model.pt" ]; then
            cp "v2/artifacts/artifacts/$BEST_EXP/model.pt" v2/model.pt
            echo "Reverted model.pt to $BEST_EXP"
        fi
    elif [ "$BEATS_ALL" = "true" ] && python3 -c "import sys; sys.exit(0 if float('$SCORE') > float('$BEST_SCORE') else 1)"; then
        echo "KEEP: $SCORE > $BEST_SCORE (beats all baselines)"
        BEST_SCORE=$SCORE
        BEST_EXP=$EXP_ID
        NO_IMPROVE=0
        echo "$BEST_SCORE" > v2/.best_score
    else
        echo "REVERT: $SCORE <= $BEST_SCORE or missed a baseline"
        NO_IMPROVE=$((NO_IMPROVE + 1))
        if [ -n "$BEST_EXP" ] && [ -f "v2/artifacts/artifacts/$BEST_EXP/model.pt" ]; then
            cp "v2/artifacts/artifacts/$BEST_EXP/model.pt" v2/model.pt
            echo "Reverted model.pt to $BEST_EXP"
        fi
    fi

    echo "Status: best=$BEST_SCORE ($BEST_EXP) no_improve=$NO_IMPROVE"

    if [ $NO_IMPROVE -ge $MAX_NO_IMPROVE ]; then
        echo "STOPPING: $NO_IMPROVE consecutive non-improvements"
        break
    fi
done

echo ""
echo "========================================"
echo "SESSION COMPLETE"
echo "Best score: $BEST_SCORE ($BEST_EXP)"
echo "========================================"
