#!/usr/bin/env bash
# Check if documentation is stale relative to code changes.
# Called by Claude Code Stop hook. Outputs JSON with systemMessage if docs need updating.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

stale_items=()

# 1. Check if lab_notebook.md is older than results.tsv
if [[ -f v2/results.tsv && -f v2/lab_notebook.md ]]; then
    results_mod=$(stat -f %m v2/results.tsv 2>/dev/null || stat -c %Y v2/results.tsv 2>/dev/null || echo 0)
    notebook_mod=$(stat -f %m v2/lab_notebook.md 2>/dev/null || stat -c %Y v2/lab_notebook.md 2>/dev/null || echo 0)
    if [[ "$results_mod" -gt "$notebook_mod" ]]; then
        # Count experiments in results but not in notebook
        n_results=$(tail -n +2 v2/results.tsv | wc -l | tr -d ' ')
        n_notebook=$(grep -c '^## Experiment\|^### exp_\|^exp_' v2/lab_notebook.md 2>/dev/null || echo 0)
        stale_items+=("lab_notebook.md is behind results.tsv ($n_results experiments vs ~$n_notebook documented)")
    fi
fi

# 2. Check if train.py or policy.py changed since last doc commit
last_doc_commit=$(git log -1 --format=%H -- v2/lab_notebook.md v2/program.md v2/docs/ 2>/dev/null || echo "")
if [[ -n "$last_doc_commit" ]]; then
    code_changes=$(git log --oneline "$last_doc_commit"..HEAD -- v2/train.py v2/core/policy.py v2/pipeline/build_v2_dataset.py 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$code_changes" -gt 0 ]]; then
        stale_items+=("$code_changes code commits since last doc update (train.py/policy.py/build_v2_dataset.py)")
    fi
fi

# 3. Check if program.md references match current architecture
if [[ -f v2/program.md ]]; then
    # Check if program.md mentions old feature count
    if grep -q "39.*features\|39 market" v2/program.md 2>/dev/null; then
        actual_features=$(python3 -c "import torch; d=torch.load('v2/data.pt',map_location='cpu',weights_only=False); print(d['X'].shape[1])" 2>/dev/null || echo "?")
        if [[ "$actual_features" != "?" && "$actual_features" != "39" ]]; then
            stale_items+=("program.md says 39 features but data.pt has $actual_features")
        fi
    fi
fi

# Output
if [[ ${#stale_items[@]} -gt 0 ]]; then
    msg="STALE DOCS: "
    for item in "${stale_items[@]}"; do
        msg+="$item. "
    done
    msg+="Update v2/lab_notebook.md and v2/program.md before ending this session."
    # Escape for JSON
    msg_escaped=$(echo "$msg" | sed 's/"/\\"/g')
    echo "{\"systemMessage\": \"$msg_escaped\"}"
fi
