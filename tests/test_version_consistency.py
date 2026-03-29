"""Version consistency tests — catch stale references when architecture changes.

train.py is the single source of truth for parameters and architecture.
These tests verify that satellite files (inner_loop.py, monitor.py, replay.py,
best_train.py, prepare.py, decision.py, run_loop.py) stay in sync.

All tests use regex on raw file text — no torch imports, runs in <100ms.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent


def _read(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def _extract_env_params(text: str) -> set[str]:
    """Extract all env-var param names from _env_float/_env_int/setdefault calls."""
    params: set[str] = set()
    for m in re.finditer(r'_env_(?:float|int)\(\s*["\'](\w+)["\']', text):
        params.add(m.group(1))
    for m in re.finditer(r'os\.environ\.setdefault\(\s*["\'](\w+)["\']', text):
        params.add(m.group(1))
    return params


def _extract_param_space_keys(text: str) -> set[str]:
    """Extract keys from inner_loop.py's _PARAM_SPACE dict."""
    keys: set[str] = set()
    in_block = False
    brace_depth = 0
    for line in text.splitlines():
        if '_PARAM_SPACE' in line and '{' in line:
            in_block = True
            brace_depth = line.count('{') - line.count('}')
            continue
        if in_block:
            brace_depth += line.count('{') - line.count('}')
            m = re.match(r'\s*["\'](\w+)["\']', line)
            if m:
                keys.add(m.group(1))
            if brace_depth <= 0:
                break
    return keys


def _extract_constant(text: str, name: str) -> int | None:
    m = re.search(rf'\b{name}\s*=\s*(\d+)', text)
    return int(m.group(1)) if m else None


def _extract_feature_groups(text: str) -> dict[str, tuple[int, int]]:
    """Parse FEATURE_GROUPS dict from source text."""
    groups: dict[str, tuple[int, int]] = {}
    for m in re.finditer(r"'(\w+)'\s*:\s*\((\d+)\s*,\s*(\d+)\)", text):
        groups[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return groups


def _extract_model_heads(text: str) -> set[str]:
    return set(re.findall(r'self\.(\w+_head)\s*=', text))


def _count_feature_names(text: str) -> int:
    """Count entries in FEATURE_NAMES list in prepare.py."""
    in_list = False
    count = 0
    for line in text.splitlines():
        if 'FEATURE_NAMES' in line and '[' in line:
            in_list = True
            continue
        if in_list:
            # Only match ] that ends the list (at start of line, not in comments)
            stripped = line.split('#')[0].strip()
            if stripped == ']':
                break
            # Count lines that contain a quoted string (feature name)
            if re.search(r"'[\w_]+'", line):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_text():
    return _read("training/train.py")


@pytest.fixture(scope="module")
def train_params(train_text):
    return _extract_env_params(train_text)


@pytest.fixture(scope="module")
def best_train_text():
    return _read("training/best_train.py")


@pytest.fixture(scope="module")
def prepare_text():
    return _read("training/prepare.py")


# Params intentionally excluded from PBT (architecture or system-level)
ARCHITECTURE_PARAMS = frozenset({
    "TRAIN_LOOKBACK", "TRAIN_D_MODEL", "TRAIN_DEPTH", "TRAIN_FF_MULT",
    "TRAIN_DAY_SEQ_BATCH", "TORCHINDUCTOR_COMPILE_THREADS",
    # v13 gate-specific: architecture or proven-best from PBT
    "TRAIN_EV_GATE_SCALE", "TRAIN_FALSE_ENTRY_PENALTY",
    "TRAIN_GATE_LABEL_SMOOTHING", "TRAIN_DIR_LABEL_SMOOTHING",
    "TRAIN_VALUE_EXIT_THRESH", "TRAIN_VALUE_LOSS_TYPE", "TRAIN_TOD_FILTER",
    # v10 experimental (default 0.0, rarely used)
    "TRAIN_RWR_WEIGHT", "TRAIN_DAY_RWR_WEIGHT",
})


# ===================================================================
# 1. PBT <-> train.py parameter sync
# ===================================================================

class TestPBTParamSync:

    def test_pbt_params_exist_in_train(self, train_params):
        """Every _PARAM_SPACE key must be consumed by train.py."""
        il_text = _read("tools/inner_loop.py")
        pbt_params = _extract_param_space_keys(il_text)
        orphans = pbt_params - train_params
        assert not orphans, (
            f"inner_loop.py _PARAM_SPACE has params not read by train.py: {sorted(orphans)}. "
            f"These env vars will be set but never consumed. "
            f"Remove from _PARAM_SPACE or add _env_float/_env_int to train.py."
        )

    def test_train_params_covered_by_pbt_or_excluded(self, train_params):
        """Every train.py env var should be in _PARAM_SPACE or the exclusion list."""
        il_text = _read("tools/inner_loop.py")
        pbt_params = _extract_param_space_keys(il_text)
        uncovered = train_params - pbt_params - ARCHITECTURE_PARAMS
        assert not uncovered, (
            f"train.py reads env vars not in _PARAM_SPACE or ARCHITECTURE_PARAMS: {sorted(uncovered)}. "
            f"Add to _PARAM_SPACE for PBT tuning, or add to ARCHITECTURE_PARAMS if intentionally excluded."
        )


# ===================================================================
# 2. Monitor display params
# ===================================================================

class TestMonitorSync:

    def test_monitor_params_are_current(self, train_params):
        """monitor.py's abbrev dict should only reference current params."""
        mon_text = _read("tools/monitor.py")
        # Extract keys from abbrev dict
        mon_params = set(re.findall(r'["\'](\w+)["\']\s*:\s*["\']', mon_text))
        # Also grab priority list entries
        priority_match = re.search(r'priority\s*=\s*\[([^\]]+)\]', mon_text, re.DOTALL)
        if priority_match:
            mon_params.update(re.findall(r'["\'](\w+)["\']', priority_match.group(1)))
        # Filter to only training-related params
        mon_params = {p for p in mon_params if p.startswith(('TRAIN_', 'WEIGHT_', 'REG_', 'WARM_'))}
        stale = mon_params - train_params
        assert not stale, (
            f"monitor.py references params not in train.py: {sorted(stale)}. "
            f"Update the abbrev dict and priority list in monitor.py."
        )


# ===================================================================
# 3. Architecture consistency
# ===================================================================

class TestArchitectureSync:

    def test_best_train_constants_match_train(self, train_text, best_train_text):
        """best_train.py must have same architecture constants as train.py."""
        for const in ('NUM_ACTION_CLASSES', 'POSITION_STATE_DIM', 'ACCOUNT_STATE_DIM'):
            train_val = _extract_constant(train_text, const)
            best_val = _extract_constant(best_train_text, const)
            if train_val is None:
                continue
            assert best_val is not None, f"{const} missing from best_train.py"
            assert train_val == best_val, (
                f"{const} mismatch: train.py={train_val}, best_train.py={best_val}."
            )

    def test_replay_model_heads_match_train(self, train_text):
        """replay.py's TradingModel must have same heads as train.py."""
        replay_text = _read("training/replay.py")
        primary = {'action_head', 'risk_head', 'gate_head', 'dir_head', 'value_head', 'exit_head'}
        train_heads = _extract_model_heads(train_text) & primary
        replay_heads = _extract_model_heads(replay_text) & primary
        assert train_heads == replay_heads, (
            f"Model head mismatch!\n"
            f"  train.py heads:  {sorted(train_heads)}\n"
            f"  replay.py heads: {sorted(replay_heads)}\n"
            f"replay.py architecture is stale — copy from train.py."
        )

    def test_replay_position_state_dim(self, train_text):
        """replay.py POSITION_STATE_DIM must match train.py."""
        replay_text = _read("training/replay.py")
        train_dim = _extract_constant(train_text, 'POSITION_STATE_DIM')
        replay_dim = _extract_constant(replay_text, 'POSITION_STATE_DIM')
        assert train_dim is not None, "POSITION_STATE_DIM missing from train.py"
        assert replay_dim is not None, "POSITION_STATE_DIM missing from replay.py"
        assert train_dim == replay_dim, (
            f"POSITION_STATE_DIM mismatch: train.py={train_dim}, replay.py={replay_dim}"
        )

    def test_num_actions_consistency(self, train_text, prepare_text):
        """Verify action space consistency between train.py and prepare.py.

        v13 (gate+dir): train.py has gate_head(2) + dir_head(14), prepare.py has NUM_ACTIONS=16.
        v12 (unified):  train.py has NUM_ACTION_CLASSES=15, prepare.py has NUM_ACTIONS=16.
        """
        num_actions = _extract_constant(prepare_text, 'NUM_ACTIONS')
        assert num_actions is not None, "NUM_ACTIONS missing from prepare.py"
        # v13: check for gate_head and dir_head presence
        has_gate = 'gate_head' in train_text
        has_dir = 'dir_head' in train_text
        num_action_classes = _extract_constant(train_text, 'NUM_ACTION_CLASSES')
        if has_gate and has_dir:
            # v13 four-head: gate(2) + dir(14) maps to NUM_ACTIONS=16 (14 options + DO_NOTHING + EXIT)
            assert num_actions == 16, f"NUM_ACTIONS should be 16 for v13 gate+dir, got {num_actions}"
        elif num_action_classes is not None:
            # v12 unified: NUM_ACTION_CLASSES + 1 (EXIT) = NUM_ACTIONS
            assert num_actions == num_action_classes + 1, (
                f"NUM_ACTIONS ({num_actions}) should be NUM_ACTION_CLASSES ({num_action_classes}) + 1"
            )


# ===================================================================
# 4. Feature consistency
# ===================================================================

class TestFeatureSync:

    def test_feature_lock_count_matches_prepare(self, prepare_text):
        """run_loop.py FEATURE_LOCK_COUNT must match prepare.py FEATURE_NAMES count."""
        feature_count = _count_feature_names(prepare_text)
        rl_text = _read("training/run_loop.py")
        lock_count = _extract_constant(rl_text, 'FEATURE_LOCK_COUNT')
        assert lock_count is not None, "FEATURE_LOCK_COUNT missing from run_loop.py"
        assert lock_count == feature_count, (
            f"FEATURE_LOCK_COUNT={lock_count} in run_loop.py != "
            f"len(FEATURE_NAMES)={feature_count} in prepare.py"
        )

    def test_feature_groups_upper_bound(self, train_text, prepare_text):
        """Max FEATURE_GROUPS index in train.py must match NUM_FEATURES."""
        groups = _extract_feature_groups(train_text)
        assert groups, "Could not parse FEATURE_GROUPS from train.py"
        max_idx = max(end for _, end in groups.values())
        feature_count = _count_feature_names(prepare_text)
        assert max_idx == feature_count, (
            f"FEATURE_GROUPS max index={max_idx} in train.py != "
            f"len(FEATURE_NAMES)={feature_count} in prepare.py. "
            f"Features were added/removed but FEATURE_GROUPS wasn't updated."
        )

    def test_feature_groups_match_best_train(self, train_text, best_train_text):
        """FEATURE_GROUPS in best_train.py must match train.py."""
        train_groups = _extract_feature_groups(train_text)
        best_groups = _extract_feature_groups(best_train_text)
        assert train_groups, "Could not parse FEATURE_GROUPS from train.py"
        assert best_groups, "Could not parse FEATURE_GROUPS from best_train.py"
        assert train_groups == best_groups, (
            f"FEATURE_GROUPS mismatch between train.py and best_train.py:\n"
            f"  Only in train.py: {set(train_groups) - set(best_groups)}\n"
            f"  Only in best_train.py: {set(best_groups) - set(train_groups)}\n"
            f"  Differing ranges: {[(k, train_groups[k], best_groups[k]) for k in train_groups if k in best_groups and train_groups[k] != best_groups[k]]}"
        )

    def test_feature_index_constants(self, train_text, prepare_text):
        """IDX_* constants in train.py must match FEATURE_NAMES positions."""
        # Extract IDX_* constants
        idx_consts = {}
        for m in re.finditer(r'^(IDX_\w+)\s*=\s*(\d+)', train_text, re.MULTILINE):
            idx_consts[m.group(1)] = int(m.group(2))

        # Build feature name -> position map from prepare.py
        positions = {}
        in_list = False
        pos = 0
        for line in prepare_text.splitlines():
            if 'FEATURE_NAMES' in line and '[' in line:
                in_list = True
                continue
            if in_list:
                if ']' in line:
                    break
                m = re.search(r"'([\w_]+)'", line)
                if m:
                    positions[m.group(1)] = pos
                    pos += 1

        for idx_name, idx_val in idx_consts.items():
            # IDX_MINUTES_TO_CLOSE -> minutes_to_close
            feat_name = idx_name[4:].lower()  # strip IDX_ prefix
            expected = positions.get(feat_name)
            assert expected is not None, (
                f"train.py has {idx_name}={idx_val} but '{feat_name}' not found in FEATURE_NAMES"
            )
            assert expected == idx_val, (
                f"train.py {idx_name}={idx_val} but '{feat_name}' is at "
                f"position {expected} in FEATURE_NAMES"
            )

    def test_decision_num_features_default(self, prepare_text):
        """decision.py num_features default must match prepare.py NUM_FEATURES."""
        dec_text = _read("training/live/decision.py")
        m = re.search(r'num_features:\s*int\s*=\s*(\d+)', dec_text)
        assert m, "Could not find num_features default in decision.py"
        dec_default = int(m.group(1))
        feature_count = _count_feature_names(prepare_text)
        assert dec_default == feature_count, (
            f"decision.py num_features default ({dec_default}) != "
            f"prepare.py NUM_FEATURES ({feature_count}). "
            f"decision.py has a stale hardcoded default."
        )


# ===================================================================
# 5. Stale env var reads in satellites
# ===================================================================

class TestStaleEnvVars:

    def test_no_stale_env_vars_in_satellites(self, train_params):
        """Satellite files must not read TRAIN_* env vars that train.py doesn't define."""
        satellites = [
            ("training/replay.py", "replay.py"),
            ("training/prepare.py", "prepare.py"),
        ]
        all_stale: dict[str, set[str]] = {}
        for relpath, label in satellites:
            sat_text = _read(relpath)
            sat_envs = set(re.findall(r'os\.environ\.get\(\s*["\'](\w+)["\']', sat_text))
            sat_envs.update(re.findall(r'_env_float\(\s*["\'](\w+)["\']', sat_text))
            sat_envs.update(re.findall(r'_env_int\(\s*["\'](\w+)["\']', sat_text))
            sat_train = {p for p in sat_envs if p.startswith('TRAIN_')}
            stale = sat_train - train_params
            if stale:
                all_stale[label] = stale

        assert not all_stale, (
            "Satellite files read TRAIN_* env vars not defined in train.py:\n" +
            "\n".join(f"  {f}: {sorted(v)}" for f, v in all_stale.items())
        )


# ===================================================================
# 6. Documentation consistency (lighter weight)
# ===================================================================

class TestDocSync:

    def test_operating_manual_env_vars(self, train_params):
        """Operating manual loss params table should list current env vars."""
        manual = _read(".claude/rules/art2-operating-manual.md")
        # Extract env var names from the loss hyperparameters table
        # Pattern: | ... | ... | ENV_VAR_NAME | ...
        table_vars = set(re.findall(r'\|\s*(TRAIN_\w+)\s*\|', manual))
        stale = table_vars - train_params
        assert not stale, (
            f"Operating manual lists env vars not in train.py: {sorted(stale)}. "
            f"Update the Loss Hyperparameters table in art2-operating-manual.md."
        )
