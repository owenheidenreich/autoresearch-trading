"""Autoresearch experiment orchestrator.

Manages the experiment cycle: parse results from run_experiment.py,
keep or revert, enforce session limits, track state.

This is NOT called by the AI directly. The AI runs run_experiment.py,
reads the results, and calls the keep/revert functions here. The session
limits are checked by the AI before each experiment.

v2 changes from v1:
- Scores using replay P&L (account curve), not prediction accuracy
- Artifact bundles replace loose checkpoint files
- Session limits enforced (6hr, 50 experiments, 8 no-improve, 3hr plateau, 3 crashes)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict, field


STATE_FILE = "v2/.inner_loop_state.json"
BEST_SCORE_FILE = "v2/.best_score"
RESULTS_TSV = "v2/results.tsv"

# Session limits
MAX_EXPERIMENTS = 50
MAX_HOURS = 6
MAX_NO_IMPROVE_STREAK = 8
MAX_PLATEAU_HOURS = 3
MAX_CRASH_STREAK = 3


@dataclass
class SessionState:
    """Persistent state across the experiment session."""
    experiment_count: int = 0
    session_start: float = 0.0
    best_score: float = -5.0
    best_artifact_id: str = ""
    best_experiment_num: int = 0
    no_improve_streak: int = 0
    crash_streak: int = 0
    last_improve_time: float = 0.0
    stopped: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SessionState:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str = STATE_FILE):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str = STATE_FILE) -> SessionState:
        if not os.path.exists(path):
            return cls(session_start=time.time(), last_improve_time=time.time())
        with open(path) as f:
            return cls.from_dict(json.load(f))


def init_session() -> SessionState:
    """Initialize a new experiment session."""
    state = SessionState(
        session_start=time.time(),
        last_improve_time=time.time(),
    )
    # Load best score from file if it exists
    if os.path.exists(BEST_SCORE_FILE):
        try:
            with open(BEST_SCORE_FILE) as f:
                state.best_score = float(f.read().strip())
        except (ValueError, IOError):
            pass
    state.save()

    # Initialize results.tsv if it doesn't exist
    if not os.path.exists(RESULTS_TSV):
        os.makedirs(os.path.dirname(RESULTS_TSV), exist_ok=True)
        with open(RESULTS_TSV, "w") as f:
            f.write("experiment\tscore\tstatus\tdescription\n")

    return state


def check_session_limits(state: SessionState) -> tuple[bool, str]:
    """Check if any session limit has been hit.

    Returns (can_continue, reason).
    """
    if state.stopped:
        return False, state.stop_reason

    elapsed_hours = (time.time() - state.session_start) / 3600

    if state.experiment_count >= MAX_EXPERIMENTS:
        return False, f"experiment_limit ({MAX_EXPERIMENTS} experiments)"

    if elapsed_hours >= MAX_HOURS:
        return False, f"time_limit ({MAX_HOURS}h)"

    if state.no_improve_streak >= MAX_NO_IMPROVE_STREAK:
        return False, f"no_improve_streak ({MAX_NO_IMPROVE_STREAK} consecutive reverts)"

    plateau_hours = (time.time() - state.last_improve_time) / 3600
    if plateau_hours >= MAX_PLATEAU_HOURS:
        return False, f"plateau ({MAX_PLATEAU_HOURS}h without improvement)"

    if state.crash_streak >= MAX_CRASH_STREAK:
        return False, f"crash_storm ({MAX_CRASH_STREAK} consecutive crashes)"

    return True, ""


def record_result(
    state: SessionState,
    experiment_id: str,
    score: float,
    status: str,
    description: str,
    beats_all_baselines: bool = False,
) -> str:
    """Record experiment result and decide keep/revert.

    Returns "keep", "revert", or "crash".
    """
    state.experiment_count += 1

    if status == "crash":
        state.crash_streak += 1
        state.no_improve_streak += 1
        decision = "crash"
    elif score > state.best_score and beats_all_baselines:
        # KEEP
        state.best_score = score
        state.best_artifact_id = experiment_id
        state.best_experiment_num = state.experiment_count
        state.no_improve_streak = 0
        state.crash_streak = 0
        state.last_improve_time = time.time()
        decision = "keep"

        # Update best score file
        os.makedirs(os.path.dirname(BEST_SCORE_FILE), exist_ok=True)
        with open(BEST_SCORE_FILE, "w") as f:
            f.write(f"{score:.6f}\n")
    else:
        # REVERT
        state.no_improve_streak += 1
        state.crash_streak = 0
        decision = "revert"

    # Check if we should stop
    can_continue, reason = check_session_limits(state)
    if not can_continue:
        state.stopped = True
        state.stop_reason = reason

    # Log to TSV
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{experiment_id}\t{score:.6f}\t{decision}\t{description}\n")

    state.save()
    return decision


def revert_mutable_files(state: SessionState | None = None):
    """Git checkout the mutable research files and restore the best model.

    Without model restoration, a reverted experiment leaves the FAILED model
    on disk at v2/model.pt, corrupting all subsequent evaluations and
    warm-starts.
    """
    mutable_files = ["v2/train.py", "v2/core/policy.py"]
    for f in mutable_files:
        if os.path.exists(f):
            subprocess.run(["git", "checkout", f], capture_output=True)

    # Restore best model.pt from the session's best artifact
    from v2.ops.artifact import ARTIFACTS_DIR
    best_id = state.best_artifact_id if state else None

    if best_id:
        best_model = os.path.join(ARTIFACTS_DIR, best_id, "model.pt")
        if os.path.exists(best_model):
            shutil.copy2(best_model, "v2/model.pt")
            print(f"Restored model.pt from artifact {best_id}")
            return

    # Fallback: no session state, scan for best artifact with compatible arch
    from v2.ops.artifact import get_best_artifact
    best_dir = get_best_artifact()
    if best_dir:
        best_model = os.path.join(best_dir, "model.pt")
        if os.path.exists(best_model):
            shutil.copy2(best_model, "v2/model.pt")
            print(f"Restored model.pt from {best_dir}")


def format_session_status(state: SessionState) -> str:
    """Format current session state for display."""
    elapsed = (time.time() - state.session_start) / 3600
    plateau = (time.time() - state.last_improve_time) / 3600

    lines = [
        f"Experiment: {state.experiment_count}/{MAX_EXPERIMENTS}",
        f"Time: {elapsed:.1f}/{MAX_HOURS}h",
        f"Best score: {state.best_score:.6f} (exp #{state.best_experiment_num})",
        f"No-improve streak: {state.no_improve_streak}/{MAX_NO_IMPROVE_STREAK}",
        f"Crash streak: {state.crash_streak}/{MAX_CRASH_STREAK}",
        f"Plateau: {plateau:.1f}/{MAX_PLATEAU_HOURS}h",
    ]
    if state.stopped:
        lines.append(f"STOPPED: {state.stop_reason}")
    return "\n".join(lines)
