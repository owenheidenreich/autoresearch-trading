"""Autoresearch experiment orchestrator.

Manages the experiment cycle: run_experiment.py per iteration,
keep or revert, enforce session limits, track state.

Can be used two ways:
1. As a library: the AI calls init_session(), record_result(), etc.
2. As a runner: `python -m v2.ops.inner_loop` runs the full loop autonomously.

The __main__ runner is the standard way to execute on the GPU node.
It runs run_experiment.py in a subprocess, parses results, does keep/revert,
writes state files (so monitor.py can observe), and enforces all session limits.

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
    kept_count: int = 0
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
        state.kept_count += 1
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
    # Restore best model.pt from the session's best artifact
    from v2.ops.artifact import ARTIFACTS_DIR, _file_fingerprint
    best_id = state.best_artifact_id if state else None

    if best_id:
        best_model = os.path.join(ARTIFACTS_DIR, best_id, "model.pt")
        manifest_path = os.path.join(ARTIFACTS_DIR, best_id, "manifest.json")

        if not os.path.exists(best_model):
            print(f"ERROR: artifact model not found at {best_model}")
        else:
            # Validate fingerprint matches manifest before restoring
            if os.path.exists(manifest_path):
                with open(manifest_path) as f:
                    manifest = json.load(f)
                expected_fp = manifest.get("model_fingerprint", "")
                actual_fp = _file_fingerprint(best_model)
                if expected_fp and actual_fp != expected_fp:
                    print(f"ERROR: model fingerprint mismatch in {best_id}!")
                    print(f"  manifest expects: {expected_fp}")
                    print(f"  file on disk:     {actual_fp}")
                    print(f"  NOT restoring -- artifact may be corrupted")
                    return
                print(f"Restored model.pt from artifact {best_id} "
                      f"(score={manifest.get('score', '?')}, fp={actual_fp})")
            else:
                print(f"WARNING: no manifest for {best_id}, restoring without validation")

            shutil.copy2(best_model, "v2/model.pt")

            # Restore mutable files: prefer git checkout, fall back to artifact snapshots
            has_git = shutil.which("git") is not None
            if has_git:
                for f in ["v2/train.py", "v2/core/policy.py"]:
                    if os.path.exists(f):
                        subprocess.run(["git", "checkout", f], capture_output=True)
            else:
                # No git (GPU container) -- restore from artifact snapshots
                snapshot_map = {
                    "train.py.snapshot": "v2/train.py",
                    "policy.py.snapshot": "v2/core/policy.py",
                }
                for snap_name, target in snapshot_map.items():
                    snap_path = os.path.join(ARTIFACTS_DIR, best_id, snap_name)
                    if os.path.exists(snap_path):
                        shutil.copy2(snap_path, target)
                        print(f"Restored {target} from artifact snapshot")
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


def _next_experiment_number(state: SessionState) -> int:
    """Return the next globally-unique experiment number.

    Reads existing results.tsv and artifact dirs to find the highest
    experiment number ever used, then returns max + 1. This prevents
    ID collisions across sessions.
    """
    import re
    max_num = state.experiment_count  # session-local fallback

    # Scan results.tsv
    if os.path.exists(RESULTS_TSV):
        try:
            with open(RESULTS_TSV) as f:
                for line in f:
                    m = re.match(r'exp_(\d+)', line.strip())
                    if m:
                        max_num = max(max_num, int(m.group(1)))
        except OSError:
            pass

    # Scan artifact dirs
    artifacts_dir = os.path.join("v2", "artifacts")
    if os.path.exists(artifacts_dir):
        for name in os.listdir(artifacts_dir):
            m = re.match(r'exp_(\d+)', name)
            if m:
                max_num = max(max_num, int(m.group(1)))

    return max_num + 1


# ===================================================================
# Autonomous loop runner (__main__)
# ===================================================================

def _run_single_experiment(experiment_id: str) -> dict:
    """Run run_experiment.py as a subprocess and parse its JSON output."""
    cmd = [
        "python3", "-m", "v2.ops.run_experiment",
        "--id", experiment_id,
    ]
    print(f"\n{'='*60}")
    print(f"  LAUNCHING: {experiment_id}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,  # 30 min max
        )
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {experiment_id} exceeded 30 minutes")
        return {"status": "crash", "error": "timeout", "score": -999.0}

    # Print stdout so it appears in run.log
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        import sys as _sys
        print(result.stderr, file=_sys.stderr)

    # Parse RESULTS_JSON from stdout
    for line in (result.stdout or "").splitlines():
        if line.startswith("RESULTS_JSON:"):
            try:
                return json.loads(line[len("RESULTS_JSON:"):])
            except json.JSONDecodeError:
                pass

    # If we got here, couldn't parse results
    if result.returncode != 0:
        return {
            "status": "crash",
            "error": f"exit code {result.returncode}",
            "score": -999.0,
        }

    return {
        "status": "crash",
        "error": "no RESULTS_JSON in output",
        "score": -999.0,
    }


def run_loop():
    """Run the full autonomous experiment loop.

    This is the standard entry point on the GPU node.
    Runs run_experiment.py repeatedly, does keep/revert, enforces limits.
    """
    import sys

    print(f"\n{'='*60}")
    print(f"  ART2 EXPERIMENT LOOP")
    print(f"  Limits: {MAX_EXPERIMENTS} experiments, {MAX_HOURS}h, "
          f"{MAX_NO_IMPROVE_STREAK} no-improve, {MAX_CRASH_STREAK} crashes")
    print(f"{'='*60}\n")

    state = init_session()
    print(format_session_status(state))

    while True:
        # Check limits
        can_continue, reason = check_session_limits(state)
        if not can_continue:
            print(f"\n*** SESSION STOPPED: {reason} ***")
            state.stopped = True
            state.stop_reason = reason
            state.save()
            break

        # Generate experiment ID (globally unique across sessions)
        exp_num = _next_experiment_number(state)
        experiment_id = f"exp_{exp_num:03d}"

        # Run experiment
        results = _run_single_experiment(experiment_id)

        # Parse results
        score = results.get("score", -999.0)
        status = results.get("status", "crash")
        beats_all = all([
            results.get("beats_random", False),
            results.get("beats_atm", False),
            results.get("beats_rules", False),
            results.get("beats_trailing", False),
        ])

        # Build description
        if status == "crash":
            description = f"CRASH: {results.get('error', 'unknown')}"
        else:
            trades = results.get("total_trades", 0)
            wr = results.get("win_rate", 0)
            description = f"score={score:.3f} trades={trades} wr={wr:.1%} baselines={'ALL' if beats_all else 'PARTIAL'}"

        # Record and decide
        decision = record_result(
            state, experiment_id, score, status, description, beats_all,
        )

        print(f"\n--- DECISION: {decision.upper()} (score={score:.4f}, best={state.best_score:.4f}) ---")

        if decision == "revert":
            print("Reverting mutable files to best known state...")
            revert_mutable_files(state)

        print(f"\n{format_session_status(state)}\n")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  SESSION COMPLETE")
    print(f"{'='*60}")
    print(format_session_status(state))
    print(f"\nBest model: experiment #{state.best_experiment_num} (score {state.best_score:.4f})")


if __name__ == "__main__":
    run_loop()
