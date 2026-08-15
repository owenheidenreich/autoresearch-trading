"""Phase 3: the known-answer campaign for the second G1 attempt.

Three questions, asked in this order, **before any real economics are read**:

1. Do matched surrogates with no predictability by construction pass the gate?
   They must not, more than 5% of the time.
2. Does row 183's shared-term shape pass? Same limit.
3. Is a real effect of known size recovered at least 80% of the time?

The first attempt failed question 3 and stopped there, which is why its
mechanisms remain legitimately available. This run reads no strategy outcome on
the real corpus: it scores surrogates and fixtures only, and it deliberately
never calls the scorer on unmodified sessions.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from v5.research.direction import fixtures, hypothesis_2026_08 as hypothesis
from v5.research.direction import loader, screen_2026_08 as screen, surrogate
from v5.research.direction.campaign import wilson_upper


@dataclass(frozen=True)
class Outcome:
    name: str
    trials: int
    passes: int
    rate: float
    wilson_upper: float
    limit: float
    is_floor: bool

    @property
    def ok(self) -> bool:
        return self.rate >= self.limit if self.is_floor else self.rate <= self.limit

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "trials": self.trials,
            "passes": self.passes,
            "rate": round(self.rate, 6),
            "wilson_upper": round(self.wilson_upper, 6),
            "limit": self.limit,
            "is_floor": self.is_floor,
            "ok": self.ok,
        }


def _campaign(name, build, *, trials, limit, is_floor, seed_base, log) -> Outcome:
    passes = 0
    started = time.time()
    for trial in range(trials):
        sessions = build(seed_base + trial)
        features = loader.session_features(sessions)
        mask = screen.eligible_mask(sessions, features)
        scores = screen.score_family(features, mask, seed_offset=trial)
        passes += screen.any_member_passed(scores)
        if (trial + 1) % 25 == 0:
            rate = passes / (trial + 1)
            each = (time.time() - started) / (trial + 1)
            log(
                f"  {name}: {trial + 1}/{trials} trials, rate {rate:.4f}, "
                f"{each:.2f}s/trial"
            )
    rate = passes / trials
    return Outcome(name, trials, passes, rate, wilson_upper(passes, trials), limit, is_floor)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1000)
    # Swept rather than fixed, so the run reports the gate's operational
    # detection floor instead of one arbitrary size. 4.01 points is the declared
    # 65.73% accuracy threshold expressed in points at the corpus's pooled
    # dispersion; the smaller sizes probe below it and the larger ones above.
    p.add_argument(
        "--recovery-points",
        type=float,
        nargs="+",
        default=[2.0, 3.0, 4.01, 6.0, 8.0],
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"hypothesis freeze {hypothesis.freeze_sha256()}")
    log(f"members {list(hypothesis.MEMBERS)}, required accuracy {hypothesis.REQUIRED_ACCURACY}")

    sessions, _features, mask = screen.build_inputs()
    log(f"corpus {len(sessions):,} sessions, eligible index {int(mask.sum()):,} — matches the freeze")

    crit = hypothesis.KNOWN_ANSWER_CRITERIA
    results: list[Outcome] = []

    log("question 1: matched surrogates, no predictability by construction")
    results.append(
        _campaign(
            "matched_surrogate_false_pass",
            lambda seed: surrogate.matched_surrogate(sessions, seed=seed)[0],
            trials=args.trials,
            limit=crit["matched_surrogate_false_pass_max"],
            is_floor=False,
            seed_base=10_000,
            log=log,
        )
    )
    log("question 2: row 183's shared-term shape")
    results.append(
        _campaign(
            "shared_term_fixture",
            lambda seed: fixtures.shared_term_sessions(sessions, seed=seed),
            trials=args.trials,
            limit=crit["shared_term_fixture_false_pass_max"],
            is_floor=False,
            seed_base=20_000,
            log=log,
        )
    )
    log(f"question 3: recovery sweep over {args.recovery_points} points/session")
    recovery: list[Outcome] = []
    for i, points in enumerate(sorted(args.recovery_points)):
        recovery.append(
            _campaign(
                f"injected_{points:g}pts",
                lambda seed, pts=points: fixtures.injected_surrogate(
                    sessions, seed=seed, points_per_session=pts, mechanism="M3"
                ),
                trials=args.trials,
                limit=crit["recovery_of_a_detectable_effect_min"],
                is_floor=True,
                seed_base=30_000 + 1_000 * i,
                log=log,
            )
        )
    results.extend(recovery)

    # The declared criterion is recovery at the threshold the option layer sets.
    # The rest of the sweep locates the floor and shows the curve is monotonic,
    # which is how a power limit is told apart from a broken gate.
    at_threshold = min(recovery, key=lambda r: abs(float(r.name.split("_")[1][:-3]) - 4.01))
    cleared = [r for r in recovery if r.ok]
    floor = min(
        (float(r.name.split("_")[1][:-3]) for r in cleared), default=float("nan")
    )

    payload = {
        "schema_version": "v5.g1-known-answer-campaign.v2",
        "hypothesis_freeze_sha256": hypothesis.freeze_sha256(),
        "eligible_index_sha256": hypothesis.ELIGIBLE_INDEX_SHA256,
        "eligible_sessions": int(mask.sum()),
        "members": list(hypothesis.MEMBERS),
        "required_accuracy": hypothesis.REQUIRED_ACCURACY,
        "trials_per_question": args.trials,
        "recovery_sweep_points": sorted(args.recovery_points),
        "results": [r.as_dict() for r in results],
        "recovery": {
            "declared_threshold_points": 4.01,
            "recovery_at_threshold": round(at_threshold.rate, 6),
            "meets_declared_criterion": at_threshold.ok,
            "operational_detection_floor_points": floor,
            "monotonic": all(
                a.rate <= b.rate for a, b in zip(recovery, recovery[1:])
            ),
        },
        "all_criteria_met": (
            results[0].ok and results[1].ok and at_threshold.ok
        ),
        "real_economics_computed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    log("")
    for r in results:
        log(f"{r.name:<32} rate {r.rate:.4f} (limit {r.limit}) -> {'OK' if r.ok else 'FAIL'}")
    log(f"all criteria met: {payload['all_criteria_met']}")
    log(f"receipt: {args.out}")
    if not payload["all_criteria_met"]:
        log("STOP. Do not read the economics. A positive number without a passing")
        log("known-answer campaign is the row-183 failure with extra steps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
