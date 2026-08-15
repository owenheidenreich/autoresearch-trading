"""Paper-trading model default registry.

The daily paper autopilot should run the current paper-approved default, not
whichever research artifact happened to be trained most recently.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DEFAULT_PAPER_TRADING_DEFAULT = Path("v4/promotion/PAPER_TRADING_DEFAULT.json")
SUPPORTED_RUNNER_KINDS = {"protocol101_persistent"}


@dataclass(frozen=True)
class PaperModelSelection:
    registry_path: Path
    model_id: str
    display_name: str
    entrypoint: str
    runner_kind: str
    run_id_template: str
    arguments: dict[str, str]
    payload: dict[str, Any]

    def default_run_id(self, session: str) -> str:
        return self.run_id_template.format(session=session, model_id=self.model_id)


def load_paper_model_selection(
    registry_path: Path = DEFAULT_PAPER_TRADING_DEFAULT,
    *,
    model_id: str | None = None,
) -> PaperModelSelection:
    payload = json.loads(Path(registry_path).read_text())
    selected_id = str(model_id or payload.get("current_model_id") or "").strip()
    if not selected_id:
        raise ValueError("paper default registry is missing current_model_id")

    models = payload.get("models")
    if not isinstance(models, dict) or selected_id not in models:
        raise ValueError(f"paper default model {selected_id!r} is not registered")

    model = models[selected_id]
    if str(model.get("paper_trading_status")) != "approved_paper_default":
        raise ValueError(f"paper default model {selected_id!r} is not paper-approved")

    runner_kind = str(model.get("runner_kind") or "")
    if runner_kind not in SUPPORTED_RUNNER_KINDS:
        raise ValueError(f"paper default runner {runner_kind!r} is not supported")

    arguments = model.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError(f"paper default model {selected_id!r} has invalid arguments")

    return PaperModelSelection(
        registry_path=Path(registry_path),
        model_id=selected_id,
        display_name=str(model.get("display_name") or selected_id),
        entrypoint=str(model["entrypoint"]),
        runner_kind=runner_kind,
        run_id_template=str(model.get("run_id_template") or "daily_paper_autopilot_{session}"),
        arguments={str(key): str(value) for key, value in arguments.items()},
        payload=payload,
    )


def selection_summary(selection: PaperModelSelection) -> dict[str, Any]:
    return {
        "feature_name": str(selection.payload.get("feature_name") or "daily paper autopilot"),
        "model_id": selection.model_id,
        "display_name": selection.display_name,
        "entrypoint": selection.entrypoint,
        "runner_kind": selection.runner_kind,
        "registry_path": str(selection.registry_path),
        "run_id_template": selection.run_id_template,
        "selection_policy": selection.payload.get("selection_policy"),
        "arguments": selection.arguments,
    }
