"""Protocol 149: separate live-log visual inspection dashboard.

This deliberately does not reuse historical `trades.html`/`equity.html`.
It renders the actual live/paper session log into a compact operational
dashboard: launchd state, event timeline, blockers, paper account state, and
linked live-shadow artifacts.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
from typing import Any

from v4.live.paper_trade_log import DEFAULT_TRADE_LOG_ROOT, flatten_trade_event, load_trade_log
from v4.scripts.run_protocol140_ibkr_autostart_prep import GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL
from v4.scripts.run_protocol148_protocol101_post_session_analyzer import analyze_session_rows, resolve_trade_log


DEFAULT_OUT_ROOT = Path("v4/audit/autoresearch/v4_aplus_hypothesis_149_protocol101_live_log_visual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-log", type=Path, default=None)
    parser.add_argument("--trade-log-root", type=Path, default=DEFAULT_TRADE_LOG_ROOT)
    parser.add_argument("--session", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trade_log = resolve_trade_log(
        trade_log=args.trade_log,
        root=args.trade_log_root,
        session=args.session,
        run_id=args.run_id,
    )
    rows = load_trade_log(trade_log)
    if not rows:
        raise SystemExit(f"no rows found in {trade_log}")
    analysis = analyze_session_rows(rows, trade_log=trade_log)
    launchd = launchd_statuses()
    out_dir = args.out_root / str(analysis["session"]) / str(analysis["run_id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = [flatten_trade_event(row) for row in rows]
    payload = {
        "protocol": "149_protocol101_live_log_visual",
        "decision": "pass_live_log_visual_written",
        "paid_data_downloaded": False,
        "live_orders": False,
        "real_money_trading": False,
        "source_trade_log": str(trade_log),
        "analysis": analysis,
        "launchd": launchd,
        "outputs": {
            "html": str(out_dir / "live_session.html"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    write_html(out_dir / "live_session.html", payload, flat_rows)
    print(json.dumps({"decision": payload["decision"], "html": str(out_dir / "live_session.html")}, indent=2))
    return 0


def launchd_statuses() -> dict[str, Any]:
    return {
        label: launchd_status(label)
        for label in (GATEWAY_LABEL, PREFLIGHT_LABEL, SESSION_LABEL)
    }


def launchd_status(label: str) -> dict[str, Any]:
    try:
        uid = subprocess.run(["id", "-u"], text=True, capture_output=True, check=True).stdout.strip()
        result = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:
        return {"loaded": False, "state": "unknown", "error": str(exc)}
    state = "unknown"
    runs = None
    schedule = {}
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if line.startswith("state = "):
            state = line.split("=", 1)[1].strip()
        elif line.startswith("runs = "):
            runs = line.split("=", 1)[1].strip()
        elif line.startswith('"Minute" =>'):
            schedule["minute"] = line.rsplit("=>", 1)[1].strip()
        elif line.startswith('"Hour" =>'):
            schedule["hour"] = line.rsplit("=>", 1)[1].strip()
    return {
        "loaded": result.returncode == 0,
        "state": state,
        "runs": runs,
        "schedule": schedule,
    }


def write_html(path: Path, payload: dict[str, Any], flat_rows: list[dict[str, Any]]) -> None:
    analysis = payload["analysis"]
    launchd = payload["launchd"]
    status_cards = [
        ("Log", analysis["validation"]["status"]),
        ("Rows", str(analysis["rows"])),
        ("API", "open" if analysis["startup_and_data"]["gateway_or_api_port_open"] else "not confirmed"),
        ("Live Parity", "ready" if analysis["startup_and_data"]["live_parity_ready"] else "blocked"),
        ("Broker Calls", str(analysis["broker_order_endpoint_called_rows"])),
        ("Paper PnL", f"${analysis['paper_fill_pnl']['total_pnl']:.2f}"),
    ]
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Protocol101 Live Session</title>
  <style>
    :root {{ color-scheme: light; --line:#d6dde8; --ink:#172033; --muted:#657083; --bg:#f6f8fb; --panel:#fff; --ok:#0f7b50; --bad:#a12b2b; }}
    body {{ margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }}
    header {{ padding:18px 24px; background:#101828; color:white; }}
    h1 {{ margin:0 0 4px; font-size:20px; }}
    main {{ padding:18px 24px 32px; max-width:1400px; margin:auto; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:14px 0; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
    .k {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.02em; }}
    .v {{ font-size:18px; font-weight:650; margin-top:4px; }}
    section {{ margin-top:18px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }}
    th,td {{ padding:8px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; white-space:nowrap; }}
    th {{ position:sticky; top:0; background:#edf2f7; z-index:1; }}
    .table-wrap {{ max-height:520px; overflow:auto; border:1px solid var(--line); border-radius:8px; }}
    .ok {{ color:var(--ok); font-weight:650; }}
    .bad {{ color:var(--bad); font-weight:650; }}
    code {{ background:#eef2f7; padding:1px 4px; border-radius:4px; }}
    pre {{ white-space:pre-wrap; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; }}
  </style>
</head>
<body>
  <header>
    <h1>Protocol101 Live Session</h1>
    <div>{esc(analysis["session"])} · {esc(analysis["run_id"])}</div>
  </header>
  <main>
    <section class="grid">
      {''.join(card(k, v) for k, v in status_cards)}
    </section>
    <section>
      <h2>Launchd Status</h2>
      <div class="grid">
        {''.join(card(label.split('.')[-1], launchd_value(row)) for label, row in launchd.items())}
      </div>
    </section>
    <section>
      <h2>Blocked Reasons</h2>
      <pre>{esc(json.dumps(analysis["risk_reason_counts"], indent=2, sort_keys=True))}</pre>
    </section>
    <section>
      <h2>Linked Live Shadow Logs</h2>
      <pre>{esc(json.dumps(analysis["linked_shadow_logs"], indent=2))}</pre>
    </section>
    <section>
      <h2>Event Timeline</h2>
      <div class="table-wrap">
        {event_table(flat_rows)}
      </div>
    </section>
  </main>
</body>
</html>
"""
    path.write_text(html_text)


def card(key: str, value: str) -> str:
    klass = "ok" if value in {"pass", "ready", "open"} else "bad" if value in {"fail", "blocked", "not confirmed"} else ""
    return f'<div class="card"><div class="k">{esc(key)}</div><div class="v {klass}">{esc(value)}</div></div>'


def launchd_value(row: dict[str, Any]) -> str:
    schedule = row.get("schedule", {})
    when = f"{schedule.get('hour', '?')}:{str(schedule.get('minute', '?')).zfill(2)}"
    return f"{row.get('state', 'unknown')} @ {when}"


def event_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "timestamp",
        "event_type",
        "model_action",
        "risk_passed",
        "risk_reason",
        "cash",
        "equity",
        "open_positions",
        "contract_id",
        "action",
        "quantity",
        "limit_price",
        "spx",
        "vix",
        "bid",
        "ask",
    ]
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{esc(row.get(column, ''))}</td>" for column in columns)
            + "</tr>"
        )
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{esc(column)}</th>" for column in columns)
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


if __name__ == "__main__":
    raise SystemExit(main())
