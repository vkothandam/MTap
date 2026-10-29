"""MODEL.md: the model card every training run writes into its run directory.

Front matter (`label`, `forecast`, `summary`, plus `kind` and `status`), then a short markdown
body: purpose, inputs, folds, result, command. The dashboard (forecast_dash/contract.py
read_card) shows the label instead of the directory name and the card in an ⓘ.

Trainers call `write` when a run starts (status running) and again when it ends (done, with
the result), so no run is ever without a card. A rewrite keeps the label, summary and purpose
already in the card unless new ones are given, so a label set at the start, by
`scripts_model_cards.py` or by hand, survives a resume.

    tft-vwap ... --label R7f --about "HGB + full pre-market" --purpose "Does HGB beat R6f?"

Shared by tft_vwap, tft_intraday and screener; imports none of them.
"""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path

CARD = "MODEL.md"
STATUS_TEXT = {"running": "Not finished: the run is still training.",
               "paused": "Paused before every fold was scored; continue with --resume.",
               "done": "Done."}


def add_args(p) -> None:
    g = p.add_argument_group("model card (MODEL.md in the run directory)")
    g.add_argument("--label", help="short name the dashboard shows, e.g. R7f (default: "
                                   "<kind> <MMDD-HHMM> of the run directory)")
    g.add_argument("--about", help="one line: what the run is (default: from its inputs)")
    g.add_argument("--purpose", help="why the run exists: the question it answers (markdown)")


def from_args(a, prog: str, argv: list[str] | None) -> dict:
    """The card fields given on the command line, plus the command itself."""
    import sys

    args = sys.argv[1:] if argv is None else list(argv)
    return {"label": a.label, "summary": a.about, "purpose": a.purpose,
            "command": shlex.join([prog, *args])}


def read(run_dir: Path) -> dict:
    """Front matter keys, and each `**Name.** text` section as a lowercase key; {} if none."""
    f = Path(run_dir) / CARD
    if not f.is_file():
        return {}
    text, meta = f.read_text(), {}
    body = text
    if text.startswith("---"):
        head, sep, rest = text[3:].partition("\n---")
        if sep:
            meta = {k.strip(): v.strip() for k, _, v in
                    (ln.partition(":") for ln in head.splitlines() if ":" in ln)}
            body = rest.lstrip("-").strip()
    parts = re.split(r"(?m)^\*\*([A-Z][\w ]*)\.\*\* ", body)
    sections = {parts[i].lower(): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if "command" in sections:
        sections["command"] = sections["command"].strip("`")
    return {**sections, **meta}


def default_label(kind: str, run_dir: Path) -> str:
    """`GBM 1029-1702` from a `20261029-170229Z-...` directory; the name itself otherwise."""
    name = Path(run_dir).name
    m = re.match(r"\d{4}(\d{4})-(\d{4})", name)
    return f"{kind} {m[1]}-{m[2]}" if m else name


def write(run_dir: Path, *, kind: str, forecast: str, inputs: str, folds: str,
          status: str = "running", label: str | None = None, summary: str | None = None,
          purpose: str | None = None, result: str | None = None,
          command: str | None = None) -> Path:
    run_dir = Path(run_dir)
    old = read(run_dir)
    label = label or old.get("label") or default_label(kind, run_dir)
    summary = " ".join((summary or old.get("summary") or f"{kind}: {inputs}").split())
    purpose = purpose or old.get("purpose")
    command = command or old.get("command")
    body = ([f"**Purpose.** {purpose}"] if purpose else []) + [
        f"**Inputs.** {inputs}", f"**Folds.** {folds}",
        f"**Result.** {result or STATUS_TEXT.get(status, status)}"]
    if command:
        body.append(f"**Command.** `{command}`")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / CARD).write_text(
        f"---\nlabel: {label}\nforecast: {forecast}\nsummary: {summary}\nkind: {kind}\n"
        f"status: {status}\n---\n\n" + "\n\n".join(body) + "\n")
    return run_dir / CARD


def label_of(run_dir: Path) -> str:
    return read(run_dir).get("label") or Path(run_dir).name


def contract_result(run_dir: Path) -> str:
    """The result line of a daily-contract run, pooled over its fold_*/metrics.json test blocks
    (row-weighted, so the pinball loss is exactly the pooled one)."""
    rows = []
    for f in sorted(Path(run_dir).glob("fold_*/metrics.json")):
        t = json.loads(f.read_text()).get("test") or {}
        if t.get("rows"):
            rows.append(t)
    if not rows:
        return "No test fold scored."
    n = sum(r["rows"] for r in rows)

    def pooled(key: str) -> float:
        return sum(r[key] * r["rows"] for r in rows) / n

    loss, naive = pooled("quantile_loss"), pooled("naive_quantile_loss")
    skill = f"Skill {1 - loss / naive:.1%} vs no change" if naive else "Skill n/a (naive loss 0)"
    text = (f"{skill} (pinball {loss:.5f}, naive {naive:.5f}), "
            f"P5–P95 coverage {pooled('interval_coverage'):.1%}")
    if all(r.get("directional_accuracy") is not None for r in rows):
        text += f", median direction right {pooled('directional_accuracy'):.1%}"
    return (text + f"; {n:,} test forecasts in {len(rows)} folds. Paired comparison with a rival: "
            "the dashboard's Compare runs tab or `tft-vwap-compare`.")
