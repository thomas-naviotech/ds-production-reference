"""Pretty terminal output with rich — tables + mini bar charts.

One place for all CLI display so every script looks the same in class.
Plain-text fallback if rich is ever missing (never crash the demo).
"""

from __future__ import annotations

try:
    from rich.console import Console
    from rich.table import Table

    _HAS_RICH = True
except ImportError:  # pragma: no cover - rich is a hard dep, this is just safety
    _HAS_RICH = False

_console = None
BAR_WIDTH = 20  # blocks in the mini F1 bar chart


def get_console():
    # Single shared console (rich handles colors/width detection).
    global _console
    if _console is None and _HAS_RICH:
        _console = Console()
    return _console


def f1_bar(value: float | None) -> str:
    # Mini bar chart: '#' = score, '-' = remainder. ASCII-only so it
    # renders on every student console (Windows cp1252 included).
    if value is None:
        return "-"
    filled = round(max(0.0, min(1.0, float(value))) * BAR_WIDTH)
    return "#" * filled + "-" * (BAR_WIDTH - filled)


def leaderboard(rows: list[dict]) -> None:
    # Classroom model race: sorted rows + F1 bar chart; winner row is green.
    con = get_console()
    if not _HAS_RICH or con is None:
        for r in rows:  # fallback: plain lines, same info
            print(f"  {r['model']:8s} acc={r['accuracy']:.2f} f1={r['f1']:.2f}")
        return
    table = Table(title="Leaderboard (by f1, then roc_auc, then accuracy)")
    table.add_column("Model", style="bold")
    for col in ("Acc", "Prec", "Rec", "F1", "ROC-AUC"):
        table.add_column(col, justify="right")
    table.add_column("F1 chart", justify="left")
    for i, r in enumerate(rows):
        table.add_row(
            ("* " if i == 0 else "  ") + r["model"],
            f"{r['accuracy']:.2f}", f"{r['precision']:.2f}", f"{r['recall']:.2f}",
            f"{r['f1']:.2f}", str(r["roc_auc"]), f1_bar(r["f1"]),
            style="bold green" if i == 0 else "",
        )
    con.print(table)


def show_metrics(title: str, metrics: dict) -> None:
    # One metric scorecard (train / evaluate / tune final). 2-decimal reads.
    con = get_console()
    keys = [k for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "n_test")
            if k in metrics]
    if not _HAS_RICH or con is None:
        print(title + ": " + "  ".join(f"{k}={metrics[k]}" for k in keys))
        return
    table = Table(title=title)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for k in keys:
        table.add_row(k, str(metrics[k]))
    con.print(table)


def show_params(title: str, params: dict) -> None:
    # Hyperparameter set (tuning stages): one row per param.
    con = get_console()
    if not _HAS_RICH or con is None:
        print(title + ": " + ", ".join(f"{k}={v}" for k, v in params.items()))
        return
    table = Table(title=title)
    table.add_column("Param", style="bold")
    table.add_column("Value", justify="right")
    for k, v in params.items():
        table.add_row(str(k), str(v))
    con.print(table)


def show_confusion(cm: list) -> None:
    # 2x2 confusion matrix [rows=actual, cols=predicted].
    con = get_console()
    if not _HAS_RICH or con is None:
        print(f"TN={cm[0][0]} FP={cm[0][1]} / FN={cm[1][0]} TP={cm[1][1]}")
        return
    table = Table(title="Confusion matrix [rows=actual, cols=predicted]")
    table.add_column("", style="bold")
    table.add_column("pred 0", justify="right")
    table.add_column("pred 1", justify="right")
    table.add_row("actual 0", str(cm[0][0]), str(cm[0][1]))
    table.add_row("actual 1", str(cm[1][0]), str(cm[1][1]))
    con.print(table)
