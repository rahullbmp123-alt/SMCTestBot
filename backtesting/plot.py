"""
Equity curve and performance visualisation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    _HAS_PLOT = True
except ImportError:
    _HAS_PLOT = False

from backtesting.engine import BacktestResult
from config import settings
from utils.logger import get_logger

log = get_logger(__name__)

_OUT_DIR = settings.LOGS_DIR


def plot_equity_curve(result: BacktestResult, save: bool = True) -> None:
    if not _HAS_PLOT:
        log.warning("matplotlib not installed — skipping plots")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    ts = result.timestamps
    eq = result.equity_curve
    min_len = min(len(ts), len(eq))
    ts = ts[:min_len]
    eq = eq[:min_len]

    # ── Equity curve ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(ts, eq, color="#00b4d8", linewidth=1.5, label="Equity")
    ax1.fill_between(ts, min(eq), eq, alpha=0.15, color="#00b4d8")
    ax1.axhline(eq[0], color="grey", linestyle="--", linewidth=0.8, label="Start")
    ax1.set_title("Equity Curve", fontweight="bold")
    ax1.set_ylabel("Balance ($)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Drawdown ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    eq_arr = np.array(eq)
    peak = np.maximum.accumulate(eq_arr)
    dd = (peak - eq_arr) / np.where(peak > 0, peak, 1) * 100
    ax2.fill_between(ts, 0, -dd, color="#e63946", alpha=0.6, label="Drawdown")
    ax2.set_title("Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("Drawdown %")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # ── Win/Loss distribution ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    labels = ["Win", "Loss", "BE"]
    sizes = [result.wins, result.losses, result.breakevens]
    colors = ["#2dc653", "#e63946", "#adb5bd"]
    non_zero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if non_zero:
        ls, ss, cs = zip(*non_zero)
        ax3.pie(ss, labels=ls, colors=cs, autopct="%1.0f%%", startangle=90)
    ax3.set_title("Win / Loss Distribution", fontweight="bold")

    # ── Summary stats ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    stats = result.summary()
    rows = [[k.replace("_", " ").title(), v] for k, v in stats.items()]
    tbl = ax4.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    ax4.set_title("Performance Summary", fontweight="bold")

    fig.suptitle(
        f"SMC Bot Backtest — {settings.SYMBOL}",
        fontsize=14, fontweight="bold"
    )

    if save:
        out = _OUT_DIR / "equity_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info(f"Chart saved → {out}")
    else:
        plt.show()

    plt.close(fig)
