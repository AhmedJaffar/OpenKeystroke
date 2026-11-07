"""Static PNG plots stored as experiment artifacts (the dashboard renders interactive versions from the same data)."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def confusion_png(cm: list[list[int]], classes: list[str], title: str) -> bytes:
    a = np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(a / np.maximum(a.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, str(int(a[i, j])), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(classes)), classes, rotation=60, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=8)
    return _png(fig)


def training_png(history: dict[str, Any], title: str) -> bytes:
    ep = history["epochs"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    x = [e["epoch"] for e in ep]
    for k, style in (("train_loss", "-"), ("val_loss", "--")):
        if ep and k in ep[0]:
            ax.plot(x, [e[k] for e in ep], style, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=8)
    return _png(fig)


def defense_png(curves: list[dict[str, Any]], chance: float) -> bytes:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for c in curves:
        pts = [p for p in c["points"] if p["x"] is not None]
        if not pts:
            continue
        x = [p["x"] for p in pts]
        label = f"{c['defense_type']} / {c['model']}"
        if all(p["static_accuracy"] is not None for p in pts):
            ax.plot(x, [p["static_accuracy"] for p in pts], "o-", label=f"{label} (static)")
        if all(p["adaptive_accuracy"] is not None for p in pts):
            ax.plot(x, [p["adaptive_accuracy"] for p in pts], "s--", label=f"{label} (adaptive)")
    ax.axhline(chance, color="grey", lw=0.8, ls=":", label="chance")
    ax.set_ylabel("held-out accuracy")
    ax.set_xlabel("defense parameter")
    ax.legend(fontsize=6)
    return _png(fig)
