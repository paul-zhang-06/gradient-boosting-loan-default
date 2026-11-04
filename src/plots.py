"""Shared plotting helpers."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIG = Path("outputs/figures")
sns.set_theme(style="whitegrid", context="notebook")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
