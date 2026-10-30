"""Plot the short CUDA execution check for OpenFerro Example 1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
NCELLS = 8**3
TEMPERATURES = (320, 300)
COLORS = ("#2563eb", "#dc2626", "#16a34a")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 220})
fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.4), sharex=True,
                         gridspec_kw={"hspace": 0.15})

for idx, temperature in enumerate(TEMPERATURES):
    # The second window follows 50 sampling and 20 equilibration steps.
    offset = 20 + idx * 70
    field = np.loadtxt(OUT / f"field_{temperature}K_avg.log",
                       comments="#", delimiter=",", ndmin=2)
    thermo = np.loadtxt(OUT / f"thermo_{temperature}K.log",
                        comments="#", delimiter=",", ndmin=2)
    steps = offset + field[:, 0]
    for component, (color, label) in enumerate(zip(COLORS, "xyz"), start=1):
        axes[0].plot(steps, field[:, component], "o-", color=color,
                     label=rf"$\langle u_{label}\rangle$" if idx == 0 else None)
    axes[1].plot(offset + thermo[:, 0], thermo[:, 14] / NCELLS, "o-",
                 color="#7c3aed")
    axes[2].plot(offset + thermo[:, 0], thermo[:, 13] / NCELLS, "o-",
                 color="#ea580c")
    for ax in axes:
        ax.axvspan(offset, offset + 40, color="#e2e8f0", alpha=0.4, zorder=-2)
    axes[0].text(offset + 20, 0.93, f"{temperature} K",
                 transform=axes[0].get_xaxis_transform(), ha="center", va="top")

axes[0].legend(ncol=3, frameon=False, loc="lower left")
axes[0].set_ylabel("Mean soft mode (Å)")
axes[1].set_ylabel("Potential energy (eV/cell)")
axes[2].set_ylabel("Volume (Å³/cell)")
axes[2].set_xlabel("Cumulative MD step")
axes[2].set_xlim(10, 140)
fig.suptitle("OpenFerro Example 1: short GPU execution check", y=0.98)
fig.text(0.5, 0.01,
         "8³ cells · CUDA:0 · 20 equilibration + 50 sampling steps per temperature · "
         "dt = 0.002 ps\n"
         "Diagnostic trace only: minimization did not meet force tolerance; "
         "trajectory is too short for phase conclusions.",
         ha="center", va="bottom", fontsize=9, color="#475569")
fig.subplots_adjust(top=0.92, bottom=0.12, left=0.16, right=0.97)
fig.savefig(OUT / "gpu_test_trace.png")
fig.savefig(OUT / "gpu_test_trace.pdf")
