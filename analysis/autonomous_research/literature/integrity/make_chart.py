"""Effective independent possessions per yardstick. Read-only aggregate plot. No match data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Independent true possessions each yardstick actually scores on (from AUDIT.md / INVESTIGATION).
yardsticks = [
    ("ext (9 external matches, fold-by-match)", 3000, "stable, other-league convention"),
    ("val_h2 (4 hand-labelled 2nd-half windows)", 57, "eye labels, selected twice"),
    ("masked (public window, 16 ball-mask seeds)", 17, "same 17 possessions; seeds vary ball only"),
    ("public (delivered 5-min window)", 17, "sanity check only"),
]
names = [y[0] for y in yardsticks]
vals = [y[1] for y in yardsticks]
notes = [y[2] for y in yardsticks]
colors = ["#2c7fb8", "#e6550d", "#9e9ac8", "#bdbdbd"]

fig, ax = plt.subplots(figsize=(9, 3.8))
bars = ax.barh(names, vals, color=colors)
ax.set_xscale("log")
ax.set_xlabel("independent true possessions scored (log scale)")
ax.set_title("Yardstick effective sample size: why ext is the steadiest gate")
for bar, v, n in zip(bars, vals, notes):
    ax.text(v * 1.15, bar.get_y() + bar.get_height() / 2, f"{v}  ({n})",
            va="center", fontsize=8)
ax.set_xlim(10, 20000)
ax.invert_yaxis()
plt.tight_layout()
out = "/Users/bytedance/Desktop/BARCA/more-than-a-hack-2027/analysis/autonomous_research/literature/integrity/yardstick_sample_size.png"
plt.savefig(out, dpi=130)
print("wrote", out)
