"""Ranked-experiment map after the executed BOCPD sweep. Pure local, no match data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, cost_days, bench_delta, adopted, note)
points = [
    ("baseline v2g", 0.0, 0.0, True, "reproduced 0.661"),
    ("BOCPD h=1/400", 0.5, -0.028, False, "tested, negative"),
    ("BOCPD h=1/25", 0.3, -0.033, False, "tested, negative"),
    ("BOCPD h=1/50", 0.3, -0.029, False, "tested, negative"),
    ("BOCPD h=1/100", 0.3, -0.030, False, "tested, negative"),
]

fig, ax = plt.subplots(figsize=(9, 5))
for label, cost, delta, adopt, note in points:
    color = "#1b7837" if adopt else "#d73027"
    ax.scatter(cost, delta, s=180, c=color, alpha=0.8, edgecolors="black", linewidths=0.6, zorder=3)
    ax.annotate(label, (cost, delta), textcoords="offset points", xytext=(8, 6), fontsize=9)

ax.axhline(0.005, color="black", linestyle="--", linewidth=1)
ax.text(1.45, 0.006, "adoption gate +0.005", fontsize=8)
ax.axhline(0.0, color="grey", linewidth=0.8)
ax.set_xlabel("cost (person-days, no GPU)")
ax.set_ylabel("bench delta vs v2g 0.661")
ax.set_title("Executed BOCPD convention-neutral smoother on cached P_R1\n"
             "all predeclared hazard values lose; masked drops 0.04-0.05")
ax.set_xlim(-0.1, 1.8)
ax.set_ylim(-0.05, 0.02)
ax.grid(alpha=0.3)
fig.tight_layout()
out = "analysis/autonomous_research/literature/sequence/ranked_experiments.png"
fig.savefig(out, dpi=150)
print("wrote", out)
