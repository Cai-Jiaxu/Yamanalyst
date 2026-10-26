# Decoder experiment log (chronological)

Frozen baseline = final_v5 reproduced from cache/P_R1.npz (no refit, no manual labels).
Adoption rule: bench delta >= +0.005 AND ext delta >= -0.005 AND masked delta >= -0.005.
venv interpreter used directly; no uv.lock mutation; no repo files modified.

## 2026-10-25

1. **R0 reproduce** (`reproduce_v5.py`) — bench 0.6612, ext 0.6163 (idsse 0.6073,
   metrica 0.6477), masked 0.7061, public sp 0.6968/mf 0.9098, val sp 0.5792/mf 0.8411.
   Errors/100: missed 22.3, drift 19.4, merge 10.6, spurious 3.6, split 2.0; pred/true 0.71.
   Matches historical 0.661/0.616/0.706 exactly.

2. **E3 duel-conditioned switch cost** (`e3_sweep.py`) — raise H<->A cost when >=1 home AND
   >=1 away within 2 m (seen & nh2>=1 & na2>=1). extra_team in {+1,+2,+4,+8}.
   Result: bench 0.661 -> 0.660 (flat/slightly down), ext -0.0007, masked 0.000. Rejected.
   The +0.011 val_h2 bump at extra>=4 is the known eye-label trap (eye labels reward rule
   timing), not a real gain. Spurious stays 3.6/100; decoder already conservative.

3. **C1-C4 transition levers** (`cycle2_levers.py`) — C1 cost_dead=8, C2 cost_dead=6,
   C3 w_bound=0.5, C4 w_bound=1.0.
   - C1: bench 0.6600 (ext 0.6147, masked 0.7052) — rejected.
   - C2: bench 0.6588 (ext 0.6151, masked 0.7024) — rejected, worse both.
   - C3: bench 0.6595 (ext 0.6177 up, masked 0.7014 down) — rejected.
   - C4: bench 0.6608 (ext 0.6202 +0.004, masked 0.7014 -0.005) — rejected.

**Diagnostic:** across every transition-cost lever the error counts are flat (missed ~22,
drift ~19, merge ~10.6, spurious ~3.6). The tested transition-cost levers did not clear
the adoption gate; this does not rule out transition smoothing broadly, and the invariant
taxonomy (computed per-variant from the custom callback's predictions, not a Yard.report
artifact) does not by itself prove emissions are the root cause. Missed short spells
(<2 s recall ~0.19) are an emission/model problem, owned by the sports/training track. Cross-track signal: w_bound (external-trained boundary head) lifts
ext +0.004 but drops masked -0.005 — external label conventions do not transfer to our
missing-ball regime; do not add boundary emission features without our-match calibration.

**Best = frozen final_v5 (0.6612).** No decoder candidate cleared the adoption bar.
Not promoted to final; gates (stop_gate=100, min_dead=25, kickoff=half) remain required.
