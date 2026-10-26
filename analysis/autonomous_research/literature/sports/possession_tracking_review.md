# Sports possession & tracking literature — audit for MTAH 2027 (families 1 & 2)

Scope. Direct football/sports possession (family 1) and open-source sports tracking, ball
reconstruction, pitch-control and event-inference implementations (family 2). Goal: only what is
genuinely new versus the local review in `docs/INVESTIGATION.md` §0, and what transfers under the
challenge rules (algorithm-only labels; no match data to a hosted LLM; no external weights; scarce
public labels = 17 possessions). Date: 2026-10-25. All claims below are `[reported]` from the cited
source unless marked `[local]`.

## 0. System state this review is aimed at (do not re-solve)

`[local]` From `docs/INVESTIGATION.md`, `docs/ROADMAP.md`, `mtah/feats.py`:

- Task: per-frame H/A/D over 199,319 tactical frames; ranked on spellF1. Current submission
  `final_v5` = gradient-boosted trees on 9 noised external matches + a fixed-cost Viterbi + a
  stoppage gate. Fold bench 0.661 (ext 0.616, masked 0.706); delivered public spellF1 0.723.
- Bottlenecks that remain:
  - **B1 convention mismatch**: IDSSE labels count ~4.3 touches/min vs our GT ~3.4/min. A learned
    temporal refiner (GRU/TCN) gained on external matches but lost on our match (masked −0.02–0.03,
    public −0.04–0.06) because it learned IDSSE's touch-happy sequence convention (§22–23).
  - **B2 H↔A boundary timing near turnovers**: the 2nd-half "drift" is near changes, mid-possession
    the model already matches the rules; errors are confident and happen with the ball visible (§18,
    §20).
  - **B3 short possessions**: recall 0.186 for spells < 2 s (§20, §25). But the ceiling analysis
    shows one spurious 0.4 s blip/min costs ~0.23 spellF1 while missing every < 2 s possession costs
    only ~0.12 (§1), so suppressing blips is correct by design. This tension is not a bug to remove.
  - **B4 ball gaps**: 41% of graded frames miss the ball, 85% in gaps > 10 s. Even a perfect ball
    restored in gaps adds only +0.005 bench (R2, §12). Ball reconstruction is a closed lever.
- Already in the feature set (`mtah/feats.py`, `mtah/shape.py`): ball-free team shape (centroid,
  spread, 2nd-deepest line, forward/adv direction, pressing distance, team speed), k-nearest
  distances, approach speed, **time-to-intercept pitch control** (static E8 and velocity-projected
  R8, the latter measured and dropped), ball/touch blocks, set-piece geometry, camera centre proxy.

## 1. What prior local work already covers (do not repeat)

From `docs/INVESTIGATION.md` §0 and `docs/REVIEWS.md`: Borghesi TCN possession (2023); PathCRF
ball-free possession path (arXiv 2602.12080); BallRadar ball-trajectory inference (arXiv 2306.08206);
Vidal-Codina automatic event detection (arXiv 2202.00804); Link & Hoernig individual ball possession
(PLOS ONE 2017); IDSSE/DFL dataset (Bassek, Sci Data 2025); MS-TCN/MS-TCN++, ASRF boundary head,
segmental MBR decoding, HSMM duration priors; Kaggle CMI sleep-states, DFL shootout, NFL contact.
Measured-and-dropped on our bench: video ball channel, DP ball linker, MBR, GRU/TCN refiner, boundary
head, in-Viterbi dwell, per-frame stacker, rule vote in decoder, noisy-student, CatBoost, velocity
pitch control (R8), touch-sequence features (R17), IDSSE relabel to our convention.

## 2. Family 1 — direct possession / control / transition methods (new)

| work | method / assumption | data | reported result | what transfers to us | licence |
|---|---|---|---|---|---|
| **Socceraction / SPADL / VAEP** (Decroos, Robberechts, KU Leuven) — repo https://github.com/ML-KULeuven/socceraction ; docs https://socceraction.readthedocs.io ; successor `silly-kicks` https://pypi.org/project/silly-kicks/ | Open-source event/action framework. SPADL normalises vendor events; VAEP values actions. **Corrected (integrity addendum, §7):** its `player_possession_time` is NOT a geometric control detector. The docstring says it "resets after a defensive interception attempt or a take-on", but the implementation only accumulates seconds when the SAME player performs consecutive actions and the previous type was a dribble. No opponent distance, no contested-ball test; it needs SPADL event types we do not have per frame. The docstring line is a narrative gloss, not an implemented rule. | Event streams (not tracking). Action-level, vendor format. | VAEP predicts goal/non-goal of actions; not a per-frame possession label. | Not a geometric controlled-possession reference. The E1 contested-moment test (below) is therefore designed from our own tracking (both teams within 2 m of the ball), not borrowed from this code. Code only, no trained weights, no data. | **MIT** (repo); `silly-kicks` MIT. Code only, no trained weights. Compatible. |
| **Renner, Koster, Bauer, Schienle 2026 — GNN-LSTM match-phase classification** — arXiv https://arxiv.org/pdf/2610.11571v1 | GNN over player graph (Delaunay triangulation = passing lanes) + SEAConv edge features + LSTM; 7-class phase per second. Integrated gradients: signal is horizontal positioning + possession + ball-status. | Rule-based labels over **203 matches**; DFB. | GNN-LSTM beats aggregated-feature baselines (XGBoost, plain LSTM) by **4.6 points macroF1**. | Negative evidence for us: the graph earns its gain only with ~203 labelled matches. We have 9, and any temporal/graph stage risks re-learning the external sequence convention (B1). Treat as a reason NOT to add a GNN, with a citation. | arXiv preprint; code not confirmed open. We would not build it. |
| **Bauer & Anzer 2021 — data-driven counterpressing detection** — Data Min. Knowl. Disc. https://link.springer.com/content/pdf/10.1007%2Fs10618-021-00763-7.pdf | XGBoost on **134 positional+event features** to label "team wins the ball back within X s after losing it" (defensive transition). Compactness, distance to ball, pressing distance, recovery time. | 20,000 hand-labelled defensive transitions, 97 pro matches, Bundesliga. | AUC 0.874 on held-out; applied across 6 seasons. | The feature *design* for the turnover regime (B2) transfers: tight marking on the carrier, defence compressing toward the ball, closing speed. We cannot borrow their 20k hand labels (rules + our own validation only), but the feature list is public physics. | Publisher PDF; read-only. Features are ideas, not code we ship. |
| **Bian 2024 — Soccer Last Touch & duel events with skeletal tracking** — MIT thesis https://dspace.mit.edu/bitstream/handle/1721.1/156773/bian-gbian-meng-eecs-2024-thesis.pdf | Decision tree for last-touch and duel/goalkeeper events using **skeletal joint data** from broadcast video. | Broadcast video poses. | Thesis-level; duel-like classification with joint context. | Last-touch = the exact possession-change instant. But it needs video pose estimation, which we do not run and which T&Cs keep off hosted models. Not transferable to a tracking-only pipeline. | Thesis; not used. |

## 3. Family 2 — open-source tracking / ball / pitch-control / event code (new)

| artifact | method | evidence | verdict for us | licence |
|---|---|---|---|---|
| **Friends-of-Tracking-Data / LaurieOnTracking — `Metrica_PitchControl.py`** — repo https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking ; file http://raw.githubusercontent.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/master/Metrica_PitchControl.py | The canonical open-source time-to-intercept pitch-control surface (each player's arrival time to a point; team control = logistic of nearest-player arrival-time gap). | Standard Metrica reference; we already load Metrica sample data. | Our `velocity_control` (`mtah/feats.py`) is this same nearest-player arrival-time model with a 0.7 s velocity projection. It is already present and was measured and dropped (R8). **No new lever**; cite as the reference our feature already matches. | Open (attribution); we already attribute Metrica. |
| **Wu & Swartz, "A New Metric for Pitch Control based on an Intuitive Motion Model"** — https://www.sfu.ca/~tswartz/papers/pitch_control.pdf | Time-to-reach model using current velocity, acceleration and player-specific max speed. | Validated: team with pitch control = observed pass recipient on **91% of 5,887 passes**. | Same family as our machinery. The only new knob is player-specific max speed and acceleration; marginal given our uniform 5 m/s + 11.5 m/s clip. Low priority. | Academic PDF; not used. |
| **Da Silva et al. 2025 — pitch control + player top speed + stamina** — arXiv https://arxiv.org/pdf/2504.18250 (LaLiga Mediacoach) | Player-specific measured top speed and a stamina decay in the arrival-time model. | Scientific Reports; role-dependent gain. | A refinement of the same arrival-time feature. Only worth trying if a pressure/transition feature (E2) first shows pitch-control inputs still have headroom. | Preprint; not used. |
| **Fernández, Bornn, Cervone 2021 — expected possession value (EPV) framework** — https://pmc.ncbi.nlm.nih.gov/articles/PMC8570314/ | Decomposes the value of a possession into pass/drive/shot sub-models with interception-probability surfaces over 22 players + ball. | FC Barcelona / UPC; calibrated DNN surfaces. | This is a *scoring-value* model, not a per-frame H/A/D label. Its interception surface is again arrival-time pitch control. Not a possession classifier; not transferable to our output. | PMC open; not used. |
| Ball reconstruction repos (BallRadar arXiv 2306.08206; TrackNet-style temporal ball trackers) | Predict/impute the ball path from player contexts. | In-review; locally measured: even a perfect ball in gaps adds **+0.005 bench** (R2). | Closed lever (B4). No new repo changes this arithmetic. | — |

## 4. Compatibility and licensing limitations

- Importable artefacts are **code/rule only**: Socceraction (MIT) and FoTD/LaurieOnTracking (open,
  attribution). We ship no weights from them and download no extra data.
- External training data already in use and within licence: IDSSE (CC BY 4.0) and Metrica sample
  (open, attribution requested; upstream LICENSE still unresolved — flagged as REVIEWS #9).
- Bauer & Anzer's 20,000 hand labels and Bian's skeletal data are **not** used: our rule is
  algorithm-only labels, and video pose is off-limits to hosted models. We borrow feature *design*
  only.
- Nothing in families 1–2 supplies a ready public per-frame possession label in our convention
  (confirms ROADMAP: PFF needs a personal sign-up; SkillCorner has no per-frame possession).

## 5. Ranked small informative experiments against the bottlenecks

Ranked by expected information per unit cost, under the rule that adopt-only-if-bench moves >= 0.005
and a borderline multi-try win is a "no". All are analysis-only; none mutates `mtah/` model code; all
print aggregates only.

- **E1 (targets B1; highest information) — Contested-duel reweighting of the IDSSE training labels.
  Our own geometric hypothesis, NOT a Socceraction rule (integrity addendum §7).** Define a "contested"
  frame purely from our tracking: ball seen AND at least one home and one away player within 2 m
  (`seen & nh2>=1 & na2>=1`). The hypothesis is that IDSSE labels a team at such frames while our GT
  attributes to the last team, so down-weighting IDSSE contested frames teaches the model not to flip
  at a 50-50. Implemented as a sample-weight on the fold fits (Metrica never reweighted). Cost: one
  analysis script + possibly one refit.
- **E3 (targets B2; cheapest, decoder-only on cached `P_R1`) — Conditional H/A switch cost at
  contested balls.** Where the ball is seen and two opposing players are both within ~2 m of it (a
  duel), raise the Viterbi team-switch cost. The challenge rule says an aerial duel without subsequent
  control does **not** switch possession, so this directly encodes a GT rule against the costliest
  error (spurious flips). No refit; reuse cached probabilities. Cost: hours.
- **E2 (targets B2; screen first) — Ball-carrier pressure / transition-compactness features.** Add,
  when the ball is seen: (a) nearest-opponent distance to the ball (carrier tightness), (b) gradient
  of the defending team's compactness toward the ball (is the defence compressing), (c) that
  opponent's closing speed. Grounded in Bauer & Anzer's counterpressing feature design. Screen on the
  fast tier; expect a small effect (R8/R17 screen gains largely vanish in full fit). Cost: screen,
  then at most one refit.
- **E4 (record as considered-and-rejected) — Do not build a Delaunay GNN / second deep family.**
  Renner et al. show the graph earns +4.6 macroF1 with ~203 labelled matches; we have 9 and a
  convention mismatch (B1), so a GNN would repeat the refiner failure. Keep a citation in the
  technical doc and move on. Cost: none.
- **E5 (lowest priority, only if E2 passes) — Player-specific arrival-time pitch control.**
  Per-slot max speed and acceleration in the existing `velocity_control` (Wu & Swartz; Da Silva
  2025). Marginal; our uniform-speed version is already measured. Cost: screen.

## 6. Bottom line

Families 1 and 2 are largely tapped locally. Socceraction is NOT the geometric control reference it
first appeared to be (its possession timer is a same-player-dribble event counter; integrity §7). The
open question it motivates is ours: whether down-weighting IDSSE frames where both teams are within
2 m of a seen ball aligns training with our convention. The other transfers are feature-design ideas
(counterpressing compactness, ball-carrier pressure) and confirmation that our pitch-control already
matches the open reference. Ball reconstruction and deep temporal/graph stages remain closed levers
under the scarce-label and convention-mismatch constraints.
