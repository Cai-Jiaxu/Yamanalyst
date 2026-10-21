# Attack plan: MTAH 2027 Phase 1 (ball possession)

Deadline **17 Nov 2026**. Registration closes **10 Nov**, so make sure all 3–5 team members are registered under the same team name before then.

---

## 0. What the repo tells us (the facts the plan is built on)

| Fact | Where | Consequence |
|---|---|---|
| Ranked on **spellF1**: possessions (maximal H or A runs) matched 1-to-1 by temporal IoU, mean F1 at IoU 0.3–0.7 | `evaluation/possession_metrics.py:517` | We're scored on **segmentation**, not per-frame accuracy. |
| `H H D D H H` = **two** possessions | CHALLENGE.md | Every stoppage starts a new spell. **Dead-ball detection is half the problem.** |
| Being 1 s late costs 0.35 spellF1; 3 frames late costs only 0.02 | EVALUATION.md | Boundary *timing* matters. Use centred (non-causal) filters only, and calibrate any lag. |
| `X` gets forward-filled before spellF1 | `possession_metrics.py:274` | For spellF1, `X` is exactly the same as copying our previous label, and macroF1 counts it as a miss. **Never write X on graded frames. Do the fill ourselves.** |
| GT is generated mechanically from **corrected** tracking: D where it marks the ball dead, otherwise the team it says owns the ball. **No smoothing** | CHALLENGE.md | Real spells can be very short. The public window has spells of 0.8 s and 1.3 s. The target is "who the corrected tracker says owns it", so learn its conventions. |
| Raw ATD gives only `x,y,speed` for 46 team-level slots plus the ball. **No ball-owner or ball-alive flag** | verified on the file format | Both ownership and alive/dead have to be inferred. |
| Ball missing on 41% of graded frames | DATA.md | Ball imputation is the biggest lever on the data side. |
| Public GT: 5 min, **17 possessions** (A 6.6 s, H 8.1 s, A 2.8 s, H 41.6 s, …, mean 14.2 s) | `evaluation/gt/` | Far too small to tune on. The baselines prove it: smooth125 wins on public and loses on the full match. Extrapolated, the full match has about 300–400 possessions. |
| 2nd half is graded from 110,808 but kicks off at 113,742, so 2,934 frames are all D | FAQ.md | Free points, but detect the kickoff **algorithmically** (rules: labels must come from an algorithm). |
| Hand-labelled data **for training/validation is allowed** | RULES.md | This is how we fix the 17-possession validation problem (§2). |
| No uploading data or extracts to a cloud LLM (including Claude) | OVERVIEW.md, T&Cs | The LLM writes code. The code runs locally, and only scores and aggregates come back. Never paste tracking rows, frames or video. |

---

## 1. Kaggle lessons that apply directly

| Lesson | From | How we apply it |
|---|---|---|
| **Trust your CV, not the public LB** | Every Kaggle shake-up | Never choose between variants on the 5-min window. Build a bigger local validation set (§2) and use `--bootstrap 1000`. |
| **Post-processing tuned to the metric beats a better model** | CMI *Detect Sleep States* (1st place: UNet+GRU; post-processing alone moved private 0.829 → 0.852); DFL *Bundesliga Data Shootout* (event detection with tolerance windows) | Separate the per-frame **probabilities** from the **decoder** (HMM/Viterbi with switching costs, minimum durations, peak-picking). Tune the decoder directly on spellF1. |
| **Predict events, not states** | CMI Sleep, SoccerNet action spotting (T-DEED, 2024 winner) | Alongside the frame model, train a "transition here?" head with Gaussian-smoothed targets, then do NMS/peak detection to place boundaries precisely. |
| **Tracking features + GBDT first, then fuse with video** | NFL *Player Contact Detection* (winning solutions combined tracking features with 2.5D/3D CNNs on video) | Start with LightGBM on hand-crafted tracking features (fast, interpretable). Only add video once that plateaus. |
| **Pseudo-labelling / noisy student** | Many Kaggle comps with little labelled data | Take high-confidence frames from the rule system over the full match, train on them, then iterate. |
| **Ensemble probabilities, then decode once** | Standard | Average the model probabilities before Viterbi. Never vote on decoded labels. |
| **Pick flat optima** | Shake-up avoidance | Choose hyperparameters from the middle of a plateau, not from a sharp peak. |
| **Reproducibility from day 1** | Required here ("not approximately: the same file") | Pin deps with `uv`, fix seeds, make the pipeline deterministic, and regenerate from a clean clone in CI before submitting. |

---

## 2. Fix the validation problem first (the single most important step)

17 possessions cannot rank anything. We need **≥150 labelled possessions**.

1. **Get the videos.** The bundle unpacked in `data/` has the ATD, homography and tagging, but **neither `.mp4`**. They come from the same enrolment download link (probably a separate, much larger file). Re-check that email or page. If the videos aren't there, email `hackathon.talentarena@mobileworldcapital.com`. Ask on the repo issues as well, since other teams will hit the same thing.
2. **Learn the GT conventions** by watching the public 5 minutes side by side with the GT timeline. Exactly when does D start: when the ball crosses the line, or at the whistle? When does D end: at the restart touch? Does a pass that gets intercepted switch at the interception touch, or once control is established? How long is the "controlled" delay? Write the answers down as a short spec.
3. **Hand-label a validation set** following that spec. Use a quick labelling tool: timeline plus video plus a key per transition, so we label transitions rather than individual frames. Aim for 4 × 5-minute windows spread across both halves (about 70 more possessions), more if time allows. Documented in the tech doc, this is allowed: "hand-labelled data used to train or validate a model is fine".
   - Labellers must label **independently of model output** so the validation set stays unbiased.
   - Keep these labels out of the public repo if they could count as "derived data that reconstructs" the match. Ask the organisers. The safe default is to describe them in the doc and keep them private.
4. **Scoring harness:** `run_evaluation.py` with `--gt <our csv>`, plus a per-transition diagnostic (missed, spurious and late transitions, and the timing-error histogram).

---

## 3. Pipeline

```
ATD (+homography, Smart Tagging, [video])
  └─ A. clean & impute ─ B. features ─ C. per-frame P(H), P(A), P(D) + P(transition)
                                           └─ D. structured decoder (tuned on spellF1) ─ E. CSV
```

### A. Cleaning and imputation
- Parse the ATD once into `npz`/parquet arrays `(frames × slots × {x, y, s})` plus the ball. The text parse takes minutes, and we only want to pay that once.
- Convert to metres (105.3 × 68.0). Remove ball outliers (implied speed > 40 m/s, sudden teleports).
- **Short ball gaps (< ~0.5–1 s):** fill with a constant-velocity or ballistic fit between the observations on either side.
- **Long ball gaps:** use the proxies in §B, and optionally a learned imputer (*Ball Radar*, Kim et al. KDD 2023: Set Transformer plus hierarchical Bi-LSTM. Its first stage is literally a "ball possessor classifier"). Train it on external data (§F).
- Slot identities swap, but we only need team-level sets, so treat each team as an unordered set (no ID stitching needed).

### B. Features (per frame, plus windowed context of about ±1/2/5 s)
- **Ball:** position, velocity, acceleration, visibility flag, time since last seen, **direction-change events** (a proxy for touches).
- **Ball and players:** distance to the nearest H player and nearest A player, the difference between them, counts within 1/2/5 m, the nearest player's velocity relative to the ball, and who was nearest at the last direction change (the last touch).
- **Team shape (still works when the ball is missing):** centroids, spread/area (the attacking team stretches, the defending team compacts), defensive-line height, mean speed, speed towards each goal, pressing intensity. Teams switch ends at half-time.
- **Tactical camera:** the tactical camera follows the ball. The homography gives the pitch point at the image centre, so camera pan and zoom are a **free ball-location proxy** when the ball is missing. Camera stillness and halftime-style reprojection error also hint at stoppages.
- **Dead-ball cues:** ball outside the pitch bounds (with a margin), ball stationary, all players slow, set-piece formations (a player at the corner flag or touchline, a wall, players clustered in the box, a goalkeeper with the ball in the 6-yard box), a kickoff formation (each team in its own half, ball on the centre spot).
- **Smart Tagging priors:** the team tag of the active phase (Build Up, Progression, …), and `SET PIECES` intervals as "restart near here". Throw-ins are tagged `N/A`, so take the side from geometry instead. Treat the tags as evidence only, never as labels.

### C. Models (in order, each must beat the last on our validation set)
1. **Rule system (week 1):** possession zones and duel zones (Vidal-Codina et al., arXiv 2202.00804; Link & Hoernig 2017, PLOS ONE). A team gains possession only after **controlled** contact: within r m for ≥ N frames, or two consecutive same-team touches. Contested or loose ball means possession stays with the last team, as the CHALLENGE.md rules say. Add a dead-ball state machine on top. This should already beat 0.2433 comfortably, and it supplies the pseudo-labels.
2. **LightGBM per-frame classifier** (H/A/D) plus a **transition head**. Train on: public 5 min + hand labels + high-confidence pseudo-labels. Use grouped CV by time block, never random frames, because neighbouring frames leak.
3. **Sequence model** (BiGRU or 1D-UNet/TCN at 5–10 Hz with set-pooled player features, CMI-Sleep style). Only if (2) plateaus and we have enough labels or external pretraining.

### D. Decoder (where most of the spellF1 is won)
- **HMM/HSMM Viterbi** over {H, A, D} with emission = model probabilities, a **switching penalty** and **minimum-duration** priors per state, and separate costs for H↔A and alive↔dead.
- Keep the asymmetry in mind: a false blip *inside* a long spell costs an FP **and** can break the matched spell into two that fail IoU. Missing a 1-second true spell costs only one FN. So the penalty should lean towards fewer switches, but the GT does contain 1-second spells, so don't over-smooth (the baseline's smooth125 lesson).
- Snap boundaries to the frame of peak transition probability (event-head NMS) instead of where the smoothed state happens to flip.
- Detect the 2nd-half kickoff algorithmically (kickoff formation plus Smart Tagging `Kick Off (Start)`) and label D before it.
- Tune with a grid or Optuna **on the hand-labelled validation set** with bootstrap CIs, and pick a plateau.

### E. Output
- Graded frames: always H/A/D (never X). Ungraded frames: X.
- Run `validate_submission.py` on the exact file to be uploaded.

### F. External data (allowed if public and documented)
Use it for pretraining the possession and ball-imputation models. Simulate *our* noise on it: drop the ball for 41% of frames in occlusion-like bursts, drop and fragment tracks, strip player IDs.
- **Metrica Sports sample data** (`github.com/metrica-sports/sample-data`): same provider and format family, tracking plus events.
- **PFF FC World Cup 2022 open data:** tracking plus events with possession, 64 matches.
- **IDSSE / Sportec open Bundesliga dataset** (Bassek et al.): tracking plus events with possession and ball status.
- **SkillCorner open data:** broadcast tracking.
- For each one, check the licence before use.

### G. Video (optional, after the tracking pipeline is solid)
- **Ball detection** on the tactical video: YOLO fine-tuned on soccer-ball data, run with tiled inference (SAHI) because the ball is tiny at 1080p. Project detections onto the pitch with the homography to fill the ATD ball gaps. All of this runs **locally** (M2 Pro with MPS: roughly 2–4 h for 199k frames at reduced fps).
- **Ball action spotting** (T-DEED, SoccerNet 2024 winner, with public code and weights): pass, drive and set-piece events give touch timing and restarts.
- **Broadcast:** skip unless there's spare time. Syncing it costs a lot. The one cheap signal is that replays only happen during stoppages.

---

## 4. Timeline (4 weeks)

| Week | Goal | Exit criterion |
|---|---|---|
| **1: 21–27 Oct** | Environment (uv, pinned deps), ATD → npz cache, scorer harness plus timeline visualiser, rule system v1, dead-ball state machine. Get the videos. Write the GT-conventions spec. Start hand-labelling. | Rule system beats 0.2433 on public 5 min **and** possession count looks sane (~3–4 per minute). First valid full CSV. |
| **2: 28 Oct–3 Nov** | Feature set, LightGBM plus transition head, HMM decoder, validation set ≥ 100 possessions, pseudo-labels. | Beats the rule system on hand-labelled validation outside the bootstrap CI. |
| **3: 4–10 Nov** | Ball imputation (external pretraining), optional YOLO ball detection, ensemble, decoder tuning. **Confirm team registration by 10 Nov.** | Ablation table: what each component adds. |
| **4: 11–17 Nov** | **Freeze code 14 Nov.** Reproduce from a clean clone, check the hash matches, write the tech doc, submit 15–16 Nov (form allows edits until the deadline). | Validated CSV, public repo with no data in it, PDF named `<team>.csv` / `<team>.pdf`. |

## 5. Suggested split (3–5 people)
- **Data/infra:** loader, cache, scorer harness, visualiser, reproducibility, final repo.
- **Football/rules:** GT conventions spec, hand-labelling lead, rule system, dead-ball logic.
- **ML:** features, LightGBM, sequence model, pseudo-labels, external data.
- **Decoder/metric:** HMM, spellF1 tuning, bootstrap, ablations.
- **CV (optional 5th):** YOLO ball detection, action spotting.

## 6. Risks
- **Overfitting the 5 minutes.** Mitigated by §2.
- **Boundary lag** from causal filters. Use centred windows and check the timing-error histogram.
- **Non-determinism** (LightGBM threads, GPU). Set `deterministic=True`, fix seeds, store models or retrain deterministically.
- **Data hygiene:** `data/`, videos and hand labels never go into git. Never send data to hosted LLMs.
