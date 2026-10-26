# Six-item follow-up coverage (working note, current)

Goal aspiration: about 0.90 spellF1 / macroF1. No promise. Development only.
MacroF1 reported excludes X (H/A/D).

| item | status | result |
|---|---|---|
| 1 fusion broadcast ball | identity restored, recovery diagnostic done | Seed0 0.7310/0.8945, seed1 0.6897/0.9273 match baseline. Recovery seed0 85/4523 hidden, max prob diff 0.0764, 0 label change, score unchanged. Seed1 0/4366, 0 change. Diagnostic only, no 16-seed gate. |
| 2 physical control latch | tested, strong negative | v2 vector velocity + carrier continuity: bench 0.3030, ext 0.1679, masked 0.4380, public 0.5120/0.8805. Strict fixed gate fails. Not promoted. |
| 3 out-of-pitch emission | tried, flat | boost weights 2/4/8 leave dev bench 0.6612/0.6163/0.7061 unchanged |
| 4 two-head transition | rejected | AR-TI0 identity exact. AR-TI1 lambda=2: bench 0.6573 (-0.0039), ext 0.6220 (+0.0057), masked 0.6926 (-0.0135), public 0.6710. Rejected. |
| 5 filter + NMS DoG | fixed, public negative | independent scorer: spellF1 0.6857, macro 0.9049 vs baseline 0.7226/0.9098. w=0 identity exact. Dev bench +0.006 but public down. Not promoted. |
| 6 pseudo anchors | availability partial, no F1 | SmartTags 638 events (kickoff 6, throw 26, corner 12, goal kick 10, free kick 5). Kickoff D agreement 0.586 includes X, biased. Not ground truth. |

Notes:
- v5 protected public: spellF1 0.722581, macroF1 0.909848.
- E1 corrected: bench 0.6581, ext 0.613, masked 0.703. Not promoted.
- Holdout g3: tied 0.6128 clean.
- No final HTML until wrapper fixed.
