# Physical Latch Availability Assessment (corrected with actual stats)

## Player tracking data (verified)
- Schema: `m.players` shape (n, 101, 3) — 101 track slots per frame
- `m.team`: (101,) int8, 0=home (47 slots), 1=away (54 slots)
- Third column (z) is all 0.0 — dummy/placeholder, NOT visibility
- `m.ball`: (n, 3) ball position

## Reliability diagnostics (idsse_J03WPY, 149k frames, 25 FPS)
- Per-slot observation rate: 19.8%
- Visible players per frame: mean 20.0, median 21, P10 19, P90 22, max 22
- Ball finite frames: 58.1%
- 2-frame continuity rate: 19.8%

## Interpretation
- ~20-21 players visible per frame (out of 22 expected)
- 101 slots = ~22 real players + ~79 extra tracking/noise points
- True per-player ball-relative velocities: FEASIBLE
- Velocity derivable via frame-to-frame difference at 25 FPS
- Active-track filtering needed (most slots NaN)

## Kickoff prior status
- `kickoff.evidence()`: mode="all" includes after-goal restarts
- mode="half" adopted in frozen v5
- Pre-kickoff dead ball imposed only when kick SEEN
- Existing comment: "restarts after goals, up to 35 min in (bench: IDSSE -0.021)"
- No explicit Smart Tagging goal events; restarts inferred from formation
