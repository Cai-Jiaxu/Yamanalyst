# Goal-Conditioned Kickoff Prior Feasibility

## Existing kickoff module
- `mtah/kickoff.py`: detect() + evidence() functions
- mode="all": dead ball before each half's first kickoff, + kicking team after every kickoff
- mode="half": only dead ball before half's first kickoff (adopted in frozen v5)
- Pre-kickoff dead ball imposed ONLY when kick itself SEEN (formation-only detection too late)

## Goal event availability
- No explicit Smart Tagging goal events in tracking data
- Goal restarts inferred from formation detection (not event labels)
- Existing comment: "restarts after goals, up to 35 min in (bench: IDSSE -0.021)"

## Tested vs new conditioning
- TESTED: half-kickoff dead ball (mode="half"), after-goal restart (mode="all")
- NOT TESTED: goal-conditioned prior (boosting team possession immediately after goal)
- The current kickoff evidence already adds team-specific log-evidence after kickoffs

## Timing prerequisites
- HALF_WAIT window for pre-kickoff dead ball
- STILL_PACE threshold for waiting players
- Formation detection: BRIDGE, FIRST_WITHIN, FORM_FRAMES parameters

## Missing data
- No explicit goal event labels (Smart Tagging)
- No team identity at goal events
- Goal restarts inferred, not observed

## Feasibility conclusion
- Goal-conditioned kickoff prior: PARTIALLY FEASIBLE
- Current kickoff evidence already adds team-specific evidence after kickoffs
- True goal event conditioning requires explicit event tags (not available)
- Existing after-goal restart detection is formation-based, not event-based
