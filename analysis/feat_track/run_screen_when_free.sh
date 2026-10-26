#!/bin/bash
# Wait until the parent rebuild (PID given as $1) exits, then run the feature screen chain.
# Fully detached so agent-message interruptions cannot kill it.
PID_WAIT="$1"
LOG="$HOME/Desktop/BARCA/more-than-a-hack-2027/analysis/feat_track/feat_screen_run.log"
cd "$HOME/Desktop/BARCA/more-than-a-hack-2027" || exit 1

echo "coord: waiting for PID $PID_WAIT to finish at $(date +%H:%M:%S)" >> "$LOG"
while kill -0 "$PID_WAIT" 2>/dev/null; do
  sleep 20
done
echo "coord: PID $PID_WAIT gone at $(date +%H:%M:%S); starting screen chain" >> "$LOG"

caffeinate -i .venv/bin/python analysis/feat_track/feat_screen.py >> "$LOG" 2>&1
echo "coord: screen chain exited $? at $(date +%H:%M:%S)" >> "$LOG"
