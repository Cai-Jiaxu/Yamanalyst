#!/usr/bin/env bash
# Public datasets used for training/validation (W8). Not match data; licences:
#   IDSSE (Bassek et al., Sci Data 2025, doi:10.1038/s41597-025-04505-y)  CC BY 4.0, via figshare API
#   Metrica Sports sample data (github.com/metrica-sports/sample-data)   open, attribution requested
# Usage: bash scripts/download_external.sh   -> external/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external/idsse external/metrica

while read -r match meta event tracking; do
  for pair in "meta:$meta" "event:$event" "tracking:$tracking"; do
    kind=${pair%%:*}; id=${pair##*:}
    out="external/idsse/${match}_${kind}.xml"
    [ -s "$out" ] && continue
    curl -sSL --retry 3 -o "$out.part" "https://api.figshare.com/v2/file/download/$id"
    mv "$out.part" "$out"
    echo "idsse $match $kind $(du -h "$out" | cut -f1)"
  done
done <<'IDS'
J03WPY 51643487 51643505 51643526
J03WN1 51643472 51643496 51643517
J03WMX 51643475 51643493 51643514
J03WOH 51643478 51643499 51643520
J03WQQ 51643484 51643508 51643529
J03WOY 51643481 51643502 51643523
J03WR9 51643490 51643511 51643532
IDS

BASE=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
for g in 1 2; do
  for f in RawEventsData RawTrackingData_Home_Team RawTrackingData_Away_Team; do
    out="external/metrica/Sample_Game_${g}_${f}.csv"
    [ -s "$out" ] || curl -sSL --retry 3 -o "$out" "$BASE/Sample_Game_${g}/Sample_Game_${g}_${f}.csv"
    echo "metrica game $g $f $(du -h "$out" | cut -f1)"
  done
done
