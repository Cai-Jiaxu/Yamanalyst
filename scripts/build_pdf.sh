#!/usr/bin/env bash
# Export the technical document to PDF for the submission form (*.pdf is gitignored).
# Needs pandoc on PATH; WeasyPrint is fetched by uv for this command only (it needs Pango:
# macOS `brew install pandoc pango`, Debian/Ubuntu `apt install pandoc libpango-1.0-0 libpangoft2-1.0-0`).
# Usage: bash scripts/build_pdf.sh [team_name]      -> <team_name>.pdf (default: Yamanalyst.pdf)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:-Yamanalyst}
TMP=$(mktemp -d)
cp docs/technical_document.css "$TMP/style.css"
pandoc Yamanalyst.md -f gfm -t html5 -s --metadata title="$NAME" -c style.css -o "$TMP/doc.html"
# pandoc repeats the metadata title above the document's own heading; hide it
sed -i.bak 's#<header id="title-block-header">#<header id="title-block-header" style="display:none">#' "$TMP/doc.html"
uv run --no-project --with weasyprint weasyprint "$TMP/doc.html" "$NAME.pdf"
rm -rf "$TMP"
echo "wrote $NAME.pdf"
