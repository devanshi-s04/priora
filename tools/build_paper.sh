#!/usr/bin/env bash
# Build paper/main.pdf. Needs a LaTeX installation; the AASTeX class is fetched by 'make external'.
set -euo pipefail
cd "$PRIORA_REPO/paper"
[ -f aastex701.cls ] || { echo "aastex701.cls missing: run 'make external' first"; exit 1; }
pdflatex -interaction=nonstopmode main >/dev/null; bibtex main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null; pdflatex -interaction=nonstopmode main >/dev/null
echo "paper/main.pdf built"
