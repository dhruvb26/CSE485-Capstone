#!/usr/bin/env bash
set -euo pipefail

payload="$(cat)"

# Run only when report/report.md changed
if echo "$payload" | rg -q '"path"\s*:\s*".*report/report\.md"'; then
  cd report
  pandoc report.md -o report.pdf --citeproc
  cd ..
fi