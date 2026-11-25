#!/usr/bin/env bash
# Download and unpack the stored simulation / NEOCP outputs (validation/output) from the Zenodo record given as $1.
set -euo pipefail
URL="$1"; cd "$PRIORA_REPO"
curl -L --fail -o /tmp/priora_outputs.tar.gz "$URL"
tar -xzf /tmp/priora_outputs.tar.gz -C "$PRIORA_REPO"
echo "stored outputs unpacked into validation/output"
