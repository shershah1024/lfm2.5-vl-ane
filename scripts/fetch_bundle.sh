#!/usr/bin/env bash
# Download the on-device model bundle (compiled CoreML/ANE) into ./bundle.
# The bundle has files >100 MB, so it's hosted rather than committed to git.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${LFM2_BUNDLE_URL:-https://github.com/shershah1024/lfm2.5-vl-ane/releases/download/v0.2.0/lfm2.5-vl-ane-20260602.tar.gz}"
cd "$ROOT"
if [ -d bundle ]; then echo "bundle/ already present — delete it to re-fetch."; exit 0; fi
echo "downloading bundle (~500 MB) from $URL ..."
curl -L --fail -o /tmp/lfm2-bundle.tar.gz "$URL"
echo "extracting bundle/ ..."
tar xzf /tmp/lfm2-bundle.tar.gz bundle
rm -f /tmp/lfm2-bundle.tar.gz
echo "done -> $ROOT/bundle ($(du -sh bundle | cut -f1))"
