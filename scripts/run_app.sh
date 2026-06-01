#!/usr/bin/env bash
# Build + launch the Mac app (GUI window + REST on :8765). Fetches the bundle if missing.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d bundle ] || ./scripts/fetch_bundle.sh
export LFM2_BUNDLE="$ROOT/bundle" LFM2_PORT="${LFM2_PORT:-8765}"
swift build -c release --package-path Lfm2VlKit
echo "launching app — window opens; REST at POST http://localhost:$LFM2_PORT/caption"
exec ./Lfm2VlKit/.build/release/lfm2vl-app
