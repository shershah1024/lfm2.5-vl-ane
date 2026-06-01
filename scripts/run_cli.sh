#!/usr/bin/env bash
# One-shot caption from the command line:  ./scripts/run_cli.sh <image> ["prompt"]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -d bundle ] || ./scripts/fetch_bundle.sh
swift build -c release --package-path Lfm2VlKit
exec ./Lfm2VlKit/.build/release/lfm2vl-cli bundle "${1:-demo.png}" "${2:-What do you see in this image?}"
