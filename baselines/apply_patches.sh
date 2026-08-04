#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
git -C "$ROOT/_vendor/FANNBench" apply --check "$ROOT/patches/serf/serf_add_load.patch"
git -C "$ROOT/_vendor/FANNBench" apply         "$ROOT/patches/serf/serf_add_load.patch"
echo "[apply_patches] SeRF load() patch applied."
