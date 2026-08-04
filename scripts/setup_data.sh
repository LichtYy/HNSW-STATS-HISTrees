#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_SRC="${DATASETS_SRC:?set DATASETS_SRC to the directory holding the datasets}"
GT_CACHE_SRC="${GT_CACHE_SRC:-$ROOT/.scratch/gt_cache}"

mkdir -p "$GT_CACHE_SRC"
mkdir -p "$ROOT/data"
ln -sfn "$DATASETS_SRC" "$ROOT/data/datasets"
ln -sfn "$GT_CACHE_SRC" "$ROOT/data/gt_cache"
echo "data/datasets -> $DATASETS_SRC"
echo "data/gt_cache -> $GT_CACHE_SRC"
