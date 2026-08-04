#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/_vendor"
mkdir -p "$VENDOR"
clone() { # url commit dir
  [ -d "$3/.git" ] || git clone "$1" "$3"
  git -C "$3" checkout --quiet "$2"
}
clone https://github.com/lmccccc/FANNBench.git ebfbbd2ab375bab64d7b0302a4b16f72198ec495 "$VENDOR/FANNBench"
clone https://github.com/YujianFu97/NHQ.git    f19ebc4747c2889f1ef7a485fbf0f8664ec75dc8 "$VENDOR/NHQ"
echo "[fetch] done. Next: ./apply_patches.sh"
