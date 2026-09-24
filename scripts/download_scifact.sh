#!/usr/bin/env bash
# SciFact 다운로드 (CC BY-NC). HuggingFace 차단 환경에서도 S3 접근 가능.
# 사용: bash scripts/download_scifact.sh
set -euo pipefail

DEST="data/raw"
mkdir -p "$DEST"
URL="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

echo "[*] downloading SciFact → $DEST"
curl -fL --retry 3 -o "$DEST/scifact.tar.gz" "$URL"
tar xzf "$DEST/scifact.tar.gz" -C "$DEST"
echo "[*] done. files at $DEST/data/"
ls -1 "$DEST/data/"
