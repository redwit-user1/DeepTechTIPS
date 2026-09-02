#!/usr/bin/env bash
# 로컬 OCR 연구노트 작업 — 원스톱 실행기.
#
# GPU·torch 불필요. Python 3.9+ 만 있으면 된다(파이프라인 전체가 표준 라이브러리).
#
# 사용:
#   bash scripts/run_local_ocr.sh /path/to/ocr            # 1~3단계 순차(dry-run)
#   bash scripts/run_local_ocr.sh /path/to/ocr --write    # 결과 저장까지
set -euo pipefail

ROOT="${1:-}"
WRITE=""
[ "${2:-}" = "--write" ] && WRITE="--write"

if [ -z "$ROOT" ]; then
  echo "사용법: bash scripts/run_local_ocr.sh /path/to/ocr [--write]"; exit 1
fi
if [ ! -d "$ROOT" ]; then
  echo "[중단] 경로를 찾을 수 없습니다: $ROOT"; exit 1
fi

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "[중단] python3 을 찾을 수 없습니다"; exit 1; }
echo "Python: $($PY --version)"

# 저장소 루트에서 실행되도록
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# 비식별화 salt — 미설정 시 경고 후 진행
if [ -z "${GOONO_DEID_SALT:-}" ]; then
  echo
  echo "⚠️  GOONO_DEID_SALT 미설정 — 기본값으로 진행합니다."
  echo "    운영에서는 조직 고유 비밀값을 설정하고 공개하지 마세요:"
  echo "      export GOONO_DEID_SALT='...'"
fi

echo
echo "══════════════════════════════════════════════════════════════"
echo " 1단계 — 프로파일링 (원문 미출력)"
echo "══════════════════════════════════════════════════════════════"
"$PY" scripts/profile_ocr_dataset.py "$ROOT" --json ocr_profile.json --sample-fields

echo
echo "══════════════════════════════════════════════════════════════"
echo " 2단계 — 표본 수확 (500건, dry-run)"
echo "══════════════════════════════════════════════════════════════"
"$PY" -m compliance_gateway.data.ocr.harvest "$ROOT" --limit 500

if [ -n "$WRITE" ]; then
  echo
  echo "══════════════════════════════════════════════════════════════"
  echo " 3단계 — 전량 수확 및 저장"
  echo "══════════════════════════════════════════════════════════════"
  "$PY" -m compliance_gateway.data.ocr.harvest "$ROOT" $WRITE --out data/real/labnote
  echo
  echo "저장 완료: data/real/labnote/notes.jsonl (비식별화 완료, gitignore 대상)"
else
  echo
  echo "──────────────────────────────────────────────────────────────"
  echo " 표본 결과를 확인한 뒤 전량 처리하려면:"
  echo "   bash scripts/run_local_ocr.sh $ROOT --write"
  echo "──────────────────────────────────────────────────────────────"
fi

echo
echo "생성 파일: ocr_profile.json (원문 미포함 — 공유 가능)"
