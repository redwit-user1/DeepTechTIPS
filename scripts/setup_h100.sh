#!/usr/bin/env bash
# AI Train(H100) 컨테이너 부트스트랩 — teleport 로 옮겨온 뒤 첫 실행.
#
# 사용:
#   bash scripts/setup_h100.sh            # 학습 스택까지 전부
#   bash scripts/setup_h100.sh --eval-only # 평가만(서빙 연동 시나리오)
set -euo pipefail

EVAL_ONLY=0
[ "${1:-}" = "--eval-only" ] && EVAL_ONLY=1

echo "=============================================================="
echo " GOONO AI — H100 컨테이너 셋업"
echo "=============================================================="

echo
echo "[1/4] 기본 패키지"
pip install -q -e . && echo "  ✅ compliance_gateway"

echo
echo "[2/4] 데이터"
if [ ! -f data/raw/data/corpus.jsonl ]; then
  bash scripts/download_scifact.sh >/dev/null && echo "  ✅ SciFact"
else
  echo "  ✅ SciFact (이미 존재)"
fi
python -m compliance_gateway.data.build_dpo >/dev/null && echo "  ✅ bioRxiv 합성 DPO"
python -m compliance_gateway.data.korean.real_eval >/dev/null && echo "  ✅ 국내 실데이터 평가셋"

if [ "$EVAL_ONLY" = "1" ]; then
  echo
  echo "[3/4] 학습 스택 — 건너뜀 (--eval-only)"
else
  echo
  echo "[3/4] 학습 스택 (torch/transformers/trl/peft/…)"
  pip install -q -e ".[train]"
  echo "  ✅ 기본 학습 스택"
  # Unsloth/vLLM 은 환경 의존성이 커서 실패해도 진행(HF 폴백 경로 존재)
  pip install -q unsloth 2>/dev/null && echo "  ✅ unsloth" || echo "  ⚠️  unsloth 설치 실패 → HF 폴백 사용"
  pip install -q vllm 2>/dev/null && echo "  ✅ vllm" || echo "  ⚠️  vllm 설치 실패 → GRPO 는 use_vllm=False 로"
fi

echo
echo "[4/4] 환경 진단"
python scripts/probe_env.py

echo
echo "=============================================================="
echo " 다음 단계"
echo "=============================================================="
if [ "$EVAL_ONLY" = "1" ]; then
  cat <<'TXT'
  서빙 엔드포인트로 KPI 측정:
    export KT_API_KEY=<키>
    python -m compliance_gateway.eval.external --split dev \
        --nli-endpoint <서빙주소>/v1 --nli-model <모델ID> --sweep
    python -m compliance_gateway.eval.korean --real \
        --nli-endpoint <서빙주소>/v1 --nli-model <모델ID>
TXT
else
  cat <<'TXT'
  M1 전체 실행(기준선 → NLI 파인튜닝 → EN/KR 재측정):
    bash scripts/run_m1_h100.sh

  기준선(이 값들이 얼마나 오르는지가 판단 근거):
    · 영어 외부(SciFact dev, θ=0.70)  F1 24.0%
    · 한국어 실데이터                  AUC 0.715 (사용 가능한 운영점 없음)
TXT
fi
