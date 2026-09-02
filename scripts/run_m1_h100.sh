#!/usr/bin/env bash
# M1 파이프라인 (H100, KT Cloud AI Nexus) — NLI 파인튜닝 후 실데이터로 KPI 재측정.
#
# 목적: "통계 NLI 로는 목표 도달 불가"(docs/EVAL_EXTERNAL.md)를 확인했으므로,
#       트랜스포머 NLI 를 학습해 KPI 가 실제로 얼마나 오르는지 측정한다.
#       이 수치가 과제 목표(90%+) 실현 가능성의 판단 근거다.
#
# 사용: bash scripts/run_m1_h100.sh [출력경로]
#
# 먼저 환경 확인:  python scripts/probe_env.py
set -euo pipefail

OUT="${1:-checkpoints/nli}"
BASE="${NLI_BASE:-deberta-mnli}"

echo "==================================================================="
echo " M1: NLI 파인튜닝 → 외부 실데이터 KPI 재측정"
echo "==================================================================="

echo
echo "[0/6] 환경 진단"
python scripts/probe_env.py || true
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "[중단] GPU 가 없습니다. AI Serv(추론 서빙)만 프로비저닝된 환경일 수 있습니다."
  echo "  → 학습은 AI Train 컨테이너에서 실행하세요."
  echo "  → 서빙만 있다면 학습 없이 평가만 가능합니다:"
  echo "      python -m compliance_gateway.eval.external --split dev \\"
  echo "          --nli-endpoint <서빙주소>/v1 --nli-model <모델ID>"
  exit 1
fi

echo
echo "[1/6] SciFact 데이터 준비"
if [ ! -f data/raw/data/corpus.jsonl ]; then
  bash scripts/download_scifact.sh
else
  echo "  이미 존재 — 건너뜀"
fi

echo
echo "[2/6] 데이터·설정 검증 (dry-run)"
python -m compliance_gateway.train.nli_finetune --dry-run

echo
echo "[3/6] 학습 전 기준선 — 통계 NLI, dev split"
echo "      (파인튜닝은 train 으로 하므로 공정 비교는 반드시 dev)"
python -m compliance_gateway.eval.external --split dev | tee /tmp/m1_before.txt

echo
echo "[4/6] NLI 파인튜닝 (train → $OUT)"
python -m compliance_gateway.train.nli_finetune \
  --base "$BASE" --split train --eval-split dev --output "$OUT"

echo
echo "[5/6] 학습 후 재측정 — dev split (누출 없음)"
python -m compliance_gateway.eval.external \
  --split dev --nli "$OUT" --device cuda --sweep | tee /tmp/m1_after.txt

echo
echo "==================================================================="
echo " 결과 비교 (외부 실데이터 dev, 정직한 일반화 성능)"
echo "==================================================================="
echo "--- 학습 전 (통계 NLI v0.5) ---"
grep -E "위반탐지|compliant PASS" /tmp/m1_before.txt || true
echo "--- 학습 후 (트랜스포머 NLI) ---"
grep -E "위반탐지|compliant PASS" /tmp/m1_after.txt || true
echo
echo
echo "[6/6] 한국어 실데이터 재측정 (국내 기관 프로토콜 원문)"
python -m compliance_gateway.eval.korean --real --threshold 0.64 \
  --nli "$OUT" --device cuda | tail -20

echo
echo "판단 기준:"
echo "  · 목표 = 규정위반 탐지 Precision 90%+"
echo "  · 영어 외부 기준선(dev, θ=0.70) = Precision 39.6% / F1 24.0%"
echo "  · 한국어 실데이터 기준선          = AUC 0.715 (사용 가능한 운영점 없음)"
echo "  · 개선폭이 작다면 → 도메인 데이터 확대 또는 더 큰 NLI 모델 검토"
