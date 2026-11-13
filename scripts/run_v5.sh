#!/usr/bin/env bash
# 2026-11-12 v5: v4 에서 실패한 '기반 vs LoRA 평가' 재수행 (연구노트 §38)
#
#   NAS 폴더를 마운트한 세션에서:  nohup bash repo/scripts/run_v5.sh > logs_v5.txt 2>&1 &
#
# v4 의 평가는 평가셋이 0건이라 수치 없이 끝났다(문서 ID 가 전부 홀수 → `int(doc,16) % 10 == 0` 이 거짓).
# sft_v1 은 그래서 1차 4,229건 **전부**로 학습됐다 → 1차 단락으로는 평가할 수 없다.
# 2차 생성(gen2)의 문서는 1차와 겹치지 않으므로, 1차와 단락이 겹치는 묶음을 뺀 2차 문서 묶음 10% 를 평가셋으로 쓴다.
#
#   1. 분할    heldout2(평가 지시 약 2,300) · sft2_train(약 35,000)
#   2. 평가    기반 Qwen3.5-9B · sft_v1 1에폭(checkpoint-265) · sft_v1 2에폭 — 같은 지시, 탐욕 디코딩
#   3. 짝 비교 McNemar · 묶음 부트스트랩 구간 · 길이(짧게 써서 통과하는지)
#   4. 순환 점검 같은 응답 400건을 LLM 단독 판정으로 재심사 — 학습 필터와 평가가 같은 캐스케이드라는 문제
#   5. (SFT_V2=1 일 때만) sft2_train 으로 1에폭 학습 → 같은 평가. 약 2시간.
#      README 원칙: 본 정렬은 검증기 사람 검수(κ ≥ 0.8) 뒤. 여기서는 데이터 규모 효과를 보는 시험 학습이다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr
. $R/venv/bin/activate
B=Qwen/Qwen3.5-9B
echo "== 0. 모델 받기 $(date)"
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B'); s('$B')" > $L/dl_v5.txt 2>&1 && echo MODELS_READY || { echo MODELS_FAIL; exit 1; }

echo "== 1. 분할 $(date)"
python scripts/eval_sft.py split --gen $I/gen2.jsonl --sft $I/sft2.jsonl --exclude $I/gen.jsonl \
  --train-out $I/sft2_train.jsonl --heldout-out $I/heldout2.jsonl || { echo SPLIT_FAIL; exit 1; }

ev() {  # 이름 [어댑터] — 결과가 이미 있으면 건너뛴다(중단 후 재개)
  [ -s $I/eval2_$1.jsonl ] && { echo "-- $1 (있음, 건너뜀)"; grep -E "충실 생성률" $L/eval2_$1.txt | tail -5; return; }
  python scripts/eval_sft.py run --model $B ${2:+--adapter $2} --heldout $I/heldout2.jsonl --out $I/eval2_$1.jsonl > $L/eval2_$1.txt 2>&1
  echo "-- $1 (exit $?)"; grep -E "충실 생성률|판정기|어댑터|Error|거부" $L/eval2_$1.txt | tail -8
}
echo "== 2. 평가 $(date)"
ev base
ev sft_v1_e1 $CK/sft_v1/checkpoint-265
ev sft_v1 $CK/sft_v1

echo "== 3. 짝 비교 $(date)"
echo "[기반 → sft_v1 2에폭]";  python scripts/eval_sft.py compare $I/eval2_base.jsonl $I/eval2_sft_v1.jsonl
echo "[기반 → sft_v1 1에폭]";  python scripts/eval_sft.py compare $I/eval2_base.jsonl $I/eval2_sft_v1_e1.jsonl
echo "[1에폭 → 2에폭]";        python scripts/eval_sft.py compare $I/eval2_sft_v1_e1.jsonl $I/eval2_sft_v1.jsonl

echo "== 4. 순환 점검: LLM 단독 재심사 400건 $(date)"
for n in base sft_v1; do
  python scripts/eval_sft.py rejudge --in $I/eval2_$n.jsonl --out $I/eval2_${n}_llm.jsonl --limit 400 > $L/eval2_${n}_llm.txt 2>&1
  echo "-- $n"; grep -E "^전체|일치|판정기" $L/eval2_${n}_llm.txt
done
echo "[LLM 단독 판정 기준 기반 → sft_v1]"; python scripts/eval_sft.py compare $I/eval2_base_llm.jsonl $I/eval2_sft_v1_llm.jsonl

if [ "${SFT_V2:-0}" = 1 ]; then
  echo "== 5. SFT v2 시험 학습(sft2_train, 1에폭) $(date)"
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft2_train.jsonl --output $CK/sft_v2 \
    --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v2.txt 2>&1
  echo "SFT_EXIT $?"; grep -E "LoRA 모듈|train_loss|saved" $L/sft_v2.txt | tail -3
  ev sft_v2 $CK/sft_v2
  echo "[기반 → sft_v2]";   python scripts/eval_sft.py compare $I/eval2_base.jsonl $I/eval2_sft_v2.jsonl
  echo "[sft_v1 → sft_v2]"; python scripts/eval_sft.py compare $I/eval2_sft_v1.jsonl $I/eval2_sft_v2.jsonl
fi
echo "V5_DONE $(date)"
