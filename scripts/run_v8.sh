#!/usr/bin/env bash
# 2026-11-13 v8 대기열 — 긴 원문(KCI 초록) 평가 · 혼합 학습 · 판정 인코더 재학습 (연구노트 §42)
#
#   bash repo/scripts/run_v8.sh chain     # v7 두 조각이 끝나기를 기다렸다가 두 줄을 띄운다(GPU 가 비지 않게 미리 걸어 둔다)
#   bash repo/scripts/run_v8.sh start     # 바로 시작
#
# 준비(공통, GPU 불필요): KCI 판정 결과 → 통과분 SFT 기록 → 원문 묶음 1/40 을 평가용으로 분리
# A줄(GPU 0)  1. 긴 원문 평가: 9B+안내 · 9B sft_v2(연구노트만 학습) — 짧은 원문에서 배운 것이 긴 원문으로 옮겨 가는가
#             2. 9B sft_v3: 연구노트 35,274 + KCI 통과분 혼합 1에폭 → 긴 원문 · 연구노트 양쪽 평가
# B줄(GPU 1)  1. 긴 원문 평가: 27B+안내 · 27B sft27_v1
#             2. 판정 인코더(학생) v4: LLM 이 확신한 생성 응답 문장으로 재학습 → 평가셋 3종 재채점(퇴행 확인) → 평가 응답 재판정
# 한 단계가 실패해도 다음으로 간다. 결과가 있으면 건너뛴다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B

launch() {
  python -c "from huggingface_hub import snapshot_download as s; s('$T'); s('$S')" > $L/dl_v8.txt 2>&1 && echo MODELS_READY || echo MODELS_FAIL
  echo "== 준비 $(date)"
  [ -s $J/kci_gen_0.jsonl ] && [ -s $J/kci_gen_1.jsonl ] || { echo "PREP_FAIL — KCI 판정 결과가 없다"; exit 1; }
  cat $J/kci_gen_0.jsonl $J/kci_gen_1.jsonl > $K/judged_all.jsonl
  python scripts/judge_all.py report $K/judged_all.jsonl
  python scripts/judge_all.py export $K/sft_all.jsonl $K/judged_all.jsonl || exit 1
  python scripts/eval_sft.py split --gen $K/judged_all.jsonl --sft $K/sft_all.jsonl --mod 40 \
    --train-out $K/sft_train.jsonl --heldout-out $K/heldout.jsonl || exit 1
  cat $I/sft3_train.jsonl $K/sft_train.jsonl | shuf --random-source=<(yes) > $I/sft_mix_train.jsonl; wc -l $I/sft_mix_train.jsonl
  LANE=A CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v8.sh > $R/logs_v8_A.txt 2>&1 < /dev/null &
  LANE=B CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_v8.sh > $R/logs_v8_B.txt 2>&1 < /dev/null &
  echo STARTED
}

if [ "$1" = chain ]; then
  ( for i in $(seq 1 2880); do
      grep -q V7_DONE_0 $R/logs_v7_0.txt 2>/dev/null && grep -q V7_DONE_1 $R/logs_v7_1.txt 2>/dev/null && break; sleep 30
    done; launch ) > $R/logs_v8.txt 2>&1 < /dev/null &
  echo CHAINED; exit 0
fi
if [ "$1" = start ]; then launch > $R/logs_v8.txt 2>&1; cat $R/logs_v8.txt; exit 0; fi

ev() {  # 접두 이름 평가지시 모델 [어댑터] [안내]
  local out=$I/$1_$2.jsonl
  [ -s $out ] && { echo "-- $1_$2 (있음, 건너뜀)"; return; }
  python scripts/eval_sft.py run --model $4 ${5:+--adapter $5} ${6:+--style-hint} $( [ "$1" = evalk ] && echo --max-new-tokens 900 ) --heldout $3 --out $out > $L/$1_$2.txt 2>&1
  echo "-- $1_$2 (exit $?)"; grep -E "충실 생성률|판정기|어댑터|Error|거부" $L/$1_$2.txt | tail -8
}
cmp() { [ -s $I/$2.jsonl ] && [ -s $I/$3.jsonl ] && { echo "[$1]"; python scripts/eval_sft.py compare $I/$2.jsonl $I/$3.jsonl; }; }

if [ "$LANE" = A ]; then
  echo "== A1. 긴 원문 평가(9B) $(date)"
  ev evalk base_hint $K/heldout.jsonl $S "" hint
  ev evalk sft_v2 $K/heldout.jsonl $S $CK/sft_v2
  cmp "긴 원문: 9B+안내 → sft_v2(연구노트만 학습)" evalk_base_hint evalk_sft_v2

  echo "== A2. 9B sft_v3 (연구노트 + KCI 혼합 1에폭) $(date)"
  if [ ! -s $CK/sft_v3/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft_mix_train.jsonl --output $CK/sft_v3 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v3.txt 2>&1
    echo "SFT_V3_EXIT $?"; grep -E "LoRA 모듈|train_loss|saved|Error|OutOfMemory" $L/sft_v3.txt | tail -3
  fi
  if [ -s $CK/sft_v3/adapter_model.safetensors ]; then
    ev evalk sft_v3 $K/heldout.jsonl $S $CK/sft_v3
    ev eval2 sft_v3 $I/heldout2.jsonl $S $CK/sft_v3
    cmp "긴 원문: 9B+안내 → sft_v3(혼합)" evalk_base_hint evalk_sft_v3
    cmp "긴 원문: sft_v2 → sft_v3" evalk_sft_v2 evalk_sft_v3
    cmp "연구노트: sft_v2 → sft_v3" eval2_sft_v2 eval2_sft_v3
  fi
  python scripts/judge_all.py --out-dir $J $(ls $I/evalk_base_hint.jsonl $I/evalk_sft_v2.jsonl $I/evalk_sft_v3.jsonl $I/eval2_sft_v3.jsonl 2>/dev/null) > $L/judge_v8a.txt 2>&1
  python scripts/judge_all.py report $J/evalk_*.jsonl $J/eval2_sft_v3.jsonl
  echo "V8A_DONE $(date)"
fi

if [ "$LANE" = B ]; then
  echo "== B1. 긴 원문 평가(27B) $(date)"
  ev evalk base27_hint $K/heldout.jsonl $T "" hint
  ev evalk sft27_v1 $K/heldout.jsonl $T $CK/sft27_v1
  cmp "긴 원문: 27B+안내 → 27B LoRA(연구노트만 학습)" evalk_base27_hint evalk_sft27_v1

  echo "== B2. 판정 인코더 v4 $(date)"
  python scripts/judge_all.py distill $D/distill/train_v4.jsonl $D/distill/train_v3.jsonl $I/heldout2.jsonl $J/sft2.jsonl.screened.jsonl $J/gen.jsonl
  if [ ! -s $CK/nli_distill_v4/config.json ]; then
    ( deactivate   # 인코더 학습·평가는 시스템 파이썬(transformers 4.x) — v3 와 같은 조건
      python -m compliance_gateway.train.nli_finetune --data $D/distill/train_v4.jsonl --labels 3 --base deberta-large \
        --split train --eval-split dev --output $CK/nli_distill_v4 > $L/ft_v4.txt 2>&1
      echo "FT_V4_EXIT $?"; rm -rf $CK/nli_distill_v4/checkpoint-*; grep -E "eval:|saved" $L/ft_v4.txt | tail -2
      for set in kr_psg_eval kr_hard_eval kr_abs_eval; do
        python -m compliance_gateway.eval.kr_nli $D/$set.jsonl --nli $CK/nli_distill_v4 --dump $R/scores/${set}_distill_v4.jsonl > $L/${set}_distill_v4.txt 2>&1
        echo "-- $set v4"; grep -E "^AUC|운영점" $L/${set}_distill_v4.txt
        echo "-- $set v3"; grep -E "^AUC|운영점" $L/${set}_distill_v3.txt
      done )
  fi
  if [ -s $CK/nli_distill_v4/config.json ]; then
    python scripts/judge_all.py --student $CK/nli_distill_v4 --out-dir data/judged_v4 $I/eval2_base_hint.jsonl $I/eval2_sft_v2.jsonl $I/eval2_base27_hint.jsonl $I/eval2_sft27_v1.jsonl > $L/judge_v8b.txt 2>&1
    echo "[학생 v4 로 재판정]"; python scripts/judge_all.py report data/judged_v4/*.jsonl
  fi
  python scripts/judge_all.py --out-dir $J $(ls $I/evalk_base27_hint.jsonl $I/evalk_sft27_v1.jsonl 2>/dev/null) > $L/judge_v8b2.txt 2>&1
  python scripts/judge_all.py report $J/evalk_*27*.jsonl
  echo "V8B_DONE $(date)"
fi
