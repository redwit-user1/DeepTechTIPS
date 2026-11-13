#!/usr/bin/env bash
# 2026-11-12 v6 대기열 — GPU 기한(11월 말) 안에 'GPU 로만 만들 수 있고 뒤에 남는 것'부터 (연구노트 §40)
#
#   GPU 2장 세션에서:  bash repo/scripts/run_v6.sh start      # 모델을 받고 두 줄을 백그라운드로 띄운다
#   한 줄만:           LANE=A CUDA_VISIBLE_DEVICES=0 bash repo/scripts/run_v6.sh > logs_v6_A.txt 2>&1
#
# 한 단계가 실패해도 다음으로 간다(밤새 GPU 가 놀지 않게). 각 단계는 결과가 있으면 건너뛴다 → 중단 후 그대로 다시 실행.
#   A줄(GPU 0)  1. 전수 판정: 2차 응답 46,093 · 1차 4,681 · v5 평가 응답 — 문장별 인코더·LLM 원점수 + 질의응답 질문 적합성
#               2. 학습셋 정리: 질문에 맞지 않는 질의응답 제외 → sft3_train
#               3. 9B v2: sft3_train 1에폭 — 데이터 규모(4천 → 3만5천) · 질문 적합성 필터 효과 → 평가 · 전수 판정
#   B줄(GPU 1)  1. 27B 기준선: Qwen3.8-27B + 안내 한 줄, 평가 지시 2,294건
#               2. 27B LoRA 20스텝 시험(속도 · 메모리)
#               3. sft3_train 을 기다렸다가 27B LoRA 1에폭(README 의 정렬 대상) → 평가 · 전수 판정
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B

if [ "$1" = start ]; then
  python -c "from huggingface_hub import snapshot_download as s; s('$T'); s('$S')" > $L/dl_v6.txt 2>&1 && echo MODELS_READY || { echo MODELS_FAIL; exit 1; }
  LANE=A CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v6.sh > $R/logs_v6_A.txt 2>&1 < /dev/null &
  LANE=B CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_v6.sh > $R/logs_v6_B.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

ev() {  # 이름 모델 [어댑터] [안내]
  [ -s $I/eval2_$1.jsonl ] && { echo "-- $1 (있음, 건너뜀)"; return; }
  python scripts/eval_sft.py run --model $2 ${3:+--adapter $3} ${4:+--style-hint} --heldout $I/heldout2.jsonl --out $I/eval2_$1.jsonl > $L/eval2_$1.txt 2>&1
  echo "-- $1 (exit $?)"; grep -E "충실 생성률|판정기|어댑터|Error|거부" $L/eval2_$1.txt | tail -8
}
judge() {  # 로그이름 파일...
  local n=$1; shift
  python scripts/judge_all.py --out-dir $J "$@" > $L/judge_$n.txt 2>&1
  echo "JUDGE_EXIT($n) $?"; tail -2 $L/judge_$n.txt
}

if [ "$LANE" = A ]; then
  echo "== A1. 전수 판정 — 2차 응답 먼저(B줄이 기다리는 학습셋 정리에 필요) $(date)"
  judge v6a $I/sft2.jsonl.screened.jsonl

  echo "== A2. 학습셋 정리 $(date)"
  python scripts/judge_all.py filter $J/sft2.jsonl.screened.jsonl $I/sft2_train.jsonl $I/sft3_train.tmp \
    || { echo "FILTER_FAIL — 필터 없이 진행"; cp $I/sft2_train.jsonl $I/sft3_train.tmp; }
  mv $I/sft3_train.tmp $I/sft3_train.jsonl; wc -l $I/sft3_train.jsonl      # B줄이 이 파일을 기다린다 — 다 쓴 뒤 이름을 바꾼다

  echo "== A1b. 전수 판정 — 평가 응답 · 1차 $(date)"
  judge v6a1 $I/eval2_base.jsonl $I/eval2_base_hint.jsonl $I/eval2_sft_v1_e1.jsonl $I/eval2_sft_v1.jsonl $I/gen.jsonl
  python scripts/judge_all.py report $J/*.jsonl

  echo "== A3. 9B v2 (sft3_train 1에폭) $(date)"
  if [ ! -s $CK/sft_v2/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft3_train.jsonl --output $CK/sft_v2 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v2.txt 2>&1
    echo "SFT_V2_EXIT $?"; grep -E "LoRA 모듈|train_loss|saved|Error" $L/sft_v2.txt | tail -3
  fi
  if [ -s $CK/sft_v2/adapter_model.safetensors ]; then
    ev sft_v2 $S $CK/sft_v2
    echo "[9B sft_v1 → sft_v2]";    python scripts/eval_sft.py compare $I/eval2_sft_v1.jsonl $I/eval2_sft_v2.jsonl
    echo "[9B 기반+안내 → sft_v2]"; python scripts/eval_sft.py compare $I/eval2_base_hint.jsonl $I/eval2_sft_v2.jsonl
    judge v6a2 $I/eval2_sft_v2.jsonl
  fi
  python scripts/judge_all.py report $J/eval2_*.jsonl
  echo "V6A_DONE $(date)"
fi

if [ "$LANE" = B ]; then
  echo "== B1. 27B 기준선(기반 + 안내 한 줄) $(date)"
  ev base27_hint $T "" hint

  echo "== B2. 27B LoRA 20스텝 시험 $(date)"
  python -m compliance_gateway.train.sft --base qwen --dataset $I/sft2_train.jsonl --output $CK/sft27_smoke --max-steps 20 > $L/sft27_smoke.txt 2>&1
  echo "SMOKE_EXIT $?"; grep -E "LoRA 모듈|train_runtime|Error|OutOfMemory" $L/sft27_smoke.txt | tail -3

  echo "== B3. 27B LoRA 1에폭 $(date)"
  for i in $(seq 1 480); do [ -s $I/sft3_train.jsonl ] && break; sleep 60; done      # A줄의 학습셋 정리를 최대 8시간 기다린다
  D=$I/sft3_train.jsonl; [ -s $D ] || { echo "sft3_train 없음 — sft2_train 으로 학습"; D=$I/sft2_train.jsonl; }
  if [ ! -s $CK/sft27_v1/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen --dataset $D --output $CK/sft27_v1 --epochs 1 > $L/sft27_v1.txt 2>&1
    echo "SFT27_EXIT $?"; grep -E "LoRA 모듈|train_loss|saved|Error|OutOfMemory" $L/sft27_v1.txt | tail -4
  fi
  if [ -s $CK/sft27_v1/adapter_model.safetensors ]; then
    ev sft27_v1 $T $CK/sft27_v1
    echo "[27B 기반+안내 → 27B LoRA]"; python scripts/eval_sft.py compare $I/eval2_base27_hint.jsonl $I/eval2_sft27_v1.jsonl
  fi
  judge v6b $(ls $I/eval2_base27_hint.jsonl $I/eval2_sft27_v1.jsonl 2>/dev/null)
  python scripts/judge_all.py report $J/eval2_*27*.jsonl
  echo "V6B_DONE $(date)"
fi
