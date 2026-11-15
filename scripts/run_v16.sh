#!/usr/bin/env bash
# 2026-11-15 v16 — 합의 필터 SFT + 합의 선호쌍 DPO 를 합친 1차년도 모델 (연구노트 §46)
#
#   bash repo/scripts/run_v16.sh start
#
# v14: 세 판정 모델 합의 통과분(55,075건)으로 학습한 9B sft_v4 가 Qwen 통과분(86,905건)으로 학습한 sft_v3 보다
# 선택에 쓰지 않은 Phi-4 기준 +4.4%p(연구노트) · +5.1%p(긴 원문) 높다. v13: 합의 선호쌍 DPO 는 +1.4 ~ +2.0%p.
#   A줄(GPU 0)  27B sft27_v2: 합의 통과분으로 1에폭 → dpo27_v3: 그 위에 합의 선호쌍 DPO → 평가 → 세 모델 재심사
#   B줄(GPU 1)  (v15 가 끝난 뒤) 9B dpo_v4: sft_v4 위에 합의 선호쌍 DPO → 평가 → 세 모델 재심사
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  LANE=A CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v16.sh > $R/logs_v16_A.txt 2>&1 < /dev/null &
  ( while ! grep -q V15_DONE $R/logs_v15.txt 2>/dev/null; do sleep 30; done
    LANE=B CUDA_VISIBLE_DEVICES=1 bash scripts/run_v16.sh ) > $R/logs_v16_B.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

jd() {
  local tag=$1 m=$2; shift 2
  for f in "$@"; do
    [ -s $f ] || continue
    [ -f data/judged_$tag/$(basename $f).done ] && { echo "-- $tag $(basename $f) (있음, 건너뜀)"; continue; }
    python scripts/judge_all.py --llm $m --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}
gen() {  # 이름 모델 어댑터 — 평가 응답 2종 생성(판정은 Qwen 캐스케이드가 함께 돈다)
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1"; grep -E "^전체|어댑터|Error|거부" $L/${pre}_$1.txt | tail -3
  done
}
judge3() { jd phi4 $PHI "$@"; jd kanana $KAN "$@"; jd midm $MID "$@"; }

if [ "$LANE" = A ]; then
  echo "== A1. 27B sft27_v2 (합의 통과분 1에폭) $(date)"
  if [ ! -s $CK/sft27_v2/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen --dataset $I/sft4_train.jsonl --output $CK/sft27_v2 --epochs 1 > $L/sft27_v2.txt 2>&1
    echo "SFT27_EXIT $?"; grep -E "LoRA 모듈|saved|Error|OutOfMemory" $L/sft27_v2.txt | tail -3 | cut -c1-200
  fi
  echo "== A2. 27B dpo27_v3 (sft27_v2 위에 합의 선호쌍) $(date)"
  if [ -s $CK/sft27_v2/adapter_model.safetensors ] && [ ! -s $CK/dpo27_v3/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/sft27_v2 --dataset $I/dpo4.jsonl --output $CK/dpo27_v3 \
      --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 > $L/dpo27_v3.txt 2>&1
    echo "DPO27_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo27_v3.txt | tail -3 | cut -c1-200
  fi
  echo "== A3. 평가 $(date)"
  [ -s $CK/sft27_v2/adapter_model.safetensors ] && gen sft27_v2 $T $CK/sft27_v2
  [ -s $CK/dpo27_v3/adapter_model.safetensors ] && gen dpo27_v3 $T $CK/dpo27_v3
  echo "== A4. 재심사 $(date)"
  judge3 $I/eval2_sft27_v2.jsonl $I/evalk_sft27_v2.jsonl $I/eval2_dpo27_v3.jsonl $I/evalk_dpo27_v3.jsonl
  echo "V16A_DONE $(date)"
fi

if [ "$LANE" = B ]; then
  echo "== B1. 9B dpo_v4 (sft_v4 위에 합의 선호쌍) $(date)"
  if [ ! -s $CK/dpo_v4/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v4 --dataset $I/dpo4.jsonl --output $CK/dpo_v4 \
      --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v4.txt 2>&1
    echo "DPO_V4_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo_v4.txt | tail -3 | cut -c1-200
  fi
  echo "== B2. 평가 $(date)"
  [ -s $CK/dpo_v4/adapter_model.safetensors ] && gen dpo_v4 $S $CK/dpo_v4
  echo "== B3. 재심사 $(date)"
  judge3 $I/eval2_dpo_v4.jsonl $I/evalk_dpo_v4.jsonl
  echo "V16B_DONE $(date)"
fi
