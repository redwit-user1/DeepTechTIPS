#!/usr/bin/env bash
# 2026-11-15 v15 — 27B 에서도 같은가: 합의 선호쌍 DPO 와 다른 판정 모델 기준 SFT 효과 (연구노트 §46)
#
#   CUDA_VISIBLE_DEVICES=1 nohup bash repo/scripts/run_v15.sh > logs_v15.txt 2>&1 &
#
#   1. 27B dpo27_v2: sft27_v1 에서 이어서, 세 판정 모델 합의 선호쌍(dpo4, 9B 표본에서 고른 3,669쌍) 1에폭
#   2. 평가 응답 생성(연구노트 2,294 · 긴 원문 1,603)
#   3. Phi-4 · Kanana · Midm 재심사: 27B 기반+안내 / sft27_v1 / dpo27_v2
#      — 27B 의 SFT 효과(§40: Qwen 판정 +2.6%p)가 다른 판정 모델에서도 있는지도 함께 본다
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci
. $R/venv/bin/activate
T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4
jd() {
  local tag=$1 m=$2; shift 2
  for f in "$@"; do
    [ -s $f ] || continue
    [ -f data/judged_$tag/$(basename $f).done ] && { echo "-- $tag $(basename $f) (있음, 건너뜀)"; continue; }
    python scripts/judge_all.py --llm $m --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}
echo "== 1. 27B dpo27_v2 (합의 선호쌍) $(date)"
if [ ! -s $CK/dpo27_v2/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/sft27_v1 --dataset $I/dpo4.jsonl --output $CK/dpo27_v2 \
    --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 > $L/dpo27_v2.txt 2>&1
  echo "DPO27_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo27_v2.txt | tail -3 | cut -c1-200
fi
echo "== 2. 평가 응답 생성 $(date)"
if [ -s $CK/dpo27_v2/adapter_model.safetensors ]; then
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_dpo27_v2.jsonl ] || python scripts/eval_sft.py run --model $T --adapter $CK/dpo27_v2 $extra --heldout $ho --out $I/${pre}_dpo27_v2.jsonl > $L/${pre}_dpo27_v2.txt 2>&1
    echo "-- ${pre}_dpo27_v2"; grep -E "^전체|어댑터|Error|거부" $L/${pre}_dpo27_v2.txt | tail -3
  done
fi
echo "== 3. 다른 판정 모델로 재심사 $(date)"
FILES="$I/eval2_base27_hint.jsonl $I/eval2_sft27_v1.jsonl $I/eval2_dpo27_v2.jsonl $I/evalk_base27_hint.jsonl $I/evalk_sft27_v1.jsonl $I/evalk_dpo27_v2.jsonl"
jd phi4 $PHI $FILES
jd kanana $KAN $FILES
jd midm $MID $FILES
echo "V15_DONE $(date)"
