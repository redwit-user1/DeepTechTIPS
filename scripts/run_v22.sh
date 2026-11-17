#!/usr/bin/env bash
# 2026-11-17 v22 — SFT 단계의 씨앗 반복 (연구노트 §53)
#
#   bash repo/scripts/run_v22.sh start      # v21 이 끝나면 GPU 0 에서 시작
#
# v21: DPO 단계만 씨앗을 바꿔도 세 과제 통과율이 1%p 안팎 흔들린다. SFT 단계의 흔들림은 아직 모른다 —
# sft_v4 대 sft_v5(−1.5%p) 같은 SFT 끼리의 비교가 흔들림 안인지 보려면 SFT 도 다시 해 봐야 한다.
#   sft_v5_s1: sft_v5 와 같은 데이터(sft5, 71,176건) · 씨앗만 1 → 평가 → 재심사
#   dpo_v6_s1b: 그 위에 조건 추출을 뺀 선호쌍(씨앗 42) → 평가 → 재심사   (SFT + DPO 를 통째로 다시 한 것)
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  ( while ! grep -q V21_DONE $R/logs_v21.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_v22.sh run ) > $R/logs_v22.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

model_of() { case $1 in phi4) echo $PHI;; kanana) echo $KAN;; midm) echo $MID;; qwen) echo $T;; esac; }
jd() {
  local tag=$1; shift
  for f in "$@"; do
    [ -s $f ] || continue
    [ -f data/judged_$tag/$(basename $f).done ] && continue
    python scripts/judge_all.py --llm $(model_of $tag) --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}
kv() {
  local tag=$1 mark=$2; shift 2
  [ -f data/kv_$tag/.$mark.done ] && return
  python scripts/judge_all.py --kv --batch 16 --llm $(model_of $tag) --out-dir data/kv_$tag "$@" > $L/kv_${tag}_$mark.txt 2>&1
  local rc=$?; echo "KV_EXIT($tag $mark) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/kv_$tag/.$mark.done
}
gen() {
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1"; grep -E "^전체|Error|거부" $L/${pre}_$1.txt | tail -1 | cut -c1-120
  done
}
rejudge() {
  for n in "$@"; do
    for tag in phi4 kanana midm; do jd $tag $I/eval2_$n.jsonl $I/evalk_$n.jsonl; done
    [ -s data/judged_phi4/eval2_$n.jsonl ] && for tag in phi4 kanana midm qwen; do
      kv $tag $n data/judged_phi4/eval2_$n.jsonl data/judged_phi4/evalk_$n.jsonl; done
  done
}

echo "== 1. 9B sft_v5_s1 (sft5, 씨앗 1) $(date)"
if [ ! -s $CK/sft_v5_s1/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft5_train.jsonl --output $CK/sft_v5_s1 --epochs 1 --batch 2 --grad-accum 8 --seed 1 > $L/sft_v5_s1.txt 2>&1
  echo "SFT_EXIT $?"; grep -E "saved|Error|OutOfMemory" $L/sft_v5_s1.txt | tail -2 | cut -c1-200
fi
[ -s $CK/sft_v5_s1/adapter_model.safetensors ] && { gen sft_v5_s1 $S $CK/sft_v5_s1; rejudge sft_v5_s1; }

echo "== 2. 9B dpo_v6_s1b (sft_v5_s1 + 조건 추출을 뺀 선호쌍) $(date)"
if [ -s $CK/sft_v5_s1/adapter_model.safetensors ] && [ ! -s $CK/dpo_v6_s1b/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v5_s1 --dataset $I/dpo4nc.jsonl --output $CK/dpo_v6_s1b \
    --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v6_s1b.txt 2>&1
  echo "DPO_EXIT $?"
fi
[ -s $CK/dpo_v6_s1b/adapter_model.safetensors ] && { gen dpo_v6_s1b $S $CK/dpo_v6_s1b; rejudge dpo_v6_s1b; }
echo "V22_DONE $(date)"
