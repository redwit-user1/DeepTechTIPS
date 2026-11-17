#!/usr/bin/env bash
# 2026-11-17 v25 — 1차년도 9B 모델(dpo_v7)을 SFT 부터 통째로 다시 한 반복 (연구노트 §55)
#
#   bash repo/scripts/run_v25.sh start      # GPU 0
#
# v21 은 DPO 단계만, v22 · v23 은 SFT 단계만 씨앗을 바꿨다. 최종 모델의 수치에 붙일 흔들림은 SFT + DPO 를 통째로 다시 한 것이어야 한다.
#   dpo_v7_s1b = sft_v5_s1 + dpo7      dpo_v7_s2b = sft_v5_s2 + dpo7      dpo_v6_s2b = sft_v5_s2 + dpo4nc
# (dpo_v6_s1b = sft_v5_s1 + dpo4nc 는 v22 에서 했다.) 선호쌍은 같은 것을 쓴다 — 선호쌍을 다시 만드는 흔들림은 재지 않는다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v25.sh run > $R/logs_v25.txt 2>&1 < /dev/null &
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

for spec in dpo_v7_s1b:sft_v5_s1:dpo7 dpo_v7_s2b:sft_v5_s2:dpo7 dpo_v6_s2b:sft_v5_s2:dpo4nc; do
  IFS=: read n base pairs <<< "$spec"
  echo "== $n ($base + $pairs) $(date)"
  if [ -s $CK/$base/adapter_model.safetensors ] && [ ! -s $CK/$n/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/$base --dataset $I/$pairs.jsonl --output $CK/$n       --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/$n.txt 2>&1
    echo "DPO_EXIT($n) $?"
  fi
  [ -s $CK/$n/adapter_model.safetensors ] && { gen $n $S $CK/$n; rejudge $n; }
done
echo "V25_DONE $(date)"
