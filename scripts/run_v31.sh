#!/usr/bin/env bash
# 2026-11-18 v31 — 새 9B 후보(dpo_v8)의 반복 (연구노트 §57)
#
#   bash repo/scripts/run_v31.sh start      # v29 가 끝나면 GPU 0 에서 시작
#
# dpo_v8 = sft_v6(네 모델 합의 SFT) + dpo7 이 dpo_v7 보다 연구노트에서 Phi-4 +4.2 · EEVE +5.8 · Kanana +3.7%p 높게 나왔다. 학습 한 번의 결과다.
# 흔들림(표준편차 1%p 안팎, §53 · §56)보다 크지만 다시 해 봐야 안다.
#   dpo_v8_s1 · dpo_v8_s2: 같은 sft_v6 위에 DPO 씨앗만 1 · 2
#   sft_v6_s1 → dpo_v8_s1b: SFT 부터 다시(씨앗 1)
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

if [ "$1" = start ]; then
  ( while ! grep -q V29_DONE $R/logs_v29.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_v31.sh run ) > $R/logs_v31.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

model_of() { case $1 in phi4) echo $PHI;; kanana) echo $KAN;; midm) echo $MID;; qwen) echo $T;; eeve) echo $EEV;; esac; }
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
    for tag in phi4 eeve kanana midm; do jd $tag $I/eval2_$n.jsonl $I/evalk_$n.jsonl; done
    [ -s data/judged_phi4/eval2_$n.jsonl ] && for tag in phi4 kanana midm qwen; do
      kv $tag $n data/judged_phi4/eval2_$n.jsonl data/judged_phi4/evalk_$n.jsonl; done
  done
}

for seed in 1 2; do
  n=dpo_v8_s$seed
  echo "== $n $(date)"
  if [ ! -s $CK/$n/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v6 --dataset $I/dpo7.jsonl --output $CK/$n       --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 --seed $seed > $L/$n.txt 2>&1
    echo "DPO_EXIT($n) $?"
  fi
  [ -s $CK/$n/adapter_model.safetensors ] && { gen $n $S $CK/$n; rejudge $n; }
done

echo "== sft_v6_s1 (sft6, 씨앗 1) $(date)"
if [ ! -s $CK/sft_v6_s1/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft6_train.jsonl --output $CK/sft_v6_s1 --epochs 1 --batch 2 --grad-accum 8 --seed 1 > $L/sft_v6_s1.txt 2>&1
  echo "SFT_EXIT $?"
fi
[ -s $CK/sft_v6_s1/adapter_model.safetensors ] && { gen sft_v6_s1 $S $CK/sft_v6_s1; rejudge sft_v6_s1; }
echo "== dpo_v8_s1b (sft_v6_s1 + dpo7) $(date)"
if [ -s $CK/sft_v6_s1/adapter_model.safetensors ] && [ ! -s $CK/dpo_v8_s1b/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v6_s1 --dataset $I/dpo7.jsonl --output $CK/dpo_v8_s1b     --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v8_s1b.txt 2>&1
  echo "DPO_EXIT(dpo_v8_s1b) $?"
fi
[ -s $CK/dpo_v8_s1b/adapter_model.safetensors ] && { gen dpo_v8_s1b $S $CK/dpo_v8_s1b; rejudge dpo_v8_s1b; }
echo "V31_DONE $(date)"
