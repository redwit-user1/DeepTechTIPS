#!/usr/bin/env bash
# 2026-11-17 v21 — 씨앗만 바꾼 반복 학습으로 실행 간 흔들림을 잰다 (연구노트 §52)
#
#   bash repo/scripts/run_v21.sh start      # GPU 0
#
# 지금까지의 모델 비교는 모두 학습 한 번씩이다. dpo_v4 → dpo_v6 의 −1.1%p 같은 차이가 학습 방법의 차이인지
# 같은 학습을 다시 해도 나오는 흔들림인지 모른다. 평가셋 재표집 구간은 '평가 지시의 흔들림'만 잰다 — '학습의 흔들림'은 안 잰다.
# 같은 출발점(sft_v5) · 같은 선호쌍에서 씨앗만 바꿔 DPO 를 두 번씩 더 한다(기존 실행의 씨앗은 42).
#   dpo_v6 (조건 추출을 뺀 선호쌍)        → 씨앗 1, 2
#   dpo_v7 (+ 질의응답 선호쌍 880쌍)      → 씨앗 1, 2
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v21.sh run > $R/logs_v21.txt 2>&1 < /dev/null &
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

for spec in dpo_v6:dpo4nc dpo_v7:dpo7; do
  name=${spec%%:*}; pairs=$I/${spec#*:}.jsonl
  for seed in 1 2; do
    n=${name}_s$seed
    echo "== $n ($pairs, 씨앗 $seed) $(date)"
    if [ ! -s $CK/$n/adapter_model.safetensors ]; then
      python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v5 --dataset $pairs --output $CK/$n \
        --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 --seed $seed > $L/$n.txt 2>&1
      echo "DPO_EXIT($n) $?"
    fi
    [ -s $CK/$n/adapter_model.safetensors ] && { gen $n $S $CK/$n; rejudge $n; }
  done
done
echo "V21_DONE $(date)"
