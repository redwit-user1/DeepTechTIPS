#!/usr/bin/env bash
# 2026-11-21 v37 — 1차년도 최종 모델의 4-bit 보존율, 27B 최종 모델의 반복 (연구노트 §60)
#
#   bash repo/scripts/run_v37.sh start      # v36 이 끝나면 GPU 1 에서 시작
#
# 연차보고서의 4-bit(NF4) 보존율은 중간 모델(9B dpo_v4, 27B sft27_v1)로 잰 값이다. 최종 모델(9B dpo_v11, 27B dpo27_v7)로 다시 잰다.
# (27B 반복은 run_v38.sh 로 옮겨 GPU 0 에서 K줄과 함께 돌린다 — 9B 학습은 GPU 를 3분의 1 만 쓴다.)
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0


if [ "$1" = start ]; then
  ( while ! grep -q V36_DONE $R/logs_v36.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v37.sh run ) > $R/logs_v37.txt 2>&1 < /dev/null &
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
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra $4 --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
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

echo "== 1. 4-bit 보존율: 9B dpo_v11 $(date)"
gen dpo_v11_4bit $S $CK/dpo_v11 --load-4bit; rejudge dpo_v11_4bit
echo "== 2. 4-bit 보존율: 27B dpo27_v7 $(date)"
gen dpo27_v7_4bit $T $CK/dpo27_v7 --load-4bit; rejudge dpo27_v7_4bit
grep -ah "GiB" $L/eval2_dpo_v11_4bit.txt $L/eval2_dpo27_v7_4bit.txt | tail -4 | cut -c1-200

echo "V37_DONE $(date)"
