#!/usr/bin/env bash
# 2026-11-18 v29 — 연구노트 보고서 문단의 학습 예를 빼면 (연구노트 §57)
#
#   bash repo/scripts/run_v29.sh start      # v27 이 끝나면 GPU 0 에서 시작
#
# §56 · §57: 학습 뒤 연구노트 보고서 문단은 두 독립 판정 모두에서 내려간다(EEVE 57 → 38%, Phi-4 79 → 72%). 문장이 1.8 → 2.3개로 늘고
# EEVE 문장 지지율이 69 → 63% 로 내려간다. 네 모델 합의로 걸러도(sft_v6) 그대로다 — 걸러서 풀리는 문제가 아니다.
# 교사가 쓴 '연구노트 → 보고서 문단' 예 자체가 원인인지 본다: 그 예만 빼고 학습(sft7 = sft6 − 연구노트 보고서 문단).
# 보고서 문단이 기반모델 수준으로 돌아오고 나머지가 유지되면, 짧은 조각 원문의 보고서 문단은 학습시키지 않는 것이 낫다는 뜻이다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

if [ "$1" = start ]; then
  ( while ! grep -q V27_DONE $R/logs_v27.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_v29.sh run ) > $R/logs_v29.txt 2>&1 < /dev/null &
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

echo "== 1. sft7 = sft6 − 연구노트 보고서 문단 $(date)"
[ -s $I/sft7_train.jsonl ] || python -c "
import json
keep = drop = 0
with open('$I/sft7_train.jsonl', 'w', encoding='utf-8') as f:
    for l in open('$I/sft6_train.jsonl', encoding='utf-8'):
        r = json.loads(l)
        if r['source']['task'] == 'report' and '[연구노트]' in r['messages'][0]['content']:
            drop += 1
            continue
        f.write(l); keep += 1
print('sft7', keep, '· 뺀 연구노트 보고서 문단', drop)"
wc -l $I/sft7_train.jsonl

echo "== 2. 9B sft_v7 $(date)"
if [ -s $I/sft7_train.jsonl ] && [ ! -s $CK/sft_v7/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft7_train.jsonl --output $CK/sft_v7 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v7.txt 2>&1
  echo "SFT_EXIT $?"
fi
[ -s $CK/sft_v7/adapter_model.safetensors ] && { gen sft_v7 $S $CK/sft_v7; rejudge sft_v7; }
echo "V29_DONE $(date)"
