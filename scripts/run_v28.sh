#!/usr/bin/env bash
# 2026-11-18 v28 — 선호쌍도 EEVE 까지 넣은 네 모델 합의로 (연구노트 §56)
#
#   bash repo/scripts/run_v28.sh start      # 학생 표집분의 EEVE 판정(run_eeve_samples.sh)이 끝나면 GPU 1 에서 시작
#
# v27 은 학습 데이터(SFT)를 네 모델 합의로 걸렀다. 선호쌍(dpo7)은 아직 세 모델 합의다. 같은 표집분을 EEVE 로도 판정했으니
# chosen = 넷 모두 지지(질의응답은 넷 모두 적합), rejected = 과반 미지지(지지 1 이하) 로 다시 고른다. 조건 추출 쌍은 뺀다(§51).
#   1. 네 모델 합의 선호쌍 dpo8
#   2. dpo_v10 = sft_v5(세 모델 합의 SFT) + dpo8   — 선호쌍만 바꾼 효과 (비교: dpo_v7 = sft_v5 + dpo7)
#   3. dpo_v9  = sft_v6(네 모델 합의 SFT) + dpo8   — 둘 다 바꾼 것 (비교: v27 의 dpo_v8 = sft_v6 + dpo7)
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

if [ "$1" = start ]; then
  ( while ! grep -q EEVE_SAMPLES_DONE $R/logs_eeve_samples.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v28.sh run ) > $R/logs_v28.txt 2>&1 < /dev/null &
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

echo "== 1. 네 모델 합의 선호쌍 $(date)"
if [ ! -s $I/dpo8.jsonl ]; then
  python scripts/judge_all.py dpo-multi $I/dpo8_all.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores note_sft_v2.jsonl,kci_sft_v2.jsonl,note_qa_sft_v5.jsonl,kci_qa_sft_v5.jsonl     qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm eeve=data/judged_eeve && python -c "
import json
rows = [l for l in open('$I/dpo8_all.tmp', encoding='utf-8') if json.loads(l)['source']['task'] != 'conditions']
open('$I/dpo8.jsonl', 'w', encoding='utf-8').writelines(rows); print('dpo8 (조건 추출 제외)', len(rows))"
fi
wc -l $I/dpo8.jsonl $I/dpo7.jsonl

for spec in dpo_v10:sft_v5 dpo_v9:sft_v6; do
  n=${spec%%:*}; base=${spec#*:}
  echo "== $n ($base + dpo8) $(date)"
  for i in $(seq 1 720); do [ -s $CK/$base/adapter_model.safetensors ] && break; sleep 30; done
  if [ -s $I/dpo8.jsonl ] && [ -s $CK/$base/adapter_model.safetensors ] && [ ! -s $CK/$n/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/$base --dataset $I/dpo8.jsonl --output $CK/$n       --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/$n.txt 2>&1
    echo "DPO_EXIT($n) $?"
  fi
  [ -s $CK/$n/adapter_model.safetensors ] && { gen $n $S $CK/$n; rejudge $n; }
done
echo "V28_DONE $(date)"
