#!/usr/bin/env bash
# 2026-11-16 v20 — 27B 를 9B 최종 후보와 같은 데이터로 (연구노트 §51)
#
#   bash repo/scripts/run_v20.sh start      # v18 B줄이 끝나면 GPU 1 에서 시작
#
# §49: 세 과제(요약 · 보고서 문단 · 질의응답)에서는 학습 뒤의 27B 가 9B 보다 낫지 않다. §51: 조건 추출은 27B 가 낫다
# (연구노트 통과율 80% vs 75%). 27B 는 아직 조건 추출 예가 5,377건뿐인 예전 합의 통과분으로 학습한 것(sft27_v2)이다.
#   27B sft27_v3: 조건 추출을 되살린 합의 통과분(sft5, 71,176건) 1에폭 → 평가 → 재심사
#   27B dpo27_v5: 그 위에 선호쌍 DPO — v19 의 질의응답 쌍이 있으면 dpo7(조건 추출 제외 + 질의응답), 없으면 dpo4nc → 평가 → 재심사
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci
. $R/venv/bin/activate
T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  ( while ! grep -q V18B_DONE $R/logs_v18_B.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v20.sh run ) > $R/logs_v20.txt 2>&1 < /dev/null &
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
    echo "-- ${pre}_$1"; grep -E "^전체|어댑터|Error|거부" $L/${pre}_$1.txt | tail -3 | cut -c1-200
  done
}
rejudge() {
  for n in "$@"; do
    for tag in phi4 kanana midm; do jd $tag $I/eval2_$n.jsonl $I/evalk_$n.jsonl; done
    [ -s data/judged_phi4/eval2_$n.jsonl ] && for tag in phi4 kanana midm qwen; do
      kv $tag $n data/judged_phi4/eval2_$n.jsonl data/judged_phi4/evalk_$n.jsonl; done
  done
}

echo "== 1. 27B sft27_v3 (조건 추출을 되살린 합의 통과분 1에폭) $(date)"
if [ ! -s $CK/sft27_v3/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen --dataset $I/sft5_train.jsonl --output $CK/sft27_v3 --epochs 1 > $L/sft27_v3.txt 2>&1
  echo "SFT27_EXIT $?"; grep -E "saved|Error|OutOfMemory" $L/sft27_v3.txt | tail -2 | cut -c1-200
fi
[ -s $CK/sft27_v3/adapter_model.safetensors ] && { gen sft27_v3 $T $CK/sft27_v3; rejudge sft27_v3; }

echo "== 2. 27B dpo27_v5 (sft27_v3 위에 선호쌍) $(date)"
D=$I/dpo4nc.jsonl; [ -s $I/dpo7.jsonl ] && D=$I/dpo7.jsonl
echo "선호쌍 $D $(wc -l < $D)"
if [ -s $CK/sft27_v3/adapter_model.safetensors ] && [ ! -s $CK/dpo27_v5/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/sft27_v3 --dataset $D --output $CK/dpo27_v5 \
    --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 > $L/dpo27_v5.txt 2>&1
  echo "DPO27_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo27_v5.txt | tail -3 | cut -c1-200
fi
[ -s $CK/dpo27_v5/adapter_model.safetensors ] && { gen dpo27_v5 $T $CK/dpo27_v5; rejudge dpo27_v5; }
echo "V20_DONE $(date)"
