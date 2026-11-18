#!/usr/bin/env bash
# 2026-11-18 v30 — 27B 를 네 모델 합의 통과분으로 (연구노트 §57)
#
#   bash repo/scripts/run_v30.sh start      # v28 이 끝나면 GPU 1 에서 시작
#
# 9B: 네 모델 합의 통과분(sft6, 55,518건)이 세 모델 합의(sft5)보다 두 독립 판정 모두에서 조금 높다(연구노트 Phi-4 +1.7 · EEVE +2.5%p).
# 27B 는 결론이 판정에 가장 덜 흔들리는 모델이다(§55). 같은 데이터로 다시: sft27_v4 = sft6 1에폭 → dpo27_v6 = + dpo7 → 평가 → 재심사(판정 넷 + 전용 판정).
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

if [ "$1" = start ]; then
  ( while ! grep -q V28_DONE $R/logs_v28.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v30.sh run ) > $R/logs_v30.txt 2>&1 < /dev/null &
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

echo "== 1. 27B sft27_v4 (sft6 1에폭) $(date)"
if [ ! -s $CK/sft27_v4/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen --dataset $I/sft6_train.jsonl --output $CK/sft27_v4 --epochs 1 > $L/sft27_v4.txt 2>&1
  echo "SFT27_EXIT $?"
fi
[ -s $CK/sft27_v4/adapter_model.safetensors ] && { gen sft27_v4 $T $CK/sft27_v4; rejudge sft27_v4; }

echo "== 2. 27B dpo27_v6 (sft27_v4 + dpo7) $(date)"
if [ -s $CK/sft27_v4/adapter_model.safetensors ] && [ ! -s $CK/dpo27_v6/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/sft27_v4 --dataset $I/dpo7.jsonl --output $CK/dpo27_v6     --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 > $L/dpo27_v6.txt 2>&1
  echo "DPO27_EXIT $?"
fi
[ -s $CK/dpo27_v6/adapter_model.safetensors ] && { gen dpo27_v6 $T $CK/dpo27_v6; rejudge dpo27_v6; }
echo "V30_DONE $(date)"
