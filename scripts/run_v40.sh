#!/usr/bin/env bash
# 2026-11-21 v40 — 두 바퀴째 전체 모델(dpo_v12)의 반복 (연구노트 §60)
#
#   bash repo/scripts/run_v40.sh start      # v38 이 끝나면 GPU 0 에서 시작
#
# dpo_v12(sft6 + 두 바퀴째 데이터 전체 + DPO)는 Phi-4 83.8 / 81.4% 로 두 평가셋 합쳐 가장 높다. 학습 한 번이라 1차년도 대표 모델로 올리지 않았다.
#   dpo_v12_s1: 같은 sft_v9 위에 DPO 씨앗 1        sft_v9_s1 → dpo_v12_s1b: SFT 부터 씨앗 1
# 재현되면 9B 대표 모델을 dpo_v12 로 바꿀지 판단한다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0




if [ "$1" = start ]; then
  ( while ! grep -q V38_DONE $R/logs_v38.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_v40.sh run ) > $R/logs_v40.txt 2>&1 < /dev/null &
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

dpo() {  # 이름 SFT어댑터 씨앗
  if [ -s $CK/$2/adapter_model.safetensors ] && [ ! -s $CK/$1/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/$2 --dataset $I/dpo7.jsonl --output $CK/$1 --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 --seed $3 > $L/$1.txt 2>&1
    echo "DPO_EXIT($1) $?"
  fi
  [ -s $CK/$1/adapter_model.safetensors ] && { gen $1 $S $CK/$1; rejudge $1; }
}
echo "== dpo_v12_s1 $(date)"
dpo dpo_v12_s1 sft_v9 1
echo "== sft_v9_s1 (sft9, 씨앗 1) $(date)"
if [ ! -s $CK/sft_v9_s1/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft9_train.jsonl --output $CK/sft_v9_s1 --epochs 1 --batch 2 --grad-accum 8 --seed 1 > $L/sft_v9_s1.txt 2>&1
  echo "SFT_EXIT $?"
fi
[ -s $CK/sft_v9_s1/adapter_model.safetensors ] && { gen sft_v9_s1 $S $CK/sft_v9_s1; rejudge sft_v9_s1; }
echo "== dpo_v12_s1b $(date)"
dpo dpo_v12_s1b sft_v9_s1 42
echo "V40_DONE $(date)"
