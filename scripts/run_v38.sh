#!/usr/bin/env bash
# 2026-11-21 v38 — 27B 최종 모델의 반복을 GPU 0 에서 v35 K줄과 함께 (연구노트 §60)
#
#   bash repo/scripts/run_v38.sh start      # 바로 시작
#
# 9B 학습 · 평가는 GPU 메모리 20 ~ 30GB, 사용률 35% 안팎만 쓴다. 27B 학습(58GB)을 같은 GPU 에 얹어도 143GB 안에 들어간다.
# 긴 표집분의 Midm 판정(최대 136GB)처럼 큰 작업은 이미 끝났다 — 남은 K줄 작업은 9B 학습 · 평가 응답 판정(각 30GB 안팎)이다.
#   sft27_v5_s1 (sft8, 씨앗 1) → dpo27_v7_s1 (+ dpo7, 씨앗 1) → 평가 → 재심사
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0



if [ "$1" = start ]; then
  CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v38.sh run > $R/logs_v38.txt 2>&1 < /dev/null &
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

echo "== 3. 27B 반복: sft27_v5_s1 (sft8, 씨앗 1) → dpo27_v7_s1 $(date)"
if [ ! -s $CK/sft27_v5_s1/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen --dataset $I/sft8_train.jsonl --output $CK/sft27_v5_s1 --epochs 1 --seed 1 > $L/sft27_v5_s1.txt 2>&1
  echo "SFT27_EXIT $?"
fi
if [ -s $CK/sft27_v5_s1/adapter_model.safetensors ] && [ ! -s $CK/dpo27_v7_s1/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/sft27_v5_s1 --dataset $I/dpo7.jsonl --output $CK/dpo27_v7_s1 --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 --seed 1 > $L/dpo27_v7_s1.txt 2>&1
  echo "DPO27_EXIT $?"
fi
[ -s $CK/dpo27_v7_s1/adapter_model.safetensors ] && { gen dpo27_v7_s1 $T $CK/dpo27_v7_s1; rejudge dpo27_v7_s1; }
echo "V38_DONE $(date)"
