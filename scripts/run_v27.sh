#!/usr/bin/env bash
# 2026-11-18 v27 — 가장 엄격한 판정(EEVE)까지 넣은 네 모델 합의로 학습 데이터를 다시 거른다 (연구노트 §55)
#
#   bash repo/scripts/run_v27.sh start      # v25 가 끝나면 GPU 0 에서 시작
#
# §55: 9B 의 연구노트 단락 개선은 Phi-4 로는 +, EEVE 로는 −(보고서 문단 56.6 → 41.3%). 세 모델(Qwen · Kanana · Midm) 합의를 통과한
# 학습 응답에 EEVE 가 탈락시키는 '풀어쓴 문장'이 남아 있다는 뜻이다. EEVE 를 합의에 넣어 걸러 내면 엄격한 판정에서도 오르는지 본다.
# 평가는 선별에 쓰지 않은 Phi-4(독립)와, 이제 선별에 쓴 EEVE 를 함께 본다.
#   1. 교사 생성분을 EEVE 로 판정 → 2. 네 모델 합의 통과분(sft6; 조건 추출은 전용 판정 그대로) → 3. 9B sft_v6 → 평가 → 재심사
#   4. dpo_v8 = sft_v6 + dpo7(조건 추출 뺀 쌍 + 질의응답 쌍) → 평가 → 재심사
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0
TEACHER="sft2.jsonl.screened.jsonl gen.jsonl gen_0.jsonl gen_1.jsonl"

if [ "$1" = start ]; then
  ( while ! grep -q V25_DONE $R/logs_v25.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_v27.sh run ) > $R/logs_v27.txt 2>&1 < /dev/null &
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

echo "== 1. 교사 생성분을 EEVE 로 판정 $(date)"
jd eeve $(for f in $TEACHER; do echo $J/$f; done)

echo "== 2. 네 모델 합의 통과분 $(date)"
[ -s $I/sft6_train.jsonl ] || { python scripts/judge_all.py export-multi $I/sft6.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores $(echo $TEACHER | tr ' ' ,) \
  qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm eeve=data/judged_eeve kv=data/kv_phi4,data/kv_kanana,data/kv_midm,data/kv_qwen && mv $I/sft6.tmp $I/sft6_train.jsonl; }
wc -l $I/sft6_train.jsonl

echo "== 3. 9B sft_v6 (네 모델 합의 통과분, 1에폭) $(date)"
if [ -s $I/sft6_train.jsonl ] && [ ! -s $CK/sft_v6/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft6_train.jsonl --output $CK/sft_v6 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v6.txt 2>&1
  echo "SFT_EXIT $?"
fi
[ -s $CK/sft_v6/adapter_model.safetensors ] && { gen sft_v6 $S $CK/sft_v6; rejudge sft_v6; }

echo "== 4. 9B dpo_v8 (sft_v6 + dpo7) $(date)"
if [ -s $CK/sft_v6/adapter_model.safetensors ] && [ ! -s $CK/dpo_v8/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v6 --dataset $I/dpo7.jsonl --output $CK/dpo_v8 \
    --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v8.txt 2>&1
  echo "DPO_EXIT $?"
fi
[ -s $CK/dpo_v8/adapter_model.safetensors ] && { gen dpo_v8 $S $CK/dpo_v8; rejudge dpo_v8; }
echo "V27_DONE $(date)"
