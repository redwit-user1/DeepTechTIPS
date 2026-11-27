#!/usr/bin/env bash
# 2026-11-27 v49 — v48 반복 (연구노트 §65)
#
#   bash repo/scripts/run_v49.sh start      # GPU 0: WF-A 두 번째 실행, GPU 1: 씨앗을 바꾼 재학습
#
# Codex · Grok 검토: 제안서 예 학습(dpo_v13)은 한 번 학습 · 한 번 실행이라 시범으로만 볼 수 있다 → 반복을 붙인다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data; I=data/instr; K=data/kci
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  ( LANE=A CUDA_VISIBLE_DEVICES=0 bash scripts/run_v49.sh run ) > $R/logs_v49_A.txt 2>&1 < /dev/null &
  ( LANE=B CUDA_VISIBLE_DEVICES=1 bash scripts/run_v49.sh run ) > $R/logs_v49_B.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 40 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --topics scripts/wf_a_topics_ext.json \
    --search-cache $D/wf_a_search_40.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?) $(date +%m-%d_%H:%M)"; tail -1 $L/wf_a_$n.txt | cut -c1-200
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1 | cut -c1-200
}
gen() {
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1 $(date +%m-%d_%H:%M)"
  done
}
jd() {
  local tag=$1 llm=$2; shift 2
  for f in "$@"; do
    [ -f data/judged_$tag/$(basename $f).done ] && continue
    python scripts/judge_all.py --llm $llm --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}

if [ "$LANE" = A ]; then
  echo "== A: dpo_v13 WF-A 두 번째 실행 $(date)"
  wf w40_9b_full_v13_b $S --adapter $CK/dpo_v13
  wf w40_9b_regen_v13_b $S --adapter $CK/dpo_v13 --adapter-on regen
  wf w40_9b_full_v11_b $S --adapter $CK/dpo_v11
  echo "V49A_DONE $(date)"
fi
if [ "$LANE" = B ]; then
  echo "== B: 씨앗 1 로 sft_v10_s1 → dpo_v13_s1 $(date)"
  if [ ! -s $CK/sft_v10_s1/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft10_train.jsonl --output $CK/sft_v10_s1 --epochs 1 --batch 2 --grad-accum 8 --seed 1 > $L/sft_v10_s1.txt 2>&1
    echo "SFT_EXIT $? $(date +%m-%d_%H:%M)"
  fi
  [ -s $CK/sft_v10_s1/adapter_model.safetensors ] || { echo "V49B_FAIL sft_v10_s1 없음"; exit 1; }
  [ -s $CK/dpo_v13_s1/adapter_model.safetensors ] || python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v10_s1 --dataset $I/dpo7.jsonl --output $CK/dpo_v13_s1 --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 --seed 1 > $L/dpo_v13_s1.txt 2>&1
  [ -s $CK/dpo_v13_s1/adapter_model.safetensors ] || { echo "V49B_FAIL dpo_v13_s1 없음"; exit 1; }
  wf w40_9b_full_v13s1 $S --adapter $CK/dpo_v13_s1
  gen dpo_v13_s1 $S $CK/dpo_v13_s1
  jd phi4 $PHI $I/eval2_dpo_v13_s1.jsonl $I/evalk_dpo_v13_s1.jsonl
  echo "V49B_DONE $(date)"
fi
