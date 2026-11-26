#!/usr/bin/env bash
# 2026-11-26 v48 — v47 다시: 제안서 예를 넣은 9B 재학습 (연구노트 §65)
#
#   bash repo/scripts/run_v48.sh start      # GPU 1
#
# v47 의 SFT 가 데이터 열이 달라 실패했다(sft8 = messages · source, 제안서 예 = + kind · topic → datasets CastError).
# DPO 가 그 사실을 모르고 SFT 어댑터 없이 기반 모델 위에서 돌아, dpo_v13 과 그 평가 둘은 무효(_invalid_v47/ 로 옮김).
# 고친 것: 제안서 예는 messages · source 만 남긴다. SFT 어댑터가 없으면 DPO 로 넘어가지 않는다.
# 제안서 예는 194개뿐이라(OpenAlex 하루 한도로 주제 232개) 다섯 번 반복해 넣는다(약 970개, 전체의 1.2%) — 시범이라는 것을 기록에 남긴다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data; I=data/instr; K=data/kci
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  ( CUDA_VISIBLE_DEVICES=1 bash scripts/run_v48.sh run ) > $R/logs_v48.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 40 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --topics scripts/wf_a_topics_ext.json \
    --search-cache $D/wf_a_search_40.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?) $(date +%m-%d_%H:%M)"; grep -aE "거부|Error" $L/wf_a_$n.txt | tail -3 | cut -c1-200; tail -1 $L/wf_a_$n.txt | cut -c1-200
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1 | cut -c1-200
}
gen() {
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1 $(date +%m-%d_%H:%M)"; grep -E "^전체|Error|거부" $L/${pre}_$1.txt | tail -1 | cut -c1-120
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

echo "== 1. 학습셋 sft10 = sft8 + 제안서 예 × 5 $(date)"
/usr/bin/python - <<'EOF'
import json
I = "data/instr/"
rows = [json.loads(l) for p in ("prop_train_a.jsonl", "prop_train_b.jsonl") for l in open(I + p, encoding="utf-8") if l.strip()]
with open(I + "prop_train.jsonl", "w", encoding="utf-8") as f:
    for r in rows * 5:
        # sft8 의 source 는 dict(doc · task · passage_sha256) — 열 형식이 다르면 datasets 가 합치지 못한다
        src = {"doc": "prop", "task": "proposal_" + r["kind"], "passage_sha256": ""}
        f.write(json.dumps({"messages": r["messages"], "source": src}, ensure_ascii=False) + "\n")
print("제안서 예", len(rows), "× 5 =", len(rows) * 5)
EOF
cat $I/sft8_train.jsonl $I/prop_train.jsonl > $I/sft10_train.jsonl; wc -l $I/sft10_train.jsonl

echo "== 2. sft_v10 → dpo_v13 $(date)"
if [ ! -s $CK/sft_v10/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft10_train.jsonl --output $CK/sft_v10 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v10.txt 2>&1
  echo "SFT_EXIT $? $(date +%m-%d_%H:%M)"; grep -aE "train_loss|Error" $L/sft_v10.txt | tail -2 | cut -c1-200
fi
[ -s $CK/sft_v10/adapter_model.safetensors ] || { echo "V48_FAIL sft_v10 없음 — DPO 로 넘어가지 않는다"; exit 1; }
if [ ! -s $CK/dpo_v13/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v10 --dataset $I/dpo7.jsonl --output $CK/dpo_v13 --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v13.txt 2>&1
  echo "DPO_EXIT $? $(date +%m-%d_%H:%M)"; grep -aE "train_loss|Error" $L/dpo_v13.txt | tail -2 | cut -c1-200
fi
[ -s $CK/dpo_v13/adapter_model.safetensors ] || { echo "V48_FAIL dpo_v13 없음"; exit 1; }

echo "== 3가. WF-A 40개 주제 $(date)"
wf w40_9b_full_v13 $S --adapter $CK/dpo_v13
wf w40_9b_regen_v13 $S --adapter $CK/dpo_v13 --adapter-on regen
echo "== 3나. 근거 충실 평가셋 $(date)"
gen dpo_v13 $S $CK/dpo_v13
jd phi4 $PHI $I/eval2_dpo_v13.jsonl $I/evalk_dpo_v13.jsonl
echo "V48_DONE $(date)"
