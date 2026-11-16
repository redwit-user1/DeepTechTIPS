#!/usr/bin/env bash
# 2026-11-16 v19 — 질의응답 선호쌍 (연구노트 §50)
#
#   bash repo/scripts/run_v19.sh start      # v18 A줄이 끝나고 sft_v5 어댑터가 생기면 GPU 0 에서 시작
#
# §50: 질의응답은 SFT 이후 한 번도 움직이지 않았고 통과율이 가장 낮다(Phi-4 연구노트 77.6%, 긴 원문 68.6%). 표집 단계에서
# 질의응답을 뽑지 않아 선호쌍이 없었기 때문이다. 교사가 만든 질문을 지시로 학생(sft_v5)의 답을 지시당 4개 표집하고,
# 같은 질문 안에서 '세 판정 모델이 모두 지지 + 질문에 맞다'와 '과반이 지지하지 않거나 질문에 맞지 않다'를 짝짓는다.
#   표집(연구노트 · KCI) → Kanana · Midm · Qwen 판정 → 질의응답 합의 선호쌍 → dpo7 = 조건 추출을 뺀 선호쌍(dpo4nc) + 질의응답 쌍
#   → 9B dpo_v7 (sft_v5 위에) → 평가 → 재심사.  비교 대상은 v18 의 dpo_v6 (sft_v5 + dpo4nc).
# 한 줄(GPU 0)로만 돈다 — 줄 사이 대기를 두지 않는다(§50 사고 기록).
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; SM=data/sampled
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4
HO="$I/heldout2.jsonl $K/heldout.jsonl"
NOTE=$SM/note_qa_sft_v5.jsonl; KCI=$SM/kci_qa_sft_v5.jsonl

if [ "$1" = start ]; then
  ( while ! grep -q V18A_DONE $R/logs_v18_A.txt 2>/dev/null || [ ! -s $CK/sft_v5/adapter_model.safetensors ]; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_v19.sh run ) > $R/logs_v19.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

model_of() { case $1 in phi4) echo $PHI;; kanana) echo $KAN;; midm) echo $MID;; qwen) echo $T;; esac; }
jd() {  # 꼬리표 파일... (qwen 은 기본 판정기, data/judged 에 쓴다)
  local tag=$1; shift
  local dir=data/judged_$tag opt="--llm $(model_of $tag)"; [ $tag = qwen ] && { dir=$J; opt=""; }
  for f in "$@"; do
    [ -s $f ] || continue
    [ -f $dir/$(basename $f).done ] && continue
    python scripts/judge_all.py $opt --out-dir $dir $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $dir/$(basename $f).done
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

echo "== 1. 질의응답 표집 (9B sft_v5, 질문당 4개) $(date)"
if [ ! -f $NOTE.done ]; then
  python scripts/eval_sft.py sample --qa --model $S --adapter $CK/sft_v5 --source $J/sft2.jsonl.screened.jsonl $J/gen.jsonl \
    --exclude $HO --out $NOTE --n 4 --batch 32 --max-new-tokens 200 > $L/sample_note_qa.txt 2>&1
  rc=$?; echo "SAMPLE_EXIT(note) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $NOTE.done
fi
if [ ! -f $KCI.done ]; then
  python scripts/eval_sft.py sample --qa --model $S --adapter $CK/sft_v5 --source $K/judged_all.jsonl --exclude $HO \
    --out $KCI --n 4 --batch 16 --max-new-tokens 300 --limit-passages 6000 > $L/sample_kci_qa.txt 2>&1
  rc=$?; echo "SAMPLE_EXIT(kci) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $KCI.done
fi
wc -l $NOTE $KCI

echo "== 2. 판정 (Kanana · Midm · Qwen) $(date)"
[ -f $NOTE.done ] && [ -f $KCI.done ] && for tag in kanana midm qwen; do jd $tag $NOTE $KCI; done

echo "== 3. 질의응답 합의 선호쌍 + 조건 추출을 뺀 기존 선호쌍 $(date)"
python scripts/judge_all.py dpo-multi $I/dpo_qa.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores $(basename $NOTE),$(basename $KCI) \
  qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm && mv $I/dpo_qa.tmp $I/dpo_qa.jsonl
[ -s $I/dpo_qa.jsonl ] && [ -s $I/dpo4nc.jsonl ] && cat $I/dpo4nc.jsonl $I/dpo_qa.jsonl > $I/dpo7.jsonl
wc -l $I/dpo_qa.jsonl $I/dpo4nc.jsonl $I/dpo7.jsonl

echo "== 4. 9B dpo_v7 (sft_v5 + 조건 추출을 뺀 선호쌍 + 질의응답 쌍) $(date)"
if [ -s $I/dpo7.jsonl ] && [ ! -s $CK/dpo_v7/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v5 --dataset $I/dpo7.jsonl --output $CK/dpo_v7 \
    --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v7.txt 2>&1
  echo "DPO_V7_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo_v7.txt | tail -3 | cut -c1-200
fi
[ -s $CK/dpo_v7/adapter_model.safetensors ] && { gen dpo_v7 $S $CK/dpo_v7; rejudge dpo_v7; }
echo "V19_DONE $(date)"
