#!/usr/bin/env bash
# 2026-11-15 v17 — 합의 선호쌍 DPO 2회차: 개선된 학생(dpo_v4)의 응답으로 선호쌍을 다시 만든다 (연구노트 §47)
#
#   bash repo/scripts/run_v17.sh start      # B줄은 바로, A줄은 v16 A줄이 끝난 뒤 시작
#
# v16: sft_v4 위에 합의 선호쌍 DPO(dpo_v4)가 선택에 쓰지 않은 Phi-4 기준 +5.2%p(연구노트) · +2.7%p(긴 원문).
# 그 선호쌍은 예전 학생(sft_v2)의 응답에서 골랐다. 학생이 달라졌으니 틀리는 곳도 달라졌을 것이다.
# dpo_v4 가 직접 쓴 응답을 다시 표집 → Qwen · Kanana · Midm 판정 → 합의 선호쌍 → dpo_v5. 평가는 계속 Phi-4.
#   B줄(GPU 1)  연구노트 · KCI 표집 → Kanana 판정 → Qwen 판정(연구노트) → [A 대기] 합의 선호쌍 → 9B dpo_v5 → 평가 → 재심사
#   A줄(GPU 0)  Midm 판정 → Qwen 판정(KCI) → [쌍 대기] 27B dpo27_v4(dpo27_v3 위에) → 평가 → 재심사
# 같은 GPU 에 판정 작업을 겹치지 않는다. 끝난 단계는 .done 표시로 건너뛴다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; SM=data/sampled
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4
HO="$I/heldout2.jsonl $K/heldout.jsonl"
NOTE=$SM/note_dpo_v4.jsonl; KCI=$SM/kci_dpo_v4.jsonl

if [ "$1" = start ]; then
  LANE=B CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_v17.sh > $R/logs_v17_B.txt 2>&1 < /dev/null &
  ( while ! grep -q V16A_DONE $R/logs_v16_A.txt 2>/dev/null; do sleep 30; done
    LANE=A CUDA_VISIBLE_DEVICES=0 bash scripts/run_v17.sh ) > $R/logs_v17_A.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wait_for() { for i in $(seq 1 2880); do [ -e $1 ] && return 0; sleep 30; done; echo "TIMEOUT $1"; return 1; }
jd() {  # 꼬리표 모델 파일... (꼬리표 qwen 은 기본 판정기 = Qwen 캐스케이드, data/judged 에 쓴다)
  local tag=$1 m=$2; shift 2
  local dir=data/judged_$tag opt="--llm $m"; [ $tag = qwen ] && { dir=$J; opt=""; }
  for f in "$@"; do
    [ -s $f ] || continue
    [ -f $dir/$(basename $f).done ] && { echo "-- $tag $(basename $f) (있음, 건너뜀)"; continue; }
    python scripts/judge_all.py $opt --out-dir $dir $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $dir/$(basename $f).done
  done
}
gen() {  # 이름 모델 어댑터
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1"; grep -E "^전체|어댑터|Error|거부" $L/${pre}_$1.txt | tail -3
  done
}
judge3() { jd phi4 $PHI "$@"; jd kanana $KAN "$@"; jd midm $MID "$@"; }

if [ "$LANE" = B ]; then
  echo "== B1. 표집 (9B dpo_v4, 지시당 4개) $(date)"
  if [ ! -f $NOTE.done ]; then
    python scripts/eval_sft.py sample --model $S --adapter $CK/dpo_v4 --source $J/sft2.jsonl.screened.jsonl $J/gen.jsonl \
      --exclude $HO --out $NOTE --n 4 --batch 16 > $L/sample_note_v17.txt 2>&1
    rc=$?; echo "SAMPLE_EXIT(note) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $NOTE.done
  fi
  if [ ! -f $KCI.done ]; then
    python scripts/eval_sft.py sample --model $S --adapter $CK/dpo_v4 --source $K/judged_all.jsonl --exclude $HO \
      --out $KCI --n 4 --batch 8 --max-new-tokens 900 --limit-passages 3000 > $L/sample_kci_v17.txt 2>&1
    rc=$?; echo "SAMPLE_EXIT(kci) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $KCI.done
  fi
  wc -l $NOTE $KCI
  echo "== B2. Kanana · Qwen(연구노트) 판정 $(date)"
  [ -f $NOTE.done ] && [ -f $KCI.done ] && { jd kanana $KAN $NOTE $KCI; jd qwen - $NOTE; }

  echo "== B3. 합의 선호쌍 (A줄의 Midm · Qwen 판정을 기다린다) $(date)"
  wait_for data/judged_midm/$(basename $KCI).done && wait_for $J/$(basename $KCI).done && \
    python scripts/judge_all.py dpo-multi $I/dpo5.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores $(basename $NOTE),$(basename $KCI) \
      qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm && mv $I/dpo5.tmp $I/dpo5.jsonl
  wc -l $I/dpo5.jsonl

  echo "== B4. 9B dpo_v5 (dpo_v4 위에 2회차 합의 선호쌍) $(date)"
  if [ -s $I/dpo5.jsonl ] && [ ! -s $CK/dpo_v5/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/dpo_v4 --dataset $I/dpo5.jsonl --output $CK/dpo_v5 \
      --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v5.txt 2>&1
    echo "DPO_V5_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo_v5.txt | tail -3 | cut -c1-200
  fi
  [ -s $CK/dpo_v5/adapter_model.safetensors ] && { gen dpo_v5 $S $CK/dpo_v5; judge3 $I/eval2_dpo_v5.jsonl $I/evalk_dpo_v5.jsonl; }
  echo "V17B_DONE $(date)"
fi

if [ "$LANE" = A ]; then
  echo "== A1. Midm · Qwen(KCI) 판정 (B줄의 표집을 기다린다) $(date)"
  wait_for $NOTE.done && jd midm $MID $NOTE
  wait_for $KCI.done && { jd midm $MID $KCI; jd qwen - $KCI; }

  echo "== A2. 27B dpo27_v4 (dpo27_v3 위에 2회차 합의 선호쌍 — 9B 표본이라 27B 에는 남의 응답) $(date)"
  wait_for $I/dpo5.jsonl
  if [ -s $I/dpo5.jsonl ] && [ -s $CK/dpo27_v3/adapter_model.safetensors ] && [ ! -s $CK/dpo27_v4/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/dpo27_v3 --dataset $I/dpo5.jsonl --output $CK/dpo27_v4 \
      --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 > $L/dpo27_v4.txt 2>&1
    echo "DPO27_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo27_v4.txt | tail -3 | cut -c1-200
  fi
  [ -s $CK/dpo27_v4/adapter_model.safetensors ] && { gen dpo27_v4 $T $CK/dpo27_v4; judge3 $I/eval2_dpo27_v4.jsonl $I/evalk_dpo27_v4.jsonl; }
  echo "V17A_DONE $(date)"
fi
