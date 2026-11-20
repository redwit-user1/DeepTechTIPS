#!/usr/bin/env bash
# 2026-11-20 v36 — 두 바퀴째의 연구노트 쪽을 먼저 (v33 과 같은 구성, 연구노트 §60)
#
#   bash repo/scripts/run_v36.sh start      # v35 N줄이 끝나면 GPU 1 에서 시작
#
#   1. KCI 질의응답 표집 분담  2. 두 번째 교사의 연구노트 합의 통과분 → sft9n = sft6 + fly2_note
#   3. 9B sft_v9n → 평가   4. dpo_v12n(+ dpo7) → 평가
#   (K줄의 판정 분담은 뺐다 — 이 줄이 닿을 무렵 K줄이 같은 판정을 돌고 있을 수 있어 겹칠 위험이 있다)
# 비교: sft_v8n(첫 바퀴 교사의 연구노트분) 대 sft_v9n(두 번째 교사의 연구노트분) — 교사만 다르다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

SM=data/sampled
if [ "$1" = start ]; then
  ( while ! grep -q V35N_DONE $R/logs_v35_N.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v36.sh run ) > $R/logs_v36.txt 2>&1 < /dev/null &
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

HO="$I/heldout2.jsonl $K/heldout.jsonl"
samp() {  # 출력 이름, 나머지는 sample 옵션
  local out=$SM/$1.jsonl; shift
  [ -f $out.done ] && return
  python scripts/eval_sft.py sample --model $T --adapter $CK/dpo27_v7 --exclude $HO --out $out --n 1 --temperature 0.3 "$@" > $L/sample_$(basename $out .jsonl).txt 2>&1
  local rc=$?; echo "SAMPLE_EXIT($(basename $out)) $rc $(date +%H:%M) $(wc -l < $out)"; [ $rc = 0 ] && touch $out.done
}
jd4() {  # 표집 파일들을 판정 넷으로(qwen 은 기본 판정기 → data/judged), 이어서 조건 추출 전용 판정
  for f in "$@"; do
    [ -f $J/$(basename $f).done ] || { python scripts/judge_all.py --out-dir $J $f > $L/judge_qwen_$(basename $f .jsonl).txt 2>&1; rc=$?; echo "JUDGE_EXIT(qwen $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $J/$(basename $f).done; }
  done
  for tag in kanana midm eeve; do jd $tag "$@"; done
}

echo "== 1. KCI 질의응답 표집 (K줄 대신) $(date)"
samp fly2_kci_qa --qa --source $K/judged_all.jsonl --batch 16 --max-new-tokens 300 --limit-passages 4000

echo "== 2. 연구노트 쪽 네 모델 합의 통과분 $(date)"
[ -s $I/fly2_note_train.jsonl ] || { python scripts/judge_all.py export-multi $I/fly2_note.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores fly2_note.jsonl,fly2_note_qa.jsonl   qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm eeve=data/judged_eeve kv=data/kv_phi4,data/kv_kanana,data/kv_midm,data/kv_qwen && mv $I/fly2_note.tmp $I/fly2_note_train.jsonl; }
cat $I/sft6_train.jsonl $I/fly2_note_train.jsonl > $I/sft9n_train.jsonl; wc -l $I/fly2_note_train.jsonl $I/sft9n_train.jsonl

echo "== 3. 9B sft_v9n (sft6 + 새 교사의 연구노트 합의 통과분) $(date)"
if [ -s $I/fly2_note_train.jsonl ] && [ ! -s $CK/sft_v9n/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft9n_train.jsonl --output $CK/sft_v9n --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v9n.txt 2>&1
  echo "SFT_EXIT $?"
fi
[ -s $CK/sft_v9n/adapter_model.safetensors ] && { gen sft_v9n $S $CK/sft_v9n; rejudge sft_v9n; }
echo "== 4. 9B dpo_v12n (sft_v9n + dpo7) $(date)"
if [ -s $CK/sft_v9n/adapter_model.safetensors ] && [ ! -s $CK/dpo_v12n/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v9n --dataset $I/dpo7.jsonl --output $CK/dpo_v12n     --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v12n.txt 2>&1
  echo "DPO_EXIT $?"
fi
[ -s $CK/dpo_v12n/adapter_model.safetensors ] && { gen dpo_v12n $S $CK/dpo_v12n; rejudge dpo_v12n; }
echo "V36_DONE $(date)"
