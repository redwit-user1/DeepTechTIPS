#!/usr/bin/env bash
# 2026-11-19 v33 — v32 의 N줄이 끝난 뒤 빈 GPU 1 을 쓴다 (연구노트 §59)
#
#   bash repo/scripts/run_v33.sh start
#
# v32 의 KCI 표집(27B, 긴 원문)이 예상보다 느려 K줄은 자정 무렵에야 학습에 들어간다. 그동안 GPU 1 이 빈다.
#   1. KCI 질의응답 표집을 대신 한다(K줄은 완료 표시를 보고 건너뛴다)
#   2. 연구노트 쪽만으로 먼저 한 바퀴: fly_note 의 네 모델 합의 통과분 → sft8n = sft6 + fly_note → 9B sft_v8n → 평가 → dpo_v11n(+ dpo7) → 평가
#      (K줄의 sft_v8 은 KCI 쪽까지 더한 것 — 둘을 비교하면 새 교사 데이터가 어느 원천에서 듣는지도 보인다)
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

SM=data/sampled
if [ "$1" = start ]; then
  ( while ! grep -q V32N_DONE $R/logs_v32_N.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v33.sh run ) > $R/logs_v33.txt 2>&1 < /dev/null &
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
  python scripts/eval_sft.py sample --model $T --adapter $CK/dpo27_v6 --exclude $HO --out $out --n 1 --temperature 0.3 "$@" > $L/sample_$(basename $out .jsonl).txt 2>&1
  local rc=$?; echo "SAMPLE_EXIT($(basename $out)) $rc $(date +%H:%M) $(wc -l < $out)"; [ $rc = 0 ] && touch $out.done
}
jd4() {  # 표집 파일들을 판정 넷으로(qwen 은 기본 판정기 → data/judged), 이어서 조건 추출 전용 판정
  for f in "$@"; do
    [ -f $J/$(basename $f).done ] || { python scripts/judge_all.py --out-dir $J $f > $L/judge_qwen_$(basename $f .jsonl).txt 2>&1; rc=$?; echo "JUDGE_EXIT(qwen $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $J/$(basename $f).done; }
  done
  for tag in kanana midm eeve; do jd $tag "$@"; done
}

echo "== 1. KCI 질의응답 표집 (K줄 대신) $(date)"
samp fly_kci_qa --qa --source $K/judged_all.jsonl --batch 16 --max-new-tokens 300 --limit-passages 4000

echo "== 2. 연구노트 쪽 네 모델 합의 통과분 $(date)"
[ -s $I/fly_note_train.jsonl ] || { python scripts/judge_all.py export-multi $I/fly_note.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores fly_note.jsonl,fly_note_qa.jsonl   qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm eeve=data/judged_eeve kv=data/kv_phi4,data/kv_kanana,data/kv_midm,data/kv_qwen && mv $I/fly_note.tmp $I/fly_note_train.jsonl; }
cat $I/sft6_train.jsonl $I/fly_note_train.jsonl > $I/sft8n_train.jsonl; wc -l $I/fly_note_train.jsonl $I/sft8n_train.jsonl

echo "== 3. 9B sft_v8n (sft6 + 새 교사의 연구노트 합의 통과분) $(date)"
if [ -s $I/fly_note_train.jsonl ] && [ ! -s $CK/sft_v8n/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft8n_train.jsonl --output $CK/sft_v8n --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v8n.txt 2>&1
  echo "SFT_EXIT $?"
fi
[ -s $CK/sft_v8n/adapter_model.safetensors ] && { gen sft_v8n $S $CK/sft_v8n; rejudge sft_v8n; }
echo "== 4. 9B dpo_v11n (sft_v8n + dpo7) $(date)"
if [ -s $CK/sft_v8n/adapter_model.safetensors ] && [ ! -s $CK/dpo_v11n/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v8n --dataset $I/dpo7.jsonl --output $CK/dpo_v11n     --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v11n.txt 2>&1
  echo "DPO_EXIT $?"
fi
[ -s $CK/dpo_v11n/adapter_model.safetensors ] && { gen dpo_v11n $S $CK/dpo_v11n; rejudge dpo_v11n; }
echo "V33_DONE $(date)"
