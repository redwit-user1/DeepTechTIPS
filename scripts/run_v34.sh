#!/usr/bin/env bash
# 2026-11-19 v34 — v33 이 끝난 GPU 1 로 v32 K줄의 판정을 나눠 지고, 플라이휠 결과를 반복으로 확인한다 (연구노트 §59)
#
#   bash repo/scripts/run_v34.sh start
#
#   1. fly_kci · fly_kci_qa 의 EEVE 판정(K줄은 Qwen → Kanana → Midm → EEVE 순서라 EEVE 에 마지막으로 닿는다 — 완료 표시를 보고 건너뛴다)
#   2. K줄의 Qwen 판정이 끝나면 fly_kci 의 조건 추출 전용 판정 넷(역시 K줄이 마지막에 하는 일)
#   3. sft_v8n 을 씨앗만 바꿔 한 번 더(sft_v8n_s1) — §59 의 연구노트 +4.6%p 가 학습 한 번의 운인지
# 같은 (판정, 파일)을 두 줄이 동시에 돌리면 출력이 깨진다. K줄이 나중에 하는 것만 가져온다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

SM=data/sampled
if [ "$1" = start ]; then
  ( while ! grep -q V33_DONE $R/logs_v33.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v34.sh run ) > $R/logs_v34.txt 2>&1 < /dev/null &
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

echo "== 1. fly_kci 의 EEVE 판정 $(date)"
jd eeve $SM/fly_kci.jsonl $SM/fly_kci_qa.jsonl
echo "== 2. fly_kci 조건 추출 전용 판정 (K줄의 Qwen 판정을 기다린다) $(date)"
for i in $(seq 1 720); do [ -f $J/fly_kci.jsonl.done ] && break; sleep 30; done
[ -f $J/fly_kci.jsonl.done ] && for tag in phi4 kanana midm qwen; do kv $tag fly_kci $J/fly_kci.jsonl; done

echo "== 3. 9B sft_v8n_s1 (sft8n, 씨앗 1) $(date)"
if [ ! -s $CK/sft_v8n_s1/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft8n_train.jsonl --output $CK/sft_v8n_s1 --epochs 1 --batch 2 --grad-accum 8 --seed 1 > $L/sft_v8n_s1.txt 2>&1
  echo "SFT_EXIT $?"
fi
[ -s $CK/sft_v8n_s1/adapter_model.safetensors ] && { gen sft_v8n_s1 $S $CK/sft_v8n_s1; rejudge sft_v8n_s1; }
echo "V34_DONE $(date)"
