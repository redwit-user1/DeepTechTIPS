#!/usr/bin/env bash
# 2026-11-19 v32 — 플라이휠 한 바퀴: 학습된 27B 를 교사로 다시 쓴다 (연구노트 §58)
#
#   bash repo/scripts/run_v32.sh start      # N줄(GPU 1)은 v30 뒤, K줄(GPU 0)은 v31 뒤
#
# 지금까지의 학습 데이터는 모두 기반 Qwen 27B 가 쓴 응답을 걸러 낸 것이다. 네 판정 모델 합의를 통과한 것은 112,443건 중 35,686건(32%)뿐이다.
# 2차년도 계획의 Compliance Data Flywheel 은 '검증기가 거른 데이터로 학습한 모델이 더 잘 걸러지는 데이터를 만든다'는 가정에 서 있다. 그 한 바퀴를 실제로 돌려 본다.
#   교사 = dpo27_v6 (27B, 네 모델 합의 SFT + 합의 선호쌍 DPO). 학습용 원문(평가 원문 제외)에 과제별 응답 1개씩.
#   N줄(GPU 1): 연구노트 원문(요약 · 조건 · 보고서) + 연구노트 질의응답 → 판정 넷 → 조건 추출 전용 판정
#   K줄(GPU 0): KCI 초록 4,000건(요약 · 조건 · 보고서) + 그 질의응답 → 판정 넷 → 조건 추출 전용 판정
#               → [N 대기] 네 모델 합의 통과분(fly) → sft8 = sft6 + fly → 9B sft_v8 → 평가 → dpo_v11 = sft_v8 + dpo7 → 평가
# 볼 것: ① 새 교사 응답의 네 모델 합의 통과율(기반 교사 32% 대비) ② 그 데이터를 더한 학생이 sft_v6 · dpo_v8 보다 나은가(Phi-4 독립).
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

SM=data/sampled
if [ "$1" = start ]; then
  ( while ! grep -q V30_DONE $R/logs_v30.txt 2>/dev/null; do sleep 30; done
    LANE=N CUDA_VISIBLE_DEVICES=1 bash scripts/run_v32.sh run ) > $R/logs_v32_N.txt 2>&1 < /dev/null &
  ( while ! grep -q V31_DONE $R/logs_v31.txt 2>/dev/null; do sleep 30; done
    LANE=K CUDA_VISIBLE_DEVICES=0 bash scripts/run_v32.sh run ) > $R/logs_v32_K.txt 2>&1 < /dev/null &
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

if [ "$LANE" = N ]; then
  echo "== N1. 연구노트 표집 (교사 dpo27_v6) $(date)"
  samp fly_note --source $J/sft2.jsonl.screened.jsonl $J/gen.jsonl --batch 16
  samp fly_note_qa --qa --source $J/sft2.jsonl.screened.jsonl $J/gen.jsonl --batch 32 --max-new-tokens 200
  echo "== N2. 판정 $(date)"
  jd4 $SM/fly_note.jsonl $SM/fly_note_qa.jsonl
  for tag in phi4 kanana midm qwen; do kv $tag fly_note $J/fly_note.jsonl; done
  touch $SM/.fly_note_all.done
  echo "V32N_DONE $(date)"
fi

if [ "$LANE" = K ]; then
  echo "== K1. KCI 표집 (교사 dpo27_v6, 4,000 원문) $(date)"
  samp fly_kci --source $K/judged_all.jsonl --batch 8 --max-new-tokens 900 --limit-passages 4000
  samp fly_kci_qa --qa --source $K/judged_all.jsonl --batch 16 --max-new-tokens 300 --limit-passages 4000
  echo "== K2. 판정 $(date)"
  jd4 $SM/fly_kci.jsonl $SM/fly_kci_qa.jsonl
  for tag in phi4 kanana midm qwen; do kv $tag fly_kci $J/fly_kci.jsonl; done

  echo "== K3. 네 모델 합의 통과분 (N줄을 기다린다) $(date)"
  for i in $(seq 1 2880); do [ -f $SM/.fly_note_all.done ] && break; sleep 30; done
  [ -s $I/fly_train.jsonl ] || { python scripts/judge_all.py export-multi $I/fly.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores fly_note.jsonl,fly_note_qa.jsonl,fly_kci.jsonl,fly_kci_qa.jsonl     qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm eeve=data/judged_eeve kv=data/kv_phi4,data/kv_kanana,data/kv_midm,data/kv_qwen && mv $I/fly.tmp $I/fly_train.jsonl; }
  cat $I/sft6_train.jsonl $I/fly_train.jsonl > $I/sft8_train.jsonl; wc -l $I/fly_train.jsonl $I/sft8_train.jsonl

  echo "== K4. 9B sft_v8 (sft6 + 새 교사의 합의 통과분) $(date)"
  if [ -s $I/fly_train.jsonl ] && [ ! -s $CK/sft_v8/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft8_train.jsonl --output $CK/sft_v8 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v8.txt 2>&1
    echo "SFT_EXIT $?"
  fi
  [ -s $CK/sft_v8/adapter_model.safetensors ] && { gen sft_v8 $S $CK/sft_v8; rejudge sft_v8; }
  echo "== K5. 9B dpo_v11 (sft_v8 + dpo7) $(date)"
  if [ -s $CK/sft_v8/adapter_model.safetensors ] && [ ! -s $CK/dpo_v11/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v8 --dataset $I/dpo7.jsonl --output $CK/dpo_v11       --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v11.txt 2>&1
    echo "DPO_EXIT $?"
  fi
  [ -s $CK/dpo_v11/adapter_model.safetensors ] && { gen dpo_v11 $S $CK/dpo_v11; rejudge dpo_v11; }
  echo "V32K_DONE $(date)"
fi
