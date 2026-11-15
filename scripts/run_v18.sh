#!/usr/bin/env bash
# 2026-11-16 v18 — 조건 추출 학습 예를 항목 줄 전용 판정으로 되살린다 (연구노트 §48)
#
#   bash repo/scripts/run_v18.sh start      # 두 줄 모두 v17 의 같은 줄이 끝난 뒤 시작
#
# §48: 기존 판정 프롬프트는 '항목: 값' 줄을 값이 원문에 그대로 있어도 3건 중 1건 탈락시켰다. 그 판정으로 거른 합의 통과분(sft_v4)은
# 조건 추출 예가 5,377건만 남았고, 모델은 줄을 덜 쓰게 됐다(응답당 통과 줄 3.14 → 2.52). 선호쌍도 같은 판정으로 골랐다.
#   A줄(GPU 0)  교사 생성분 조건 추출을 Phi-4 · Kanana 전용 판정 → dpo_v4nc(sft_v4 + 조건 추출 쌍을 뺀 선호쌍) → 평가 → 재심사 → 4-bit 보존율
#   B줄(GPU 1)  교사 생성분 조건 추출을 Midm · Qwen 전용 판정 → [A 대기] 합의 통과분 다시 내기(sft5) → 9B sft_v5 → 평가 → 재심사
#               → dpo_v6(sft_v5 + 조건 추출 쌍을 뺀 선호쌍) → 평가 → 재심사
# 평가 응답의 조건 추출은 네 판정 모델의 전용 판정으로도 심사한다(data/kv_<꼬리표>/).
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4
TEACHER="sft2.jsonl.screened.jsonl gen.jsonl gen_0.jsonl gen_1.jsonl"

if [ "$1" = start ]; then
  ( while ! grep -q V17A_DONE $R/logs_v17_A.txt 2>/dev/null; do sleep 30; done
    LANE=A CUDA_VISIBLE_DEVICES=0 bash scripts/run_v18.sh ) > $R/logs_v18_A.txt 2>&1 < /dev/null &
  ( while ! grep -q V17B_DONE $R/logs_v17_B.txt 2>/dev/null; do sleep 30; done
    LANE=B CUDA_VISIBLE_DEVICES=1 bash scripts/run_v18.sh ) > $R/logs_v18_B.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

model_of() { case $1 in phi4) echo $PHI;; kanana) echo $KAN;; midm) echo $MID;; qwen) echo $T;; esac; }
wait_for() { for i in $(seq 1 2880); do [ -e $1 ] && return 0; sleep 30; done; echo "TIMEOUT $1"; return 1; }
jd() {  # 꼬리표 파일...
  local tag=$1; shift
  for f in "$@"; do
    [ -s $f ] || continue
    [ -f data/judged_$tag/$(basename $f).done ] && continue
    python scripts/judge_all.py --llm $(model_of $tag) --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}
kv() {  # 꼬리표 이름표 판정된파일... — 조건 추출 항목 줄만. 끝나면 data/kv_<꼬리표>/.<이름표>.done
  local tag=$1 mark=$2; shift 2
  [ -f data/kv_$tag/.$mark.done ] && return
  python scripts/judge_all.py --kv --batch 16 --llm $(model_of $tag) --out-dir data/kv_$tag "$@" > $L/kv_${tag}_$mark.txt 2>&1
  local rc=$?; echo "KV_EXIT($tag $mark) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/kv_$tag/.$mark.done
}
gen() {  # 이름 모델 어댑터 [추가 옵션]
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra="$4"; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="$4 --max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1"; grep -E "^전체|어댑터|Error|거부" $L/${pre}_$1.txt | tail -3 | cut -c1-200
  done
}
rejudge() {  # 이름... — 세 판정 모델 재심사 + 네 판정 모델의 항목 줄 전용 판정
  for n in "$@"; do
    for tag in phi4 kanana midm; do jd $tag $I/eval2_$n.jsonl $I/evalk_$n.jsonl; done
    [ -s data/judged_phi4/eval2_$n.jsonl ] && for tag in phi4 kanana midm qwen; do
      kv $tag $n data/judged_phi4/eval2_$n.jsonl data/judged_phi4/evalk_$n.jsonl; done
  done
}
dpo_nc() {  # 조건 추출 쌍을 뺀 선호쌍 — 그 쌍은 기존 프롬프트의 항목 줄 판정으로 골랐다
  [ -s $I/dpo4nc.jsonl ] || python -c "
import json
rows = [l for l in open('$I/dpo4.jsonl', encoding='utf-8') if json.loads(l)['source']['task'] != 'conditions']
open('$I/dpo4nc.jsonl', 'w', encoding='utf-8').writelines(rows); print('dpo4nc', len(rows))"
}

if [ "$LANE" = A ]; then
  echo "== A1. 교사 생성분 조건 추출 — Phi-4 · Kanana 전용 판정 $(date)"
  for tag in phi4 kanana; do kv $tag teacher $(for f in $TEACHER; do echo $J/$f; done); done

  echo "== A2. 9B dpo_v4nc (sft_v4 + 조건 추출 쌍을 뺀 선호쌍) $(date)"
  dpo_nc; wc -l $I/dpo4nc.jsonl
  if [ -s $I/dpo4nc.jsonl ] && [ ! -s $CK/dpo_v4nc/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v4 --dataset $I/dpo4nc.jsonl --output $CK/dpo_v4nc \
      --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v4nc.txt 2>&1
    echo "DPO_V4NC_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo_v4nc.txt | tail -3 | cut -c1-200
  fi
  [ -s $CK/dpo_v4nc/adapter_model.safetensors ] && { gen dpo_v4nc $S $CK/dpo_v4nc; rejudge dpo_v4nc; }

  echo "== A3. 이미 있는 평가 응답의 항목 줄 전용 판정 (27B · v17) $(date)"
  rejudge base27_hint sft27_v1 sft27_v2 dpo27_v2 dpo27_v3 dpo_v5 dpo27_v4

  echo "== A4. 4-bit 보존율 (9B dpo_v4) $(date)"
  gen dpo_v4_4bit $S $CK/dpo_v4 --load-4bit; rejudge dpo_v4_4bit
  echo "V18A_DONE $(date)"
fi

if [ "$LANE" = B ]; then
  echo "== B1. 교사 생성분 조건 추출 — Midm · Qwen 전용 판정 $(date)"
  for tag in midm qwen; do kv $tag teacher $(for f in $TEACHER; do echo $J/$f; done); done

  echo "== B2. 합의 통과분 다시 내기 — 조건 추출은 전용 판정으로 (A줄을 기다린다) $(date)"
  wait_for data/kv_phi4/.teacher.done && wait_for data/kv_kanana/.teacher.done && [ -f data/kv_midm/.teacher.done ] && [ -f data/kv_qwen/.teacher.done ] && \
    python scripts/judge_all.py export-multi $I/sft5.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores $(echo $TEACHER | tr ' ' ,) \
      qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm kv=data/kv_phi4,data/kv_kanana,data/kv_midm,data/kv_qwen && mv $I/sft5.tmp $I/sft5_train.jsonl
  wc -l $I/sft5_train.jsonl

  echo "== B3. 9B sft_v5 (합의 통과분 + 되살린 조건 추출, 1에폭) $(date)"
  if [ -s $I/sft5_train.jsonl ] && [ ! -s $CK/sft_v5/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft5_train.jsonl --output $CK/sft_v5 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v5.txt 2>&1
    echo "SFT_V5_EXIT $?"; grep -E "LoRA 모듈|saved|Error|OutOfMemory" $L/sft_v5.txt | tail -2 | cut -c1-200
  fi
  [ -s $CK/sft_v5/adapter_model.safetensors ] && { gen sft_v5 $S $CK/sft_v5; rejudge sft_v5; }

  echo "== B4. 9B dpo_v6 (sft_v5 + 조건 추출 쌍을 뺀 선호쌍) $(date)"
  dpo_nc
  if [ -s $CK/sft_v5/adapter_model.safetensors ] && [ -s $I/dpo4nc.jsonl ] && [ ! -s $CK/dpo_v6/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v5 --dataset $I/dpo4nc.jsonl --output $CK/dpo_v6 \
      --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v6.txt 2>&1
    echo "DPO_V6_EXIT $?"; grep -E "accepted|saved|Error|OutOfMemory" $L/dpo_v6.txt | tail -3 | cut -c1-200
  fi
  [ -s $CK/dpo_v6/adapter_model.safetensors ] && { gen dpo_v6 $S $CK/dpo_v6; rejudge dpo_v6; }
  echo "V18B_DONE $(date)"
fi
