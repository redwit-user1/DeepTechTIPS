#!/usr/bin/env bash
# 2026-11-14 v11 — 자기 응답 표집으로 선호쌍 확대 → DPO v2 (연구노트 §44)
#
#   bash repo/scripts/run_v11.sh start      # A줄은 바로, B줄은 v10 이 끝난 뒤 시작
#
# v9 의 DPO 는 교사 응답 693쌍으로 +2.0%p(보고서 문단 +6.9%p)였다. 쌍이 적고, 탈락 응답이 학생의 오류가 아니다.
# 학생(9B sft_v2)이 학습용 원문에 직접 쓴 응답을 지시당 4개 표집 → 전수 판정 → 두 판정기가 합의한 쌍만 → DPO.
#   A줄(GPU 0)  연구노트 원문 표집(약 1.9만 지시 × 4) → 판정 → [B 대기] 합의 선호쌍 → 9B dpo_v2 → 평가 2종
#   B줄(GPU 1)  KCI 초록 3,000건 표집(× 3과제 × 4) → 판정 → [쌍 대기] 27B dpo27_v1(sft27_v1 에서 이어서) → 평가 2종
# 같은 GPU 에 27B 판정 모델을 쓰는 작업을 겹치지 않는다. 결과가 있으면 건너뛴다(표집·판정은 이어서 진행).
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; SM=data/sampled
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
HO="$I/heldout2.jsonl $K/heldout.jsonl"

if [ "$1" = start ]; then
  mkdir -p $SM
  LANE=A CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v11.sh > $R/logs_v11_A.txt 2>&1 < /dev/null &
  ( while ! grep -q V10_DONE $R/logs_v10.txt 2>/dev/null; do sleep 30; done
    LANE=B CUDA_VISIBLE_DEVICES=1 bash scripts/run_v11.sh ) > $R/logs_v11_B.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

ev() {  # 접두 이름 평가지시 모델 어댑터
  local out=$I/$1_$2.jsonl
  [ -s $out ] && { echo "-- $1_$2 (있음, 건너뜀)"; return; }
  python scripts/eval_sft.py run --model $4 --adapter $5 $( [ "$1" = evalk ] && echo --max-new-tokens 900 ) --heldout $3 --out $out > $L/$1_$2.txt 2>&1
  echo "-- $1_$2 (exit $?)"; grep -E "충실 생성률|판정기|어댑터|Error|거부" $L/$1_$2.txt | tail -8
}
cmp() { [ -s $I/$2.jsonl ] && [ -s $I/$3.jsonl ] && { echo "[$1]"; python scripts/eval_sft.py compare $I/$2.jsonl $I/$3.jsonl; }; }
judge() {  # 로그이름 파일 — 판정이 끝나면 <파일>.done 을 남긴다
  [ -f $J/$(basename $2).done ] && { echo "-- 판정 $1 (있음, 건너뜀)"; return; }
  python scripts/judge_all.py --out-dir $J $2 > $L/judge_$1.txt 2>&1
  local rc=$?; echo "JUDGE_EXIT($1) $rc"; [ $rc = 0 ] && touch $J/$(basename $2).done
  python scripts/judge_all.py report $J/$(basename $2)
}

if [ "$LANE" = A ]; then
  echo "== A1. 연구노트 원문 표집 (9B sft_v2, 지시당 4개) $(date)"
  python scripts/eval_sft.py sample --model $S --adapter $CK/sft_v2 --source $J/sft2.jsonl.screened.jsonl $J/gen.jsonl \
    --exclude $HO --out $SM/note_sft_v2.jsonl --n 4 --batch 16 > $L/sample_note.txt 2>&1
  echo "SAMPLE_EXIT $?"; wc -l $SM/note_sft_v2.jsonl
  echo "== A2. 판정 $(date)"
  judge v11a $SM/note_sft_v2.jsonl

  echo "== A3. 합의 선호쌍 (B줄의 KCI 판정을 최대 2시간 기다린다 — 없으면 연구노트 표본만으로) $(date)"
  for i in $(seq 1 240); do [ -f $J/kci_sft_v2.jsonl.done ] && break; sleep 30; done
  python scripts/judge_all.py dpo $I/dpo3.tmp $I/heldout2.jsonl,$K/heldout.jsonl \
    $(ls $J/note_sft_v2.jsonl $J/kci_sft_v2.jsonl $J/sft2.jsonl.screened.jsonl $J/gen.jsonl 2>/dev/null) && mv $I/dpo3.tmp $I/dpo3.jsonl
  wc -l $I/dpo3.jsonl

  echo "== A4. 9B dpo_v2 (sft_v2 에서 이어서, 1에폭) $(date)"
  if [ ! -s $CK/dpo_v2/adapter_model.safetensors ] && [ -s $I/dpo3.jsonl ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v2 --dataset $I/dpo3.jsonl --output $CK/dpo_v2 \
      --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v2.txt 2>&1
    echo "DPO_V2_EXIT $?"; grep -E "accepted|train_loss|saved|Error|Traceback" $L/dpo_v2.txt | tail -4 | cut -c1-300
  fi
  if [ -s $CK/dpo_v2/adapter_model.safetensors ]; then
    ev eval2 dpo_v2 $I/heldout2.jsonl $S $CK/dpo_v2
    ev evalk dpo_v2 $K/heldout.jsonl $S $CK/dpo_v2
    [ -s $I/evalk_dpo_v1.jsonl ] || ev evalk dpo_v1 $K/heldout.jsonl $S $CK/dpo_v1
    cmp "연구노트: sft_v2 → dpo_v2" eval2_sft_v2 eval2_dpo_v2
    cmp "연구노트: dpo_v1 → dpo_v2" eval2_dpo_v1 eval2_dpo_v2
    cmp "긴 원문: sft_v2 → dpo_v2" evalk_sft_v2 evalk_dpo_v2
    cmp "긴 원문: sft_v2 → dpo_v1" evalk_sft_v2 evalk_dpo_v1
  fi
  echo "V11A_DONE $(date)"
fi

if [ "$LANE" = B ]; then
  echo "== B1. KCI 초록 3,000건 표집 (9B sft_v2, 지시당 4개) $(date)"
  python scripts/eval_sft.py sample --model $S --adapter $CK/sft_v2 --source $K/judged_all.jsonl --exclude $HO \
    --out $SM/kci_sft_v2.jsonl --n 4 --batch 8 --max-new-tokens 900 --limit-passages 3000 > $L/sample_kci.txt 2>&1
  echo "SAMPLE_EXIT $?"; wc -l $SM/kci_sft_v2.jsonl
  echo "== B2. 판정 $(date)"
  judge v11b $SM/kci_sft_v2.jsonl

  echo "== B3. 27B dpo27_v1 (sft27_v1 에서 이어서, 1에폭 — 선호쌍은 9B 표본이라 27B 에는 남의 응답이다) $(date)"
  for i in $(seq 1 1440); do [ -s $I/dpo3.jsonl ] && break; sleep 30; done
  if [ ! -s $CK/dpo27_v1/adapter_model.safetensors ] && [ -s $I/dpo3.jsonl ]; then
    python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/sft27_v1 --dataset $I/dpo3.jsonl --output $CK/dpo27_v1 \
      --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 > $L/dpo27_v1.txt 2>&1
    echo "DPO27_EXIT $?"; grep -E "accepted|train_loss|saved|Error|Traceback|OutOfMemory" $L/dpo27_v1.txt | tail -4 | cut -c1-300
  fi
  if [ -s $CK/dpo27_v1/adapter_model.safetensors ]; then
    ev eval2 dpo27_v1 $I/heldout2.jsonl $T $CK/dpo27_v1
    ev evalk dpo27_v1 $K/heldout.jsonl $T $CK/dpo27_v1
    cmp "연구노트: 27B LoRA → 27B DPO" eval2_sft27_v1 eval2_dpo27_v1
    cmp "긴 원문: 27B LoRA → 27B DPO" evalk_sft27_v1 evalk_dpo27_v1
  fi
  echo "V11B_DONE $(date)"
fi
