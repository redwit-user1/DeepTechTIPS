#!/usr/bin/env bash
# 2026-11-15 v14 — 학습 데이터를 여러 판정 모델의 합의로 거르면 달라지는가 (연구노트 §45)
#
#   bash repo/scripts/run_v14.sh start
#
# v13: 선호쌍을 세 판정 모델(Qwen · Kanana · Midm)의 합의로 고른 DPO 는 선택에 쓰지 않은 Phi-4 에서도 이득이 났다
# (+1.4 ~ +2.0%p). Qwen 하나로 고른 쌍은 0 이었다. 같은 원리를 SFT 학습 데이터에 적용한다:
# 지금까지의 학습셋은 Qwen 캐스케이드 통과분(86,905건)이고, 짧은 연구노트 단락에서는 다른 판정 모델 기준 이득이 0 이었다(§44).
#   B줄(GPU 1)  Midm 으로 교사 생성분(연구노트 50,774 · KCI 61,669) 판정 → sft_v3 평가 응답을 세 모델로 재심사
#   A줄(GPU 0)  (Phi-4 표집분 판정이 끝난 뒤) Kanana 로 교사 생성분 판정 → 합의 통과분만 SFT 기록으로 →
#               9B sft_v4 1에폭 → 평가 응답 생성 → Phi-4 · Kanana · Midm 재심사
# 비교: sft_v3(Qwen 통과 86,905건) vs sft_v4(같은 원천에서 세 모델 합의 통과분). 같은 GPU 에 큰 작업을 겹치지 않는다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4
TEACHER="$I/sft2.jsonl.screened.jsonl $I/gen.jsonl $K/gen_0.jsonl $K/gen_1.jsonl"

if [ "$1" = start ]; then
  # Qwen 판정본을 KCI 조각 이름으로도 둔다(행 순서 대조용 — kci_gen_N 과 같은 파일)
  for g in 0 1; do [ -e $J/gen_$g.jsonl ] || ln -s kci_gen_$g.jsonl $J/gen_$g.jsonl; done
  LANE=B CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_v14.sh > $R/logs_v14_B.txt 2>&1 < /dev/null &
  ( while ! grep -q V13P_DONE $R/logs_v13_phi4.txt 2>/dev/null; do sleep 30; done
    LANE=A CUDA_VISIBLE_DEVICES=0 bash scripts/run_v14.sh ) > $R/logs_v14_A.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

jd() {  # 꼬리표 모델 파일... — 끝나면 data/judged_<꼬리표>/<파일>.done
  local tag=$1 m=$2; shift 2
  for f in "$@"; do
    [ -f data/judged_$tag/$(basename $f).done ] && { echo "-- $tag $(basename $f) (있음, 건너뜀)"; continue; }
    python scripts/judge_all.py --llm $m --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}

if [ "$LANE" = B ]; then
  echo "== B1. Midm 으로 교사 생성분 판정 $(date)"
  jd midm $MID $TEACHER
  touch data/judged_midm/.teacher_done
  echo "== B2. sft_v3 평가 응답 재심사 $(date)"
  jd phi4 $PHI $I/eval2_sft_v3.jsonl $I/evalk_sft_v3.jsonl
  jd kanana $KAN $I/eval2_sft_v3.jsonl $I/evalk_sft_v3.jsonl
  jd midm $MID $I/eval2_sft_v3.jsonl $I/evalk_sft_v3.jsonl
  echo "V14B_DONE $(date)"
fi

if [ "$LANE" = A ]; then
  echo "== A1. Kanana 로 교사 생성분 판정 $(date)"
  jd kanana $KAN $TEACHER

  echo "== A2. 합의 통과 학습셋 (B줄을 기다린다) $(date)"
  for i in $(seq 1 1440); do [ -f data/judged_midm/.teacher_done ] && break; sleep 30; done
  python scripts/judge_all.py export-multi $I/sft4.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores     sft2.jsonl.screened.jsonl,gen.jsonl,gen_0.jsonl,gen_1.jsonl qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm && mv $I/sft4.tmp $I/sft4_train.jsonl
  wc -l $I/sft4_train.jsonl

  echo "== A3. 9B sft_v4 (합의 통과분, 1에폭) $(date)"
  if [ ! -s $CK/sft_v4/adapter_model.safetensors ] && [ -s $I/sft4_train.jsonl ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft4_train.jsonl --output $CK/sft_v4 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v4.txt 2>&1
    echo "SFT_V4_EXIT $?"; grep -E "LoRA 모듈|train_loss|saved|Error|OutOfMemory" $L/sft_v4.txt | tail -3 | cut -c1-300
  fi
  if [ -s $CK/sft_v4/adapter_model.safetensors ]; then
    for pre in eval2 evalk; do
      ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
      [ -s $I/${pre}_sft_v4.jsonl ] || python scripts/eval_sft.py run --model $S --adapter $CK/sft_v4 $extra --heldout $ho --out $I/${pre}_sft_v4.jsonl > $L/${pre}_sft_v4.txt 2>&1
      echo "-- ${pre}_sft_v4"; grep -E "^전체|어댑터|Error|거부" $L/${pre}_sft_v4.txt | tail -3
    done
    echo "== A4. 다른 판정 모델로 재심사 $(date)"
    jd phi4 $PHI $I/eval2_sft_v4.jsonl $I/evalk_sft_v4.jsonl
    jd kanana $KAN $I/eval2_sft_v4.jsonl $I/evalk_sft_v4.jsonl
    jd midm $MID $I/eval2_sft_v4.jsonl $I/evalk_sft_v4.jsonl
  fi
  echo "V14A_DONE $(date)"
fi
