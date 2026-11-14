#!/usr/bin/env bash
# 2026-11-14 v13 — 여러 판정 모델의 합의로 고른 선호쌍으로 DPO, 평가는 쓰지 않은 판정 모델로 (연구노트 §44)
#
#   bash repo/scripts/run_v13.sh start
#
# v12: DPO 의 이득(Qwen 판정 +3.0%p)이 계열이 다른 판정 모델 3종에서는 사라졌다(−0.8 ~ +0.8%p). 선호쌍을 Qwen 하나로
# 골랐기 때문에 Qwen 에 맞추는 법을 배운 것이다. 선호쌍을 Qwen · Kanana · Midm 세 모델의 합의로 고르고,
# 평가는 선호쌍 선택에 **쓰지 않은** Phi-4(라벨 평가셋 AUC 가 가장 높다)로 한다.
#   A줄(GPU 0)  Kanana 로 학생 표집분(연구노트 76,676 · KCI 35,988) 판정
#   B줄(GPU 1)  Midm 으로 같은 표집분 판정
#   이어서(GPU 0) 합의 선호쌍 → 9B dpo_v3(sft_v2 에서 이어서) → 평가 응답 생성 → Phi-4 · Kanana · Midm 재심사
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; SM=data/sampled
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('$KAN', '$MID', '$PHI', '$S', 'Qwen/Qwen3.8-27B')]" > $L/dl_v13.txt 2>&1 && echo MODELS_READY || echo MODELS_FAIL
  LANE=A CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_v13.sh > $R/logs_v13_A.txt 2>&1 < /dev/null &
  LANE=B CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_v13.sh > $R/logs_v13_B.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

jd() {  # 꼬리표 모델 파일... — 끝나면 data/judged_<꼬리표>/<파일>.done
  local tag=$1 m=$2; shift 2
  for f in "$@"; do
    [ -f data/judged_$tag/$(basename $f).done ] && { echo "-- $tag $(basename $f) (있음, 건너뜀)"; continue; }
    python scripts/judge_all.py --llm $m --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}

if [ "$LANE" = B ]; then
  echo "== B. Midm 으로 표집분 판정 $(date)"
  jd midm $MID $SM/note_sft_v2.jsonl $SM/kci_sft_v2.jsonl
  echo "V13B_DONE $(date)"
fi

if [ "$LANE" = A ]; then
  echo "== A1. Kanana 로 표집분 판정 $(date)"
  jd kanana $KAN $SM/note_sft_v2.jsonl $SM/kci_sft_v2.jsonl

  echo "== A2. 다중 판정 합의 선호쌍 (B줄을 기다린다) $(date)"
  for i in $(seq 1 1440); do [ -f data/judged_midm/kci_sft_v2.jsonl.done ] && break; sleep 30; done
  python scripts/judge_all.py dpo-multi $I/dpo4.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores note_sft_v2.jsonl,kci_sft_v2.jsonl \
    qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm && mv $I/dpo4.tmp $I/dpo4.jsonl
  wc -l $I/dpo4.jsonl

  echo "== A3. 9B dpo_v3 (sft_v2 에서 이어서, 1에폭) $(date)"
  if [ ! -s $CK/dpo_v3/adapter_model.safetensors ] && [ -s $I/dpo4.jsonl ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v2 --dataset $I/dpo4.jsonl --output $CK/dpo_v3 \
      --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v3.txt 2>&1
    echo "DPO_V3_EXIT $?"; grep -E "accepted|train_loss|saved|Error|Traceback" $L/dpo_v3.txt | tail -4 | cut -c1-300
  fi
  if [ -s $CK/dpo_v3/adapter_model.safetensors ]; then
    for pre in eval2 evalk; do
      ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
      [ -s $I/${pre}_dpo_v3.jsonl ] || python scripts/eval_sft.py run --model $S --adapter $CK/dpo_v3 $extra --heldout $ho --out $I/${pre}_dpo_v3.jsonl > $L/${pre}_dpo_v3.txt 2>&1
      echo "-- ${pre}_dpo_v3"; grep -E "^전체|어댑터|Error|거부" $L/${pre}_dpo_v3.txt | tail -3
      echo "[Qwen 캐스케이드: sft_v2 → dpo_v3]"; python scripts/eval_sft.py compare $I/${pre}_sft_v2.jsonl $I/${pre}_dpo_v3.jsonl | head -1
    done
    echo "== A4. 다른 판정 모델로 재심사 $(date)"
    jd phi4 $PHI $I/eval2_dpo_v3.jsonl $I/evalk_dpo_v3.jsonl
    jd kanana $KAN $I/eval2_dpo_v3.jsonl $I/evalk_dpo_v3.jsonl
    jd midm $MID $I/eval2_dpo_v3.jsonl $I/evalk_dpo_v3.jsonl
  fi
  echo "V13A_DONE $(date)"
fi
