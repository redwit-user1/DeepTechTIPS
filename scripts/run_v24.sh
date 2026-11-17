#!/usr/bin/env bash
# 2026-11-17 v24 — 합의 증류 인코더의 섞는 비율 (연구노트 §54)
#
#   bash repo/scripts/run_v24.sh start      # v23 이 끝나면 GPU 1 에서 시작
#
# v5(기존 7,859 + 합의 60,000)는 실제 응답에서 Phi-4 와 잘 맞지만(AUC 0.70 → 0.81) 설계한 위반 평가셋에서 내려갔다(초록 0.970 → 0.912).
# 설계한 위반 예가 합의 예에 묻히지 않게 비율을 바꿔 본다. 두 평가를 같이 본다.
#   v6a  기존 ×1 + 합의 15,000        v6b  기존 ×3 + 합의 30,000        v6c  기존 ×3 + 합의 15,000
# 인코더 학습 · 라벨 평가는 시스템 파이썬(/usr/bin/python), 그 밖은 가상환경.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; D=$R/data
PY=$R/venv/bin/python
FILES=sft2.jsonl.screened.jsonl,gen.jsonl,gen_0.jsonl,gen_1.jsonl,note_sft_v2.jsonl,kci_sft_v2.jsonl,note_dpo_v4.jsonl,kci_dpo_v4.jsonl

if [ "$1" = start ]; then
  ( while ! grep -q V23_DONE $R/logs_v23.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v24.sh run ) > $R/logs_v24.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

EVALS=""; for m in base_hint dpo_v6 dpo27_v4; do EVALS="$EVALS data/judged_phi4/eval2_$m.jsonl data/judged_phi4/evalk_$m.jsonl"; done
for spec in v6a:15000:1 v6b:30000:3 v6c:15000:3; do
  IFS=: read v cap bx <<< "$spec"
  echo "== $v (기존 ×$bx + 합의 $cap) $(date)"
  T=$D/distill/train_$v.jsonl
  [ -s $T ] || { $PY scripts/judge_all.py distill-multi $T.tmp $D/distill/train_v3.jsonl $I/heldout2.jsonl,$K/heldout.jsonl $R/scores $FILES \
    qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm cap=$cap base_x=$bx | tail -1 && mv $T.tmp $T; }
  if [ -s $T ] && [ ! -s $CK/nli_distill_$v/config.json ]; then
    /usr/bin/python -m compliance_gateway.train.nli_finetune --data $T --labels 3 --base deberta-large \
      --split train --eval-split dev --output $CK/nli_distill_$v > $L/ft_$v.txt 2>&1
    echo "FT_EXIT($v) $?"; rm -rf $CK/nli_distill_$v/checkpoint-*
  fi
  [ -s $CK/nli_distill_$v/config.json ] || continue
  for set in kr_psg_eval kr_hard_eval kr_abs_eval; do
    /usr/bin/python -m compliance_gateway.eval.kr_nli $D/$set.jsonl --nli $CK/nli_distill_$v --dump $R/scores/${set}_distill_$v.jsonl > $L/${set}_distill_$v.txt 2>&1
    echo "-- $set $v: $(grep -aE '^AUC' $L/${set}_distill_$v.txt)"
  done
  $PY scripts/judge_all.py --enc-only --student $CK/nli_distill_$v --out-dir data/judged_enc_$v $EVALS > $L/enc_$v.txt 2>&1
  echo "ENC_EXIT($v) $? $(date +%H:%M)"
done
echo "V24_DONE $(date)"
