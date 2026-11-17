#!/usr/bin/env bash
# 학생 표집분(선호쌍의 재료)을 EEVE 로도 판정한다 — 네 모델 합의 선호쌍을 만들기 위한 준비 (연구노트 §56). GPU 1.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1; L=$R/logs; SM=data/sampled
M=yanolja/EEVE-Korean-Instruct-10.8B-v1.0
for f in $SM/note_sft_v2.jsonl $SM/kci_sft_v2.jsonl $SM/note_qa_sft_v5.jsonl $SM/kci_qa_sft_v5.jsonl; do
  b=$(basename $f); [ -f data/judged_eeve/$b.done ] && continue
  $R/venv/bin/python scripts/judge_all.py --llm $M --out-dir data/judged_eeve $f > $L/judge_eeve_${b%.jsonl}.txt 2>&1
  rc=$?; echo "JUDGE_EXIT(eeve $b) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_eeve/$b.done
done
echo EEVE_SAMPLES_DONE
