#!/usr/bin/env bash
# EEVE(선별에 쓰지 않은 두 번째 판정 모델)로 나머지 평가 응답도 모두 판정한다 (연구노트 §55). Phi-4 로 판정한 모델 전부.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1; L=$R/logs; I=data/instr
M=yanolja/EEVE-Korean-Instruct-10.8B-v1.0
for round in 1 2; do      # 2회차: 도는 중에 새로 생긴 평가 응답(v25)
  for f in data/judged_phi4/eval2_*.jsonl data/judged_phi4/evalk_*.jsonl; do
    b=$(basename $f); [ -s $I/$b ] || continue
    [ -f data/judged_eeve/$b.done ] && continue
    $R/venv/bin/python scripts/judge_all.py --llm $M --out-dir data/judged_eeve $I/$b > $L/judge_eeve_${b%.jsonl}.txt 2>&1
    rc=$?; echo "JUDGE_EXIT(eeve $b) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_eeve/$b.done
  done
done
echo EEVE_ALL_DONE
