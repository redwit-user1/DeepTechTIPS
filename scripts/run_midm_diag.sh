#!/usr/bin/env bash
# Midm 판정의 재현성 점검(연구노트 §52): 같은 파일을 같은 설정으로 다시(r2), 그리고 패딩 없이(배치 1) 판정한다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; . $R/venv/bin/activate; export CUDA_VISIBLE_DEVICES=0
M=K-intelligence/Midm-2.0-Base-Instruct; I=data/instr
python scripts/judge_all.py --llm $M --out-dir data/judged_midm_r2 $I/eval2_dpo_v7.jsonl $I/eval2_dpo_v6.jsonl > $R/logs/midm_r2.txt 2>&1; echo "R2_EXIT $? $(date +%H:%M)"
python scripts/judge_all.py --llm $M --batch 1 --out-dir data/judged_midm_b1 $I/eval2_dpo_v7.jsonl $I/eval2_dpo_v6.jsonl > $R/logs/midm_b1.txt 2>&1; echo "B1_EXIT $? $(date +%H:%M)"
echo MIDM_DIAG_DONE
