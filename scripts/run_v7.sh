#!/usr/bin/env bash
# 2026-11-13 v7 대기열 — KCI 국문 초록(이공·의약·농수해양)으로 Instruction 생성 → 전수 판정 (연구노트 §41)
#
#   GPU 2장 세션에서:  bash repo/scripts/run_v7.sh start
#
# 긴 원문(초록 중앙값 약 1,000자) · 공개 자료. 두 GPU 가 초록을 반씩 나눠 생성하고, 끝나는 대로 문장별 원점수로 판정한다.
# 생성·판정 모두 중단 후 재개된다(생성은 문서 단위, 판정은 행 단위) — 세션이 회수되면 같은 명령을 다시 실행.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; K=data/kci; J=data/judged
. $R/venv/bin/activate

if [ "$1" = start ]; then
  python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B')" > $L/dl_v7.txt 2>&1 && echo MODELS_READY || { echo MODELS_FAIL; exit 1; }
  [ -s $K/passages_0.jsonl ] || python scripts/build_kci_passages.py --corpus $R/data/corpus/kci --out $K --shards 2 || exit 1
  wc -l $K/passages_*.jsonl
  for g in 0 1; do
    SHARD=$g CUDA_VISIBLE_DEVICES=$g nohup bash scripts/run_v7.sh > $R/logs_v7_$g.txt 2>&1 < /dev/null &
  done
  echo STARTED; exit 0
fi

s=$SHARD
echo "== 생성 (조각 $s) $(date)"
python scripts/build_instructions.py gen --passages $K/passages_$s.jsonl --out $K/gen_$s.jsonl --kind abstract \
  --batch 16 --max-new-tokens 1800 > $L/kci_gen_$s.txt 2>&1
echo "GEN_EXIT $?"; wc -l $K/gen_$s.jsonl
echo "== 전수 판정 (조각 $s) $(date)"
[ -s $J/kci_gen_$s.jsonl ] || python scripts/judge_all.py --out-dir $J $K/gen_$s.jsonl > $L/kci_judge_$s.txt 2>&1
echo "JUDGE_EXIT $?"; mv $J/gen_$s.jsonl $J/kci_gen_$s.jsonl 2>/dev/null
python scripts/judge_all.py report $J/kci_gen_$s.jsonl
echo "V7_DONE_$s $(date)"
