#!/usr/bin/env bash
# 2026-11-17 Midm 판정 다시 — 날짜를 고정하고 (연구노트 §52)
#
# Midm 2.0 의 대화 틀이 시스템 프롬프트에 오늘 날짜를 넣어, 판정한 날이 다르면 같은 문장의 점수가 달랐다.
# 임계값을 보정한 11-14 로 날짜를 고정하고(llm_judge.JUDGE_DATE) 11-15 이후에 판정한 평가 응답을 다시 판정한다.
# 결과는 data/judged_midm_d14/ · data/kv_midm_d14/ 에 따로 둔다(원래 결과는 남긴다). sft_v2 는 대조용 — 11-14 판정과 같아야 한다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; . $R/venv/bin/activate; export CUDA_VISIBLE_DEVICES=${GPU:-0}
M=K-intelligence/Midm-2.0-Base-Instruct; I=data/instr; L=$R/logs
OLD="sft_v2 sft_v3 sft_v4 dpo_v4 base27_hint sft27_v1 dpo27_v2 sft27_v2 dpo27_v3 dpo_v5 dpo27_v4 dpo_v4nc dpo_v4_4bit sft_v5 dpo_v6 dpo_v7"
ALL="base_hint dpo_v3 $OLD"
for n in $OLD; do
  for pre in eval2 evalk; do
    f=$I/${pre}_$n.jsonl; [ -s $f ] || continue
    [ -f data/judged_midm_d14/${pre}_$n.jsonl.done ] && continue
    python scripts/judge_all.py --llm $M --out-dir data/judged_midm_d14 $f > $L/midm_d14_${pre}_$n.txt 2>&1
    rc=$?; echo "REJUDGE_EXIT(${pre}_$n) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_midm_d14/${pre}_$n.jsonl.done
  done
done
F=""; for n in $ALL; do for pre in eval2 evalk; do [ -s data/judged_phi4/${pre}_$n.jsonl ] && F="$F data/judged_phi4/${pre}_$n.jsonl"; done; done
python scripts/judge_all.py --kv --batch 16 --llm $M --out-dir data/kv_midm_d14 $F > $L/kv_midm_d14.txt 2>&1; echo "KV_EXIT $? $(date +%H:%M)"
echo MIDM_REJUDGE_DONE
