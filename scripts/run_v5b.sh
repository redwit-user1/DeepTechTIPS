#!/usr/bin/env bash
# 2026-11-12 v5b: 프롬프트 기준선 — 기반모델에 형식·충실 안내 한 줄만 붙였을 때 (연구노트 §39)
#   nohup bash repo/scripts/run_v5b.sh > logs_v5b.txt 2>&1 &
# v5 에서 기반 48.5% → sft_v1 89.5% 였지만 기반 탈락의 상당수가 형식(마크다운·머리말·해설)이었다.
# 학습 없이 지시 한 줄로 얻는 몫을 빼야 SFT 의 순효과가 나온다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; I=data/instr
. $R/venv/bin/activate
echo "== 기반 + 안내 한 줄 $(date)"
python scripts/eval_sft.py run --model Qwen/Qwen3.5-9B --style-hint --heldout $I/heldout2.jsonl --out $I/eval2_base_hint.jsonl > $L/eval2_base_hint.txt 2>&1
echo "exit $?"; grep -E "충실 생성률|판정기|Error|거부" $L/eval2_base_hint.txt | tail -8
echo "[기반 → 기반+안내]";    python scripts/eval_sft.py compare $I/eval2_base.jsonl $I/eval2_base_hint.jsonl
echo "[기반+안내 → sft_v1]";  python scripts/eval_sft.py compare $I/eval2_base_hint.jsonl $I/eval2_sft_v1.jsonl
python scripts/eval_sft.py rejudge --in $I/eval2_base_hint.jsonl --out $I/eval2_base_hint_llm.jsonl --limit 400 > $L/eval2_base_hint_llm.txt 2>&1
grep -E "^전체|일치|판정기" $L/eval2_base_hint_llm.txt
echo "[LLM 단독 판정: 기반+안내 → sft_v1]"; python scripts/eval_sft.py compare $I/eval2_base_hint_llm.jsonl $I/eval2_sft_v1_llm.jsonl
echo "V5B_DONE $(date)"
