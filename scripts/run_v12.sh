#!/usr/bin/env bash
# 2026-11-14 v12 — 다른 계열 판정 모델로 재심사 (연구노트 §44)
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash repo/scripts/run_v12.sh > logs_v12.txt 2>&1 &
#
# 지금까지의 판정은 전부 Qwen3.8-27B 와 그 증류 인코더다. 학습 데이터를 만든 교사도, DPO 선호를 정한 것도 같은 모델이라
# 'DPO 가 판정기에 맞추는 법을 배웠다'는 설명을 배제할 수 없다. 계열이 다른 공개 모델 3종으로 같은 응답을 다시 심사한다.
#   1. 판정 모델 자체 검증: 라벨이 있는 평가셋 3종(연구노트 단락 · 어려운 셋 · 외부 초록)
#   2. 평가 응답 재심사: 9B 기반+안내 / sft_v2 / dpo_v1 / dpo_v2 — 연구노트 2,294건과 긴 원문 1,603건
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; I=data/instr; D=$R/data
. $R/venv/bin/activate
# Mistral-Small-3.2 는 AutoTokenizer 로는 대화 틀·패딩 토큰이 없어 제외했다(mistral-common 필요)
for spec in "midm K-intelligence/Midm-2.0-Base-Instruct" "kanana kakaocorp/kanana-1.5-8b-instruct-2505" "phi4 microsoft/phi-4"; do
  set -- $spec; tag=$1; M=$2
  [ -f data/judged_$tag/.done ] && { echo "== $tag (있음, 건너뜀)"; continue; }
  echo "== $tag ($M) $(date)"
  python -c "from huggingface_hub import snapshot_download as s; s('$M')" > $L/dl_$tag.txt 2>&1 && echo MODEL_READY || { echo MODEL_FAIL; tail -2 $L/dl_$tag.txt | cut -c1-300; continue; }
  for s in kr_psg_eval kr_hard_eval kr_abs_eval; do
    python -m compliance_gateway.eval.kr_nli $D/$s.jsonl --llm $M --dump $R/scores/${s}_$tag.jsonl > $L/${s}_$tag.txt 2>&1
    echo "-- $s (exit $?)"; grep -E "^AUC|운영점|Error" $L/${s}_$tag.txt | cut -c1-300
  done
  python scripts/judge_all.py --llm $M --out-dir data/judged_$tag \
    $I/eval2_base_hint.jsonl $I/eval2_sft_v2.jsonl $I/eval2_dpo_v1.jsonl $I/eval2_dpo_v2.jsonl \
    $I/evalk_base_hint.jsonl $I/evalk_sft_v2.jsonl $I/evalk_dpo_v2.jsonl > $L/judge_$tag.txt 2>&1
  rc=$?; echo "JUDGE_EXIT($tag) $rc"; tail -1 $L/judge_$tag.txt | cut -c1-200; [ $rc = 0 ] && touch data/judged_$tag/.done
done
echo "V12_DONE $(date)"
