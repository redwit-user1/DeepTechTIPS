#!/usr/bin/env bash
# 항목 줄 전용 판정(연구노트 §48) — 조건 추출 줄만 다시 묻는다. 표집 중인 GPU 1 에 얹는다 — 배치 32 는 103GB 까지 잡아 표집을 위협했다. 배치 4 로.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; . $R/venv/bin/activate; export CUDA_VISIBLE_DEVICES=1
F=""; for m in base_hint sft_v2 sft_v3 sft_v4 dpo_v4; do F="$F data/judged_phi4/eval2_$m.jsonl data/judged_phi4/evalk_$m.jsonl"; done
for spec in phi4=microsoft/phi-4 kanana=kakaocorp/kanana-1.5-8b-instruct-2505 midm=K-intelligence/Midm-2.0-Base-Instruct qwen=Qwen/Qwen3.8-27B; do
  tag=${spec%%=*}; python scripts/judge_all.py --kv --batch 4 --llm ${spec#*=} --out-dir data/kv_$tag $F > $R/logs/kv_$tag.txt 2>&1; echo "KV_EXIT($tag) $? $(date +%H:%M)"
done
echo KV_ALL_DONE
