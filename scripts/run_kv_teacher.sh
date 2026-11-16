#!/usr/bin/env bash
# v17 B줄이 A줄을 기다리는 동안 빈 GPU 1 에서 v18 B1(교사 생성분 조건 추출 전용 판정)을 미리 돈다. 표시 파일은 run_v18.sh 와 같다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; . $R/venv/bin/activate; export CUDA_VISIBLE_DEVICES=1
F="data/judged/sft2.jsonl.screened.jsonl data/judged/gen.jsonl data/judged/gen_0.jsonl data/judged/gen_1.jsonl"
for spec in midm=K-intelligence/Midm-2.0-Base-Instruct qwen=Qwen/Qwen3.8-27B; do
  tag=${spec%%=*}; [ -f data/kv_$tag/.teacher.done ] && continue
  python scripts/judge_all.py --kv --batch 16 --llm ${spec#*=} --out-dir data/kv_$tag $F > $R/logs/kv_${tag}_teacher.txt 2>&1
  rc=$?; echo "KV_EXIT($tag teacher) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/kv_$tag/.teacher.done
done
echo KVT_DONE
