#!/usr/bin/env bash
# 2026-11-14 v10 — 1차년도 마일스톤 '통합 데모' · 27B 4-bit 보존율 (연구노트 §43)
#
#   CUDA_VISIBLE_DEVICES=1 nohup bash repo/scripts/run_v10.sh > logs_v10.txt 2>&1 &
#
#   1. 통합 실행   WF-A 10개 주제를 같은 검색 캐시로: 9B 기반 / 9B LoRA(dpo_v1) — SLM v1 + Gateway v1 + 에이전트 v1
#   2. 27B 4-bit   sft27_v1 을 4-bit(NF4)로 올려 평가 지시 2,294건 → bf16 대비 비율 · 메모리(RTX 4090 24GB 에 들어가는가)
# 같은 GPU 에 27B 판정 모델을 쓰는 작업을 겹치지 않는다(v9 의 OOM). 결과가 있으면 건너뛴다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B

wf() {  # 이름 [추가 인자...]
  local n=$1; shift
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 10 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $S --out $L/wf_a_$n.jsonl --search-cache $D/wf_a_search.json "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?)"; grep -E "어댑터|자동화율|거부|Error" $L/wf_a_$n.txt | tail -3; tail -11 $L/wf_a_$n.txt | head -10
  python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1
}
echo "== 1. 통합 실행: WF-A (9B) $(date)"
wf 9b_base
wf 9b_dpo_v1 --adapter $CK/dpo_v1

echo "== 2. 27B LoRA 4-bit $(date)"
if [ ! -s $I/eval2_sft27_v1_4bit.jsonl ]; then
  python scripts/eval_sft.py run --model $T --adapter $CK/sft27_v1 --load-4bit --heldout $I/heldout2.jsonl --out $I/eval2_sft27_v1_4bit.jsonl > $L/eval2_sft27_v1_4bit.txt 2>&1
  echo "-- sft27_v1_4bit (exit $?)"; grep -E "충실 생성률|판정기|어댑터|4-bit|Error|거부" $L/eval2_sft27_v1_4bit.txt | tail -9
fi
[ -s $I/eval2_sft27_v1_4bit.jsonl ] && { echo "[27B LoRA bf16 → 4-bit]"; python scripts/eval_sft.py compare $I/eval2_sft27_v1.jsonl $I/eval2_sft27_v1_4bit.jsonl; }
echo "V10_DONE $(date)"
