#!/usr/bin/env bash
# 2026-11-22 v41 — 1차년도 마일스톤 '통합 실행'을 최종 모델로 다시 (연구노트 §61)
#
#   bash repo/scripts/run_v41.sh start      # v39 가 끝나면 GPU 1 에서 시작
#
# §43 의 통합 실행(SLM v1 + Gateway v1 + 에이전트 v1, WF-A 10개 주제 · 같은 검색 캐시)은 초기 어댑터(dpo_v1)로 했다:
# 통과 5 · 차단 5, 인용 DOI 가 117 → 27 로 줄었다(단락 과제로만 학습해 제안서 초안이 짧아짐). 최종 모델로 같은 조건에서 다시 잰다.
#   9B: 기반 / dpo_v11        27B: 기반 / dpo27_v7
# 10개 주제 1회라 통과 수 한두 개 차이는 잡음이다(§38). 보는 것은 파이프라인이 끝까지 도는가, 인용 수 · 출처 정확률이 어떻게 달라지는가.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B

if [ "$1" = start ]; then
  ( while ! grep -q V39_DONE $R/logs_v39.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v41.sh run ) > $R/logs_v41.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 10 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --search-cache $D/wf_a_search.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?) $(date +%H:%M)"; grep -aE "어댑터|자동화율|거부|Error" $L/wf_a_$n.txt | tail -3 | cut -c1-200; tail -11 $L/wf_a_$n.txt | head -10 | cut -c1-200
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1 | cut -c1-200
}
echo "== 통합 실행: WF-A 10개 주제 $(date)"
wf f9b_base $S
wf f9b_dpo_v11 $S --adapter $CK/dpo_v11
wf f27b_base $T
wf f27b_dpo27_v7 $T --adapter $CK/dpo27_v7
echo "V41_DONE $(date)"
