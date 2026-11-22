#!/usr/bin/env bash
# 2026-11-22 v44 — 통합 실행 반복: 같은 구성을 다시 돌리면 얼마나 달라지나 (연구노트 §61)
#
#   bash repo/scripts/run_v44.sh start      # GPU 0 바로, GPU 1 은 v43 뒤
#
# v41 ~ v43: 같은 기반 모델이 같은 프롬프트로 초안을 쓰는 구성끼리도 주제별 통과/차단이 다르다
# (9B 재작성 없이 통과: 기반 2, 어댑터 끔 2, 재작성만 어댑터 5, 초안만 기반 5 — 뒤의 둘도 초안은 기반이 쓴다).
# 탐욕 디코딩인데도 실행마다 초안이 달라진다. 10개 주제 1회로는 통과 수를 비교할 수 없다 → 같은 구성을 두 번 더.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B

if [ "$1" = start ]; then
  ( CUDA_VISIBLE_DEVICES=0 bash scripts/run_v44.sh k0 ) > $R/logs_v44_0.txt 2>&1 < /dev/null &
  ( while ! grep -q V43_DONE $R/logs_v43.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v44.sh k1 ) > $R/logs_v44_1.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 10 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --search-cache $D/wf_a_search.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?) $(date +%H:%M)"; grep -aE "거부|Error" $L/wf_a_$n.txt | tail -3 | cut -c1-200; tail -1 $L/wf_a_$n.txt | cut -c1-200
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1 | cut -c1-200
}
if [ "$1" = k0 ]; then
  echo "== K0: 9B 반복 $(date)"
  for r in r2 r3; do
    wf f9b_v11_off_$r $S --adapter $CK/dpo_v11 --adapter-on none
    wf f9b_v11_regen_$r $S --adapter $CK/dpo_v11 --adapter-on regen
    wf f9b_dpo_v11_$r $S --adapter $CK/dpo_v11
  done
  echo "V44_K0_DONE $(date)"
fi
if [ "$1" = k1 ]; then
  echo "== K1: 27B 반복 $(date)"
  for r in r2 r3; do
    wf f27b_v7_off_$r $T --adapter $CK/dpo27_v7 --adapter-on none
    wf f27b_v7_regen_$r $T --adapter $CK/dpo27_v7 --adapter-on regen
  done
  echo "V44_K1_DONE $(date)"
fi
