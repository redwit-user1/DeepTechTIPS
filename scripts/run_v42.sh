#!/usr/bin/env bash
# 2026-11-22 v42 — 통합 실행: 단계별로 어댑터를 켜고 끈다 (연구노트 §61)
#
#   bash repo/scripts/run_v42.sh start      # GPU 1
#
# v41: 최종 어댑터를 WF-A 의 세 호출(갭 분석 · 초안 · 재작성) 모두에 쓰면 제안서 형식이 무너진다
# (9B dpo_v11: 통과 3 → 1, 인용 DOI 84 → 8, 초안 길이 2,892 → 773자. 27B dpo27_v7: 통과 5 → 5, 인용 118 → 67).
# 어댑터는 짧은 근거 충실 문단을 쓰도록 학습됐다. 구조를 잡는 초안은 기반 모델이, 근거에 맞춰 고쳐 쓰는 단계만 어댑터가 하면?
#   regen      : 갭 분석 · 초안 = 기반, 재작성 = 어댑터
#   gap,regen  : 초안만 기반
# 10개 주제 1회 — 통과 수 한두 개는 잡음. 인용 수 · 초안 길이 · 형식이 유지되는지를 본다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B

if [ "$1" = start ]; then
  ( CUDA_VISIBLE_DEVICES=1 bash scripts/run_v42.sh run ) > $R/logs_v42.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 10 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --search-cache $D/wf_a_search.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?) $(date +%H:%M)"; grep -aE "어댑터|거부|Error" $L/wf_a_$n.txt | tail -3 | cut -c1-200; tail -11 $L/wf_a_$n.txt | cut -c1-200
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1 | cut -c1-200
}
echo "== 통합 실행, 단계별 어댑터 $(date)"
wf f9b_v11_regen $S --adapter $CK/dpo_v11 --adapter-on regen
wf f27b_v7_regen $T --adapter $CK/dpo27_v7 --adapter-on regen
wf f9b_v11_gapregen $S --adapter $CK/dpo_v11 --adapter-on gap,regen
wf f27b_v7_gapregen $T --adapter $CK/dpo27_v7 --adapter-on gap,regen
echo "V42_DONE $(date)"
