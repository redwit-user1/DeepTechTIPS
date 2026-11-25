#!/usr/bin/env bash
# 2026-11-25 v46 — 통합 실행을 40개 주제로 (연구노트 §62)
#
#   bash repo/scripts/run_v46.sh start      # GPU 0
#
# §61: 10개 주제로는 같은 구성의 통과 수가 2 ~ 6 으로 흔들려 구성 사이의 차이를 말하기 어려웠다.
# 주제를 40개(기존 10 + 공학 · ICT · 바이오 · 연구데이터 30, scripts/wf_a_topics_ext.json)로 늘려 두 구성을 다시 잰다.
#   off   : 어댑터를 올리고 세 단계 모두 끔(= 기반 모델)
#   regen : 갭 분석 · 초안 = 기반, 재작성 = 어댑터 (§61 의 권장 구성)
# 검색 결과는 고정(data/wf_a_search_40.json, 기존 10개는 v41 ~ v45 와 같은 캐시).
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B

if [ "$1" = start ]; then
  ( CUDA_VISIBLE_DEVICES=0 bash scripts/run_v46.sh run ) > $R/logs_v46.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 40 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --topics scripts/wf_a_topics_ext.json \
    --search-cache $D/wf_a_search_40.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?) $(date +%m-%d_%H:%M)"; grep -aE "거부|Error" $L/wf_a_$n.txt | tail -3 | cut -c1-200; tail -1 $L/wf_a_$n.txt | cut -c1-200
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1 | cut -c1-200
}
echo "== 통합 실행 40개 주제 $(date)"
for r in a b; do
  wf w40_9b_off_$r $S --adapter $CK/dpo_v11 --adapter-on none
  wf w40_9b_regen_$r $S --adapter $CK/dpo_v11 --adapter-on regen
  wf w40_27b_off_$r $T --adapter $CK/dpo27_v7 --adapter-on none
  wf w40_27b_regen_$r $T --adapter $CK/dpo27_v7 --adapter-on regen
done
echo "V46_DONE $(date)"
