#!/usr/bin/env bash
# 2026-11-22 v45 — 통합 실행 반복 더: 9B 의 '재작성만 어댑터' 대 '어댑터 끔' 을 두 번씩 더 (연구노트 §61)
#
#   bash repo/scripts/run_v45.sh start      # v44 K0 뒤 GPU 0
#
# v44 까지: 9B 통과 수가 어댑터 끔 2 · 3 · 5 · 5, 재작성만 어댑터 8 · 8 · 5. 세 번째에서 겹쳤다. 다섯 번씩으로 늘린다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B

if [ "$1" = start ]; then
  ( while ! grep -q V44_K0_DONE $R/logs_v44_0.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_v45.sh run ) > $R/logs_v45.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  [ -s $L/wf_a_$n.jsonl ] && [ "$(wc -l < $L/wf_a_$n.jsonl)" -ge 10 ] && { echo "-- wf_a_$n (있음, 건너뜀)"; return; }
  python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --search-cache $D/wf_a_search.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
  echo "-- wf_a_$n (exit $?) $(date +%H:%M)"; grep -aE "거부|Error" $L/wf_a_$n.txt | tail -3 | cut -c1-200; tail -1 $L/wf_a_$n.txt | cut -c1-200
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl 2>&1 | tail -1 | cut -c1-200
}
echo "== 9B 반복 $(date)"
for r in r4 r5; do
  wf f9b_v11_off_$r $S --adapter $CK/dpo_v11 --adapter-on none
  wf f9b_v11_regen_$r $S --adapter $CK/dpo_v11 --adapter-on regen
done
echo "V45_DONE $(date)"
