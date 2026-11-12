#!/usr/bin/env bash
# AI Nexus 세션에서 코퍼스 수집을 백그라운드로 띄운다. GPU 불필요(CPU·네트워크만).
#
#   bash ~/research-data-analysis/harvest_repo/scripts/run_harvest.sh        # 전체 기동
#   bash ~/research-data-analysis/harvest_repo/scripts/run_harvest.sh kci    # 한 작업만(포그라운드)
#
# 다시 돌려도 안전하다: 도는 작업은 건너뛰고, 멈춘 작업은 끝난 곳부터 이어받는다.
# 진행 확인: tail -f ~/research-data-analysis/logs/harvest_*.log
set -u
R=~/research-data-analysis
OUT=$R/data/corpus
LOG=$R/logs
export PYTHONPATH=$R/harvest_repo PYTHONUNBUFFERED=1
cd "$R/harvest_repo"
mkdir -p "$OUT" "$LOG"

KR='AFF:"Korea" AND OPEN_ACCESS:y AND HAS_FT:y'
PHARMA='("good laboratory practice" OR "good manufacturing practice" OR "data integrity" OR "good clinical practice" OR "ALCOA" OR "21 CFR" OR "preclinical" OR "toxicity study" OR "pharmacokinetics" OR "clinical trial")'

case "${1:-all}" in
  kci)    # ① KCI 국문 초록 전체 — 월 단위 이어받기, 1초 간격
    python scripts/harvest.py kci --out "$OUT/kci" ;;
  epmc)   # ② 한국 소속 OA 메타·초록 전체 → 제약·바이오 CC BY/CC0 전문
    python scripts/harvest.py epmc --out "$OUT/epmc_kr_meta" --query "$KR" &&
    python scripts/harvest.py epmc --out "$OUT/epmc_kr_pharma" --fulltext \
      --query "$KR AND (LICENSE:\"cc by\" OR LICENSE:\"cc0\") AND $PHARMA" ;;
  law)    # ③ 행정규칙(HF, CC BY 4.0). /tmp 에 받고 NAS 로 복사 — NAS(NFSv3)에선 HF filelock 이 멈춘다
    [ -f "$OUT/law_admin_rules/.done" ] && { echo "law: 이미 완료"; exit 0; }
    # 시스템 파이썬(hf_hub 0.36). NAS venv 의 hf_hub 2.x+httpx2 는 다운로드 중 TypeError(2026-11-10 실측)
    HF_HOME=/tmp/hf python -c "from huggingface_hub import snapshot_download as s; s('mosshoon/korean-admin-rules', repo_type='dataset', local_dir='/tmp/law')" &&
    mkdir -p "$OUT/law_admin_rules" && cp -r /tmp/law/. "$OUT/law_admin_rules/" && touch "$OUT/law_admin_rules/.done" && echo "law: 완료" ;;
  mfds)   # ③ 식약처 민원인안내서 원문(PDF/HWP) — 2초 간격
    python scripts/harvest.py mfds --out "$OUT/mfds_guide" ;;
  all)
    for j in kci epmc law mfds; do
      if pgrep -f "run_harvest.sh $j" >/dev/null; then echo "skip $j (실행 중)"; continue; fi
      nohup bash "$0" "$j" >>"$LOG/harvest_$j.log" 2>&1 &
      echo "start $j (pid $!)"
    done ;;
  *) echo "usage: $0 [kci|epmc|law|mfds]"; exit 2 ;;
esac
