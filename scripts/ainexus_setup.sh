#!/usr/bin/env bash
# AI Nexus(H200) 세션 준비 — 세션을 새로 띄울 때마다 1회.
#
#   NAS 폴더 research-data-analysis 를 마운트한 세션에서:
#     bash repo/scripts/ainexus_setup.sh
#
# - venv 는 NAS 에 둔다(세션이 바뀌어도 남는다). 시스템 torch(NGC, constraint 고정)를 공유한다.
# - transformers 5.x 는 Qwen3.8 용. 인코더 학습·평가는 시스템 파이썬(transformers 4.55)을 쓴다.
#   datasets 업그레이드는 NGC constraint 와 충돌하므로 하지 않는다.
# - 모델 캐시는 /tmp(컨테이너 루트, 2.7TB). /home/work 는 49GB 라 Qwen3.8-27B(52GB)가 안 들어가고,
#   NAS 는 NFSv3 라 HF filelock 이 멈춘다. /tmp 는 세션 종료 시 지워지므로 매번 다시 받는다(약 1분).
set -e
R=~/research-data-analysis
[ -d $R/venv ] || python -m venv --system-site-packages $R/venv
. $R/venv/bin/activate
pip install -q -U "transformers>=5.3" accelerate "huggingface_hub>=1.3,<2" flash-linear-attention 2>&1 | grep -iE "^error" || true
python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"
export HF_HOME=/tmp/hf
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.8-27B')" && echo MODEL_READY
