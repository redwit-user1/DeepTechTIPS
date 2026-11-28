#!/usr/bin/env bash
# 2026-11-28 v50 — 27B 에도 제안서 예를 넣어 다시 학습 (연구노트 §67 다음)
#
#   bash repo/scripts/run_v50.sh check      # 시작 전 점검만(모델, 데이터, 어댑터, GPU). 아무것도 학습하지 않음
#   bash repo/scripts/run_v50.sh start      # 점검 통과 시 GPU 0: 씨앗 42, GPU 1: 씨앗 1
#
# 9B 에서는 제안서 예 194개(× 5)를 넣은 dpo_v13 이 학습 두 번, 실행 세 번에서 재현됨(§67).
# 27B 도 같은 데이터(sft10 = sft8 + 제안서 예 × 5)와 같은 DPO 데이터(dpo7)로 다시 학습해 본다.
#   sft27_v6 → dpo27_v8 (씨앗 42),  sft27_v6_s1 → dpo27_v8_s1 (씨앗 1)
# 27B 학습 설정은 dpo27_v7 과 같음(run_v35.sh: SFT 기본값, DPO lr 2e-5 batch 1 grad-accum 16). 데이터만 sft8 → sft10.
# 씨앗은 학습 순서와 Trainer 초기화에 들어감. Unsloth LoRA 초기화 씨앗은 따로 넘기지 않음(9B, 27B 의 이전 씨앗 반복과 같은 조건).
# 비교 기준: 27B 기반만 24, 32 / 재작성만 dpo27_v7 27, 23 (§65). 세 단계 모두 dpo27_v7 은 GPU 0 마지막에 잰다.
#
# 감사(§66)와 실행 전 검토(Codex, Grok) 반영: 단계마다 종료 코드와 산출물(어댑터 파일, 줄 수)을 함께 확인하고, 어긋나면 그 레인은 멈춤.
R=~/research-data-analysis; cd $R/repo || exit 1; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data; I=data/instr; K=data/kci
. $R/venv/bin/activate || exit 1
T=Qwen/Qwen3.8-27B; PHI=microsoft/phi-4

lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
fail() { echo "V50${LANE}_FAIL $* $(date +%m-%d_%H:%M)"; exit 1; }
have_adapter() { [ -s $CK/$1/adapter_model.safetensors ] && [ -s $CK/$1/adapter_config.json ]; }

if [ "$1" = check ] || [ "$1" = start ]; then
  ok=1
  if pgrep -f "run_v50.sh run" > /dev/null; then echo "이미 v50 레인이 돌고 있음"; ok=0; fi
  # 모델: 실행에 쓰는 venv python 으로 설정, 토크나이저, 모든 가중치 조각을 확인
  python - "$T" "$PHI" <<'PY' || ok=0
import json, sys, os
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
bad = 0
for m in sys.argv[1:]:
    try:
        p = snapshot_download(m, local_files_only=True)
        AutoConfig.from_pretrained(p); AutoTokenizer.from_pretrained(p)
        idx = os.path.join(p, "model.safetensors.index.json")
        shards = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
        miss = [s for s in shards if not os.path.exists(os.path.join(p, s)) or os.path.getsize(os.path.join(p, s)) == 0]
        print(f"모델 {m}: 조각 {len(shards)}개, 빠짐 {len(miss)}")
        bad += bool(miss)
    except Exception as e:
        print(f"모델 {m}: 실패 {type(e).__name__} {e}"[:300]); bad += 1
# 판정용 증류 인코더(judge_all 기본값)
p = "checkpoints/nli_distill"
try:
    AutoConfig.from_pretrained(p); AutoTokenizer.from_pretrained(p)
    w = [f for f in os.listdir(p) if f.endswith((".safetensors", ".bin")) and os.path.getsize(os.path.join(p, f)) > 0]
    print(f"nli_distill: 가중치 {w}"); bad += not w
except Exception as e:
    print(f"nli_distill: 실패 {e}"[:300]); bad += 1
import torch
print("CUDA", torch.cuda.device_count()); bad += torch.cuda.device_count() < 2
sys.exit(1 if bad else 0)
PY
  n8=$(lines $I/sft8_train.jsonl); n10=$(lines $I/sft10_train.jsonl); np=$(lines $I/prop_train.jsonl)
  echo "sft8 $n8, 제안서 예 $np, sft10 $n10"
  [ "$n8" -eq 79520 ] && [ "$np" -eq 970 ] && [ "$n10" -eq 80490 ] || { echo "학습셋 줄 수가 v48 과 다름"; ok=0; }
  cmp -s <(cat $I/sft8_train.jsonl $I/prop_train.jsonl) $I/sft10_train.jsonl || { echo "sft10 이 sft8 + 제안서 예와 내용이 다름"; ok=0; }
  [ "$(lines $I/dpo7.jsonl)" -eq 3506 ] || { echo "dpo7 줄 수 다름"; ok=0; }
  [ "$(lines $I/heldout2.jsonl)" -eq 2294 ] && [ "$(lines $K/heldout.jsonl)" -eq 1603 ] || { echo "평가셋 줄 수 다름"; ok=0; }
  for f in $D/wf_a_search_40.json scripts/wf_a_topics_ext.json; do [ -s $f ] || { echo "없음 $f"; ok=0; }; done
  have_adapter dpo27_v7 || { echo "dpo27_v7 없음(GPU 0 마지막 비교용)"; ok=0; }
  for c in sft27_v6 dpo27_v8 sft27_v6_s1 dpo27_v8_s1; do [ -e $CK/$c ] && { echo "이미 있음 $CK/$c (지우거나 이름을 바꿀 것)"; ok=0; }; done
  for f in $L/wf_a_w40_27b_full_v8.jsonl $L/wf_a_w40_27b_full_v8s1.jsonl $L/wf_a_w40_27b_full_v7.jsonl $I/eval2_dpo27_v8.jsonl $I/eval2_dpo27_v8_s1.jsonl; do
    [ -e $f ] && { echo "이미 있음 $f"; ok=0; }
  done
  grep -q "지정한 어댑터가 없으면 멈춤" compliance_gateway/train/dpo.py || { echo "dpo.py 가 어댑터 누락 시 멈추는 판이 아님(repo 동기화 필요)"; ok=0; }
  touch $CK/.v50_write_test && rm -f $CK/.v50_write_test || { echo "체크포인트 폴더에 쓸 수 없음"; ok=0; }
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
  df -h /tmp $R | tail -2
  [ $ok = 1 ] && echo CHECK_OK || { echo CHECK_FAIL; exit 1; }
  [ "$1" = check ] && exit 0
  ( LANE=A CUDA_VISIBLE_DEVICES=0 bash scripts/run_v50.sh run; echo "LANE_A_EXIT $?" ) > $R/logs_v50_A.txt 2>&1 < /dev/null &
  ( LANE=B CUDA_VISIBLE_DEVICES=1 bash scripts/run_v50.sh run; echo "LANE_B_EXIT $?" ) > $R/logs_v50_B.txt 2>&1 < /dev/null &
  echo "STARTED (끝났는지는 로그의 V50A_DONE, V50B_DONE 으로 확인)"; exit 0
fi
[ "$1" = run ] || { echo "사용법: run_v50.sh check|start"; exit 1; }

wf() {  # 이름 모델 [추가 인자...]
  local n=$1 m=$2; shift 2
  if [ "$(lines $L/wf_a_$n.jsonl)" -lt 40 ]; then
    rm -f $L/wf_a_$n.jsonl
    python scripts/run_wf_a_local.py --model $m --out $L/wf_a_$n.jsonl --topics scripts/wf_a_topics_ext.json \
      --search-cache $D/wf_a_search_40.json --limit 10 "$@" > $L/wf_a_$n.txt 2>&1
    local rc=$?
    echo "-- wf_a_$n (exit $rc, $(lines $L/wf_a_$n.jsonl)줄) $(date +%m-%d_%H:%M)"; grep -aE "거부|Error" $L/wf_a_$n.txt | tail -3 | cut -c1-200; tail -1 $L/wf_a_$n.txt | cut -c1-200
    [ $rc = 0 ] && [ "$(lines $L/wf_a_$n.jsonl)" -ge 40 ] || return 1
  fi
  HF_HUB_OFFLINE=0 python scripts/citation_accuracy.py $L/wf_a_$n.jsonl > $L/citation_$n.txt 2>&1 || { echo "citation_accuracy 실패 $n"; return 1; }
  tail -1 $L/citation_$n.txt | cut -c1-200
}
gen() {  # 이름 모델 어댑터
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    local out=$I/${pre}_$1.jsonl want=$(lines $ho)
    if [ "$(lines $out)" -ne "$want" ]; then
      rm -f $out
      python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $out > $L/${pre}_$1.txt 2>&1 || { echo "eval_sft 실패 ${pre}_$1 (exit $?)"; return 1; }
    fi
    echo "-- ${pre}_$1 ($(lines $out)/$want줄) $(date +%m-%d_%H:%M)"; grep -E "^전체|Error|거부" $L/${pre}_$1.txt | tail -1 | cut -c1-120
    [ "$(lines $out)" -eq "$want" ] || return 1
  done
}
jd() {  # 판정 결과도 입력과 줄 수가 같아야 완료로 봄
  local tag=$1 llm=$2; shift 2
  for f in "$@"; do
    local o=data/judged_$tag/$(basename $f)
    [ -f $o.done ] && [ "$(lines $o)" -eq "$(lines $f)" ] && continue
    python scripts/judge_all.py --llm $llm --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(lines $o)/$(lines $f)줄 $(date +%H:%M)"
    [ $rc = 0 ] && [ "$(lines $o)" -eq "$(lines $f)" ] && touch $o.done || return 1
  done
}
train() {  # sft이름 dpo이름 씨앗
  local sft=$1 dpo=$2 seed=$3 rc
  if ! have_adapter $sft; then
    python -m compliance_gateway.train.sft --base qwen --dataset $I/sft10_train.jsonl --output $CK/$sft --epochs 1 --seed $seed > $L/$sft.txt 2>&1
    rc=$?; echo "SFT_EXIT($sft) $rc $(date +%m-%d_%H:%M)"; grep -aE "train_loss|Error" $L/$sft.txt | tail -2 | cut -c1-200
    [ $rc = 0 ] || fail "$sft 학습 실패(exit $rc) — DPO 로 넘어가지 않음"
  fi
  have_adapter $sft || fail "$sft 어댑터 없음 — DPO 로 넘어가지 않음"
  if ! have_adapter $dpo; then
    python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/$sft --dataset $I/dpo7.jsonl --output $CK/$dpo --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 --seed $seed > $L/$dpo.txt 2>&1
    rc=$?; echo "DPO_EXIT($dpo) $rc $(date +%m-%d_%H:%M)"; grep -aE "train_loss|Error|SFT 어댑터에서 이어서" $L/$dpo.txt | tail -3 | cut -c1-200
    [ $rc = 0 ] || fail "$dpo 학습 실패(exit $rc)"
    grep -q "SFT 어댑터에서 이어서 학습" $L/$dpo.txt || fail "$dpo 로그에 SFT 어댑터에서 이어 학습한 기록이 없음"
  fi
  have_adapter $dpo || fail "$dpo 어댑터 없음"
  sha256sum $I/sft10_train.jsonl $I/dpo7.jsonl $CK/$sft/adapter_model.safetensors $CK/$dpo/adapter_model.safetensors > $CK/$dpo/v50_inputs.sha256
}

if [ "$LANE" = A ]; then
  echo "== A: 씨앗 42 로 sft27_v6 → dpo27_v8 $(date)"
  train sft27_v6 dpo27_v8 42
  wf w40_27b_full_v8 $T --adapter $CK/dpo27_v8 || fail "wf w40_27b_full_v8"
  gen dpo27_v8 $T $CK/dpo27_v8 || fail "gen dpo27_v8"
  jd phi4 $PHI $I/eval2_dpo27_v8.jsonl $I/evalk_dpo27_v8.jsonl || fail "judge dpo27_v8"
  echo "V50A_DONE $(date)"
  # 아래는 비교 기준. 실패하거나 시간이 모자라도 본 결과(위)는 유효하므로 레인을 실패로 만들지 않음(Grok 검토)
  echo "== A: 비교 기준 dpo27_v7 세 단계 모두 $(date)"
  wf w40_27b_full_v7 $T --adapter $CK/dpo27_v7 && echo "V50A_EXTRA_DONE $(date)" || echo "V50A_EXTRA_SKIP $(date +%m-%d_%H:%M)"
fi

if [ "$LANE" = B ]; then
  echo "== B: 씨앗 1 로 sft27_v6_s1 → dpo27_v8_s1 $(date)"
  train sft27_v6_s1 dpo27_v8_s1 1
  wf w40_27b_full_v8s1 $T --adapter $CK/dpo27_v8_s1 || fail "wf w40_27b_full_v8s1"
  gen dpo27_v8_s1 $T $CK/dpo27_v8_s1 || fail "gen dpo27_v8_s1"
  jd phi4 $PHI $I/eval2_dpo27_v8_s1.jsonl $I/evalk_dpo27_v8_s1.jsonl || fail "judge dpo27_v8_s1"
  echo "V50B_DONE $(date)"
  echo "== B: 시간이 남으면 재작성만 새 어댑터 $(date)"
  wf w40_27b_regen_v8s1 $T --adapter $CK/dpo27_v8_s1 --adapter-on regen && echo "V50B_EXTRA_DONE $(date)" || echo "V50B_EXTRA_SKIP $(date +%m-%d_%H:%M)"
fi
