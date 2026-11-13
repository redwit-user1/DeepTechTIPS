#!/usr/bin/env bash
# 2026-11-14 v9 — 1차년도 남은 항목: DPO 초기정렬 · 4-bit 양자화 보존율 (연구노트 §43)
#
#   CUDA_VISIBLE_DEVICES=1 nohup bash repo/scripts/run_v9.sh > logs_v9.txt 2>&1 &
#
#   1. 합의 선호쌍    두 판정기(캐스케이드 · LLM 0.5)가 모두 동의한 쌍만. 평가용 원문 제외
#   2. DPO 초기정렬   9B sft_v2 어댑터에서 이어서(기준 정책 = sft_v2 사본). 10스텝 시험 → 본 학습
#   3. 평가           연구노트 평가 지시 2,294건: sft_v2 → dpo_v1 짝 비교, 전수 판정
#   4. 4-bit 보존율   sft_v2 를 4-bit(NF4)로 올려 같은 지시로 평가 → bf16 대비 비율
# 한 단계가 실패해도 다음으로 간다. 결과가 있으면 건너뛴다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B
ev() {  # 이름 [추가 인자...]
  local n=$1; shift
  [ -s $I/eval2_$n.jsonl ] && { echo "-- $n (있음, 건너뜀)"; return; }
  python scripts/eval_sft.py run --model $S --heldout $I/heldout2.jsonl --out $I/eval2_$n.jsonl "$@" > $L/eval2_$n.txt 2>&1
  echo "-- $n (exit $?)"; grep -E "충실 생성률|판정기|어댑터|4-bit|Error|거부" $L/eval2_$n.txt | tail -9
}

echo "== 1. 합의 선호쌍 $(date)"
python scripts/judge_all.py dpo $I/dpo2.jsonl $I/heldout2.jsonl,$K/heldout.jsonl $J/sft2.jsonl.screened.jsonl $J/gen.jsonl $K/judged_all.jsonl || echo DPO_PAIRS_FAIL

echo "== 2. DPO 초기정렬 (9B, sft_v2 에서 이어서) $(date)"
if [ ! -s $CK/dpo_v1/adapter_model.safetensors ] && [ -s $I/dpo2.jsonl ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v2 --dataset $I/dpo2.jsonl --output $CK/dpo_smoke \
    --max-steps 10 --batch 2 --grad-accum 4 > $L/dpo_smoke.txt 2>&1
  echo "DPO_SMOKE_EXIT $?"; grep -E "accepted|이어서|LoRA 모듈|train_runtime|Error|Traceback" $L/dpo_smoke.txt | tail -5 | cut -c1-300
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v2 --dataset $I/dpo2.jsonl --output $CK/dpo_v1 \
    --epochs 2 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v1.txt 2>&1
  echo "DPO_EXIT $?"; grep -E "accepted|train_loss|saved|Error|Traceback" $L/dpo_v1.txt | tail -4 | cut -c1-300
fi

echo "== 3. DPO 평가 $(date)"
if [ -s $CK/dpo_v1/adapter_model.safetensors ]; then
  ev dpo_v1 --adapter $CK/dpo_v1
  echo "[sft_v2 → dpo_v1]"; python scripts/eval_sft.py compare $I/eval2_sft_v2.jsonl $I/eval2_dpo_v1.jsonl
fi

echo "== 4. 4-bit 양자화 보존율 (sft_v2) $(date)"
python -c "import bitsandbytes" 2>/dev/null || env PIP_CONSTRAINT= pip install -q bitsandbytes > $L/bnb_install.txt 2>&1
ev sft_v2_4bit --adapter $CK/sft_v2 --load-4bit
[ -s $I/eval2_sft_v2_4bit.jsonl ] && { echo "[sft_v2 bf16 → 4-bit]"; python scripts/eval_sft.py compare $I/eval2_sft_v2.jsonl $I/eval2_sft_v2_4bit.jsonl; }

echo "== 5. 전수 판정 $(date)"
python scripts/judge_all.py --out-dir $J $(ls $I/eval2_dpo_v1.jsonl $I/eval2_sft_v2_4bit.jsonl 2>/dev/null) > $L/judge_v9.txt 2>&1
python scripts/judge_all.py report $(ls $J/eval2_sft_v2.jsonl $J/eval2_dpo_v1.jsonl $J/eval2_sft_v2_4bit.jsonl 2>/dev/null)
echo "V9_DONE $(date)"
