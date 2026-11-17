#!/usr/bin/env bash
# 2026-11-17 v23b — 합의 증류 인코더(v5) 다시. v23 의 2 ~ 4 단계가 두 가지로 틀렸다:
#   ① 가상환경 안에서 `deactivate` 로는 시스템 파이썬이 되지 않았다(warmup_ratio 오류, v4 때와 같은 실수) → /usr/bin/python 을 직접 부른다
#   ② '셋 모두 미지지'만 neutral 로 쓰니 1,116문장뿐 → 과반 미지지로(distill-multi)
# 인코더는 작아서 sft_v5_s2 학습(v23 의 5단계)이 도는 GPU 1 에 얹는다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; D=$R/data
export CUDA_VISIBLE_DEVICES=1
FILES=sft2.jsonl.screened.jsonl,gen.jsonl,gen_0.jsonl,gen_1.jsonl,note_sft_v2.jsonl,kci_sft_v2.jsonl,note_dpo_v4.jsonl,kci_dpo_v4.jsonl
echo "== 1. 합의 증류 학습셋 $(date)"
$R/venv/bin/python scripts/judge_all.py distill-multi $D/distill/train_v5.tmp $D/distill/train_v3.jsonl $I/heldout2.jsonl,$K/heldout.jsonl $R/scores $FILES \
  qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm && mv $D/distill/train_v5.tmp $D/distill/train_v5.jsonl
echo "== 2. 판정 인코더 v5 (시스템 파이썬) $(date)"
rm -rf $CK/nli_distill_v5
/usr/bin/python -m compliance_gateway.train.nli_finetune --data $D/distill/train_v5.jsonl --labels 3 --base deberta-large \
  --split train --eval-split dev --output $CK/nli_distill_v5 > $L/ft_v5.txt 2>&1
echo "FT_V5_EXIT $?"; rm -rf $CK/nli_distill_v5/checkpoint-*; grep -aE "eval:|saved" $L/ft_v5.txt | tail -2 | cut -c1-250
echo "== 3. 라벨 평가셋 $(date)"
for set in kr_psg_eval kr_hard_eval kr_abs_eval; do
  /usr/bin/python -m compliance_gateway.eval.kr_nli $D/$set.jsonl --nli $CK/nli_distill_v5 --dump $R/scores/${set}_distill_v5.jsonl > $L/${set}_distill_v5.txt 2>&1
  echo "-- $set v5"; grep -aE "^AUC|운영점" $L/${set}_distill_v5.txt
  echo "-- $set v3"; grep -aE "^AUC|운영점" $L/${set}_distill_v3.txt
done
echo "== 4. Phi-4 와의 일치 — v5 인코더로 평가 응답 판정 $(date)"
if [ -s $CK/nli_distill_v5/config.json ]; then
  $R/venv/bin/python scripts/judge_all.py --student $CK/nli_distill_v5 --llm microsoft/phi-4 --batch 16 --out-dir data/judged_enc5 \
    $I/eval2_base_hint.jsonl $I/evalk_base_hint.jsonl $I/eval2_dpo_v6.jsonl $I/evalk_dpo_v6.jsonl $I/eval2_dpo27_v4.jsonl $I/evalk_dpo27_v4.jsonl > $L/judge_enc5.txt 2>&1
  echo "ENC5_JUDGE_EXIT $? $(date +%H:%M)"
fi
echo "V23B_DONE $(date)"
