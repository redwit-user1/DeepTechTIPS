#!/usr/bin/env bash
# 2026-11-17 v23 — 판정 인코더를 세 판정 모델의 합의로 다시 증류한다 (연구노트 §53)
#
#   bash repo/scripts/run_v23.sh start      # v20 이 끝나면 GPU 1 에서 시작
#
# 게이트웨이 캐스케이드의 첫 단(증류 DeBERTa)은 Qwen 하나의 판정을 라벨로 배웠다. Qwen 은 다른 판정 모델보다 관대하다(§44).
# 교사 생성분 + 학생 표집분(판정 모델 3종의 문장별 점수가 모두 있다)에서 셋이 일치한 문장만 라벨로 써서 v5 를 만든다.
#   1. 합의 증류 학습셋(distill-multi) → 2. DeBERTa-large 미세조정(시스템 파이썬) → 3. 라벨 평가셋 3종 AUC(v3 와 비교)
#   4. 선별에 쓰지 않은 Phi-4 와의 일치: 평가 응답을 v5 인코더 + Phi-4 로 판정(data/judged_enc5/) — 기존 인코더 점수는 judged_phi4/ 에 있다
#   5. (남는 시간) SFT 씨앗 반복 한 번 더: sft_v5_s2 → 평가 → 재심사
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; D=$R/data
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  ( while ! grep -q V20_DONE $R/logs_v20.txt 2>/dev/null; do sleep 30; done
    CUDA_VISIBLE_DEVICES=1 bash scripts/run_v23.sh run ) > $R/logs_v23.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

echo "== 1. 합의 증류 학습셋 $(date)"
FILES=sft2.jsonl.screened.jsonl,gen.jsonl,gen_0.jsonl,gen_1.jsonl,note_sft_v2.jsonl,kci_sft_v2.jsonl,note_dpo_v4.jsonl,kci_dpo_v4.jsonl
[ -s $D/distill/train_v5.jsonl ] || { python scripts/judge_all.py distill-multi $D/distill/train_v5.tmp $D/distill/train_v3.jsonl $I/heldout2.jsonl,$K/heldout.jsonl $R/scores $FILES \
  qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm && mv $D/distill/train_v5.tmp $D/distill/train_v5.jsonl; }
wc -l $D/distill/train_v5.jsonl

echo "== 2. 판정 인코더 v5 학습 · 3. 라벨 평가셋 $(date)"
if [ -s $D/distill/train_v5.jsonl ] && [ ! -s $CK/nli_distill_v5/config.json ]; then
  ( deactivate   # 인코더 학습 · 평가는 시스템 파이썬(transformers 4.x) — v3 와 같은 조건
    python -m compliance_gateway.train.nli_finetune --data $D/distill/train_v5.jsonl --labels 3 --base deberta-large \
      --split train --eval-split dev --output $CK/nli_distill_v5 > $L/ft_v5.txt 2>&1
    echo "FT_V5_EXIT $?"; rm -rf $CK/nli_distill_v5/checkpoint-*; grep -aE "eval:|saved" $L/ft_v5.txt | tail -2 | cut -c1-250
    for set in kr_psg_eval kr_hard_eval kr_abs_eval; do
      python -m compliance_gateway.eval.kr_nli $D/$set.jsonl --nli $CK/nli_distill_v5 --dump $R/scores/${set}_distill_v5.jsonl > $L/${set}_distill_v5.txt 2>&1
      echo "-- $set v5"; grep -aE "^AUC|운영점" $L/${set}_distill_v5.txt
      echo "-- $set v3"; grep -aE "^AUC|운영점" $L/${set}_distill_v3.txt
    done )
fi

echo "== 4. Phi-4 와의 일치 — v5 인코더로 평가 응답 판정 $(date)"
if [ -s $CK/nli_distill_v5/config.json ] && [ ! -f data/judged_enc5/.done ]; then
  python scripts/judge_all.py --student $CK/nli_distill_v5 --llm $PHI --out-dir data/judged_enc5 \
    $I/eval2_base_hint.jsonl $I/evalk_base_hint.jsonl $I/eval2_dpo_v6.jsonl $I/evalk_dpo_v6.jsonl $I/eval2_dpo27_v4.jsonl $I/evalk_dpo27_v4.jsonl > $L/judge_enc5.txt 2>&1
  rc=$?; echo "ENC5_JUDGE_EXIT $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_enc5/.done
fi

echo "== 5. 9B sft_v5_s2 (sft5, 씨앗 2) $(date)"
model_of() { case $1 in phi4) echo $PHI;; kanana) echo $KAN;; midm) echo $MID;; qwen) echo $T;; esac; }
if [ ! -s $CK/sft_v5_s2/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft5_train.jsonl --output $CK/sft_v5_s2 --epochs 1 --batch 2 --grad-accum 8 --seed 2 > $L/sft_v5_s2.txt 2>&1
  echo "SFT_EXIT $?"
fi
if [ -s $CK/sft_v5_s2/adapter_model.safetensors ]; then
  n=sft_v5_s2
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$n.jsonl ] || python scripts/eval_sft.py run --model $S --adapter $CK/$n $extra --heldout $ho --out $I/${pre}_$n.jsonl > $L/${pre}_$n.txt 2>&1
    echo "-- ${pre}_$n"; grep -E "^전체|Error|거부" $L/${pre}_$n.txt | tail -1 | cut -c1-120
  done
  for tag in phi4 kanana midm; do
    for f in $I/eval2_$n.jsonl $I/evalk_$n.jsonl; do
      [ -f data/judged_$tag/$(basename $f).done ] && continue
      python scripts/judge_all.py --llm $(model_of $tag) --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
      rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
    done
  done
  for tag in phi4 kanana midm qwen; do
    python scripts/judge_all.py --kv --batch 16 --llm $(model_of $tag) --out-dir data/kv_$tag data/judged_phi4/eval2_$n.jsonl data/judged_phi4/evalk_$n.jsonl > $L/kv_${tag}_$n.txt 2>&1
    echo "KV_EXIT($tag $n) $? $(date +%H:%M)"
  done
fi
echo "V23_DONE $(date)"
