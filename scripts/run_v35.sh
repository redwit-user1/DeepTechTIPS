#!/usr/bin/env bash
# 2026-11-20 v35 — 플라이휠 두 바퀴째 (연구노트 §59)
#
#   bash repo/scripts/run_v35.sh start      # N줄(GPU 1)은 v34 뒤, K줄(GPU 0)은 v32 K줄 뒤 + 두 번째 교사가 생긴 뒤
#
# 첫 바퀴(v32 · v33): 교사를 학습된 27B(dpo27_v6)로 바꾸니 연구노트에서 네 모델 합의 통과율이 39 → 50%, 그 데이터를 더한 9B 가 독립 판정(Phi-4)에서 +4.6%p.
# 한 번 더 돌린다. 두 번째 교사 = 첫 바퀴 데이터까지 넣어 학습한 27B(sft27_v5 = sft8 → dpo27_v7).
#   볼 것: ① 두 번째 교사의 합의 통과율이 첫 바퀴(50%)보다 또 오르는가 ② sft_v9(= sft6 + 두 바퀴째 데이터)가 sft_v8(= sft6 + 첫 바퀴 데이터)보다 나은가
#   — 오르면 플라이휠은 계속 돌고, 멈추면 한 바퀴가 한계다. 판정 넷에 맞춰져 가는 것인지는 Phi-4(독립)로만 가릴 수 있다.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B
KAN=kakaocorp/kanana-1.5-8b-instruct-2505; MID=K-intelligence/Midm-2.0-Base-Instruct; PHI=microsoft/phi-4; EEV=yanolja/EEVE-Korean-Instruct-10.8B-v1.0

SM=data/sampled
if [ "$1" = start ]; then
  ( while ! grep -q V34_DONE $R/logs_v34.txt 2>/dev/null; do sleep 30; done
    LANE=N CUDA_VISIBLE_DEVICES=1 bash scripts/run_v35.sh run ) > $R/logs_v35_N.txt 2>&1 < /dev/null &
  ( while ! grep -q V32K_DONE $R/logs_v32_K.txt 2>/dev/null; do sleep 30; done
    LANE=K CUDA_VISIBLE_DEVICES=0 bash scripts/run_v35.sh run ) > $R/logs_v35_K.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

model_of() { case $1 in phi4) echo $PHI;; kanana) echo $KAN;; midm) echo $MID;; qwen) echo $T;; eeve) echo $EEV;; esac; }
jd() {
  local tag=$1; shift
  for f in "$@"; do
    [ -s $f ] || continue
    [ -f data/judged_$tag/$(basename $f).done ] && continue
    python scripts/judge_all.py --llm $(model_of $tag) --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}
kv() {
  local tag=$1 mark=$2; shift 2
  [ -f data/kv_$tag/.$mark.done ] && return
  python scripts/judge_all.py --kv --batch 16 --llm $(model_of $tag) --out-dir data/kv_$tag "$@" > $L/kv_${tag}_$mark.txt 2>&1
  local rc=$?; echo "KV_EXIT($tag $mark) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/kv_$tag/.$mark.done
}
gen() {
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1"; grep -E "^전체|Error|거부" $L/${pre}_$1.txt | tail -1 | cut -c1-120
  done
}
rejudge() {
  for n in "$@"; do
    for tag in phi4 eeve kanana midm; do jd $tag $I/eval2_$n.jsonl $I/evalk_$n.jsonl; done
    [ -s data/judged_phi4/eval2_$n.jsonl ] && for tag in phi4 kanana midm qwen; do
      kv $tag $n data/judged_phi4/eval2_$n.jsonl data/judged_phi4/evalk_$n.jsonl; done
  done
}

HO="$I/heldout2.jsonl $K/heldout.jsonl"
samp() {  # 출력 이름, 나머지는 sample 옵션
  local out=$SM/$1.jsonl; shift
  [ -f $out.done ] && return
  python scripts/eval_sft.py sample --model $T --adapter $CK/dpo27_v7 --exclude $HO --out $out --n 1 --temperature 0.3 "$@" > $L/sample_$(basename $out .jsonl).txt 2>&1
  local rc=$?; echo "SAMPLE_EXIT($(basename $out)) $rc $(date +%H:%M) $(wc -l < $out)"; [ $rc = 0 ] && touch $out.done
}
jd4() {  # 표집 파일들을 판정 넷으로(qwen 은 기본 판정기 → data/judged), 이어서 조건 추출 전용 판정
  for f in "$@"; do
    [ -f $J/$(basename $f).done ] || { python scripts/judge_all.py --out-dir $J $f > $L/judge_qwen_$(basename $f .jsonl).txt 2>&1; rc=$?; echo "JUDGE_EXIT(qwen $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch $J/$(basename $f).done; }
  done
  for tag in kanana midm eeve; do jd $tag "$@"; done
}

if [ "$LANE" = N ]; then
  echo "== N0. 두 번째 교사: 27B sft27_v5 (sft8 = sft6 + 첫 바퀴 데이터) → dpo27_v7 (+ dpo7) $(date)"
  for i in $(seq 1 2880); do [ -s $I/sft8_train.jsonl ] && break; sleep 30; done
  if [ ! -s $CK/sft27_v5/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen --dataset $I/sft8_train.jsonl --output $CK/sft27_v5 --epochs 1 > $L/sft27_v5.txt 2>&1
    echo "SFT27_EXIT $?"
  fi
  if [ -s $CK/sft27_v5/adapter_model.safetensors ] && [ ! -s $CK/dpo27_v7/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen --sft-adapter $CK/sft27_v5 --dataset $I/dpo7.jsonl --output $CK/dpo27_v7 --epochs 1 --lr 2e-5 --batch 1 --grad-accum 16 > $L/dpo27_v7.txt 2>&1
    echo "DPO27_EXIT $?"
  fi
  [ -s $CK/dpo27_v7/adapter_model.safetensors ] || { echo "교사 없음 — 중단"; exit 1; }
  echo "== N1. 연구노트 표집 (교사 dpo27_v7) $(date)"
  samp fly2_note --source $J/sft2.jsonl.screened.jsonl $J/gen.jsonl --batch 16
  samp fly2_note_qa --qa --source $J/sft2.jsonl.screened.jsonl $J/gen.jsonl --batch 32 --max-new-tokens 200
  echo "== N2. 판정 $(date)"
  jd4 $SM/fly2_note.jsonl $SM/fly2_note_qa.jsonl
  for tag in phi4 kanana midm qwen; do kv $tag fly2_note $J/fly2_note.jsonl; done
  touch $SM/.fly2_note_all.done
  echo "== N3. 두 번째 교사 자신의 평가 $(date)"
  gen dpo27_v7 $T $CK/dpo27_v7; rejudge dpo27_v7
  echo "V35N_DONE $(date)"
fi

if [ "$LANE" = K ]; then
  # 두 번째 교사(27B 학습 약 10시간)를 기다리는 동안: 첫 바퀴 모델의 반복(씨앗만 바꿈). sft9 에는 첫 바퀴 데이터(fly)를 넣지 않는다 — 교사만 바뀐 비교.
  echo "== K0. 첫 바퀴 모델의 반복 $(date)"
  if [ -s $I/sft8_train.jsonl ] && [ ! -s $CK/sft_v8_s1/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft8_train.jsonl --output $CK/sft_v8_s1 --epochs 1 --batch 2 --grad-accum 8 --seed 1 > $L/sft_v8_s1.txt 2>&1
    echo "SFT_EXIT(sft_v8_s1) $?"
  fi
  [ -s $CK/sft_v8_s1/adapter_model.safetensors ] && { gen sft_v8_s1 $S $CK/sft_v8_s1; rejudge sft_v8_s1; }
  if [ -s $CK/sft_v8/adapter_model.safetensors ] && [ ! -s $CK/dpo_v11_s1/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v8 --dataset $I/dpo7.jsonl --output $CK/dpo_v11_s1 --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 --seed 1 > $L/dpo_v11_s1.txt 2>&1
    echo "DPO_EXIT(dpo_v11_s1) $?"
  fi
  [ -s $CK/dpo_v11_s1/adapter_model.safetensors ] && { gen dpo_v11_s1 $S $CK/dpo_v11_s1; rejudge dpo_v11_s1; }
  for i in $(seq 1 5760); do [ -s $CK/dpo27_v7/adapter_model.safetensors ] && break; sleep 30; done
  echo "== K1. KCI 표집 (교사 dpo27_v7, 4,000 원문) $(date)"
  samp fly2_kci --source $K/judged_all.jsonl --batch 8 --max-new-tokens 900 --limit-passages 4000
  samp fly2_kci_qa --qa --source $K/judged_all.jsonl --batch 16 --max-new-tokens 300 --limit-passages 4000
  echo "== K2. 판정 $(date)"
  jd4 $SM/fly2_kci.jsonl $SM/fly2_kci_qa.jsonl
  for tag in phi4 kanana midm qwen; do kv $tag fly2_kci $J/fly2_kci.jsonl; done

  echo "== K3. 네 모델 합의 통과분 (N줄을 기다린다) $(date)"
  for i in $(seq 1 2880); do [ -f $SM/.fly2_note_all.done ] && break; sleep 30; done
  [ -s $I/fly2_train.jsonl ] || { python scripts/judge_all.py export-multi $I/fly2.tmp $I/heldout2.jsonl,$K/heldout.jsonl $R/scores fly2_note.jsonl,fly2_note_qa.jsonl,fly2_kci.jsonl,fly2_kci_qa.jsonl     qwen=$J=qwen2 kanana=data/judged_kanana midm=data/judged_midm eeve=data/judged_eeve kv=data/kv_phi4,data/kv_kanana,data/kv_midm,data/kv_qwen && mv $I/fly2.tmp $I/fly2_train.jsonl; }
  cat $I/sft6_train.jsonl $I/fly2_train.jsonl > $I/sft9_train.jsonl; wc -l $I/fly2_train.jsonl $I/sft9_train.jsonl

  echo "== K4. 9B sft_v9 (sft6 + 새 교사의 합의 통과분) $(date)"
  if [ -s $I/fly2_train.jsonl ] && [ ! -s $CK/sft_v9/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft9_train.jsonl --output $CK/sft_v9 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v9.txt 2>&1
    echo "SFT_EXIT $?"
  fi
  [ -s $CK/sft_v9/adapter_model.safetensors ] && { gen sft_v9 $S $CK/sft_v9; rejudge sft_v9; }
  echo "== K5. 9B dpo_v12 (sft_v9 + dpo7) $(date)"
  if [ -s $CK/sft_v9/adapter_model.safetensors ] && [ ! -s $CK/dpo_v12/adapter_model.safetensors ]; then
    python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v9 --dataset $I/dpo7.jsonl --output $CK/dpo_v12       --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v12.txt 2>&1
    echo "DPO_EXIT $?"
  fi
  [ -s $CK/dpo_v12/adapter_model.safetensors ] && { gen dpo_v12 $S $CK/dpo_v12; rejudge dpo_v12; }
  echo "V35K_DONE $(date)"
fi
