#!/usr/bin/env bash
# 2026-11-25 v47 — 제안서 과제를 학습 데이터에 넣으면 어댑터가 제안서 초안을 쓰는가 (연구노트 §62, 2차년도 선행 시범)
#
#   bash repo/scripts/run_v47.sh start      # GPU 1
#
# §61: 단락 과제로만 학습한 어댑터(dpo_v11)를 WF-A 의 세 단계 모두에 쓰면 제목 · 절 구성과 DOI 인용이 사라진다(인용 84 → 8).
# 1. 교사 = 기반 27B. KCI 이공 · 의약 논문 제목 900개를 주제로, OpenAlex 근거 5건(WF-A 와 같은 입력)에 갭 분석 → 초안 → 게이트웨이.
#    통과 · 인용 3개 이상 · 인용 모두 근거 안 · 제목으로 시작하는 것만 학습 예로(걸린 초안은 한 번 재작성해서 통과하면 재작성 예로).
# 2. sft10 = sft8 + 제안서 예 → 9B sft_v10 → dpo_v13(+ dpo7). dpo_v11 과 데이터 차이는 제안서 예뿐.
# 3. 평가: (가) 근거 충실 평가셋(조건 추출 제외 세 과제, Phi-4) — 떨어지면 안 된다
#          (나) WF-A 40개 주제, 어댑터를 세 단계 모두에 — dpo_v11 과 비교(인용 수 · 형식 · 통과)
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf HF_HUB_OFFLINE=1; L=$R/logs; CK=$R/checkpoints; D=$R/data; I=data/instr; K=data/kci
. $R/venv/bin/activate
S=Qwen/Qwen3.5-9B; T=Qwen/Qwen3.8-27B; PHI=microsoft/phi-4

if [ "$1" = start ]; then
  ( CUDA_VISIBLE_DEVICES=1 bash scripts/run_v47.sh run ) > $R/logs_v47.txt 2>&1 < /dev/null &
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
gen() {
  for pre in eval2 evalk; do
    ho=$I/heldout2.jsonl; extra=""; [ $pre = evalk ] && { ho=$K/heldout.jsonl; extra="--max-new-tokens 900"; }
    [ -s $I/${pre}_$1.jsonl ] || python scripts/eval_sft.py run --model $2 --adapter $3 $extra --heldout $ho --out $I/${pre}_$1.jsonl > $L/${pre}_$1.txt 2>&1
    echo "-- ${pre}_$1 $(date +%m-%d_%H:%M)"; grep -E "^전체|Error|거부" $L/${pre}_$1.txt | tail -1 | cut -c1-120
  done
}
jd() {
  local tag=$1 llm=$2; shift 2
  for f in "$@"; do
    [ -f data/judged_$tag/$(basename $f).done ] && continue
    python scripts/judge_all.py --llm $llm --out-dir data/judged_$tag $f > $L/judge_${tag}_$(basename $f .jsonl).txt 2>&1
    local rc=$?; echo "JUDGE_EXIT($tag $(basename $f)) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/$(basename $f).done
  done
}

# OpenAlex 무료 한도(IP 단위, 하루)가 900개 검색 도중 바닥났다(11/25). 그래서 두 묶음으로 나눈다:
#   a = 지금 초록이 갖춰진 주제(약 150개) → 바로 생성. b = 한도가 풀린 뒤(11/26 09:00 KST) 다시 찾은 주제 → $D/prop_search_b.done 이 생기면 생성.
echo "== 1a. 제안서 예 생성(기반 27B), 묶음 a $(date)"
gp() {  # 묶음 이름, 나머지는 gen_proposals 옵션
  local b=$1; shift
  [ -s $I/prop_train_$b.jsonl ] && return
  python scripts/gen_proposals.py --model $T --topics $D/prop_topics.json --search-cache $D/prop_search.json \
    --out $I/prop_train_$b.jsonl --log $L/prop_gen_$b.jsonl "$@" > $L/prop_gen_$b.txt 2>&1
  echo "GEN_EXIT($b) $? $(date +%m-%d_%H:%M)"; tail -1 $L/prop_gen_$b.txt
}
gp a
echo "== 1b. 묶음 b: 다시 찾은 검색 결과를 기다림 $(date)"
for i in $(seq 1 1440); do [ -f $D/prop_search_b.done ] && break; sleep 60; done
[ -f $D/prop_search_b.done ] && gp b --skip-log $L/prop_gen_a.jsonl
cat $I/prop_train_a.jsonl $I/prop_train_b.jsonl 2>/dev/null > $I/prop_train.jsonl
[ -s $I/prop_train.jsonl ] || { echo "V47_FAIL 제안서 예 없음"; exit 1; }

echo "== 2. sft10 = sft8 + 제안서 예 → sft_v10 → dpo_v13 $(date)"
cat $I/sft8_train.jsonl $I/prop_train.jsonl > $I/sft10_train.jsonl; wc -l $I/sft8_train.jsonl $I/prop_train.jsonl $I/sft10_train.jsonl
if [ ! -s $CK/sft_v10/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.sft --base qwen-small --dataset $I/sft10_train.jsonl --output $CK/sft_v10 --epochs 1 --batch 2 --grad-accum 8 > $L/sft_v10.txt 2>&1
  echo "SFT_EXIT $? $(date +%m-%d_%H:%M)"; grep -aE "train_loss|Error" $L/sft_v10.txt | tail -2 | cut -c1-200
fi
if [ ! -s $CK/dpo_v13/adapter_model.safetensors ]; then
  python -m compliance_gateway.train.dpo --base qwen-small --sft-adapter $CK/sft_v10 --dataset $I/dpo7.jsonl --output $CK/dpo_v13 --epochs 1 --lr 2e-5 --batch 2 --grad-accum 8 > $L/dpo_v13.txt 2>&1
  echo "DPO_EXIT $? $(date +%m-%d_%H:%M)"; grep -aE "train_loss|Error" $L/dpo_v13.txt | tail -2 | cut -c1-200
fi

echo "== 3가. WF-A 40개 주제, 어댑터를 세 단계 모두에 $(date)"
wf w40_9b_full_v13 $S --adapter $CK/dpo_v13
wf w40_9b_full_v11 $S --adapter $CK/dpo_v11
wf w40_9b_regen_v13 $S --adapter $CK/dpo_v13 --adapter-on regen

echo "== 3나. 근거 충실 평가셋 $(date)"
gen dpo_v13 $S $CK/dpo_v13
jd phi4 $PHI $I/eval2_dpo_v13.jsonl $I/evalk_dpo_v13.jsonl
echo "V47_DONE $(date)"
