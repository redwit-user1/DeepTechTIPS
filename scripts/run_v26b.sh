#!/usr/bin/env bash
# 2026-11-17 v26b — 선별에 쓰지 않은 판정 모델을 하나 더 (v26 의 뒷부분 다시)
# v26: EXAONE 3.5 는 저장소의 코드를 실행해야 올라간다(trust_remote_code) — 쓰지 않는다. EEVE 는 내려받다 끊겼다 → 다시 받기 3회, 후보 추가.
#
#   bash repo/scripts/run_v26b.sh start      # GPU 1
#
# ① v24: v6a(기존 + 합의 15,000)는 어려운 사례 AUC 0.934, 같은 합의 예에 기존 예만 3번 반복한 v6c 는 0.887. 섞는 비율의 효과인지
#    인코더 학습의 흔들림인지 모른다 → v6a 를 씨앗만 바꿔 두 번 더(NLI_SEED).
# ② 지금까지 선별에 쓰지 않은 판정은 Phi-4 하나다. 계열이 다른 판정 모델을 하나 더 들여 같은 결론이 나오는지 본다.
#    들이기 전 점검(§52): 대화 틀을 찍어 날짜 · 기본 시스템 프롬프트를 확인, 라벨 평가셋 3종으로 임계값 보정, 같은 답 재판정 뒤집힘.
R=~/research-data-analysis; cd $R/repo; export HF_HOME=/tmp/hf; L=$R/logs; CK=$R/checkpoints; I=data/instr; K=data/kci; J=data/judged; D=$R/data
PY=$R/venv/bin/python; export HF_HUB_OFFLINE=1

if [ "$1" = start ]; then
  CUDA_VISIBLE_DEVICES=1 nohup bash scripts/run_v26b.sh run > $R/logs_v26b.txt 2>&1 < /dev/null &
  echo STARTED; exit 0
fi

echo "== 다섯 번째 판정 모델 후보 $(date)"
for spec in eeve=yanolja/EEVE-Korean-Instruct-10.8B-v1.0 varco=NCSOFT/Llama-VARCO-8B-Instruct bllossom=MLP-KTLim/llama-3-Korean-Bllossom-8B; do
  tag=${spec%%=*}; M=${spec#*=}
  # 가상환경의 허브 라이브러리는 내려받다 죽는다(httpx2 압축 해제 오류) — 시스템 파이썬으로 받고, 그 뒤는 오프라인으로 캐시만 쓴다
  for try in 1 2 3; do HF_HUB_OFFLINE=0 /usr/bin/python -c "from huggingface_hub import snapshot_download as s; s('$M')" > $L/dl_$tag.txt 2>&1 && break; sleep 20; done
  $PY -c "

from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$M')
tpl = t.chat_template or ''
print('TEMPLATE_OK', bool(tpl), '날짜 흔적:', [k for k in ('strftime_now', 'date_string', 'Today', 'today') if k in tpl])
print(repr(t.apply_chat_template([{'role': 'user', 'content': 'X'}], tokenize=False, add_generation_prompt=True))[:500])" > $L/probe_$tag.txt 2>&1
  echo "PROBE($tag) $? $(grep -a TEMPLATE_OK $L/probe_$tag.txt | cut -c1-120)"
  grep -aq "TEMPLATE_OK True" $L/probe_$tag.txt || continue
  ok=1
  for s in kr_psg_eval kr_hard_eval kr_abs_eval; do
    [ -s $R/scores/${s}_$tag.jsonl ] || $PY -m compliance_gateway.eval.kr_nli $D/$s.jsonl --llm $M --dump $R/scores/${s}_$tag.jsonl > $L/${s}_$tag.txt 2>&1 || ok=0
    echo "-- $s $tag: $(grep -aE '^AUC' $L/${s}_$tag.txt)"
  done
  [ $ok = 1 ] || { echo "CALIB_FAIL($tag)"; continue; }
  for n in base_hint sft_v5 dpo_v6 dpo_v7 dpo_v5 base27_hint dpo27_v5; do
    for pre in eval2 evalk; do
      f=$I/${pre}_$n.jsonl; [ -s $f ] || continue
      [ -f data/judged_$tag/${pre}_$n.jsonl.done ] && continue
      $PY scripts/judge_all.py --llm $M --out-dir data/judged_$tag $f > $L/judge_${tag}_${pre}_$n.txt 2>&1
      rc=$?; echo "JUDGE_EXIT($tag ${pre}_$n) $rc $(date +%H:%M)"; [ $rc = 0 ] && touch data/judged_$tag/${pre}_$n.jsonl.done
    done
  done
  # 같은 파일을 한 번 더 — 재판정 뒤집힘 점검
  $PY scripts/judge_all.py --llm $M --out-dir data/judged_${tag}_r2 $I/eval2_dpo_v7.jsonl > $L/judge_${tag}_r2.txt 2>&1; echo "R2_EXIT($tag) $?"
  echo "JUDGE5_DONE($tag) $(date)"
  break      # 하나만 쓴다 — 먼저 되는 쪽
done
echo "V26B_DONE $(date)"
