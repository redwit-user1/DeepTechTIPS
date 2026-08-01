# A100 6개월 실행 계획

지원사업으로 **A100 2장 × 6개월** 확보. 7B 모델 기준 **컴퓨트는 병목이 아니다**
(LoRA/DPO는 A100 1장 수 시간, GRPO 수일). 따라서 병목은 ① 베이스 모델 확보(HF 가능 ✅)
② 도메인 데이터 양·질(GOONO 일부 + 합성 병행) ③ **KPI 측정 체계**로 이동한다.
→ 데이터·평가에 과투자한다.

## 자원 배분 원칙

| 자원 | 상태 | 전략 |
|---|---|---|
| A100 2장 | 6개월 | 1장 학습 / 1장 서빙·평가, GRPO 시 vLLM 생성+학습 분리 |
| 베이스 모델 | HF 접근 ✅ | EXAONE 3.5 7.8B 우선, Qwen2.5 7B 대조 |
| GOONO 데이터 | 일부만 | 합성 파이프라인(bioRxiv) 대폭 확대로 보완 |
| KPI 평가 | 스캐폴드 완료 | 학습 전/후 동일 코드로 측정 → 개선 증명 |

## KPI ↔ 작업 ↔ 측정 매핑

| KPI (목표) | 담당 작업 | 측정 |
|---|---|---|
| 규정위반 탐지 정밀도 90%+ | NLI 파인튜닝 + Gateway | `eval/kpi.py` (precision/recall/F1) |
| 출처 정확률 90%+ | NLI + DOI resolver(DB 대조) | `eval/kpi.py`, SciFact AUC |
| 보고서 자동생성 정확도 88%+ | SLM SFT+DPO + RAG | 외부 전문가 블라인드(과제 30건) |
| 도메인 질의 정확도 2배+ | 도메인 LoRA | 벤치 3도메인×100 A/B |
| VCR 0.85+ | DPO→GRPO | VCR 자기평가 분포 |
| 양자화 보존율 80%+ | GPTQ/AWQ 4-bit | FP16 vs 4-bit 동일 벤치 |

## 6개월 타임라인

### M1 (1개월) — 인프라 + 측정 체계  ← 이 레포에 스캐폴드 완료
- [x] 학습 파이프라인 스캐폴드: `train/{config,data_format,sft,dpo,nli_finetune}.py`
- [x] KPI 측정 하니스: `eval/kpi.py` (baseline: 통계 NLI)
- [ ] **(A100) NLI 파인튜닝**: `python -m compliance_gateway.train.nli_finetune`
      → `TransformerNLI(model_name="checkpoints/nli")` 를 Gateway 에 주입 → KPI 재측정
- [ ] 합성 데이터 확대(수천 쌍) + FDA Warning Letters/CiteAudit 외부 평가셋 연동

### M2–3 (2개월) — Compliance-Aligned SLM
- [ ] 데이터: 합성 DPO 수만 + GOONO 일부 → SFT/DPO 데이터셋 빌드
      `python -m compliance_gateway.train.data_format` (SFT/DPO 변환)
- [ ] SFT: `python -m compliance_gateway.train.sft --base exaone`
- [ ] DPO: `python -m compliance_gateway.train.dpo --sft-adapter checkpoints/sft --vcr-accept 0.7`
- [ ] 목표: VCR 0.60 기준선, 규정위반 탐지·보고서 정확도 KPI 1차 달성

### M4 (1개월) — 에이전트 + RAG
- [ ] 탐색·기획 에이전트 v1 + LangGraph DAG (모든 출력 Gateway 경유)
- [ ] R&D 특화 RAG(구조 인식 청킹) + ScienceON/NTIS 연동
- [ ] 목표: 워크플로우 자동화율 KPI

### M5 (1개월) — RLVR + 온프레미스
- [ ] GRPO 전환: `train/grpo.py` (질의당 N=8, VCR 보상, 롤백) — VCR 0.7→0.85
- [ ] GPTQ/AWQ 4-bit 양자화, 단일 GPU(RTX 4090급) 추론 검증
- [ ] 목표: VCR 0.85+, 양자화 보존율 80%+

### M6 (1개월) — 통합·측정·산출
- [ ] 전 KPI 최종 측정(TTA V&V 대응 포맷), 온프레미스 번들
- [ ] 특허(VCR/Gateway) · 논문 초안(KISTI 공동)

## A100 셋업 (환경 준비되면)

```bash
git clone <repo> && cd DeepTechTIPS
pip install -e ".[train]"                       # torch/transformers/trl/peft/bitsandbytes
bash scripts/download_scifact.sh                # NLI 학습·평가셋

# 1) NLI 파인튜닝 (M1 핵심 — 규정위반/출처 KPI)
python -m compliance_gateway.train.nli_finetune --base deberta-mnli --output checkpoints/nli
python -m compliance_gateway.eval.kpi           # 트랜스포머 NLI 주입 후 재측정

# 2) 데이터 빌드 → SLM SFT → DPO (M2-3)
python -m compliance_gateway.data.build_dpo     # 합성 DPO/SFT 생성
python -m compliance_gateway.train.sft  --base exaone --dataset data/synth/sft.jsonl
python -m compliance_gateway.train.dpo  --base exaone --sft-adapter checkpoints/sft --vcr-accept 0.7
```

## 리스크

| # | 리스크 | 대응 |
|---|---|---|
| R1 | 합성 데이터가 실제보다 쉬움(KPI 낙관 편향) | FDA Warning Letters/CiteAudit 등 **외부 실데이터** 평가셋 필수 |
| R2 | GOONO 데이터 일부만 → 도메인 편중 | 합성 카테고리·연도 다양화, 3도메인(공학/ICT/바이오) 균형 |
| R3 | 6개월 내 GRPO까지 무리 | DPO(VCR 0.7)를 확정 목표, GRPO는 스트레치 |
| R4 | polarity(결론 역전) 탐지 약점 | 트랜스포머 NLI로 M1에서 우선 해소, 재측정 |
