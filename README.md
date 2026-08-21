# DeepTechTIPS — GOONO AI

> 연구데이터 **규정준수(Compliance) 기반 Agentic AI 오케스트레이션 엔진**
> 주식회사 레드윗 · 딥테크 TIPS (2026.04 ~ 2029.03)

범용 LLM이 R&D 현장에 바로 쓰이지 못하는 두 가지 구조적 결함을 정면으로 해결합니다.

| 문제 | 범용 LLM의 한계 | GOONO AI의 해법 |
|---|---|---|
| **AI 신뢰성** | RLHF는 "유해성 감소"만 정렬한다. "출처 없는 주장 = 비준수"라는 R&D 고유 기준이 없다 → 가짜 DOI·가짜 논문·수치 변조형 환각 | **VCR 보상함수**로 "출처 없는 생성 = 비준수"를 *파라미터 수준*에 내재화 |
| **AI 불투명성** | 생성 과정이 블랙박스 → ALCOA+ / FDA 21 CFR Part 11 감사 대응 불가 | **Compliance Gateway**(생성 중 검증) + 블록체인 감사추적 |

## 핵심 자산 (왜 레드윗인가)

- **데이터 해자**: GOONO ELN 2,500+ 기관 · 70,000+ 구조화 연구노트
- **규제 자산**: 영업비밀 원본인증기관(특허청), 블록체인 시점인증 특허 10건, GLP/ALCOA+/FDA Part 11 운영 5년
- **유통 해자**: 기존 2,500개 고객 Cross-sell, 정출연 5곳 온프레미스 납품(에어갭 시장 선점)
- **흑자 기업**: 2025년 매출 21.9억 / 영업이익 +1.3억(최초 흑자)

## 기술 4대 축

```
GOONO AI 오케스트레이션 엔진
├─ ① Compliance-Aligned SLM   : 7B + LoRA + (DPO→RLVR/GRPO), VCR 정렬
├─ ② Compliance Gateway       : 출처바인딩 → 환각 게이팅 → ALCOA+ 체크 → 블록체인 기록   ◀ 1순위 개발
├─ ③ R&D 에이전트 + DAG        : 탐색·기획 / 실험분석·설계 / 규정준수·재현성 (HITL)
└─ ④ Harnessing 아키텍처       : Compliance Data Flywheel, R&D특화 RAG, L1/L2 메모리, 듀얼배포
   + Compliance Ontology (OWL2/BFO) — ②③의 지식 기반
```

## 개발 전략

- **Gateway-First**: 모든 모듈이 경유하는 공통 인프라이자 사업화 BM(Compliance-as-a-Service)의 핵심. API 모델(GPT/Claude) 위에서 *지금 당장* 동작·데모 가능 → 파일럿/투자 트리거 조기 확보.
- **초기 집중 도메인 = 제약·바이오**: GLP / FDA 21 CFR Part 11 / ALCOA+. 규제 강제성이 가장 크고(고단가, 건당 2억 On-premise), 본과제 매출 비중 최대(2029년 30%). "깊게 한 도메인 증명" 전략.
- **Demo-Early / Data-Driven**: Gateway에서 쌓인 검증 기록 = SLM 학습용 Preference 데이터(Compliance Data Flywheel의 시작점).

자세한 내용은 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/ROADMAP.md`](docs/ROADMAP.md), [`docs/VCR_SPEC.md`](docs/VCR_SPEC.md) 참고.

## 레포 구조

```
compliance_gateway/      # ② Compliance Gateway (1순위 모듈)
  vcr/                    #   VCR 보상함수 (SourceExist / SourceMatch / ALCOA / Halluc)
  nli/                    #   NLI 백엔드 (lexical / statistical v0.5 / transformer)
  verify/                 #   3단계 인용 검증 (3-class, CrossRef/OpenAlex/로컬)
  data/                   #   합성 데이터 파이프라인 (bioRxiv → DPO) + 시드
    korean/               #     국내 R&D 한국어 데이터셋 (실기관 과제 + ALCOA+ 변조)
  train/                  #   학습 스캐폴드 (sft/dpo/grpo/nli_finetune, Unsloth) — A100용
  eval/                   #   SciFact 벤치마크 + KPI 측정 하니스(kpi.py)
  models.py              #   데이터 모델 (의존성 없음)
  pipeline.py            #   7단계 게이트웨이 파이프라인 오케스트레이션
  demo.py                #   VCR/Gateway 데모
docs/                    # 설계·로드맵·스펙·평가 문서
scripts/                 # 데이터 다운로드·시드 생성
tests/                   # 단위 테스트
```

## 데이터 / 평가

학습용 자체 데이터(GOONO)가 없는 단계에서, 공개·실데이터로 부트스트랩한다(초기 도메인: 제약·바이오).

```bash
# (A) bioRxiv 실논문 → DPO Preference 쌍 + Gateway 평가셋 (VCR 자기검증)
python -m compliance_gateway.data.build_dpo            # docs/SYNTH_PIPELINE.md

# (B) SciFact 로 NLI 백엔드 벤치마크
bash scripts/download_scifact.sh                       # CC BY-NC, S3(HF 차단 무관)
python -m compliance_gateway.eval.benchmark --split train   # docs/EVAL_SCIFACT.md

# (C) 외부 실데이터 KPI — 합성 낙관편향 제거. 성능 주장의 진짜 근거
python -m compliance_gateway.eval.external --split dev --sweep     # docs/EVAL_EXTERNAL.md

# (D) 국내 R&D 실데이터 — 국내 기관 실제 프로토콜 원문, 변조 없음
python -m compliance_gateway.data.korean.real_eval                 # docs/DATASET_KR.md
python -m compliance_gateway.eval.korean --real --threshold 0.64

# (D-2) 국내 R&D 합성셋 (회귀 테스트 전용)
python -m compliance_gateway.data.korean.build_kr
```

> ⚠️ **성능 보고 원칙**: 합성 평가셋(EN F1 98.2% / KR F1 100%)은 회귀 테스트 전용이며
> KPI 근거로 쓰지 않는다. **실데이터 기준 현재 상태**:
>
> | 평가셋 | 성격 | 결과 |
> |---|---|---|
> | 외부 EN (SciFact) | 전문가 주석 실데이터 | F1 **24.0%** (dev, θ=0.70) |
> | **실데이터 KR** (국내 기관 프로토콜 원문) | 교차 귀속 오류, 변조 없음 | **AUC 0.715, 사용 가능한 운영점 없음** |
> | 합성 EN / KR | 규칙 변조 | F1 98.2% / 100% ← 인공물 |
>
> 한국어 실데이터에서 위반 35건 **전부**가 정상 최저점 위에 있어 분포가 겹친다.
> 최대 약점은 ALCOA+ `Accurate`(수치·결과변수 오귀속) **29.4%**.
> 목표 90%까지는 트랜스포머 NLI 도입이 필수 조건.
> [`docs/EVAL_EXTERNAL.md`](docs/EVAL_EXTERNAL.md) · [`docs/DATASET_KR.md`](docs/DATASET_KR.md)

벤치마크 결과·해석: [`docs/EVAL_SCIFACT.md`](docs/EVAL_SCIFACT.md), [`docs/SYNTH_PIPELINE.md`](docs/SYNTH_PIPELINE.md).
활용 가능한 공개 데이터셋 카탈로그(용도·라이선스·접근): [`docs/DATASETS.md`](docs/DATASETS.md).

## A100 학습 계획

A100 2장 × 6개월 확보. 컴퓨트는 병목이 아니므로 데이터·평가에 과투자한다.
6개월 로드맵·KPI 매핑·실행 커맨드는 [`docs/A100_PLAN.md`](docs/A100_PLAN.md).

```bash
# KPI 측정(현재 baseline — A100 없이 동작). --compare 로 서지검증 백엔드 비교
python -m compliance_gateway.eval.kpi --compare

# A100 환경(HF 접근 가능)
pip install -e ".[train]" && pip install unsloth vllm
bash scripts/run_m1_a100.sh                             # M1 전체: 기준선→NLI 파인튜닝→재측정
torchrun --nproc_per_node 2 -m compliance_gateway.train.sft   # 도메인 LoRA (2 GPU DDP)
python -m compliance_gateway.train.dpo --vcr-accept 0.7 # VCR 정렬
python -m compliance_gateway.train.grpo --vllm-mode server  # RLVR (VCR = 보상함수)
```

최신 업스트림(Unsloth/TRL-vLLM/인용검증) 조사·적용 기록과 측정치: [`docs/UPSTREAM_TECH.md`](docs/UPSTREAM_TECH.md).

**적용 성과** — 3단계 인용 검증 도입으로 서지 변조 탐지 **9.1% → 100%**,
위반탐지 F1 **0.766 → 0.982** (오탐 증가 없음).

## 빠른 시작

```bash
pip install -e .
pytest
python -m compliance_gateway.demo   # VCR 스코어링 데모
```
