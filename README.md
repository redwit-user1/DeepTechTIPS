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
    labnote/              #     연구노트 가상 데이터 (한국어 자연 문체, SYNTHETIC 표식)
    ocr/                  #     OCR 실데이터 수확 (비식별화·노이즈 내성 파서)
  integrity/              #   기록 무결성 검사 — 1차 기록(연구노트)용, VCR과 역할 분담
  train/                  #   학습 스캐폴드 (sft/dpo/grpo/nli_finetune, Unsloth) — H100용
  serving/                #   OpenAI 호환 서빙 연동 (AI Serv 등, 학습 없이 평가)
  eval/                   #   SciFact 벤치마크 + KPI 하니스 + 외부/국내 실데이터 평가
  models.py              #   데이터 모델 (의존성 없음)
  pipeline.py            #   7단계 게이트웨이 파이프라인 오케스트레이션
  demo.py                #   VCR/Gateway 데모
docs/                    # 설계·로드맵·스펙·평가 문서
scripts/                 # 환경진단(probe_env)·데이터 다운로드·H100 런북
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

# (E) 연구노트 가상 데이터 + 기록 무결성 검사 (한국어 자연 문체)
python -m compliance_gateway.data.labnote.generate --notes 40   # docs/DATASET_LABNOTE.md
python -m compliance_gateway.eval.labnote --compare-vcr

# (F) 보유 중인 OCR 연구노트 실데이터 수확 — 로컬에서 실행
#     GPU·torch 불필요(표준 라이브러리만). docs/OCR_HARVEST.md
export GOONO_DEID_SALT='조직 고유 비밀값'
bash scripts/run_local_ocr.sh /path/to/ocr          # 프로파일링 → 표본 수확
bash scripts/run_local_ocr.sh /path/to/ocr --write  # 전량 수확·저장
```

> 🔍 **VCR 로는 연구노트를 평가할 수 없다.** 연구노트는 1차 기록이라 인용이 없어
> `SourceExist`/`SourceMatch` 가 구조적으로 0에 가깝고, 정상 노트조차 전부 차단된다.
> → 기록 무결성 검사(`integrity/`)로 분리했다: AUC **1.000** vs VCR 0.607.

> ⚠️ **성능 보고 원칙**: 합성 평가셋(EN F1 98.2% / KR F1 100%)은 회귀 테스트 전용이며
> KPI 근거로 쓰지 않는다. **실데이터 기준 현재 상태**:
>
> | 평가셋 | 성격 | 결과 |
> |---|---|---|
> | 외부 EN (SciFact) | 전문가 주석 실데이터 | F1 **24.0%** (dev, θ=0.70) |
> | **실데이터 KR** (국내 8개기관 24과제 원문, n=186) | 교차 귀속 오류, 변조 없음 | ~~AUC 0.704~~ **판정 불가 평가** — 근거에 정답 값이 거의 없다 |
> | 실데이터 KR `--full-record` (근거에 과제 레코드 포함) | 판정 가능판 | 통계 0.868 / DeBERTa-large 0.900 / bge-m3 **0.969** (zero-shot) |
> | **한국어 연구노트** (원문 단락→보고서 문장, 97단락 388행, Claude 작성·판정) | 바꿔쓰기·수치/방향 변조·근거 없는 덧붙임 | 통계 AUC 0.394 / KLUE FT F1 **96.4%** / Qwen3.8-27B F1 **99.5%** |
> | **한국어 어려운 셋** (반올림·단위바꿈·부등호·단정화, 189행) | 인코더가 무너지는 경우 | 인코더 AUC 0.70~0.77 / Qwen v2 **0.996** · **CascadeNLI(증류 학생+새 숫자 라우팅+Qwen, 호출 24%) 전체 F1 97.6%, 어려운 셋 정밀도 93.0%** |
> | 합성 EN / KR | 규칙 변조 | F1 98.2% / 100% ← 인공물 |
>
> ⚠️ **정정(2026-11-09)**: 0.704 는 모델 한계가 아니라 **평가 설계 결함**이었다. 근거(과제 서술)에
> 정답 값이 있는 비율이 결과변수·선정기준 0%, 등록례수 26%, 수행기관 9% 다. `--full-record` 도
> 정답이 근거에 문자 그대로 있는 쉬운 조건이라 KPI 근거가 아니다(패러프레이즈 판 필요).
> SciFact 파인튜닝(2분류)은 SciFact dev F1 92.9% 지만 한국어 귀속 과제는 움직이지 않았다(0.900→0.898).
> 병목은 모델이 아니라 **사람이 라벨링한 한국어 평가셋**이다. [`docs/notebook/2026-11-09.md`](docs/notebook/2026-11-09.md)
> [`docs/EVAL_EXTERNAL.md`](docs/EVAL_EXTERNAL.md) · [`docs/DATASET_KR.md`](docs/DATASET_KR.md)

벤치마크 결과·해석: [`docs/EVAL_SCIFACT.md`](docs/EVAL_SCIFACT.md), [`docs/SYNTH_PIPELINE.md`](docs/SYNTH_PIPELINE.md).
활용 가능한 공개 데이터셋 카탈로그(용도·라이선스·접근): [`docs/DATASETS.md`](docs/DATASETS.md).
차단된 소스의 확보 절차(브라우저 에이전트 활용 포함): [`docs/DATA_ACQUISITION.md`](docs/DATA_ACQUISITION.md).

## GPU 실행 (KT Cloud AI Nexus · H200 141GB × 2)

NIPA 고성능 컴퓨팅 지원사업으로 받은 **Backend.AI 컨테이너 세션**이다(문서 초안의 "H100 × 2 / AI Train·Serv / teleport" 가정은 틀렸다).
환경 실측·제약은 [`docs/H100_PLAN.md`](docs/H100_PLAN.md) 상단.

1. 웹 UI → 세션 시작: 인터랙티브, **NGC PyTorch 25.03**, 프리셋 `H200-141G-1GPU`, 폴더 **`research-data-analysis`** 마운트
2. 세션 앱 → **SSH/SFTP** → `id_container` 받기. 첫 연결이 끊기면 아이콘을 **한 번 더** 눌러 데몬을 깨운다
   (호스트 `proxy{1,2,3}.ainexus.ktcloud.com`·포트가 세션마다 다르다)
3. 레포는 git 추적 파일만 tar 로 보낸다(`git ls-files | tar -czf - -T -`) — 고객 데이터(`data/`, `*.csv`)는 보내지 않는다
4. `bash repo/scripts/ainexus_setup.sh` — NAS venv + Qwen3.8-27B
5. **GPU util 6시간 평균 1% 미만이면 세션이 회수된다.** 체크포인트는 NAS 폴더에 둔다. 끝나면 세션 종료(공용 GPU)

```bash
# 인코더 NLI 학습 (시스템 파이썬)
python -m compliance_gateway.train.nli_finetune --data klue --labels 3 --base MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
# 한국어 연구노트 평가 (data/labeling/kr_psg_eval.jsonl · kr_hard_eval.jsonl)
python -m compliance_gateway.eval.kr_nli <평가셋> --nli cascade            # 규칙→DeBERTa→Qwen
. venv/bin/activate && python -m compliance_gateway.eval.kr_nli <평가셋> --llm Qwen/Qwen3.8-27B
# 캐스케이드 비용-성능 (kr_nli --dump 결과로)
python scripts/cascade_sim.py --scores data/scores --enc deberta --llm qwen2
```

**SLM 정렬(SFT/DPO/GRPO, 베이스 Qwen3.8-27B)** 은 검증기가 사람 검수(κ ≥ 0.8)를 통과한 뒤 착수한다 —
검증기가 틀리면 정렬이 틀린 목표를 학습한다. GRPO 전 `reward_check` 관문은 그대로 유효하다.

## 빠른 시작

```bash
pip install -e .
pytest
python -m compliance_gateway.demo   # VCR 스코어링 데모
```
