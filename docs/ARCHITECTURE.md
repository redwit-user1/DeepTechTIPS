# 아키텍처 (Architecture)

GOONO AI는 **규정준수를 코어에 내재화한** Agentic AI 오케스트레이션 엔진이다.
범용 프레임워크(LangChain 등)가 "사후 검증" 구조인 것과 달리, 본 시스템은 **생성 중(in-process) 검증**과
**파라미터 수준 정렬(VCR)** 을 결합해 "출처 없는 생성"을 구조적으로 차단한다.

## 1. 4대 축 개요

### ① Compliance-Aligned SLM
- 7B 베이스(EXAONE 3.5 7.8B / Qwen 3.5 7B / Gemma 후보, 모두 Apache 2.0) + 도메인 LoRA.
- 정렬: **DPO → RLVR(GRPO)** 단계적 전환. 보상함수로 **VCR**(아래) 사용.
- 데이터: GOONO 70K+ 연구노트 → (질문, 답변, 출처) 트리플 자동추출.

### ② Compliance Gateway  ◀ 본 레포 1순위
모든 AI 출력이 반드시 통과하는 7단계 파이프라인. **Model-Agnostic** (자체 LoRA + 외부 API 모두 대응).

```
01 사용자 질의
02 Pre-Gen 제약 주입       (개인정보/규정위반 가능성 사전 스크리닝)
03 SLM/API 생성
04 출처 바인딩             (문장 수준 인용 자동 태깅)
05 NLI 게이팅              (신뢰도 임계값 θ 미달 → 차단 / 자동 재생성 1~3회)
06 ALCOA+ 체크포인트       (Attributable / Accurate / Complete / Consistent ...)
07 블록체인 감사추적       (생성시점·모델ID·출처·NLI점수·ALCOA점수 불변 기록)
```

- **자체 LoRA**: 로짓 접근 가능 → in-process 직접 개입(최고 정밀도).
- **API 모델(GPT/Claude)**: 로짓 미접근 → 텍스트 기반 준실시간 비동기 검증(Adapter 패턴). 사용자에게 우선 노출 + "검증 중" 배지 → 백그라운드 NLI 완료 후 확정.

### ③ R&D 에이전트 + DAG 오케스트레이션
3종 전문 에이전트, 모든 입출력은 Gateway 경유. **Human-in-the-loop 필수**.

| 에이전트 | 역할 | 외부 연동 |
|---|---|---|
| ① 탐색·기획 | 선행연구 갭 식별, RFP 매칭, 제안서 초안 | ScienceON / NTIS / KIPRIS |
| ② 실험분석·설계 | 통계분석, 이상치, DoE 실험설계 | GOONO DB |
| ③ 규정준수·재현성 | ALCOA+ 누락 식별, 출처 오류 제안(판단 X, 연구자 승인 필수) | Compliance Gateway |

DAG 엔진(LangGraph): 순차·병렬·조건부 분기 실행. 인지 사이클 4단계(목표설정→계획→실행→피드백).

### ④ Harnessing 아키텍처
- **Compliance Data Flywheel**: 사용 → 검증기록 → Preference 데이터 → 분기별 재학습 → 품질향상(구조적 해자).
- **R&D 특화 RAG**: 수식·표·그래프·구조식 경계 인식 청킹, Hybrid retrieval + Reranker.
- **프로젝트 메모리 L1/L2**: L1(세션 컨텍스트) / L2(장기 영구보존, 핵심 산출물 승급).
- **듀얼배포**: SaaS + On-premise(4-bit 양자화 GPTQ/AWQ, RTX 4090 단일 GPU, 에어갭).

### + Compliance Ontology (지식 기반)
- W3C OWL 2 / BFO 상위온톨로지 / Protégé / SPARQL / HermiT·Pellet reasoner.
- L1 규정주체(GLP·FDA Part 11·ALCOA+·GMP·EU Annex 11·ICH E6) → L2 객체 → L3 속성(ALCOA+ 9속성) → L4 관계(requires/violates/supersedes).
- **초기 전략**: 풀 온톨로지는 후순위. M1~M4 동안은 규칙기반으로 ALCOA+ 커버 후 점진 전환.

## 2. VCR — 핵심 IP

```
VCR(y|x) = w1·SourceExist(y) + w2·SourceMatch(y) + w3·ALCOA_Score(y) + w4·(1 − Halluc(y))
```

| 항목 | 의미 | 1차 구현 | 목표 구현 |
|---|---|---|---|
| `SourceExist` | 인용 존재 여부(DOI/URL/논문명 식별) | 정규식/식별자 파서 | 패턴 + DB 대조 |
| `SourceMatch` | 인용과 실제 내용 간 일치도 | 휴리스틱(키워드 중첩) | NLI 모델(entailment) |
| `ALCOA_Score` | 귀속가능·동시적·원본 등 우선 속성 | 규칙기반 체크 | 온톨로지 추론 |
| `Halluc` | 환각 탐지 점수 | 휴리스틱 | NLI 기반 |

> 상세: [`VCR_SPEC.md`](VCR_SPEC.md)

## 3. 모델-애그노스틱 검증의 솔직한 한계

API 모델은 로짓 접근이 불가하므로 "in-process 게이팅"이 물리적으로 불가능하다 → **사후 NLI 검증**으로 후퇴한다.
따라서 `gating` 단계는 두 경로를 명시적으로 분기한다.

- `mode="logit"` (자체 LoRA): 토큰 생성 중 개입.
- `mode="post"` (API): 생성 완료 후 텍스트 기반 검증 + 자동 재생성.

이 갭을 제품/마케팅에서 정확히 커뮤니케이션하는 것이 신뢰성 확보의 전제다.

## 4. 데이터 흐름 (Gateway 중심)

```
        ┌─────────────── Compliance Data Flywheel ───────────────┐
        ▼                                                         │
  사용자 질의 → [Gateway 7단계] → 검증된 출력 → 사용자 피드백 ──┘
                     │                              │
                     ▼                              ▼
              블록체인 감사로그            Preference 데이터(VCR 채점)
                                                    │
                                            분기별 DPO/RLVR 재학습 → SLM
```
