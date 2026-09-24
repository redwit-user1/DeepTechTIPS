# VCR — Verifiable Compliance Reward (스펙)

> 검증 가능한 규정준수 보상함수. "출처 존재 여부"라는 **객관적·자동생성 가능한** 기준으로
> Preference 데이터를 대규모 자동 구축하고, SLM 정렬(DPO/RLVR)의 보상함수로 사용한다.
> **본 과제의 핵심 IP.**

## 정의

```
VCR(y | x) = w1·SourceExist(y)
           + w2·SourceMatch(y)
           + w3·ALCOA_Score(y)
           + w4·(1 − Halluc(y))
```

- `x`: 사용자 질의 / 컨텍스트
- `y`: 모델 생성 응답
- 가중치 `w1..w4`: 합 = 1. 초기값은 도메인 중립, VCR v2에서 도메인별 자동 최적화(파일럿 피드백 기반).
- 출력 범위: `[0, 1]`. DPO 기준선 0.60, 목표 0.85+.

## 구성 요소

### SourceExist(y) — 인용 존재 여부
- 응답 내 식별 가능한 출처(DOI, URL, 논문 제목/저자) 존재 비율.
- **1차 구현**: 정규식 기반 식별자 파서 (DOI `10.xxxx/...`, URL, `Author et al. (YYYY)` 패턴).
- **목표**: 패턴 추출 + 외부 DB(ScienceON/NTIS) 실재 대조.

### SourceMatch(y) — 인용-내용 일치도
- 인용한 출처가 실제로 그 주장을 뒷받침하는지 (NLI entailment).
- **1차 구현**: 휴리스틱(주장-출처 컨텍스트 키워드 중첩률).
- **목표**: NLI 모델(entailment/neutral/contradiction) 문장 수준 평가.

### ALCOA_Score(y) — 데이터 무결성
ALCOA+ 9속성 중 우선 4속성(제약·바이오 도메인 핵심):
- **Attributable** (귀속가능): 생성 주체·시점·모델ID 식별 가능
- **Accurate** (정확): 수치·단위·조건 일관
- **Complete** (완전): 필수 항목 누락 없음
- **Consistent** (일관): 동일 사실의 모순 없음
- **1차 구현**: 규칙기반 체크. **목표**: 온톨로지 추론(OWL2/SPARQL).

### Halluc(y) — 환각 점수 (낮을수록 좋음)
사업계획서가 정의한 3가지 복합기만형 환각 유형을 탐지:
- **유형 A**: 가짜 DOI + 실존 저자 (DOI는 유효하나 해당 논문에 그 저자 없음)
- **유형 B**: 정교한 가짜 논문 제목 (전문용어 조합)
- **유형 C**: 수치 변조 (예: 37°C → 25°C)
- **1차 구현**: 출처 미바인딩 주장 비율 + 수치 일관성 휴리스틱.
- **목표**: NLI 기반 + 외부 DB 대조.

## DPO Preference 쌍 예시

```
y_w (선호): "Kim et al. (2024)에 따르면, pH 7.4에서 반응속도가 2.3배 향상"  ← 출처 ✓ 정확 ✓
y_l (비선호): "pH 7.4에서 반응속도가 향상되는 것으로 알려져 있습니다"        ← 출처 ✗
```

## 보상 해킹 방지 (RLVR 전환 시)

- 형식적 출처(존재하나 무관) 방지 → `w2 SourceMatch`가 `w1 SourceExist`를 견제.
- VCR 점수가 DPO 기준선(0.60) 미만으로 떨어지면 자동 롤백(Graceful Degradation).
- GRPO: 질의당 N=8 샘플링 → VCR 자동평가 → 상대 어드밴티지 학습 (`R(o) = VCR`).

## 구현 위치

- `compliance_gateway/vcr/reward.py` — `compute_vcr()` 집계
- `compliance_gateway/vcr/source_exist.py` / `source_match.py` / `alcoa.py` / `hallucination.py`
