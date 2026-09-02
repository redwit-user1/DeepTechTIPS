"""국내 연구데이터 특화 데이터셋.

목적: 기존 평가는 **영어·바이오(SciFact)** 한정이었다. 과제 KPI 는 한국어 R&D 문서가
대상이므로, 한국어·국내 연구 맥락의 평가·학습 데이터가 필요하다.

## 데이터 출처 현황 (본 환경 네트워크 정책 기준)

| 출처 | 상태 | 비고 |
|---|---|---|
| ClinicalTrials.gov (국내 기관) | **확보** (MCP) | 실제 국내 연구기관 연구과제 레코드 |
| bioRxiv/medRxiv | 확보 (MCP) | 국내 기관 소속 프리프린트 |
| arXiv | 차단 | egress 정책 — 어댑터만 준비 |
| ScienceON / NTIS / KCI | 차단 | egress 정책 — 어댑터만 준비 |
| AI Hub 논문요약 | 차단 | 가입·약관 필요 |

차단된 출처는 `sources.py` 에 **API 명세 기반 어댑터**를 준비해 두었다.
외부망 환경(A100 등)에서 키/네트워크만 주어지면 즉시 수집이 시작된다.

## 한국어 텍스트의 성격 (정직한 기술)

`kr_render.py` 가 만드는 한국어 문장은 **실제 국내 연구과제 메타데이터**(기관·과제번호·
등록례수·시작일)를 한국어 R&D 문체로 렌더링한 것이다.
- 사실(fact)은 실데이터다 — 기관명·NCT번호·등록례수·날짜 모두 실제 값.
- 문장(surface form)은 템플릿이다 — 자연 한국어 코퍼스를 수집한 것이 **아니다**.
따라서 한국어 토크나이제이션·수치/귀속 검증 평가에는 유효하나,
자연 한국어 문체 다양성 평가로는 부족하다. 후자는 ScienceON/KCI 수집이 필요하다.
"""

from compliance_gateway.data.korean.models import KRResearchRecord
from compliance_gateway.data.korean.sources import KR_INSTITUTIONS, is_korean_institution, load_kr_seed

__all__ = ["KRResearchRecord", "KR_INSTITUTIONS", "is_korean_institution", "load_kr_seed"]
