"""기록 무결성(Record Integrity) 검사 — 연구노트 등 **1차 기록**용.

## VCR 과의 역할 분담

| | 검증 대상 | 질문 | 모듈 |
|---|---|---|---|
| **VCR** | AI 생성물 | 출처에 근거하는가 | `vcr/` |
| **RecordIntegrity** | 1차 기록(연구노트) | 기록 자체가 무결한가 | `integrity/` |

연구노트는 외부 출처를 인용하지 않는다 — 그 자체가 원본 증빙이다.
따라서 `SourceExist`/`SourceMatch` 기반 VCR 로는 평가할 수 없고
(실제로 정상 노트도 VCR 0.42 로 전부 차단됐다), 기록 무결성 관점의
별도 검사가 필요하다. 국가연구개발 연구노트 지침의 필수 요건을 그대로 구현한다.

텍스트 기반으로 동작한다 — 운영에서는 스캔/OCR 된 노트도 대상이 되기 때문이다.
"""

from compliance_gateway.integrity.record import (
    IntegrityReport,
    RecordIntegrityChecker,
    check_lab_note,
)

__all__ = ["IntegrityReport", "RecordIntegrityChecker", "check_lab_note"]
