"""OCR 연구노트 실데이터 수확 파이프라인.

로컬에 보유한 OCR 연구노트 데이터셋을 본 프로젝트의 평가·학습 데이터로 전환한다.
지금까지의 최대 공백인 **자연 한국어 R&D 실텍스트**를 채우는 경로다.

## 처리 순서

1. `scripts/profile_ocr_dataset.py` — 내용 노출 없이 구조·품질·PII 위험 파악
2. `deidentify.py` — **비식별화**(연구노트는 미공개 연구내용·개인정보 포함)
3. `parse.py` — OCR 텍스트 → 구조화 레코드(연구노트 구획 인식, OCR 노이즈 내성)
4. `harvest.py` — 평가셋/DPO 생성 + 기록 무결성 검사 적용

## 원칙

- **비식별화 없이 학습·공유 금지.** 연구노트는 영업비밀·개인정보를 담는다.
- 원문은 저장소에 커밋하지 않는다(`data/` 는 gitignore).
- 실데이터 결과는 합성 데이터 결과와 **분리해서** 보고한다.
"""

from compliance_gateway.data.ocr.deidentify import DeidentifyResult, deidentify
from compliance_gateway.data.ocr.parse import ParsedNote, parse_note

__all__ = ["deidentify", "DeidentifyResult", "parse_note", "ParsedNote"]
