"""연구노트(Lab Notebook) 가상 데이터셋.

⚠️ **전부 가상(SYNTHETIC) 데이터다.** 실제 연구기록이 아니며, 모든 레코드에
`synthetic: true` 와 가상 표식(과제번호 `SYN-`, 가상 연구자명)이 박혀 있다.
연구노트는 **법적 증빙 기록**이므로 실제 기록으로 오인되어서는 안 된다.

## 왜 만드는가

GOONO 의 핵심 제품이 전자연구노트(ELN)이고, 사업계획서의 학습 데이터도
연구노트 70K+ 다. 그러나 실데이터 접근이 없으므로, **국가연구개발 연구노트 지침의
실제 구조**를 참고한 가상 데이터로 파이프라인을 검증한다.

또한 지금까지 확보한 국내 실데이터(임상 프로토콜)는 텍스트가 영어였다.
이 데이터셋은 **자연스러운 한국어 R&D 문체**를 다루는 첫 데이터다.

## 연구노트 지침 → ALCOA+ 매핑

국가연구개발사업 연구노트 지침의 필수 요건은 ALCOA+ 와 1:1 대응한다.

| 지침 요건 | ALCOA+ | 위반 유형 |
|---|---|---|
| 연구자 서명·기록일 | Attributable | `missing_signature` |
| 점검자(입회자) 서명 | Attributable | `missing_reviewer` |
| 실험 당일 기록 | Contemporaneous | `backdated` |
| 수정 시 원본 식별 가능(한 줄 긋기·정정 사유) | Original | `overwritten` |
| 정확한 수치·단위 | Accurate | `missing_units` |
| 누락 없는 기재(여백 사선 처리 포함) | Complete | `incomplete` |
| 본문과 요약의 일치 | Consistent | `inconsistent` |

**KPI 근거로 쓰지 말 것** — 규칙 기반 생성물이므로 낙관 편향이 있다.
회귀 테스트와 한국어 문체 처리 검증용이다.
"""

from compliance_gateway.data.labnote.generate import build_dataset, generate_notes
from compliance_gateway.data.labnote.models import LabNote

__all__ = ["LabNote", "generate_notes", "build_dataset"]
