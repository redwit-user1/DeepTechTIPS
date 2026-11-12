# -*- coding: utf-8 -*-
"""민감정보(건강) 문서 표지 — 비식별화가 못 거르는 것을 **문서 단위**로 거른다.

`deidentify` 는 이름·식별번호 **패턴**만 지운다. 그런데 연구노트에는 특정 아동의 청력·언어
평가, 임상 부작용, 환자 경과 같은 **개인 건강정보**가 서술로 들어 있다(2026-11-09 평가셋
작성 중 확인). 개인정보보호법 제23조 민감정보라 이름을 지워도 학습에 넣으면 안 된다.

판정: **개인 대상 표지 + 건강 표지가 함께** 나오거나, 이상반응 표지가 나오면 민감.
세포·마우스 실험은 대상 표지가 없으므로 걸리지 않는다.

ponytail: 정규식 표지 — 재현율 우선. 오탐(일반 임상 문헌 서술)은 버려도 손실이 작다.
정밀하게 가르려면 사람 검수 표본으로 표지를 조정할 것.
"""

from __future__ import annotations

import re
from typing import Iterable

# 개인(사람) 대상 — 실험 대상이 사람임을 뜻하는 표지
SUBJECT = re.compile(
    r"환자|피험자|대상자|참여자|수진자|내담자|아동|영아|유아|소아|신생아|남아|여아|아이(?:가|는|의|에게|들)|"
    r"\d{1,2}\s*세(?![대기트])\s*(?:남|여|아동|환자)?|\d{1,2}\s*개월\s*(?:된|아|영아)|"
    r"(?:남|여)성\s*\d{1,2}\s*세|patient|subject\s*\d|participant|child(?:ren)?\b",
    re.IGNORECASE,
)
# 건강·임상 서술
HEALTH = re.compile(
    r"진단|병력|증상|질환|질병|투약|투여|처방|복용|수술|입원|외래|재활|치료|"
    r"청력|난청|언어\s*발달|발달\s*(?:평가|지연|검사)|인지\s*(?:검사|기능)|"
    r"검체|조직\s*검사|병리|통증|요통|부상|종양|암\s*(?:진단|환자|조직|병기|재발)|"
    r"(?:폐|위|간|유방|대장|갑상선|췌장|전립선|자궁경부)암|"
    r"혈압|혈당|체중\s*감소|BMI|diagnos|symptom|medical history|treatment",
    re.IGNORECASE,
)
# 이것만으로 민감 — 사람에 대한 안전성 서술
ADVERSE = re.compile(r"부작용|이상\s*반응|이상\s*사례|중대한\s*이상|adverse\s+event|side\s+effect|\bSAE\b",
                     re.IGNORECASE)


# 대상 표지 없이도 개인 병력으로 읽히는 서술 — 질환명과 임상 동사가 가까이 붙는다("홍길동은 당뇨 진단 후 투약").
# SUBJECT 가 없으면 놓치던 경우(Codex 교차검증 지적). 동물·세포·치료제 개발 맥락은 뺀다.
_DISEASE = (r"당뇨|고혈압|고지혈증|천식|우울증|공황\s*장애|불안\s*장애|조현병|치매|뇌졸중|심근경색|협심증|"
            r"간염|결핵|아토피|ADHD|자폐|파킨슨|알츠하이머|관절염|디스크")
CLINICAL = re.compile(rf"(?:{_DISEASE})[^.\n]{{0,15}}?(?:진단|투약|처방|복용|입원|수술|병력)"
                      rf"|(?:진단|병력)[^.\n]{{0,8}}?(?:{_DISEASE})", re.IGNORECASE)
# 10만 행 표본 12건 중 진단기기·키트 개발 서술이 절반 — 문서를 빼는 쪽이라 오탐은 손실만 작다
NONHUMAN = re.compile(r"마우스|생쥐|랫드|래트|동물|세포|모델|유발|치료제|후보\s*물질|신약|in\s*vitro|키트|"
                      r"진단\s*(?:기기|용|시약|기술|장비|모니터링)"
                      r"|\bmice\b|\bmouse\b|\brats?\b|\bcells?\b", re.IGNORECASE)


def sensitive_signals(text: str) -> list[str]:
    """민감 사유 목록(비면 통과)."""
    out = []
    if ADVERSE.search(text or ""):
        out.append("adverse")
    # 건강 표지가 더 드물다 — 먼저 보고 없으면 대상 표지는 검사하지 않는다(전수 처리 속도)
    if HEALTH.search(text or "") and SUBJECT.search(text or ""):
        out.append("subject_health")
    elif CLINICAL.search(text or "") and not NONHUMAN.search(text or ""):
        out.append("clinical")
    return out


def sensitive_notes(rows: Iterable, key) -> frozenset[int]:
    """한 행이라도 민감하면 그 **문서 전체**를 막는다. `key(note_id) -> int`."""
    return frozenset(key(r.note_id) for r in rows if sensitive_signals(r.text))


def _demo() -> None:
    assert sensitive_signals("만 5세 남아의 청력 검사 결과 우측 난청 소견") == ["subject_health"]
    assert sensitive_signals("투여군에서 이상반응 2건이 보고되었다") == ["adverse"]
    assert sensitive_signals("환자 12명 중 3명에서 증상 완화") == ["subject_health"]
    assert sensitive_signals("아이가 4세부터 청력 재활을 받았다") == ["subject_health"]
    assert not sensitive_signals("새로운 아이디어로 공정 치료 단계를 줄였다")
    assert sensitive_signals("환자 검체를 1:100 희석 후 염색") == ["subject_health"]
    assert sensitive_signals("43세 운동 후 허리 통증 지속") == ["subject_health"]
    assert not sensitive_signals("2세대 배터리 셀의 진단 알고리즘 정확도 95%"), "1세대·2세대는 나이가 아니다"
    assert not sensitive_signals("3세트 반복 후 장비 진단 결과 정상"), "세트는 나이가 아니다"
    assert not sensitive_signals("15세기 문헌의 질병 기록 분석"), "세기는 나이가 아니다"
    assert sensitive_signals("홍길동은 당뇨 진단 후 투약 중") == ["clinical"]
    assert sensitive_signals("고혈압 병력 있음, 약 복용") == ["clinical"]
    assert not sensitive_signals("당뇨 유발 마우스에 후보물질 투약")
    assert not sensitive_signals("당뇨 치료제 후보물질 스크리닝 및 처방 가능성 검토")
    # 세포·동물 실험, 공정 측정은 통과
    assert not sensitive_signals("HeLa 세포에 10 uM 처리 후 생존율 62% 감소")
    assert not sensitive_signals("마우스 투여군의 체중 변화는 없었다")
    assert not sensitive_signals("용접 전류 270A 에서 열영향부 폭이 좁아짐")
    print("sensitive demo ok")


if __name__ == "__main__":
    _demo()
