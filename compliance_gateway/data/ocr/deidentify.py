# -*- coding: utf-8 -*-
"""연구노트 비식별화 — 학습·공유 전 필수 단계.

연구노트는 **미공개 연구내용·개인정보·영업비밀**을 담는 법적 기록이다.
그대로 학습에 쓰거나 외부로 내보내면 안 된다.

## 설계 원칙

1. **되돌릴 수 없게(비가역)** — 해시 기반 가명 치환. 원본 복원 불가.
2. **구조는 보존** — `연구자: 김민준` → `연구자: 연구자_A3F2`.
   ALCOA+ `Attributable` 검사가 "서명이 있는가"를 계속 판정할 수 있어야 한다.
   단순 삭제하면 정상 노트가 위반으로 오판된다.
3. **일관성** — 같은 이름은 같은 가명으로. 문서 간 동일인 추적이 가능해야
   `Consistent` 검사가 의미를 갖는다.
4. **감사 가능** — 무엇을 몇 건 치환했는지 보고한다(값은 남기지 않음).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# 치환 규칙: (이름, 패턴, 가명 접두)  — 그룹 1이 있으면 그 부분만 치환
RULES: list[tuple[str, re.Pattern, str]] = [
    ("주민등록번호", re.compile(r"\d{6}\s*[-–]\s*[1-4]\d{6}"), "주민번호"),
    ("휴대전화", re.compile(r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}"), "연락처"),
    ("이메일", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "이메일"),
    # 역할 라벨 뒤의 인명만 치환한다(본문 속 일반 한글은 건드리지 않음)
    ("연구자명", re.compile(r"(?<=연구자[:： ])\s*([가-힣]{2,4})"), "연구자"),
    ("작성자명", re.compile(r"(?<=작성자[:： ])\s*([가-힣]{2,4})"), "작성자"),
    ("점검자명", re.compile(r"(?<=점검자[:： ])\s*([가-힣]{2,4})"), "점검자"),
    ("확인자명", re.compile(r"(?<=확인자[:： ])\s*([가-힣]{2,4})"), "확인자"),
    ("성명", re.compile(r"(?<=성명[:： ])\s*([가-힣]{2,4})"), "성명"),
]


@dataclass
class DeidentifyResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)
    mapping_size: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _pseudonym(value: str, prefix: str, salt: str) -> str:
    """되돌릴 수 없는 가명. 같은 값 → 같은 가명(문서 간 일관성)."""
    h = hashlib.sha256((salt + "::" + value.strip()).encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:4].upper()}"


def deidentify(text: str, salt: str = "goono-deid-v1") -> DeidentifyResult:
    """개인정보를 비가역 가명으로 치환한다.

    salt 는 배포마다 바꾸고 **외부에 공개하지 않는다**
    (공개되면 후보군 대조로 재식별 시도가 가능해진다).
    """
    counts: dict[str, int] = {}
    mapping: dict[str, str] = {}

    for name, pat, prefix in RULES:
        def repl(m: re.Match) -> str:
            raw = m.group(1) if m.groups() else m.group(0)
            key = f"{prefix}:{raw.strip()}"
            if key not in mapping:
                mapping[key] = _pseudonym(raw, prefix, salt)
            counts[name] = counts.get(name, 0) + 1
            whole = m.group(0)
            return whole.replace(raw, mapping[key]) if m.groups() else mapping[key]

        text = pat.sub(repl, text)

    return DeidentifyResult(text=text, counts=counts, mapping_size=len(mapping))


# 이미 치환된 가명(연구자_9565 등)은 위험이 아니다.
# 인명 패턴은 한글만 캡처하므로 가명의 앞부분("연구자")만 잡힌다
# → 매치 **직후** 에 `_XXXX` 가 오는지로 판정해야 한다.
_PSEUDO_TAIL = re.compile(r"^_[0-9A-F]{4}\b")


def audit(text: str) -> dict[str, int]:
    """치환하지 않고 **잔존 위험만** 센다(비식별화 후 검증용).

    가명 자체를 위험으로 세면 비식별화가 끝나도 경고가 남아 판단을 흐린다.
    """
    out: dict[str, int] = {}
    for name, pat, _prefix in RULES:
        remaining = 0
        for m in pat.finditer(text):
            if _PSEUDO_TAIL.match(text[m.end():m.end() + 8]):
                continue                       # 이미 가명 처리됨
            remaining += 1
        if remaining:
            out[name] = remaining
    return out
