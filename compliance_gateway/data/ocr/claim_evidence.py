# -*- coding: utf-8 -*-
"""근거·주장 쌍 추출 — 행 분류기에 의존하지 않는다.

## 왜 문장 단위인가

원래 계획은 `experimental`(근거) 행과 `analysis`(주장) 행을 문서 안에서 짝짓는
것이었다. 그런데 실측에서 `analysis` **정밀도가 28.6%** 로 나왔다. 임계값을
어떻게 조정해도 고쳐지지 않았다(v6 실험: 정밀도 23p 손실, 재현율 3p 이득).

근본 원인은 한 행 안에 측정값과 해석이 **같이** 들어 있다는 것이다.
행 전체를 하나의 범주로 밀어넣는 것 자체가 틀린 모델이었다.

→ 행을 분류하지 않고 **문장**을 본다. 같은 행 안에서도 수치가 있는 문장은
근거, 결론 표현이 있는 문장은 주장으로 나뉜다. 분류기 정확도 60% 에
의존하던 것이 문장 정규식으로 내려가 훨씬 견고해진다.

## 만드는 것

    양성        주장 + 같은 문서의 근거
    음성(쉬움)   주장 + 다른 문서의 근거
    음성(어려움) 주장 + 수치를 교란한 근거   ← 사업계획서 환각 유형 C

유형 C(수치 변조, 37°C → 25°C)는 `VCR_SPEC.md` 가 명시한 탐지 대상이다.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# 수치+단위 — 근거 문장의 표지.
# 뒤에 영숫자·하이픈이 이어지면 단위가 아니라 **식별자**다. 실측에서
# 제품코드 `F-3340M-20` 의 M 을 몰농도 M 으로 잡아 `F-783M` 으로 교란한 사례가
# 나왔다. 식별자 변조는 수치 변조(환각 유형 C)가 아니므로 반드시 배제한다.
NUM_UNIT = re.compile(
    r"(?<![A-Za-z0-9-])(\d+(?:\.\d+)?)\s*"
    r"(℃|°C|%|㎍|μg|ug|mg|kg|g|㎖|mL|L|μM|uM|mM|nM|M|"
    r"nm|㎛|μm|mm|cm|rpm|hr|min|sec|시간|분|초|일|주|개월|배|명|건|회|"
    r"ppm|kPa|MPa|GPa|W|kW|V|A|Hz|kHz|㎡|m2|mol|eq|wt%|vol%)"
    r"(?![A-Za-z0-9-])"
)
# 주장이 아닌 것 — 파일명·커밋 메시지·경로. 실측 샘플에서 claim 으로 잘못 뽑혔다.
NOT_CLAIM = re.compile(
    r"\.(?:pptx?|docx?|xlsx?|pdf|hwp|jpe?g|png|zip|csv)\b|"
    r"^\s*\[?(?:Commit|Merge|Revert|feat|fix|chore|refactor)\b|"
    r"refs?\s*#\d|https?://|[A-Za-z]:\\|/[a-z]+/[a-z]+/",
    re.IGNORECASE,
)
# 근거가 아닌 것 — base64·해시·난수 나열. neutral 근거에서 다수 나왔다.
NOT_EVIDENCE = re.compile(r"[A-Za-z0-9+/]{40,}|(?:\b[0-9a-f]{16,}\b)")
# 내용어 — 어휘 중첩 판정용. 한글 2자 이상 또는 영문 3자 이상.
CONTENT_WORD = re.compile(r"[가-힣]{2,}|[A-Za-z]{3,}")
_STOP = {"그리", "그러", "때문", "경우", "대한", "통해", "위한", "이상", "이하", "결과", "확인", "사용", "진행", "다음",
         "the", "and", "for", "with", "was", "were", "this", "that", "from"}
# 결론 표현 — 주장 문장의 표지
CLAIM = re.compile(
    r"증가|감소|향상|저하|개선|악화|유의(?:하게|한|함|차)|"
    r"확인(?:되|됨|하였|했)|나타났|나타냄|보였|보임|관찰(?:되|됨)|"
    r"더\s*(?:높|낮|크|작|우수)|우수(?:하|했|함)|효과적|"
    r"경향(?:을|이)|결론|시사|의미한다|판단(?:된|됨)|"
    r"significant|increase[sd]?|decrease[sd]?|improve[sd]?|"
    r"higher|lower|superior|observ(?:ed|ation)|suggest",
    re.IGNORECASE,
)
# 문장 경계. 소수점(3.14)과 약어(et al.)를 문장 끝으로 보지 않는다.
_SPLIT = re.compile(r"(?<![0-9A-Za-z])\.(?=\s)|[!?。]|\n+|(?<=[다음함됨])\.\s")

MIN_SENT, MAX_SENT = 15, 400


def sentences(text: str) -> list[str]:
    out = []
    for s in _SPLIT.split(text or ""):
        s = re.sub(r"\s+", " ", (s or "")).strip()
        if MIN_SENT <= len(s) <= MAX_SENT:
            out.append(s)
    return out


def words(s: str) -> set[str]:
    """내용어 집합. 한글은 **앞 2자만** 취해 조사·어미를 턴다.

    형태소 분석기 없이 `수율이`와 `수율`을 같게 보려면 이 정도가 필요하다.
    거칠지만 `min_overlap=2` 요건이 우연 일치를 걸러 준다.
    """
    out = set()
    for w in CONTENT_WORD.findall(s):
        out.add(w[:2] if "가" <= w[0] <= "힣" else w.lower())
    return out - _STOP


def is_evidence(s: str) -> bool:
    """수치+단위가 둘 이상이고, 난수 나열이 아니며, 내용어가 있는 문장."""
    if NOT_EVIDENCE.search(s) or len(NUM_UNIT.findall(s)) < 2:
        return False
    return len(words(s)) >= 2


def is_claim(s: str) -> bool:
    """결론 표현이 있고, 수치 나열도 파일명·커밋도 아닌 문장."""
    if not CLAIM.search(s) or NOT_CLAIM.search(s):
        return False
    # 수치가 너무 많으면 주장이 아니라 측정값 표다
    return len(NUM_UNIT.findall(s)) <= 2 and len(words(s)) >= 3


def numbers(s: str) -> set[str]:
    """문장에 등장한 (값, 단위) 쌍."""
    return {f"{v}{u}" for v, u in NUM_UNIT.findall(s)}


def supports(claim: str, evidence: str) -> bool:
    """근거가 주장을 **실제로** 뒷받침할 만한가.

    같은 문서라는 것만으로 짝지었더니 `entail` 의 상당수가 거짓이었다
    (실측 샘플: "품질이 향상되었다" ↔ "TE-201 96.8→95.8"). 그대로 학습하면
    "같은 문서면 뒷받침한다"를 배우는데, 이는 VCR 이 잡아야 할 것과 정반대다.

    두 가지 중 하나를 요구한다.

    - 내용어 **2개** 이상 공유, 또는
    - 내용어 1개 + **수치 공유** — 주장의 수치가 근거에 있다는 것은
      VCR 이 검증하려는 관계 그 자체이므로 강한 신호다.

    문장이 평균 47자로 짧아 내용어가 3개 남짓뿐이라, 2개 요구만으로는
    정당한 쌍까지 잘려 나간다. 수치 공유를 대체 근거로 인정한다.
    """
    w = len(words(claim) & words(evidence))
    if w >= 2:
        return True
    return w >= 1 and bool(numbers(claim) & numbers(evidence))


def perturb(evidence: str, rng: random.Random) -> tuple[str, list[str]]:
    """수치를 바꾼 하드 네거티브. 환각 유형 C 를 그대로 재현한다.

    자릿수를 유지하되 값을 확실히 다르게 만든다. 단위는 건드리지 않는다 —
    바꾸면 단위 불일치라는 **다른** 오류가 되어 무엇을 학습하는지 흐려진다.
    """
    changes: list[str] = []

    def one(m: re.Match) -> str:
        val, unit = m.group(1), m.group(2)
        if "." in val:
            head, tail = val.split(".", 1)
            new = f"{head}.{(int(tail) + rng.randint(1, 8)) % (10 ** len(tail)):0{len(tail)}d}"
            if new == val:
                new = f"{int(head) + 1}.{tail}"
        else:
            delta = max(1, int(int(val) * rng.uniform(0.2, 0.8)))
            new = str(int(val) + delta * rng.choice((1, -1)))
            if new == val or new.startswith("-"):
                new = str(int(val) + delta)
        changes.append(f"{val}{unit}->{new}{unit}")
        return f"{new} {unit}"

    return NUM_UNIT.sub(one, evidence), changes


@dataclass
class NoteBuffer:
    """문서 하나에서 모은 근거·주장 문장. 원문 전체는 들고 있지 않는다."""

    claims: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    rows: int = 0

    def add(self, text: str, cap: int = 12) -> None:
        self.rows += 1
        for s in sentences(text):
            if len(self.claims) < cap and is_claim(s):
                self.claims.append(s)
            elif len(self.evidence) < cap and is_evidence(s):
                self.evidence.append(s)

    @property
    def usable(self) -> bool:
        return bool(self.claims) and bool(self.evidence)


def _demo() -> None:
    txt = ("본 실험은 37 ℃ 에서 24 시간 배양하였다. "
           "대조군 대비 수율이 12.4 % 에서 18.7 % 로 유의하게 증가하였다. "
           "시료 3 개를 각각 5 mL 씩 분주하였다.")
    ss = sentences(txt)
    assert len(ss) == 3, ss
    ev = [s for s in ss if is_evidence(s)]
    cl = [s for s in ss if is_claim(s)]
    assert len(ev) >= 2, ev
    assert len(cl) == 1, cl
    assert "증가" in cl[0]

    rng = random.Random(0)
    bad, ch = perturb(ev[0], rng)
    assert bad != ev[0] and ch, (bad, ch)
    # 단위는 보존되어야 한다
    assert [u for _, u in NUM_UNIT.findall(bad)] == [u for _, u in NUM_UNIT.findall(ev[0])]

    nb = NoteBuffer()
    nb.add(txt)
    assert nb.usable and nb.rows == 1

    empty = NoteBuffer()
    empty.add("표 3-1")
    assert not empty.usable

    # 식별자를 단위로 오인하지 않는다
    assert not NUM_UNIT.findall("F-3340M-20, F-3344A-21"), NUM_UNIT.findall("F-3340M-20")
    assert NUM_UNIT.findall("농도 10 mM 에서 37 ℃")

    # 파일명·커밋은 주장이 아니다
    assert not is_claim("ASD 결과 향상 보고서 v1.1 230801 final.pptx")
    assert not is_claim("[Commit] 설정 변경으로 성능 향상 (refs #585)")

    # base64 난수는 근거가 아니다
    assert not is_evidence("eyJOeXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9eyJ1cmwiOiJodHRwcw 10 mg 5 mL")

    # 어휘가 안 겹치면 뒷받침으로 보지 않는다
    assert supports("수율이 12.4 % 에서 18.7 % 로 증가하였다",
                    "수율 측정 결과 12.4 % 와 18.7 % 로 나타남")
    assert not supports("품질이 크게 향상되었습니다", "TE-201 온도 96.8 ℃ 95.8 ℃")
    # 내용어 2개면 수치 없이도 통과
    assert supports("배양 온도를 올리자 수율이 증가하였다", "배양 조건 37 ℃ 수율 18 %")
    print("claim_evidence demo ok —", len(ss), "문장 /", len(ev), "근거 /", len(cl), "주장")
    print("  교란 예:", ch)


if __name__ == "__main__":
    _demo()
