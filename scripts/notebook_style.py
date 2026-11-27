#!/usr/bin/env python3
"""연구노트 문체 정리(기계적 부분만) — .claude/skills/goono-research-note 지침의 금지 기호와 끝맺음.

내용과 숫자는 건드리지 않는다. 바꾸는 것:
- 가운뎃점: ' · ' → ', ', 붙은 '·' → '/'
- 줄표(—, –, ―): ' — ' → ': ' (앞말이 제목 · 결론이고 뒷말이 설명인 꼴이 대부분이라)
- 서술형 끝맺음: 문장 끝의 '~다'를 노트 말투로(했다 → 했음, 한다 → 함, 이다 → 임, 있다 → 있음, 오른다 → 오름 ...)
- 굵은 글씨(**): 모두 뺀다(지침: 한 노트에 두세 곳)
코드 블록(```) 안, 인라인 코드(`...`) 안은 그대로 둔다.

    python scripts/notebook_style.py docs/notebook/2026-11-09.md [--dry]
"""

from __future__ import annotations

import re
import sys

BASE, CHO, JUNG = 0xAC00, 588, 28
J_N, J_M, J_SS = 4, 16, 20          # 종성 ㄴ, ㅁ, ㅆ


def jong(ch: str) -> int:
    return (ord(ch) - BASE) % JUNG if "가" <= ch <= "힣" else -1


def with_jong(ch: str, j: int) -> str:
    return chr(ord(ch) - jong(ch) + j)


# 끝 글자로 바로 바꾸는 것: 서술격 조사, 불규칙(ㅂ · ㅅ), ㄹ 어간 동사
SPECIAL = {"아니다": "아님", "이다": "임", "하다": "함",
           "쉽다": "쉬움", "어렵다": "어려움", "가깝다": "가까움", "가볍다": "가벼움", "무겁다": "무거움", "새롭다": "새로움",
           "낫다": "나음", "든다": "듦", "단다": "닮", "연다": "엶", "산다": "삶", "판다": "팖", "논다": "놂"}
# 모음으로 끝나는 어간의 형용사(이것만 -ㅁ). 그 밖의 '모음 + 다'는 명사 + 서술격 조사로 보고 '~임'(하나다 → 하나임)
VOWEL_ADJ = ("크", "다르", "빠르", "느리", "이르", "바르", "나쁘", "아프", "바쁘", "기쁘", "예쁘", "고르", "모르")


def nominal(word: str) -> str:
    """'~다'로 끝나는 서술어를 노트 말투 명사형으로. 바꿀 수 없으면 그대로."""
    if not word.endswith("다") or len(word) < 2:
        return word
    for k, v in SPECIAL.items():
        if word.endswith(k):
            return word[: -len(k)] + v
    stem = word[:-1]
    last = stem[-1]
    if stem.endswith("는") and len(stem) >= 2 and jong(stem[-2]) > 0:   # 넣는다 / 않는다 → 넣음 / 않음
        return stem[:-1] + "음"
    j = jong(last)
    if j < 0:
        return word
    if j == J_N:                                                          # 한다 / 오른다 / 잰다 → 함 / 오름 / 잼
        return stem[:-1] + with_jong(last, J_M)
    if j == 0:                                                            # 크다 / 다르다 → 큼 / 다름, 하나다 → 하나임
        if stem.endswith(VOWEL_ADJ):
            return stem[:-1] + with_jong(last, J_M)
        return stem + "임"
    return stem + "음"                                                    # 했다 / 있다 / 없다 / 같다 → 했음 / 있음 / 없음 / 같음


END = re.compile(r"([가-힣]+다)(?=[.!]?\s*(?:$|\)|\]|[.!]\s|\.\s*\*|\s*\||:\s))")


def fix_line(line: str) -> str:
    parts = re.split(r"(`[^`]*`)", line)
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:
            out.append(p)
            continue
        p = p.replace("**", "")
        p = p.replace(" · ", ", ").replace("·", "/")
        p = re.sub(r"\|\s*[—–―]\s*(?=\|)", "| - ", p)            # 표의 빈 칸 표시
        p = re.sub(r"(?<=\S)[—–―](?=\S)", "/", p)                 # 붙은 줄표는 짝(주장–근거 → 주장/근거)
        p = re.sub(r"\s*[—–―]\s*", ": ", p)
        p = END.sub(lambda m: nominal(m.group(1)), p)
        out.append(p)
    return "".join(out)


def main() -> None:
    path = sys.argv[1]
    dry = "--dry" in sys.argv
    lines = open(path, encoding="utf-8").read().split("\n")
    code = False
    changed = 0
    new = []
    for ln in lines:
        if ln.strip().startswith("```"):
            code = not code
            new.append(ln)
            continue
        f = ln if code else fix_line(ln)
        changed += f != ln
        if dry and f != ln and changed <= 40:
            print("-", ln[:150])
            print("+", f[:150])
        new.append(f)
    if not dry:
        open(path, "w", encoding="utf-8", newline="\n").write("\n".join(new))
    print(f"바뀐 줄 {changed}")


if __name__ == "__main__":
    main()
