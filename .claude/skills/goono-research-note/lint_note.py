"""구노 연구노트 작성본 검사 (지침 v2).

사용: python lint_note.py <작성본.md> [원본.md]
- 실패(종료 코드 1): 금지 기호, 서술형 종결, 그림 없음, 개요 표, 구분선, 소절 번호
- 경고: 표 3개 이상, 굵은 글씨 4곳 이상, 계획서 단어, 원본에 없는 수치
"""
import re
import sys

BANNED = {
    "·": "가운뎃점",
    "・": "가운뎃점",
    "‧": "가운뎃점",
    "—": "줄표(em dash)",
    "–": "줄표(en dash)",
    "―": "줄표",
    "✓": "장식 기호",
    "★": "장식 기호",
    "▸": "장식 기호",
    "※": "장식 기호",
    "ᴿ": "위첨자 R",
}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-⛿✀-➿]")
HYPHEN_SEP = re.compile(r"(?<=\S) - (?=\S)")
# 서술형 종결: 줄 끝 또는 문장 끝의 '다.' / '요.' ('필요'는 허용)
DECL = re.compile(r"(다|(?<!필)요)[.]?\s*$|(다|(?<!필)요)\.\s+\S")
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
PLAN_WORDS = ["수립", "도출", "제고", "고도화", "확립", "구현함으로써", "를 통해", "을 통해", "라는 점에서", "결론적으로"]
OVERVIEW = re.compile(r"^\|\s*(항목|구분)\s*\|\s*(내용|설명)\s*\|")
SUBSECTION = re.compile(r"^#{1,6}\s*\d+\.\d+")


def strip_code(line: str) -> str:
    return re.sub(r"`[^`]*`", "", line)


def numbers(text: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in NUM.findall(text)}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = open(sys.argv[1], encoding="utf-8").read()
    errors, warns = [], []
    in_code = False
    tables = 0
    prev_table = False
    for i, raw in enumerate(out.splitlines(), 1):
        if raw.strip().startswith("```"):
            in_code = not in_code
            continue
        line = raw if in_code else strip_code(raw)
        for ch, name in BANNED.items():
            if ch in raw:
                errors.append(f"{i}: {name} '{ch}' : {raw.strip()[:70]}")
        if EMOJI.search(raw):
            errors.append(f"{i}: 이모지/장식 기호 : {raw.strip()[:70]}")
        if in_code:
            continue
        is_table = raw.strip().startswith("|")
        if is_table and not prev_table:
            tables += 1
        prev_table = is_table
        if OVERVIEW.match(raw.strip()):
            errors.append(f"{i}: 개요 표(항목/내용 2열 표) : {raw.strip()[:70]}")
        if re.fullmatch(r"\s*([-*_]\s*){3,}", raw):
            errors.append(f"{i}: 구분선")
        if raw.lstrip().startswith(">"):
            errors.append(f"{i}: 인용 상자 : {raw.strip()[:70]}")
        if SUBSECTION.match(raw.strip()):
            errors.append(f"{i}: 소절 번호 : {raw.strip()[:70]}")
        if raw.strip().startswith("!["):
            continue
        body = re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)
        if HYPHEN_SEP.search(body) and not re.fullmatch(r"[\s|:-]+", body):
            errors.append(f"{i}: 덧붙임 하이픈 ' - ' : {raw.strip()[:70]}")
        text = re.sub(r"[*_|]", "", body).strip()
        if text and not text.startswith("#") and DECL.search(text):
            errors.append(f"{i}: 서술형 종결 : {raw.strip()[:70]}")
        for w in PLAN_WORDS:
            if w in text:
                warns.append(f"{i}: 계획서 단어 '{w}' : {raw.strip()[:60]}")

    if not re.search(r"^!\[", out, re.M):
        errors.append("그림이 한 장도 없음")
    if tables >= 3:
        warns.append(f"표 {tables}개 (데이터 표만 남길 것)")
    bold = len(re.findall(r"\*\*[^*]+\*\*", out))
    if bold >= 4:
        warns.append(f"굵은 글씨 {bold}곳")

    if len(sys.argv) > 2:
        src = open(sys.argv[2], encoding="utf-8").read()
        extra = sorted(
            n for n in numbers(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", out)) - numbers(src)
            if not (n.isdigit() and int(n) < 10)
        )
        if extra:
            warns.append("원본에 없는 수치: " + ", ".join(extra))

    for e in errors:
        print("ERROR", e)
    for w in warns:
        print("WARN", w)
    print(f"{sys.argv[1]}: 오류 {len(errors)}건, 경고 {len(warns)}건")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
