#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""출판 논문 혼입률 — 코퍼스의 고유성을 직접 깎는 요소.

분포 측정에서 영문중심 행이 57.7% 로 나왔다. 여기서 갈림길이 생긴다:

- 연구자가 **직접 쓴** 영문 기록이면 → 고유 데이터. 어디에도 없다.
- 참고용으로 **올린 남의 논문**이면 → 고유 가치 0.
  이미 모든 LLM 의 사전학습에 들어가 있는 텍스트다.

이 둘을 가르지 않으면 "650만 행"이라는 숫자가 부풀려진 것이 된다.

출판물에만 있고 연구 기록에는 없는 표지를 센다 — 투고/게재 이력, 저작권,
DOI/ISSN, 초록·키워드 블록, 참고문헌 목록, 교신저자. 반대로 1차 기록에만
있는 표지(오늘·측정·시료·조건)도 함께 세어 대조한다.

원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_notes, iter_rows

SCAN = 6000
# 문서 단위 판정 시 훑을 최대 분량. 논문 표지는 앞·뒤에 몰려 있으므로
# 앞 3페이지와 뒤 2페이지의 각 2,500자만 본다. 6,000자 x 9페이지로 잡았더니
# 정규식 부하가 과해 전수 패스가 1시간을 넘겼다.
HEAD_PAGES, TAIL_PAGES, NOTE_SCAN = 3, 2, 2500
# 비싼 표지 검사를 걸기 전 통과시킬 싼 사전필터. 아무 표지 후보도 없으면 건너뛴다.
PREFILTER = re.compile(
    r"10\.\d{4}|ISSN|arXiv|©|\(c\)\s*20|rights\s+reserved|Elsevier|Springer|Wiley|"
    r"IEEE|MDPI|Received|Accepted|Published|접수일|게재|심사일|Correspond|교신저자|"
    r"@|Abstract|ABSTRACT|초\s*록|요\s*약|Keywords?|주제어|핵심어|References?|REFERENCES|"
    r"참고\s*문헌|인용\s*문헌|et\s+al|\[\d|오늘|금일|내일|어제|작업\s*내용|진행\s*상황|"
    r"특이\s*사항|비고|시료|샘플|측정값|계측|칭량|투입량|재료비|장비|배치|lot|batch|"
    r"확인\s*요|확인\s*바|재실험|재측정|보완|검토\s*요|승인|결재|담당자|작성자|"
    r"과제\s*번호|사업\s*번호|내부\s*문서|대외비|보안|기밀|회의록|보고",
    re.IGNORECASE,
)

# ── 출판물 표지 — 1차 기록에는 없다 ──────────────────────────────────
PAPER = {
    "doi_issn": re.compile(r"\b10\.\d{4,9}/\S+|\bISSN\b|\barXiv:\s*\d{4}\.\d{4,5}",
                           re.IGNORECASE),
    "저작권": re.compile(r"©|\(c\)\s*20\d\d|All\s+rights\s+reserved|"
                      r"Elsevier|Springer|Wiley|IEEE|MDPI|Taylor\s*&\s*Francis|"
                      r"대한민국\s*저작권|한국\w{0,6}학회지",
                      re.IGNORECASE),
    "투고이력": re.compile(r"Received\b[^\n]{0,60}Accepted|Received\s*:\s*\d|"
                       r"Accepted\s*:\s*\d|Published\s*(?:online)?\s*:\s*\d|"
                       r"접수일|게재\s*확정|심사일", re.IGNORECASE),
    "교신저자": re.compile(r"Corresponding\s+author|교신저자|\*\s*Correspond|"
                       r"E-?mail\s*:\s*\S+@", re.IGNORECASE),
    "초록키워드": re.compile(r"^\s*(?:Abstract|ABSTRACT|초\s*록|요\s*약)\s*$|"
                        r"\bKeywords?\s*[:：]|주제어\s*[:：]|핵심어\s*[:：]",
                        re.MULTILINE),
    "참고문헌블록": re.compile(r"^\s*(?:References?|REFERENCES|참고\s*문헌|인용\s*문헌)\s*$",
                         re.MULTILINE),
    "인용밀도": re.compile(r"et\s+al\.|\[\d{1,3}(?:[,–-]\s*\d{1,3})*\]|"
                       r"\(\s*[A-Z][a-z]+\s+(?:et\s+al\.|and\s+[A-Z][a-z]+)?,?\s*(?:19|20)\d\d\s*\)"),
}
# ── 1차 기록 표지 — 출판물에는 없다 ──────────────────────────────────
RECORD = {
    "일지어": re.compile(r"오늘|금일|내일|어제|다음\s*주|이번\s*주|익일|"
                      r"작업\s*내용|진행\s*상황|특이\s*사항|비고\s*[:：]"),
    "실측어": re.compile(r"시료|샘플\s*번호|측정값|계측|칭량|투입량|재료비|"
                      r"장비\s*번호|배치\s*번호|lot\s*no|batch\s*no", re.IGNORECASE),
    "지시어": re.compile(r"확인\s*요|확인\s*바[람랍]|재실험|재측정|보완\s*필요|"
                      r"검토\s*요[망청]|승인|결재|담당자|작성자\s*[:：]"),
    "내부문서": re.compile(r"과제\s*번호|사업\s*번호|내부\s*문서|대외비|"
                       r"보안|기밀|회의록|주간\s*보고|월간\s*보고"),
}

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")

for _d in (PAPER, RECORD):
    for _k, _r in _d.items():
        assert not _r.search(""), f"{_k} 빈 대안"


def by_note(a) -> None:
    """문서 단위 판정 — 행 단위의 과소추정을 바로잡는다.

    한 문서의 어느 페이지든 표지가 있으면 그 문서 전체가 그 성격이다.
    코퍼스 가치에 영향을 주는 건 문서 수가 아니라 **그 문서가 차지하는 행 수**이므로
    둘 다 낸다.
    """
    docs = 0
    verdict: Counter[str] = Counter()
    v_rows: Counter[str] = Counter()
    v_pages: Counter[str] = Counter()
    marker: Counter[str] = Counter()
    ext_v: Counter[tuple[str, str]] = Counter()
    lang_v: Counter[tuple[str, str]] = Counter()
    order = ["출판 논문 추정", "논문 가능성", "1차 기록", "표지 없음"]

    seen_notes = 0
    for note in iter_notes(Path(a.csv_path)):
        seen_notes += 1
        if seen_notes % a.note_stride:
            continue
        docs += 1
        pages = sorted(note.pages, key=lambda r: r.page)
        look = pages[:HEAD_PAGES] + (pages[-TAIL_PAGES:] if len(pages) > HEAD_PAGES else [])
        blob = "\n".join(p.text[:NOTE_SCAN] for p in look)

        if PREFILTER.search(blob):
            p = [k for k, r in PAPER.items() if r.search(blob)]
            rc = [k for k, r in RECORD.items() if r.search(blob)]
        else:
            p = rc = []
        for k in p:
            marker[k] += 1

        if len(p) >= 2 and len(p) >= len(rc):
            v = "출판 논문 추정"
        elif p and not rc:
            v = "논문 가능성"
        elif rc:
            v = "1차 기록"
        else:
            v = "표지 없음"
        verdict[v] += 1
        v_rows[v] += len(pages)
        v_pages[v] += len(pages)

        han, lat = len(_HANGUL.findall(blob)), len(_LATIN.findall(blob))
        letters = han + lat
        lang = ("문자없음" if not letters else
                "한글중심" if han / letters > 0.7 else
                "혼합" if han / letters > 0.3 else "영문중심")
        lang_v[(lang, v)] += len(pages)
        ext_v[(Path(note.file_name).suffix.lower() or "(없음)", v)] += len(pages)

    tot_rows = sum(v_rows.values())
    print("=" * 74)
    print(f" 출판 논문 혼입률 — **문서 단위** 판정")
    print(f" 전체 문서 {seen_notes:,}개 중 {docs:,}개 표본(1/{a.note_stride})"
          f" · 표본 행 {tot_rows:,}")
    print("=" * 74)

    print(f"\n[판정] 문서 기준 / 행 기준")
    for v in order:
        d, r = verdict[v], v_rows[v]
        print(f"  {v:12s} 문서 {d:>8,} ({d/max(1,docs)*100:5.1f}%)   "
              f"행 {r:>9,} ({r/max(1,tot_rows)*100:5.1f}%)"
              f" {'#'*int(r/max(1,tot_rows)*30)}")
    print(f"  문서당 평균 페이지: " + "  ".join(
        f"{v}={v_pages[v]/max(1,verdict[v]):.1f}" for v in order))

    print("\n[언어 x 판정 — 행 기준] 영문중심 행의 정체")
    print(f"  {'언어':8s} {'행':>9s} " + " ".join(f"{v[:6]:>8s}" for v in order))
    for L in ["영문중심", "혼합", "한글중심", "문자없음"]:
        t = sum(lang_v[(L, v)] for v in order)
        if t:
            print(f"  {L:8s} {t:>9,} " +
                  " ".join(f"{lang_v[(L,v)]/t*100:>7.1f}%" for v in order))

    print("\n[확장자 x 판정 — 행 기준]")
    exts = sorted({e for e, _ in ext_v},
                  key=lambda e: -sum(ext_v[(e, v)] for v in order))[:6]
    print(f"  {'확장자':10s} {'행':>9s} " + " ".join(f"{v[:6]:>8s}" for v in order))
    for e in exts:
        t = sum(ext_v[(e, v)] for v in order)
        print(f"  {e:10s} {t:>9,} " +
              " ".join(f"{ext_v[(e,v)]/max(1,t)*100:>7.1f}%" for v in order))

    print("\n[출판물 표지 — 문서 기준 출현률]")
    for k, c in marker.most_common():
        print(f"  {k:12s} {c:>8,} ({c/max(1,docs)*100:5.1f}%)")

    pr = v_rows["출판 논문 추정"] / max(1, tot_rows)
    mr = (v_rows["출판 논문 추정"] + v_rows["논문 가능성"]) / max(1, tot_rows)
    print("\n" + "=" * 74)
    print(f" 고유성 보정: 출판 논문 {pr*100:.1f}% 제외 → 약 {int((1-pr)*tot_rows):,}행")
    print(f"            '논문 가능성'까지 제외(보수적) → 약 {int((1-mr)*tot_rows):,}행")
    print("=" * 74)

    Path(a.json).write_text(json.dumps({
        "mode": "by_note", "documents": docs, "total_rows": tot_rows,
        "verdict_docs": dict(verdict), "verdict_rows": dict(v_rows),
        "verdict_row_pct": {k: round(v / max(1, tot_rows), 4) for k, v in v_rows.items()},
        "avg_pages": {v: round(v_pages[v] / max(1, verdict[v]), 2) for v in order},
        "marker_doc_pct": {k: round(v / max(1, docs), 4) for k, v in marker.items()},
        "lang_x_verdict_rows": {f"{L}|{v}": lang_v[(L, v)]
                                for L in ["영문중심", "혼합", "한글중심", "문자없음"]
                                for v in order if lang_v[(L, v)]},
        "ext_x_verdict_rows": {f"{e}|{v}": ext_v[(e, v)] for e in exts
                               for v in order if ext_v[(e, v)]},
        "paper_row_pct": round(pr, 4),
        "paper_or_maybe_row_pct": round(mr, 4),
        "unique_rows_estimated": int((1 - pr) * tot_rows),
        "unique_rows_conservative": int((1 - mr) * tot_rows),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--note-stride", type=int, default=8,
                    help="문서 표본 간격. 비율 추정이므로 전수가 필요 없다")
    ap.add_argument("--by-note", action="store_true",
                    help="문서(NoteId) 단위로 판정. 논문 표지가 첫 페이지에만 "
                         "있어 행 단위로는 심하게 과소추정되므로 이쪽이 옳다.")
    ap.add_argument("--json", default="ocr_paper_contam.json")
    a = ap.parse_args()

    if a.by_note:
        return by_note(a)

    n = m = 0
    verdict: Counter[str] = Counter()
    paper_hits: Counter[str] = Counter()
    record_hits: Counter[str] = Counter()
    # 언어 x 판정 교차 — 영문중심이 정말 논문인가
    cross: Counter[tuple[str, str]] = Counter()
    ext_verdict: Counter[tuple[str, str]] = Counter()
    chars: Counter[str] = Counter()

    for row in iter_rows(Path(a.csv_path)):
        n += 1
        if n % a.stride:
            continue
        m += 1
        t = row.text[:SCAN]
        p = [k for k, r in PAPER.items() if r.search(t)]
        rc = [k for k, r in RECORD.items() if r.search(t)]
        for k in p:
            paper_hits[k] += 1
        for k in rc:
            record_hits[k] += 1

        if len(p) >= 3 and len(p) > len(rc):
            v = "출판 논문 추정"
        elif len(p) >= 1 and not rc:
            v = "논문 가능성"
        elif rc:
            v = "1차 기록"
        else:
            v = "표지 없음"
        verdict[v] += 1
        chars[v] += len(row.text)

        han, lat = len(_HANGUL.findall(t)), len(_LATIN.findall(t))
        letters = han + lat
        lang = ("문자없음" if not letters else
                "한글중심" if han / letters > 0.7 else
                "혼합" if han / letters > 0.3 else "영문중심")
        cross[(lang, v)] += 1
        ext_verdict[(Path(row.file_name).suffix.lower() or "(없음)", v)] += 1

    TOT = 5_654_358
    order = ["출판 논문 추정", "논문 가능성", "1차 기록", "표지 없음"]
    print("=" * 70)
    print(f" 출판 논문 혼입률 — 전수 {n:,}행 / 정밀 {m:,}행 (1/{a.stride})")
    print("=" * 70)

    print("\n[판정]")
    for v in order:
        c = verdict[v]
        print(f"  {v:12s} {c:>8,} ({c/max(1,m)*100:5.1f}%) 전체 약 "
              f"{int(c/max(1,m)*TOT):>9,}행  평균 {chars[v]//max(1,c):>5,}자"
              f" {'#'*int(c/max(1,m)*30)}")

    print("\n[언어 x 판정] — 영문중심 행의 정체")
    langs = ["영문중심", "혼합", "한글중심", "문자없음"]
    print(f"  {'언어':8s} {'행':>8s} " + " ".join(f"{v[:6]:>8s}" for v in order))
    for L in langs:
        tot = sum(cross[(L, v)] for v in order)
        if not tot:
            continue
        print(f"  {L:8s} {tot:>8,} " +
              " ".join(f"{cross[(L,v)]/tot*100:>7.1f}%" for v in order))

    print("\n[출판물 표지 출현률]")
    for k, c in paper_hits.most_common():
        print(f"  {k:12s} {c:>8,} ({c/max(1,m)*100:5.1f}%)")
    print("\n[1차 기록 표지 출현률]")
    for k, c in record_hits.most_common():
        print(f"  {k:12s} {c:>8,} ({c/max(1,m)*100:5.1f}%)")

    print("\n[확장자 x 판정]")
    exts = sorted({e for e, _ in ext_verdict}, key=lambda e: -sum(
        ext_verdict[(e, v)] for v in order))[:6]
    print(f"  {'확장자':10s} {'행':>8s} " + " ".join(f"{v[:6]:>8s}" for v in order))
    for e in exts:
        tot = sum(ext_verdict[(e, v)] for v in order)
        print(f"  {e:10s} {tot:>8,} " +
              " ".join(f"{ext_verdict[(e,v)]/max(1,tot)*100:>7.1f}%" for v in order))

    pc = verdict["출판 논문 추정"] / max(1, m)
    print("\n" + "=" * 70)
    print(f" 고유성 보정: 출판 논문 추정 {pc*100:.1f}% 를 제외하면 "
          f"약 {int((1-pc)*TOT):,}행")
    print("=" * 70)

    Path(a.json).write_text(json.dumps({
        "total_rows": n, "profiled_rows": m, "stride": a.stride,
        "verdict": dict(verdict),
        "verdict_pct": {k: round(v / max(1, m), 4) for k, v in verdict.items()},
        "verdict_estimated": {k: int(v / max(1, m) * TOT) for k, v in verdict.items()},
        "avg_chars": {k: chars[k] // max(1, verdict[k]) for k in verdict},
        "paper_marker_pct": {k: round(v / max(1, m), 4) for k, v in paper_hits.items()},
        "record_marker_pct": {k: round(v / max(1, m), 4) for k, v in record_hits.items()},
        "lang_x_verdict": {f"{L}|{v}": cross[(L, v)] for L in langs for v in order
                           if cross[(L, v)]},
        "ext_x_verdict": {f"{e}|{v}": ext_verdict[(e, v)] for e in exts for v in order
                          if ext_verdict[(e, v)]},
        "paper_pct": round(pc, 4),
        "unique_rows_estimated": int((1 - pc) * TOT),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {a.json}")


if __name__ == "__main__":
    main()
