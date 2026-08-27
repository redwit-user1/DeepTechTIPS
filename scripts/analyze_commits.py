#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""커밋 메시지로 '실제 연구 활동'을 분류한다.

가설(사용자): GitHub 연동으로 들어온 커밋 내용을 보면 그것이 연구 활동인지
아닌지 가를 수 있다. 이를 검증한다.

커밋 메시지는 **연구자가 자기 작업을 스스로 요약한 한 줄**이다.
연구노트의 '오늘 무엇을 했는가'와 기능적으로 같고, 타임스탬프·작성자가
붙어 있어 ALCOA+ 의 Contemporaneous·Attributable 를 기계로 검증할 수 있다.

원문(커밋 메시지 본문)은 출력하지 않는다. 분류 통계만 낸다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from triage_ocr_content import SCAN_CHARS, classify_content

# 커밋 메시지처럼 보이는 줄 — 해시/Author/Date 헤더 또는 conventional commit
_COMMIT_LINE = re.compile(
    r"^\s*(?:commit\s+[0-9a-f]{7,40}"
    r"|Author:|Date:|Merge:"
    r"|(?:feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert)"
    r"\s*(?:\([^)]*\))?\s*:)",
    re.IGNORECASE | re.MULTILINE,
)
_CONV_TYPE = re.compile(
    r"\b(feat|fix|docs|style|refactor|perf|test|chore|build|ci|revert)\b"
    r"\s*(?:\([^)]*\))?\s*:",
    re.IGNORECASE,
)
_DIFF = re.compile(r"^(?:diff --git|@@ |[+-]{3} [ab]/)", re.MULTILINE)

# 연구 활동 유형 — 커밋이 무슨 연구 단계를 기록하는가
ACTIVITY: dict[str, str] = {
    "데이터·실험": r"data|데이터|raw|측정|measure|실험|experiment|sample|시료|수집|collect|logging|계측",
    "전처리·정제": r"preprocess|전처리|clean|정제|filter|필터|normali|정규화|parse|파싱|convert|변환",
    "분석·시각화": r"analy[sz]|분석|plot|그래프|figure|chart|시각화|visuali|statist|통계|correlat|histogram",
    "모델·알고리즘": r"model|모델|train|학습|algorithm|알고리즘|network|신경망|optimiz|최적화|hyperparam|inference|추론|loss|epoch",
    "평가·결과": r"evaluat|평가|benchmark|벤치마크|result|결과|metric|지표|accuracy|정확도|score|검증|validat|실험\s*결과",
    "문서·보고": r"paper|논문|manuscript|report|보고서|docs?\b|문서|readme|slide|발표|thesis|초록|abstract",
    "인프라·환경": r"docker|kubernetes|k8s|\bci\b|pipeline|deploy|배포|depend|의존성|bump|upgrade|env|환경설정|setup|install|build",
    "유지보수": r"\bfix\b|버그|bug|typo|오타|refactor|리팩토|rename|이름\s*변경|lint|format|cleanup|정리|revert|merge\s*branch",
}
_ACT_RE = {k: re.compile(v, re.IGNORECASE) for k, v in ACTIVITY.items()}
_RESEARCH_ACT = {"데이터·실험", "전처리·정제", "분석·시각화", "모델·알고리즘", "평가·결과"}


def commit_activity(text: str) -> tuple[str, dict[str, int]]:
    t = text[:SCAN_CHARS]
    s = {k: len(r.findall(t)) for k, r in _ACT_RE.items()}
    top = max(s, key=lambda k: s[k])
    return (top if s[top] >= 1 else "미분류"), s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--json", default="ocr_commits.json")
    a = ap.parse_args()

    n = n_commitish = 0
    act: Counter[str] = Counter()
    conv: Counter[str] = Counter()
    has_diff = has_author = has_date = has_hash = 0
    act_len: Counter[str] = Counter()

    # 순수 숫자열(1234567)을 해시로 오인하지 않도록 a-f 를 최소 1자 요구한다
    _HASH = re.compile(r"\b(?=[0-9a-f]{7,40}\b)(?=[0-9a-f]*[a-f])[0-9a-f]+\b")

    for row in iter_rows(Path(a.csv_path), limit=a.rows):
        n += 1
        bucket, _ = classify_content(row.text)
        # 커밋 버킷 + 커밋 헤더가 보이는 소스/문서까지 포함
        if bucket != "code_commit" and not _COMMIT_LINE.search(row.text[:SCAN_CHARS]):
            continue
        n_commitish += 1
        t = row.text[:SCAN_CHARS]

        if _DIFF.search(t):
            has_diff += 1
        if re.search(r"^\s*Author:", t, re.MULTILINE):
            has_author += 1
        if re.search(r"^\s*Date:", t, re.MULTILINE):
            has_date += 1
        if _HASH.search(t):
            has_hash += 1

        for m in _CONV_TYPE.finditer(t):
            conv[m.group(1).lower()] += 1

        k, _ = commit_activity(t)
        act[k] += 1
        act_len[k] += len(row.text)

    EST = 6_520_029
    research = sum(act[k] for k in _RESEARCH_ACT)
    print("=" * 68)
    print(f" 커밋 내용 분석 — 표본 {n:,}행 중 커밋성 {n_commitish:,}행")
    print("=" * 68)

    print("\n[연구 활동 유형]")
    for k, c in act.most_common():
        p = c / max(1, n_commitish)
        mark = "★" if k in _RESEARCH_ACT else " "
        avg = act_len[k] // max(1, c)
        print(f" {mark}{k:12s} {c:>6,} ({p*100:5.1f}%) 평균 {avg:>5,}자 {'█'*int(p*32)}")
    print(f"\n  ★ = 연구 활동  →  {research:,}/{n_commitish:,} = "
          f"{research/max(1,n_commitish)*100:.1f}% "
          f"(전체 약 {int(research/max(1,n)*EST):,}행)")

    print("\n[ALCOA+ 기계 검증 가능 요소 — 커밋성 행 대비]")
    for label, c in (("커밋 해시(Original)", has_hash), ("Author(Attributable)", has_author),
                     ("Date(Contemporaneous)", has_date), ("diff(변경 근거)", has_diff)):
        print(f"  {label:24s} {c:>6,} ({c/max(1,n_commitish)*100:5.1f}%)")

    if conv:
        print("\n[Conventional Commit 타입]")
        for k, c in conv.most_common(10):
            print(f"  {k:10s} {c:,}")

    Path(a.json).write_text(json.dumps({
        "sampled_rows": n,
        "commitish_rows": n_commitish,
        "estimated_total": EST,
        "commitish_estimated": int(n_commitish / max(1, n) * EST),
        "activity_counts": dict(act),
        "activity_pct": {k: round(v / max(1, n_commitish), 4) for k, v in act.items()},
        "activity_avg_len": {k: act_len[k] // max(1, act[k]) for k in act},
        "research_activity_rows": research,
        "research_activity_pct": round(research / max(1, n_commitish), 4),
        "research_activity_estimated": int(research / max(1, n) * EST),
        "alcoa_signals": {"hash": has_hash, "author": has_author,
                          "date": has_date, "diff": has_diff},
        "conventional_types": dict(conv),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
