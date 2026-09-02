#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR 코퍼스 **내용 중심** 판정 — 양식이 아니라 연구 내용으로 분류한다.

## 왜 다시 하는가

앞선 `triage_ocr_csv.py` 는 국가연구개발 연구노트 **지침 양식**(과제명·연구자
서명·점검자·구획 헤더)을 기준으로 삼았다. 그 결과 구조성 지수 0.07 이 나와
"연구노트가 아니다" 라고 판정했는데, **이 잣대가 틀렸다.**

연구노트의 본질은 담긴 **연구 내용**이고, 서명·양식 같은 형식 요소는
부차적이며 이 코퍼스에는 애초에 담기지 않는다. 업로드된 PDF·PPTX·XLSX 가
곧 연구 기록 그 자체다.

## 무엇으로 분류하는가

연구 활동의 실제 산출물 유형으로 나눈다.

| 버킷 | 내용 |
|---|---|
| `experimental` | 실험 조건·측정값 — 수치와 단위가 실제로 있는 기록 |
| `analysis` | 결과 해석·통계·비교 |
| `literature` | 문헌·인용·선행연구 |
| `planning` | 과제 계획·제안·일정·예산 |
| `admin` | 행정·서식·정산 — 연구 내용이 아닌 것 |
| `fragment` | 조각(표 헤더·캡션·목차) — 짧지만 버릴 것은 아님 |

원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows

# ── 내용 신호 ────────────────────────────────────────────────────────
# 실험: 수치+단위가 실제로 존재 (연구 기록의 핵심)
_NUM_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:℃|°C|%|㎍|μg|ug|mg|kg|g|㎖|mL|L|μM|uM|mM|nM|M\b|"
    r"nm|㎛|μm|mm|cm|rpm|hr|min|sec|시간|분|초|일|주|개월|배|명|건|회|"
    r"ppm|kPa|MPa|GPa|W|kW|V|A|Hz|kHz|㎡|m2|mol|eq|wt%|vol%)"
)
_EXP_TERMS = re.compile(
    r"실험|측정|시료|샘플|처리군|대조군|반복|조건|투여|배양|합성|정제|"
    r"수율|농도|온도|압력|점도|강도|효율|assay|sample|control|treatment"
)
_ANALYSIS = re.compile(
    r"결과|분석|비교|평가|유의|경향|증가|감소|개선|향상|저하|"
    r"p\s*[<=]\s*0\.\d|평균|표준편차|SD\b|SEM\b|n\s*=\s*\d|"
    r"그림\s*\d|표\s*\d|Fig(?:ure)?\.?\s*\d|Table\s*\d"
)
_LITERATURE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+|references?|참고\s*문헌|선행\s*연구|"
    r"[A-Z][A-Za-z]+\s+et\s+al\.?|[가-힣]{2,4}\s*(?:외|등)\s*\(?\d{4}|"
    r"\[\d{1,3}\]|논문|저널|journal|doi", re.IGNORECASE
)
_PLANNING = re.compile(
    r"계획|목표|추진|일정|로드맵|마일스톤|예산|소요|연차|과제\s*제안|"
    r"기대\s*효과|활용\s*방안|필요성|배경"
)
_ADMIN = re.compile(
    r"정산|증빙|영수|지출|품의|기안|결재|청구|세금계산서|출장|"
    r"인건비|재료비|간접비|협약|규정\s*제\d|별지|서식"
)

# ── GitHub 연동 콘텐츠 ───────────────────────────────────────────────
# GOONO 는 GitHub 연동을 제공하므로 코드·커밋·README 가 섞여 들어온다.
# 연구 서술과 성격이 완전히 달라 별도 버킷으로 뺀다.
_CODE = re.compile(
    r"^\s*(?:def |class |import |from \w+ import|function |const |let |var |"
    r"public |private |#include|package |func |fn |async def)|"
    r"=>|\{\s*$|\}\s*$|;\s*$|</\w+>|\breturn\b|\bif\s*\(|\bfor\s*\(",
    re.MULTILINE,
)
# GOONO GitHub 연동 내보내기는 `commit <ISO 타임스탬프>` 형식이고 해시는 URL 끝에 붙는다.
# 기존 패턴은 `commit <해시>` 만 봐서 이 형식을 통째로 놓쳤다. 그 결과 커밋 목록이
# code_meta 나 fragment 로 밀렸다(실측: code_meta 예측 10건 중 9건이 실제 커밋).
_COMMIT = re.compile(
    r"\bcommit\s+[0-9a-f]{7,40}\b|"
    r"\bcommit\s+20\d\d-\d\d-\d\dT|"            # commit 2024-12-09T08:26:22.000Z
    r"/git/commits/[0-9a-f]{7,40}|"             # api.github.com/.../git/commits/<hash>
    r"^Author:\s|^Date:\s|^\+\+\+ |^--- |^@@ .* @@|"
    r"Merge\s+(?:pull\s+request|branch)|Signed-off-by:|"
    r"\bcommit\s+message\b|\bauthored\b|"
    r"\bgit\s+(?:add|commit|push|pull|merge|checkout)\b",
    re.MULTILINE | re.IGNORECASE,
)
# 설정·메타 파일의 **양성** 표지. 이게 없으면 code_meta 로 보내지 않는다.
_CONFIG = re.compile(
    r"package\.json|package-lock|yarn\.lock|Dockerfile|docker-compose|requirements\.txt|"
    r"pyproject\.toml|setup\.py|pom\.xml|build\.gradle|tsconfig|\.eslintrc|\.gitignore|"
    r"Makefile|CMakeLists|Chart\.yaml|values\.yaml|nginx\.conf|"
    r"^\s*(?:version|dependencies|devDependencies|scripts|image|ports|volumes)\s*:",
    re.MULTILINE | re.IGNORECASE,
)
_MARKDOWN = re.compile(
    r"^#{1,6}\s+\S|^```|\[[^\]]+\]\([^)]+\)|^\s*[-*]\s+\S|^\|.*\|.*\|",
    re.MULTILINE,
)
_REPO_HINT = re.compile(
    r"github\.com|gitlab|\.git\b|README|LICENSE|CHANGELOG|requirements\.txt|"
    r"package\.json|Dockerfile|\.ipynb|pull\s*request|repository|branch|"
    r"\b\w+\.(?:py|js|ts|tsx|java|cpp|c|h|go|rs|rb|php|sh|yaml|yml|json|md)\b",
    re.IGNORECASE,
)

MIN_CONTENT = 80          # 이보다 짧으면 조각으로 본다(200 → 80 으로 완화)


def code_subtype(sig: dict[str, int]) -> str:
    """GitHub 콘텐츠 세부 유형.

    주의: 진입 조건이 `commit>=2 or code>=4 or (repo>=3 and markdown>=3)` 이므로
    아래 세 분기 중 하나는 반드시 참이다 → `code_meta` 는 **도달 불가**였다.
    설정·메타 파일(package.json, Dockerfile 등)을 실제로 잡으려면
    저장소 힌트만 강한 경우를 따로 봐야 한다.
    """
    if sig["commit"] >= 2:
        return "code_commit"
    if sig["code"] >= 3:
        return "code_source"
    if sig["markdown"] >= 3:
        return "code_docs"
    # 남은 것을 무조건 code_meta 로 보내면 안 된다 — 실측 10건 중 9건이 실제로는
    # 커밋 목록이었다(정밀도 0%). 설정 파일의 **양성 표지**를 요구한다.
    if sig["config"] >= 1:
        return "code_meta"
    return "code_commit"


# 분류에는 앞부분만으로 충분하다. 전문을 매번 스캔하면 10만 행에서 수 분이 걸린다.
SCAN_CHARS = 4000
# 임계값은 5 이상이면 판정이 갈리지 않으므로 그 지점에서 세기를 멈춘다.
_COUNT_CAP = 6


def _count(pat: re.Pattern, text: str) -> int:
    """히트 수를 세되 `_COUNT_CAP` 에서 조기 종료한다."""
    n = 0
    for _ in pat.finditer(text):
        n += 1
        if n >= _COUNT_CAP:
            break
    return n


def classify_content(text: str) -> tuple[str, dict[str, int]]:
    """연구 내용 유형으로 분류. (버킷, 신호별 히트수)"""
    full = text.strip()
    t = full[:SCAN_CHARS]
    sig = {
        "num_unit": _count(_NUM_UNIT, t),
        "exp": _count(_EXP_TERMS, t),
        "analysis": _count(_ANALYSIS, t),
        "lit": _count(_LITERATURE, t),
        "plan": _count(_PLANNING, t),
        "admin": _count(_ADMIN, t),
        "code": _count(_CODE, t),
        "commit": _count(_COMMIT, t),
        "markdown": _count(_MARKDOWN, t),
        "repo": _count(_REPO_HINT, t),
        "config": _count(_CONFIG, t),
    }
    if len(full) < MIN_CONTENT:
        return "fragment", sig

    # GitHub 연동 콘텐츠를 먼저 분리한다 — 연구 서술과 성격이 다르다.
    # `repo>=4` 단독 조건을 넣어 설정·메타 파일(package.json, Dockerfile 등)도 잡는다.
    if (sig["commit"] >= 2 or sig["code"] >= 4
            or (sig["repo"] >= 3 and sig["markdown"] >= 3)
            or (sig["repo"] >= 4 and sig["num_unit"] + sig["exp"] < 3)):
        return code_subtype(sig), sig

    # 행정은 연구 내용이 아니므로 걸러낸다(단, 실험 신호가 강하면 제외하지 않음)
    if sig["admin"] >= 2 and sig["num_unit"] + sig["exp"] < 3:
        return "admin", sig
    # 실험 기록: 수치·단위가 실제로 있고 실험 용어가 동반
    # 무작위 표본 실측에서 예측 32.9% / 실제 32.9% 로 이 조건만은 정확하다. 건드리지 않는다.
    if sig["num_unit"] >= 3 and sig["exp"] >= 2:
        return "experimental", sig
    # 문헌·계획은 6.0배·5.0배 **과소** 예측이었다(실측). 임계값을 2 로 낮춘다.
    # 분석은 1.8배 과대였으므로 3 을 유지한다.
    if sig["lit"] >= 2:
        return "literature", sig
    if sig["analysis"] >= 3:
        return "analysis", sig
    if sig["plan"] >= 2:
        return "planning", sig
    # 수치만 많은 경우도 연구 데이터로 본다(표·측정값 목록)
    if sig["num_unit"] >= 5:
        return "experimental", sig

    # ── 흩어진 신호 구제 ────────────────────────────────────────────
    # 위 조건들은 모두 "한 종류의 신호가 3개 이상"을 요구한다. 그래서 신호가
    # 여러 종류에 1~2개씩 흩어지면 전부 fragment 로 밀려난다.
    # 실측: fragment 로 분류된 것 중 21.7% 가 신호 3개 이상이었고,
    # 64.4% 는 숫자를 10개 이상 담고 있었다(표·측정값).
    # → 총합이 임계값을 넘으면 **가장 강한 신호**의 범주로 배정한다.
    research_sig = {
        "experimental": sig["num_unit"] + sig["exp"],
        "analysis": sig["analysis"],
        "literature": sig["lit"],
        "planning": sig["plan"],
    }
    # 실측(무작위 표본 170건): fragment 예측 39.4% vs 실제 17.1% — 2.3배 과대였다.
    # 즉 이 구제가 여전히 부족하다. 임계값을 3 → 2 로 낮춘다.
    if sum(research_sig.values()) >= 2:
        return max(research_sig, key=lambda k: research_sig[k]), sig
    # 수치가 조금이라도 있고 본문이 충분히 길면 측정 기록으로 본다
    if sig["num_unit"] >= 2 and len(full) >= 300:
        return "experimental", sig
    return "fragment", sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=100000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    EST_TOTAL = 6_520_029
    buckets: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    by_ext: dict[str, Counter] = {}
    frag_len: list[int] = []
    n = 0

    for row in iter_rows(Path(a.csv_path), limit=a.rows):
        n += 1
        b, _sig = classify_content(row.text)
        buckets[b] += 1
        chars[b] += len(row.text)
        by_ext.setdefault(row.ext, Counter())[b] += 1
        if b == "fragment":
            frag_len.append(len(row.text.strip()))

    print("=" * 68)
    print(f" 연구 내용 중심 판정 — 표본 {n:,}행")
    print(" (양식이 아니라 담긴 연구 내용으로 분류)")
    print("=" * 68)

    label = {
        "experimental": "실험 기록 — 조건·측정값",
        "analysis":     "분석·해석 — 결과 논의",
        "literature":   "문헌·인용 — 선행연구",
        "planning":     "계획·제안 — 과제 기획",
        "code_source":  "[GitHub] 소스코드",
        "code_commit":  "[GitHub] 커밋·diff",
        "code_docs":    "[GitHub] README·문서",
        "code_meta":    "[GitHub] 설정·메타",
        "admin":        "행정·서식 — 연구내용 아님",
        "fragment":     "조각 — 표헤더·캡션·목차",
    }
    order = ["experimental", "analysis", "literature", "planning",
             "code_source", "code_commit", "code_docs", "code_meta",
             "fragment", "admin"]
    print(f"\n{'버킷':28s} {'행수':>9s} {'비율':>7s} {'전체추정':>12s} {'평균길이':>8s}")
    print("-" * 68)
    for b in order:
        c = buckets[b]
        pct = c / max(1, n)
        avg = chars[b] // c if c else 0
        print(f"{label[b]:28s} {c:>9,} {pct*100:6.1f}% {int(pct*EST_TOTAL):>11,}행 {avg:>7,}자")

    research = sum(buckets[b] for b in ("experimental", "analysis", "literature", "planning"))
    code = sum(buckets[b] for b in ("code_source", "code_commit", "code_docs", "code_meta"))
    print(f"\n연구 서술 내용:  {research:>8,}행 ({research/max(1,n)*100:5.1f}%) "
          f"→ 전체 약 {int(research/max(1,n)*EST_TOTAL):,}행")
    print(f"GitHub 연동:     {code:>8,}행 ({code/max(1,n)*100:5.1f}%) "
          f"→ 전체 약 {int(code/max(1,n)*EST_TOTAL):,}행")

    if frag_len:
        frag_len.sort()
        print(f"\n[조각 버킷 길이] 중앙값 {frag_len[len(frag_len)//2]:,}자 / "
              f"75분위 {frag_len[int(len(frag_len)*0.75)]:,}자")
        print("  → 짧아도 표 헤더·캡션 등 맥락이 있어 버리기 전 재검토 필요")

    print("\n[확장자별 주요 내용]")
    for ext, cc in sorted(by_ext.items(), key=lambda kv: -sum(kv[1].values()))[:7]:
        tot = sum(cc.values())
        top = ", ".join(f"{k} {v/tot*100:.0f}%" for k, v in cc.most_common(3))
        print(f"  {ext:10s} ({tot:>6,}행) → {top}")

    if a.json:
        Path(a.json).write_text(json.dumps({
            "sampled_rows": n,
            "buckets": dict(buckets),
            "bucket_pct": {k: round(v / max(1, n), 4) for k, v in buckets.items()},
            "estimated_by_bucket": {k: int(v / max(1, n) * EST_TOTAL) for k, v in buckets.items()},
            "research_content_rows_est": int(research / max(1, n) * EST_TOTAL),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
