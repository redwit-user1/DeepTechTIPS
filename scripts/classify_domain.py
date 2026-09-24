#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""연구 분야 분류 + GitHub 커밋의 연구 관련성 판별.

## 두 축

1. **연구 분야** — 국가과학기술표준분류체계를 참고한 10개 대분류.
   어느 분야의 연구가 GOONO 에 쌓이는지 알아야 도메인 LoRA 를 어디에 걸지 정한다.

2. **커밋의 연구 관련성** — GitHub 콘텐츠라고 전부 인프라 코드는 아니다.
   분석 스크립트·실험 코드·모델 학습은 **연구 기록 그 자체**이고,
   빌드·배포·의존성 갱신은 아니다. 이 둘을 갈라야 한다.

원문은 출력하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from triage_ocr_content import SCAN_CHARS, classify_content

# ── 연구 분야 (국가과학기술표준분류 대분류 참고) ──────────────────────
DOMAINS: dict[str, str] = {
    "생명·보건의료": (
        r"세포|유전자|단백질|DNA|RNA|항체|백신|임상|환자|질환|암\b|종양|"
        r"면역|미생물|균주|배양|병원체|진단|치료|약물|약효|독성|생물|"
        r"protein|gene|cell|clinical|patient|tumor|antibody|vaccine|assay|"
        r"in\s*vitro|in\s*vivo|PCR|ELISA|western\s*blot"
    ),
    "화학·화공": (
        r"합성|촉매|반응|용매|정제|수율|당량|몰\b|시약|화합물|고분자|중합|"
        r"크로마토|스펙트럼|NMR|GC-MS|HPLC|산화|환원|pH\b|농도|"
        r"synthesis|catalyst|solvent|reagent|polymer|compound|reaction"
    ),
    "재료·소재": (
        r"소재|박막|나노|결정|합금|세라믹|복합재|코팅|기판|증착|소결|"
        r"미세구조|경도|인장|강도|탄성|열처리|SEM|TEM|XRD|"
        r"material|thin\s*film|nano|alloy|ceramic|composite|coating|substrate"
    ),
    "정보·통신": (
        r"알고리즘|모델|학습|딥러닝|머신러닝|신경망|데이터셋|정확도|추론|"
        r"네트워크|프로토콜|서버|데이터베이스|API\b|소프트웨어|"
        r"algorithm|neural|deep\s*learning|machine\s*learning|dataset|"
        r"accuracy|inference|model|training|LLM|transformer"
    ),
    "전기·전자·반도체": (
        r"반도체|웨이퍼|소자|트랜지스터|회로|전압|전류|저항|커패시|"
        r"센서|칩\b|전극|배터리|전지|충전|방전|주파수|"
        r"semiconductor|wafer|transistor|circuit|voltage|electrode|battery"
    ),
    "기계·제조": (
        r"기계|가공|절삭|공정|장비|설비|조립|금형|용접|베어링|모터|"
        r"진동|토크|하중|피로|응력|CAD|CAE|시뮬레이션|"
        r"machining|manufactur|assembly|torque|stress|fatigue|vibration"
    ),
    "에너지·환경": (
        r"에너지|발전|태양광|풍력|수소|연료|온실가스|탄소|배출|폐기물|"
        r"오염|정화|재생|효율|열효율|환경|대기|수질|"
        r"energy|solar|hydrogen|fuel|emission|carbon|waste|renewable"
    ),
    "농림수산·식품": (
        r"작물|재배|품종|토양|비료|어류|양식|수산|산림|목재|"
        r"식품|발효|저장|가공|영양|관능|축산|가축|사료|"
        r"crop|cultivar|soil|fishery|forest|food|fermentation"
    ),
    "건설·교통": (
        r"구조물|교량|터널|콘크리트|철근|지반|내진|시공|건축|"
        r"도로|철도|차량|교통|물류|"
        r"concrete|bridge|tunnel|seismic|construction|traffic|vehicle"
    ),
    "기초·자연과학": (
        r"물리|양자|입자|광학|레이저|천문|지질|기상|해양|수학|통계모형|"
        r"quantum|photon|optic|laser|geolog|astronom|atmospher"
    ),
}
_DOMAIN_RE = {k: re.compile(v, re.IGNORECASE) for k, v in DOMAINS.items()}

# ── 커밋·코드의 연구 관련성 ──────────────────────────────────────────
# 연구 산출물로서의 코드 — 분석·실험·모델
RESEARCH_CODE = re.compile(
    r"analy[sz]|experiment|실험|분석|측정|측정값|simulat|시뮬레이|"
    r"model|모델|train|학습|fit\b|regress|classif|cluster|"
    r"plot|figure|graph|그래프|시각화|statist|통계|"
    r"dataset|데이터셋|preprocess|전처리|feature|피처|"
    r"notebook|ipynb|numpy|pandas|scipy|sklearn|torch|tensorflow|matplotlib|"
    r"결과|result|평가|evaluat|benchmark|벤치마크",
    re.IGNORECASE,
)
# 인프라·운영 — 연구 내용이 아님
INFRA_CODE = re.compile(
    r"\bci\b|pipeline|deploy|배포|docker|kubernetes|k8s|nginx|"
    r"depend|의존성|bump|upgrade|version\s*up|lint|린트|format|포맷|"
    r"refactor|리팩토|typo|오타|readme|문서\s*수정|"
    r"merge\s*branch|revert|rollback|"
    r"package\.json|requirements\.txt|Dockerfile|\.gitignore|"
    # 끝에 `|` 를 남기면 빈 대안이 되어 모든 위치에 매치된다 — 절대 금지
    r"test\s*fix|build\s*fix|hotfix|chore",
    re.IGNORECASE,
)
assert not INFRA_CODE.search(""), "빈 대안 — 정규식 끝의 `|` 를 확인하라"
assert not RESEARCH_CODE.search(""), "빈 대안 — 정규식 끝의 `|` 를 확인하라"

CODE_BUCKETS = {"code_source", "code_commit", "code_docs", "code_meta"}
RESEARCH_BUCKETS = {"experimental", "analysis", "literature", "planning"}


def classify_domain(text: str) -> tuple[str, dict[str, int]]:
    """가장 강한 분야 신호를 반환. 신호가 약하면 '미분류'."""
    t = text[:SCAN_CHARS]
    scores = {k: len(r.findall(t)) for k, r in _DOMAIN_RE.items()}
    top = max(scores, key=lambda k: scores[k])
    return (top if scores[top] >= 2 else "미분류"), scores


def code_research_relevance(text: str) -> str:
    """코드·커밋이 연구 산출물인가 인프라인가."""
    t = text[:SCAN_CHARS]
    r, i = len(RESEARCH_CODE.findall(t)), len(INFRA_CODE.findall(t))
    if r >= 2 and r > i:
        return "연구 코드"
    if i >= 2 and i > r:
        return "인프라 코드"
    if r > 0 and r >= i:
        return "연구 코드"
    return "판별 불가"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--json", default="ocr_domain.json")
    a = ap.parse_args()

    dom_all: Counter[str] = Counter()
    dom_by_bucket: dict[str, Counter] = {}
    code_rel: Counter[str] = Counter()
    code_rel_domain: dict[str, Counter] = {}
    n = n_research = n_code = 0

    for row in iter_rows(Path(a.csv_path), limit=a.rows):
        n += 1
        bucket, _sig = classify_content(row.text)
        dom, _ = classify_domain(row.text)

        if bucket in RESEARCH_BUCKETS:
            n_research += 1
            dom_all[dom] += 1
            dom_by_bucket.setdefault(bucket, Counter())[dom] += 1
        elif bucket in CODE_BUCKETS:
            n_code += 1
            rel = code_research_relevance(row.text)
            code_rel[rel] += 1
            code_rel_domain.setdefault(rel, Counter())[dom] += 1

    EST = 6_520_029
    print("=" * 66)
    print(f" 연구 분야 분류 — 표본 {n:,}행 (연구 내용 {n_research:,} / 코드 {n_code:,})")
    print("=" * 66)

    print(f"\n[연구 내용의 분야 분포] (연구 버킷 {n_research:,}행 대비)")
    for d, c in dom_all.most_common():
        pct = c / max(1, n_research)
        est = int(c / max(1, n) * EST)
        print(f"  {d:14s} {c:>7,} ({pct*100:5.1f}%) 전체 약 {est:>9,}행 {'█'*int(pct*30)}")

    print(f"\n[GitHub 코드의 연구 관련성] (코드 버킷 {n_code:,}행 대비)")
    for r, c in code_rel.most_common():
        pct = c / max(1, n_code)
        est = int(c / max(1, n) * EST)
        print(f"  {r:12s} {c:>7,} ({pct*100:5.1f}%) 전체 약 {est:>9,}행 {'█'*int(pct*30)}")

    print("\n[연구 코드의 분야]")
    for d, c in code_rel_domain.get("연구 코드", Counter()).most_common(6):
        print(f"  {d:14s} {c:,}")

    print("\n[내용 유형별 주요 분야]")
    for b, cc in dom_by_bucket.items():
        top = ", ".join(f"{k} {v/sum(cc.values())*100:.0f}%" for k, v in cc.most_common(3))
        print(f"  {b:14s} → {top}")

    Path(a.json).write_text(json.dumps({
        "sampled_rows": n,
        "research_rows": n_research,
        "code_rows": n_code,
        "estimated_total": EST,
        "domain_counts": dict(dom_all),
        "domain_pct": {k: round(v / max(1, n_research), 4) for k, v in dom_all.items()},
        "domain_estimated": {k: int(v / max(1, n) * EST) for k, v in dom_all.items()},
        "code_relevance": dict(code_rel),
        "code_relevance_pct": {k: round(v / max(1, n_code), 4) for k, v in code_rel.items()},
        "code_relevance_estimated": {k: int(v / max(1, n) * EST) for k, v in code_rel.items()},
        "research_code_domains": dict(code_rel_domain.get("연구 코드", Counter())),
        "domain_by_bucket": {b: dict(cc) for b, cc in dom_by_bucket.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {a.json}")


if __name__ == "__main__":
    main()
