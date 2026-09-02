"""국내 연구데이터 소스 어댑터.

접근 가능 소스는 즉시 사용하고, 차단된 소스는 **API 명세 기반 어댑터**를 준비해
외부망 환경에서 네트워크/키만 주어지면 바로 동작하도록 한다.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Iterable, Optional

from compliance_gateway.data.korean.models import KRResearchRecord

SEED_PATH = Path("compliance_gateway/data/korean/seed/kr_trials.json")

# 국내 연구기관 식별 키워드.
# ClinicalTrials sponsor 검색은 퍼지 매칭이라 해외 기관이 섞인다 → 이 목록으로 필터링.
KR_INSTITUTIONS: tuple[str, ...] = (
    # 대학·병원
    "seoul national university", "yonsei", "severance", "korea university",
    "sungkyunkwan", "samsung medical center", "asan medical center",
    "hanyang", "kyung hee", "chung-ang", "ewha", "inha", "ajou",
    "chungnam national", "chonnam national", "kyungpook national",
    "pusan national", "jeonbuk national", "gachon", "catholic university of korea",
    "kangbuk", "konkuk", "dankook", "wonkwang", "soonchunhyang",
    # 과기특성화·정출연
    "kaist", "korea advanced institute", "postech", "gist", "unist", "dgist",
    "kist", "korea institute of science", "kribb", "korea research institute",
    "kisti", "krict", "kaeri", "kier", "etri", "kigam", "kfri",
    # 기업·기타
    "gnt pharma", "samsung", "lg chem", "sk bio", "celltrion", "hanmi",
    "yuhan", "daewoong", "green cross", "chong kun dang",
    "republic of korea", ", korea", "south korea",
)


def is_korean_institution(name: str) -> bool:
    """기관명이 국내 기관인지 판정."""
    low = (name or "").lower()
    return any(kw in low for kw in KR_INSTITUTIONS)


def load_kr_seed(path: Optional[Path] = None, korean_only: bool = True) -> list[KRResearchRecord]:
    """확보된 국내 연구과제 시드 로드."""
    p = path or SEED_PATH
    rows = json.loads(Path(p).read_text(encoding="utf-8"))
    recs = [KRResearchRecord.from_dict(r) for r in rows]
    if korean_only:
        recs = [r for r in recs if is_korean_institution(r.sponsor)]
    return recs


# ---------------------------------------------------------------------------
# 차단된 소스 어댑터 — 외부망 환경에서 활성화
# ---------------------------------------------------------------------------

class ScienceONAdapter:
    """KISTI ScienceON OpenAPI 어댑터 (논문 186.8만 · 보고서 24.7만 · 특허 102만).

    본 환경: egress 정책으로 차단. 외부망 + 발급키에서 동작.
    신청: https://scienceon.kisti.re.kr  (R&D데이터 신청 / OpenAPI 키 발급)
    """

    BASE = "https://apigateway.kisti.re.kr/openapicall.do"
    name = "scienceon"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, rows: int = 20) -> list[dict]:
        if not self.available():
            raise RuntimeError(
                "ScienceON API 키 없음. 외부망 환경에서 KISTI 발급키를 지정하세요. "
                "(본 개발 환경은 egress 정책으로 scienceon.kisti.re.kr 차단)"
            )
        import urllib.parse, urllib.request  # noqa: E401  (지연 임포트)

        params = {
            "client_id": self.api_key, "token": self.api_key,
            "version": "1.0", "action": "search", "target": "ARTI",
            "searchQuery": json.dumps({"BI": query}, ensure_ascii=False),
            "displayCount": str(rows),
        }
        url = f"{self.BASE}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("recordList", [])


class NTISAdapter:
    """NTIS 국가R&D 정보 어댑터. 본 환경 차단 — 외부망 + 신청키에서 동작."""

    BASE = "https://www.ntis.go.kr/rndopen/openApi/public_rndtask"
    name = "ntis"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key

    def available(self) -> bool:
        return bool(self.api_key)


class ArxivAdapter:
    """arXiv API 어댑터 — 국내 기관 소속 논문 수집용.

    본 환경: egress 정책으로 export.arxiv.org 차단(403 CONNECT).
    외부망에서 키 없이 즉시 사용 가능(무인증 API).
    """

    BASE = "http://export.arxiv.org/api/query"
    name = "arxiv"

    _ENTRY = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
    _FIELD = {
        "id": re.compile(r"<id>(.*?)</id>", re.DOTALL),
        "title": re.compile(r"<title>(.*?)</title>", re.DOTALL),
        "summary": re.compile(r"<summary>(.*?)</summary>", re.DOTALL),
        "published": re.compile(r"<published>(.*?)</published>", re.DOTALL),
    }
    _AUTHOR = re.compile(r"<name>(.*?)</name>", re.DOTALL)

    def __init__(self, timeout: float = 20.0) -> None:
        self.timeout = timeout

    def search(self, query: str, max_results: int = 50) -> list[dict]:
        """arXiv 검색. 국내 기관 예: `all:"KAIST"`, `all:"Seoul National University"`."""
        import urllib.parse, urllib.request  # noqa: E401

        params = {"search_query": query, "max_results": str(max_results),
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        url = f"{self.BASE}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            return self.parse(resp.read().decode("utf-8"))

    @classmethod
    def parse(cls, xml: str) -> list[dict]:
        """Atom 응답 파싱(의존성 없이 정규식). 오프라인 테스트 가능."""
        out = []
        for raw in cls._ENTRY.findall(xml):
            rec = {}
            for key, pat in cls._FIELD.items():
                m = pat.search(raw)
                # XML 엔티티 해제 필수: "R&amp;D" → "R&D" (본 도메인 필수 용어)
                rec[key] = html.unescape(" ".join(m.group(1).split())) if m else ""
            rec["authors"] = tuple(
                html.unescape(" ".join(a.split())) for a in cls._AUTHOR.findall(raw)
            )
            rec["arxiv_id"] = rec["id"].rsplit("/", 1)[-1] if rec.get("id") else ""
            out.append(rec)
        return out


def available_sources() -> dict[str, bool]:
    """소스별 현재 사용 가능 여부(진단용)."""
    return {
        "clinicaltrials_seed": SEED_PATH.exists(),
        "scienceon": ScienceONAdapter().available(),
        "ntis": NTISAdapter().available(),
        "arxiv": False,   # 본 환경 egress 차단
    }
