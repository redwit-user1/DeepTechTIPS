"""온라인 서지 DB 백엔드 — CrossRef / OpenAlex.

둘 다 무료·무인증 REST API. 본 개발 환경은 외부 API 가 차단되어 사용 불가하며,
A100/운영 환경에서 `CitationVerifier([LocalRegistry(...), CrossRefBackend()])`
처럼 주입한다(로컬 캐시 우선 → 온라인 폴백).

레이트리밋(업스트림 조사 기준): CrossRef 50 req/s, OpenAlex 100K req/day,
Semantic Scholar 100 req/s. 대량 검증 시 캐시·백오프 필수.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Optional

from compliance_gateway.verify.models import PaperRecord

DEFAULT_TIMEOUT = 10.0
# CrossRef 는 mailto 를 주면 우선 처리(polite pool)
DEFAULT_MAILTO = "research@redwit.io"


def _get_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": f"GOONO-AI (mailto:{DEFAULT_MAILTO})"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        # 네트워크 차단·레이트리밋·404 → 조회 실패(UNVERIFIED 유도)
        return None


class CrossRefBackend:
    """CrossRef REST API (130M+ records)."""

    name = "crossref"
    BASE = "https://api.crossref.org/works"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    @staticmethod
    def _to_record(item: dict) -> PaperRecord:
        authors = tuple(
            f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
            for a in item.get("author", []) or []
        )
        title_list = item.get("title") or [""]
        parts = (item.get("issued", {}).get("date-parts") or [[None]])[0]
        year = parts[0] if parts else None
        return PaperRecord(
            doi=item.get("DOI"), title=title_list[0] if title_list else "",
            authors=authors, year=year, source="crossref",
        )

    def by_doi(self, doi: str) -> Optional[PaperRecord]:
        data = _get_json(f"{self.BASE}/{urllib.parse.quote(doi)}", self.timeout)
        if not data or "message" not in data:
            return None
        return self._to_record(data["message"])

    def search(self, title: str, author: str = "", limit: int = 5) -> list[PaperRecord]:
        params = {"rows": str(limit)}
        if title:
            params["query.bibliographic"] = title
        if author:
            params["query.author"] = author
        data = _get_json(f"{self.BASE}?{urllib.parse.urlencode(params)}", self.timeout)
        if not data:
            return []
        return [self._to_record(i) for i in data.get("message", {}).get("items", [])]


class OpenAlexBackend:
    """OpenAlex API (250M+ works)."""

    name = "openalex"
    BASE = "https://api.openalex.org/works"

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    @staticmethod
    def _to_record(item: dict) -> PaperRecord:
        authors = tuple(
            (a.get("author", {}) or {}).get("display_name", "")
            for a in item.get("authorships", []) or []
        )
        doi = (item.get("doi") or "").replace("https://doi.org/", "") or None
        return PaperRecord(
            doi=doi, title=item.get("title") or "", authors=authors,
            year=item.get("publication_year"), source="openalex",
        )

    def by_doi(self, doi: str) -> Optional[PaperRecord]:
        data = _get_json(f"{self.BASE}/https://doi.org/{urllib.parse.quote(doi)}", self.timeout)
        if not data or "id" not in data:
            return None
        return self._to_record(data)

    def search(self, title: str, author: str = "", limit: int = 5) -> list[PaperRecord]:
        if not title:
            return []
        params = {"search": title, "per-page": str(limit)}
        data = _get_json(f"{self.BASE}?{urllib.parse.urlencode(params)}", self.timeout)
        if not data:
            return []
        return [self._to_record(i) for i in data.get("results", [])]
