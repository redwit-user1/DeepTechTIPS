"""국내 연구데이터 소스 어댑터.

접근 가능 소스는 즉시 사용하고, 차단된 소스는 **API 명세 기반 어댑터**를 준비해
외부망 환경에서 네트워크/키만 주어지면 바로 동작하도록 한다.
"""

from __future__ import annotations

import functools
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional
from urllib.robotparser import RobotFileParser

from compliance_gateway.data.korean.models import KRResearchRecord
from compliance_gateway.verify.backends import DEFAULT_MAILTO

USER_AGENT = f"GOONO-AI research (mailto:{DEFAULT_MAILTO})"


def _get(url: str, timeout: float = 20.0, binary: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return body if binary else body.decode("utf-8")


@functools.lru_cache(maxsize=None)
def _robots(origin: str) -> RobotFileParser:
    try:
        txt = _get(f"{origin}/robots.txt")
    except urllib.error.HTTPError as e:
        if not 400 <= e.code < 500:
            raise
        txt = ""   # robots.txt 없음(4xx) = 제한 없음 (RFC 9309 §2.3.1.3)
    rp = RobotFileParser()
    rp.parse(txt.splitlines())
    return rp


def polite_get(url: str, timeout: float = 20.0, binary: bool = False):
    """robots.txt 를 지키는 GET. 금지 경로면 조용히 넘기지 않고 RuntimeError.

    NTIS(`Disallow: /`)·ScienceON(검색봇 외 전면 차단)은 여기서 막힌다 — 의도된 동작.
    """
    p = urllib.parse.urlsplit(url)
    if not _robots(f"{p.scheme}://{p.netloc}").can_fetch(USER_AGENT, url):
        raise RuntimeError(f"robots.txt 가 수집을 금지한 경로: {url}")
    return _get(url, timeout, binary)


_TAG = re.compile(r"<[^>]+>")


def _text(s: str) -> str:
    """태그 제거 + 엔티티 해제 + 공백 정규화."""
    return html.unescape(" ".join(_TAG.sub("", s).split()))

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


class KoreaScienceAdapter:
    """KoreaScience(KISTI) 검색결과 HTML — 국내 학술지 국문 초록 수집.

    공식 API 없음. robots.txt 는 PDF 만 막고 HTML 은 허용(2026-11 확인) → `polite_get`.
    검색결과 한 페이지에 초록 전문이 실려 있어 논문 페이지는 열지 않는다.
    초록 저작권은 각 학회 — 평가·RAG 용. 학습 투입은 학술지별 라이선스 확인 후.
    """

    BASE = "https://koreascience.kr/search.page"
    name = "koreascience"

    _BOX = re.compile(r'<article class="srched-box">(.*?)</article>', re.DOTALL)
    _ID = re.compile(r"/article/((?:JAKO|CFKO)\d{15})\.page")
    _FIELD = {
        "title": re.compile(r'articleTitleKor">(.*?)</span>', re.DOTALL),
        "title_en": re.compile(r'artileTitleEng">(.*?)</span>', re.DOTALL),   # 사이트 오타 그대로
        "authors": re.compile(r'authorNames">(.*?)</li>', re.DOTALL),
        "journal": re.compile(r'journalTitle">(.*?)</li>', re.DOTALL),
        "year": re.compile(r'pubYear">(.*?)</li>', re.DOTALL),
        "abstract": re.compile(r'class="g-mb-15 abstract">(.*?)</p>', re.DOTALL),
        "doi": re.compile(r'href="https://doi\.org/([^"]+)"'),
    }
    _BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
    _TAG = re.compile(r"<[^>]+>")   # 인라인 태그(<i>,<sup>)는 붙여서 제거 — "<i>X</i>을" → "X을"

    def search(self, query: str, page: int = 0, size: int = 10) -> list[dict]:
        # pageNo 는 1부터(0 은 1 과 같은 페이지를 준다 — 2026-11 실측). page 는 0부터 받는다.
        params = {"keywords": query, "lang": "ko", "pageSize": str(size), "pageNo": str(page + 1)}
        return self.parse(polite_get(f"{self.BASE}?{urllib.parse.urlencode(params)}"))

    @classmethod
    def parse(cls, page_html: str) -> list[dict]:
        out = []
        for box in cls._BOX.findall(page_html):
            m = cls._ID.search(box)
            if not m:
                continue
            rec = {"id": m.group(1), "url": f"https://koreascience.kr/article/{m.group(1)}.page"}
            for key, pat in cls._FIELD.items():
                f = pat.search(box)
                txt = cls._TAG.sub("", cls._BR.sub(" ", f.group(1))) if f else ""
                rec[key] = html.unescape(" ".join(txt.split()))
            rec["title"] = rec["title"] or rec["title_en"]
            rec["authors"] = tuple(a.strip() for a in rec["authors"].split(";") if a.strip())
            out.append(rec)
        return out


class KCIOAIAdapter:
    """KCI OAI-PMH — 국내 학술지 전체 서지·초록. 키 없음(2026-11 확인).

    OAI-PMH 는 수집용 표준 프로토콜이다. KCI 웹(www.kci.go.kr)은 robots 로 막혀 있으니 이 경로만 쓴다.
    from/until 은 **레코드 갱신일**(datestamp) 기준이지 발행연도가 아니다.
    초록 저작권은 각 학회 — 평가·RAG 용.
    """

    BASE = "https://open.kci.go.kr/oai/request"
    name = "kci"

    _REC = re.compile(r"<record>(.*?)</record>", re.DOTALL)
    _TOKEN = re.compile(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>")
    _ID = re.compile(r'article-id="([^"]+)"')
    _FIELD = {
        "journal": re.compile(r"<journal-name>(.*?)</journal-name>", re.DOTALL),
        "year": re.compile(r"<pub-year>(.*?)</pub-year>"),
        "category": re.compile(r"<article-categories>(.*?)</article-categories>", re.DOTALL),
        "language": re.compile(r"<language>(.*?)</language>"),
        "title": re.compile(r'<article-title lang="original">(.*?)</article-title>', re.DOTALL),
        "title_en": re.compile(r'<article-title lang="english">(.*?)</article-title>', re.DOTALL),
        "abstract": re.compile(r'<abstract lang="original">(.*?)</abstract>', re.DOTALL),
        "abstract_en": re.compile(r'<abstract lang="english">(.*?)</abstract>', re.DOTALL),
        "doi": re.compile(r"<doi>(.*?)</doi>"),
        "url": re.compile(r"<url>(.*?)</url>"),
    }
    # <author-name> 쪽만(이름·소속 분리). <author-group> 의 "이름(소속)" 문자열은 버린다.
    _AUTHOR = re.compile(r"<author>\s*<name>(.*?)</name>\s*(?:<affiliation>(.*?)</affiliation>)?", re.DOTALL)

    def url(self, frm: str = "", until: str = "", token: str = "") -> str:
        if token:
            # KCI 는 OAI 표준과 다르다(2026-11 실측): 토큰("from:until:offset:lastid")에 형식이 없어
            # metadataPrefix 를 빼면 oai_dc 로 돌아오고, ':' 를 %3A 로 인코딩하면 0건이 온다.
            return (f"{self.BASE}?verb=ListRecords&metadataPrefix=oai_kci"
                    f"&resumptionToken={urllib.parse.quote(token, safe=':')}")
        q = {"verb": "ListRecords", "metadataPrefix": "oai_kci", "set": "ARTI", "from": frm, "until": until}
        return f"{self.BASE}?{urllib.parse.urlencode(q)}"

    @classmethod
    def parse(cls, xml: str) -> tuple[list[dict], str]:
        """(레코드, 다음 resumptionToken 또는 "")."""
        out = []
        for raw in cls._REC.findall(xml):
            m = cls._ID.search(raw)
            if not m:   # 삭제 레코드(header status="deleted")는 metadata 가 없다
                continue
            rec = {"id": m.group(1)}
            for key, pat in cls._FIELD.items():
                f = pat.search(raw)
                rec[key] = _text(f.group(1)) if f else ""
            rec["authors"] = [{"name": _text(n), "affiliation": _text(a or "")} for n, a in cls._AUTHOR.findall(raw)]
            out.append(rec)
        t = cls._TOKEN.search(xml)
        return out, (t.group(1).strip() if t else "")


class MFDSGuideAdapter:
    """식약처 민원인안내서 게시판 — 국문 규제 가이드(GLP·GMP·데이터 완전성 등) 원문.

    robots.txt 허용(2026-11 확인). 공공누리 표시는 상세 페이지에 없음 → 건별 확인 전 학습 투입 금지.
    """

    ROOT = "https://www.mfds.go.kr/brd/m_1060"
    name = "mfds_guide"

    _SEQ = re.compile(r"view\.do\?seq=(\d+)")
    _TITLE = re.compile(r'<div class="bv_title">(.*?)</div>', re.DOTALL)
    _META = re.compile(r"<li><span>\s*(.*?)\s*</span>(.*?)</li>", re.DOTALL)
    _FILE = re.compile(r'<strong>([^<]*)</strong>\s*<a href="\./down\.do\?[^"]*?file_seq=(\d+)"')

    def list_url(self, page: int) -> str:
        return f"{self.ROOT}/list.do?page={page}"

    def view_url(self, seq: str) -> str:
        return f"{self.ROOT}/view.do?seq={seq}"

    def down_url(self, seq: str, file_seq: str) -> str:
        return f"{self.ROOT}/down.do?brd_id=data0011&seq={seq}&data_tp=A&file_seq={file_seq}"

    @classmethod
    def parse_list(cls, page_html: str) -> list[str]:
        return list(dict.fromkeys(cls._SEQ.findall(page_html)))   # 순서 유지 dedupe

    @classmethod
    def parse_view(cls, page_html: str) -> dict:
        t = cls._TITLE.search(page_html)
        meta = {_text(k): _text(v) for k, v in cls._META.findall(page_html)}
        return {
            "title": _text(t.group(1)) if t else "",
            "reg_no": meta.get("등록번호", ""),
            "date": meta.get("등록일", ""),
            "files": [{"file_seq": fs, "name": _text(n).lstrip("★ ").strip()} for n, fs in cls._FILE.findall(page_html)],
        }


class ArxivAdapter:
    """arXiv API 어댑터 — 국내 기관 소속 논문 수집용. 무인증.

    이용약관: 3초에 1회 이하 호출(수집 루프에서 지킨다). 메타데이터·초록 CC0,
    본문은 논문별 라이선스.
    """

    BASE = "https://export.arxiv.org/api/query"
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

    def search(self, query: str, max_results: int = 50, start: int = 0) -> list[dict]:
        """arXiv 검색. 국내 기관 예: `all:"KAIST"`, `all:"Seoul National University"`."""
        params = {"search_query": query, "start": str(start), "max_results": str(max_results),
                  "sortBy": "submittedDate", "sortOrder": "descending"}
        # ponytail: API 엔드포인트라 robots 대상 아님 — 레이트리밋만 지킨다
        return self.parse(_get(f"{self.BASE}?{urllib.parse.urlencode(params)}", self.timeout))

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
        "arxiv": True,          # 무인증 — 네트워크만 있으면 동작
        "koreascience": True,   # 무인증 — robots 허용 HTML
    }
