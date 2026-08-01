"""3단계 인용 검증 파이프라인.

  Stage 1 exact  : DOI 직접 조회 → 가중 메타데이터 유사도 판정
  Stage 2 fuzzy  : DOI 없거나 미스 시 제목/저자 후보 검색 → 상위 후보 판정
  Stage 3 llm    : (선택) 모호할 때 LLM 판별 — 어댑터 주입식

백엔드는 `LookupBackend` 프로토콜로 분리해 오프라인(LocalRegistry)과
온라인(CrossRef/OpenAlex)을 동일 코드로 다룬다. 본 개발 환경은 외부 API 가
차단되어 LocalRegistry 로 동작하고, A100/운영 환경에서는 CrossRefBackend 를 주입한다.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Protocol

from compliance_gateway.models import Citation
from compliance_gateway.verify.models import (
    CitationStatus,
    PaperRecord,
    VerificationResult,
)
from compliance_gateway.verify.scoring import (
    THRESHOLD_PARTIAL,
    THRESHOLD_VALID,
    weighted_score,
)

# (premise, hypothesis) 대신 인용 판별용 LLM 어댑터: 질의 → 판정 문자열
LLMJudge = Callable[[Citation, list[PaperRecord]], Optional[CitationStatus]]


class LookupBackend(Protocol):
    """서지 DB 백엔드."""

    name: str

    def by_doi(self, doi: str) -> Optional[PaperRecord]: ...

    def search(self, title: str, author: str = "", limit: int = 5) -> list[PaperRecord]: ...


class LocalRegistry:
    """오프라인 로컬 서지 레지스트리(시드·캐시 기반).

    외부 API 가 막힌 환경, 에어갭 온프레미스 배포에서 1차 백엔드로 사용한다.
    """

    name = "local"

    def __init__(self, records: Iterable[PaperRecord] = ()) -> None:
        from compliance_gateway.verify.scoring import normalize

        self._records = list(records)
        self._by_doi = {r.doi: r for r in self._records if r.doi}
        # 정확 제목 인덱스 — 대규모 레지스트리에서 선형 스캔을 회피
        self._by_title = {normalize(r.title): r for r in self._records if r.title}

    def add(self, record: PaperRecord) -> None:
        from compliance_gateway.verify.scoring import normalize

        self._records.append(record)
        if record.doi:
            self._by_doi[record.doi] = record
        if record.title:
            self._by_title[normalize(record.title)] = record

    def by_doi(self, doi: str) -> Optional[PaperRecord]:
        return self._by_doi.get(doi)

    def search(self, title: str, author: str = "", limit: int = 5) -> list[PaperRecord]:
        from compliance_gateway.verify.scoring import normalize, title_similarity

        # 1) 정확 제목 일치(정규화 후) → O(1)
        exact = self._by_title.get(normalize(title))
        if exact is not None:
            return [exact]
        # 2) 폴백: 유사도 선형 스캔
        scored = [(title_similarity(title, r.title), r) for r in self._records]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for s, r in scored[:limit] if s > 0]

    @classmethod
    def from_seed(cls) -> "LocalRegistry":
        """bioRxiv 시드에서 레지스트리 구성(합성 평가·테스트용)."""
        from compliance_gateway.data.synth import load_seed

        recs = []
        for pp in load_seed():
            authors = tuple(a.strip() for a in pp.authors.split(";") if a.strip())
            recs.append(
                PaperRecord(doi=pp.doi, title=pp.title, authors=authors,
                            year=pp.year, source="local")
            )
        return cls(recs)


class CitationVerifier:
    """3단계 인용 검증기.

    Parameters
    ----------
    backends: 조회 순서대로 시도할 백엔드들(로컬 → 온라인 순 권장).
    llm_judge: Stage 3 판별 어댑터(선택).
    """

    def __init__(
        self,
        backends: Iterable[LookupBackend],
        llm_judge: Optional[LLMJudge] = None,
    ) -> None:
        self.backends = list(backends)
        self.llm_judge = llm_judge

    # ---- Stage 1: DOI 직접 조회 ----------------------------------------
    def _exact(self, citation: Citation) -> Optional[VerificationResult]:
        if not citation.doi:
            return None
        for backend in self.backends:
            rec = backend.by_doi(citation.doi)
            if rec is None:
                continue
            score, mismatches = weighted_score(
                citation.title or "", citation.authors, citation.year, rec
            )
            # DOI 가 실존하면 최소 PARTIALLY_VALID. 메타데이터까지 맞으면 VALID.
            if score >= THRESHOLD_VALID or not (citation.title or citation.authors):
                status = CitationStatus.VALID
            elif score >= THRESHOLD_PARTIAL:
                status = CitationStatus.PARTIALLY_VALID
            else:
                # DOI 는 있으나 저자/제목이 전혀 다름 → 서지 변조(유형 A 변형)
                status = CitationStatus.PARTIALLY_VALID
            return VerificationResult(
                status=status, score=round(score, 4), matched=rec,
                stage="exact", mismatches=mismatches,
                detail={"backend": backend.name, "doi": citation.doi},
            )
        # 어떤 백엔드에도 DOI 가 없음 → 환각(유형 A)
        return VerificationResult(
            status=CitationStatus.HALLUCINATED, score=0.0, stage="exact",
            detail={"reason": "doi_not_found", "doi": citation.doi},
        )

    # ---- Stage 2: 제목/저자 퍼지 검색 -----------------------------------
    def _fuzzy(self, citation: Citation) -> Optional[VerificationResult]:
        query = citation.title or ""
        author = citation.authors[0] if citation.authors else ""
        if not query and not author:
            return None
        best: Optional[tuple[float, PaperRecord, tuple[str, ...], str]] = None
        for backend in self.backends:
            for rec in backend.search(query, author=author, limit=5):
                score, mism = weighted_score(query, citation.authors, citation.year, rec)
                if best is None or score > best[0]:
                    best = (score, rec, mism, backend.name)
        if best is None:
            return None
        score, rec, mism, bname = best
        if score >= THRESHOLD_VALID:
            status = CitationStatus.VALID
        elif score >= THRESHOLD_PARTIAL:
            status = CitationStatus.PARTIALLY_VALID
        else:
            status = CitationStatus.HALLUCINATED
        return VerificationResult(
            status=status, score=round(score, 4), matched=rec, stage="fuzzy",
            mismatches=mism, detail={"backend": bname},
        )

    def verify(self, citation: Citation) -> VerificationResult:
        """인용 1건을 3단계로 검증한다."""
        if not self.backends:
            return VerificationResult(status=CitationStatus.UNVERIFIED, stage="none")

        result = self._exact(citation)
        # DOI 미스 시 퍼지로 구제(제목/저자가 실제 논문과 맞을 수 있음)
        if result is None or result.is_hallucinated:
            fuzzy = self._fuzzy(citation)
            if fuzzy is not None and not fuzzy.is_hallucinated:
                return fuzzy
            result = result or fuzzy

        if result is None:
            return VerificationResult(status=CitationStatus.UNVERIFIED, stage="none")

        # Stage 3: 모호(부분 유효)할 때만 LLM 판별
        if self.llm_judge is not None and result.status is CitationStatus.PARTIALLY_VALID:
            candidates = [result.matched] if result.matched else []
            verdict = self.llm_judge(citation, candidates)
            if verdict is not None:
                result.status = verdict
                result.stage = "llm"
        return result

    def verify_all(self, citations: list[Citation]) -> list[VerificationResult]:
        return [self.verify(c) for c in citations]

    def as_doi_resolver(self) -> Callable[[str], bool]:
        """기존 `DOIResolver = Callable[[str], bool]` 인터페이스 어댑터(하위 호환)."""

        def resolve(doi: str) -> bool:
            return self.verify(Citation(raw=doi, doi=doi)).status is not CitationStatus.HALLUCINATED

        return resolve
