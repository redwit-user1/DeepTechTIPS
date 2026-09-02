# -*- coding: utf-8 -*-
"""GOONO OCR CSV 리더 — 대용량 단일 파일 스트리밍.

실제 내보내기 형식:
    id, NoteId, fileName, page, text

특징과 대응:

| 특징 | 대응 |
|---|---|
| 단일 파일 수 GB | **스트리밍**(전체 적재 금지) |
| `text` 에 리터럴 `\\n` | 실제 개행으로 복원 |
| NoteId 당 여러 page | 페이지 그룹핑 지원 |
| 연구노트 외 문서 혼재(xlsx/pdf 등) | `fileName` 확장자·본문 단서로 분류 |
| 대형 필드 | `csv.field_size_limit` 상향 |
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

# OCR 본문 한 셀이 매우 클 수 있다(기본 131,072자 제한이면 터진다)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# 내보내기에서 개행이 리터럴 "\n" 으로 이스케이프돼 있다
_ESCAPED_NL = re.compile(r"\\r\\n|\\n|\\r")
_ESCAPED_TAB = re.compile(r"\\t")


@dataclass
class OcrRow:
    """CSV 한 행 = 문서 한 페이지."""

    id: str
    note_id: str
    file_name: str
    page: int
    text: str

    @property
    def ext(self) -> str:
        return Path(self.file_name).suffix.lower() or "(없음)"


@dataclass
class OcrNote:
    """NoteId 로 묶은 문서 한 건(여러 페이지)."""

    note_id: str
    file_name: str
    pages: list[OcrRow] = field(default_factory=list)

    @property
    def text(self) -> str:
        """페이지 순서대로 이어붙인 전체 본문."""
        return "\n\n".join(p.text for p in sorted(self.pages, key=lambda r: r.page))

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def unescape(text: str) -> str:
    """리터럴 `\\n` / `\\t` 를 실제 제어문자로 복원."""
    return _ESCAPED_TAB.sub("\t", _ESCAPED_NL.sub("\n", text or ""))


def iter_rows(
    path: str | Path,
    limit: Optional[int] = None,
    skip: int = 0,
    encoding: str = "utf-8",
) -> Iterator[OcrRow]:
    """CSV 를 스트리밍하며 행 단위로 내보낸다(전체 적재 없음)."""
    with open(path, newline="", encoding=encoding, errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < skip:
                continue
            if limit is not None and (i - skip) >= limit:
                return
            try:
                page = int(row.get("page") or 0)
            except (TypeError, ValueError):
                page = 0
            yield OcrRow(
                id=(row.get("id") or "").strip(),
                note_id=(row.get("NoteId") or "").strip(),
                file_name=(row.get("fileName") or "").strip(),
                page=page,
                text=unescape(row.get("text") or ""),
            )


def iter_notes(
    path: str | Path,
    limit_rows: Optional[int] = None,
    encoding: str = "utf-8",
) -> Iterator[OcrNote]:
    """NoteId 로 묶어 문서 단위로 내보낸다.

    같은 NoteId 의 행이 인접해 있다고 가정하지 않는다. 인접하지 않은 경우를
    대비해 **직전 NoteId 가 바뀌면 flush** 하되, 이미 내보낸 NoteId 가 다시
    나타나면 별도 문서로 취급한다(전량 메모리 적재를 피하기 위한 절충).
    """
    current: Optional[OcrNote] = None
    for row in iter_rows(path, limit=limit_rows, encoding=encoding):
        if current is None or row.note_id != current.note_id:
            if current is not None:
                yield current
            current = OcrNote(note_id=row.note_id, file_name=row.file_name)
        current.pages.append(row)
    if current is not None:
        yield current


def count_rows(path: str | Path, encoding: str = "utf-8") -> int:
    """전체 행 수(전수 스캔 — 대용량에서는 시간이 걸린다)."""
    n = 0
    with open(path, newline="", encoding=encoding, errors="replace") as f:
        for _ in csv.DictReader(f):
            n += 1
    return n
