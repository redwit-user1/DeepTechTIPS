"""프로젝트 메모리 L1/L2 (아키텍처 ④).

- **L1** = 실행 중 세션 컨텍스트(그냥 dict). 실행이 끝나면 사라진다.
- **L2** = 장기 보존. 핵심 산출물만 `promote()` 로 승급하며, 추가만 하는(append-only) JSONL 이다.
  기존 줄은 고치지 않는다 — 감사추적과 같은 원칙.

L2 는 디스크에 남고 다음 과제 기획의 근거로 다시 쓰이므로, 들어가는 텍스트는 학습 데이터와
같은 관문을 지난다: 건강 민감정보 표지(`sensitive_signals`)가 하나라도 있으면 **거부**하고,
나머지는 `deidentify()` 로 가명 치환한 뒤 `audit()` 잔존이 0일 때만 쓴다.

기본 경로 `data/agent_memory/` 는 .gitignore(`/data/`) 대상이다 — 커밋되지 않는다.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from compliance_gateway.data.ocr.deidentify import audit, deidentify
from compliance_gateway.data.ocr.sensitive import sensitive_signals

DEFAULT_PATH = Path("data/agent_memory/l2.jsonl")


class SensitiveContentError(ValueError):
    """L2 승급 거부 — 민감정보 표지 또는 비식별화 후 개인정보 잔존. 메시지에 원문을 넣지 않는다."""


class ProjectMemory:
    def __init__(self, path: Union[str, Path] = DEFAULT_PATH, salt: Optional[str] = None) -> None:
        self.l1: dict = {}
        self.path = Path(path)
        # OCR 수확과 같은 salt — 같은 사람은 같은 가명(문서 간 일관성)
        self.salt = salt or os.getenv("GOONO_DEID_SALT") or "goono-deid-v1"

    def promote(self, key: str, value: str, reason: str) -> dict:
        """L1 산출물을 L2 로 승급한다. 쓴 레코드(비식별화본)를 돌려준다.

        민감 표지가 있거나 비식별화 후에도 개인정보가 남으면 `SensitiveContentError`.
        """
        fields = {"key": key, "value": value, "reason": reason}
        signals = sorted({s for v in fields.values() for s in sensitive_signals(v)})
        if signals:
            raise SensitiveContentError(f"민감정보 표지 {signals} — L2 승급 거부")
        clean, replaced = {}, 0
        for name, text in fields.items():
            r = deidentify(text, salt=self.salt)
            residual = audit(r.text)
            if residual:
                raise SensitiveContentError(f"비식별화 후 개인정보 잔존 {sorted(residual)} — L2 승급 거부")
            clean[name] = r.text
            replaced += r.total
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **clean,
               "deidentified": replaced}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        """L2 에서 질의 단어(공백 분리)를 부분 문자열로 찾는다. 겹친 단어 수 → 최근 순.

        한국어 조사가 붙어도 걸린다("수율" ⊂ "수율은"). 의미 검색은 RAG 단계에서.
        """
        words = [w.lower() for w in query.split()]
        if not words or not self.path.exists():
            return []
        scored = []
        with self.path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                hay = f"{rec['key']} {rec['value']} {rec['reason']}".lower()
                hits = sum(w in hay for w in words)
                if hits:
                    scored.append((hits, i, rec))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return [rec for _, _, rec in scored[:limit]]
