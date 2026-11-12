"""해시 체인 감사추적 — 시각 자동기록 · 위·변조 확인 · 서명-기록 연결.

Part 11 §11.10(e)(컴퓨터 생성·시각 기록 감사추적, 변경이 이전 기록을 가리지 않음), §11.70(서명-기록 연결),
Part 58 §58.130(e)(자동 수집 데이터의 입력 책임자·변경 사유), 연구노트 지침 제7조①2·3(날짜·시각 자동기록,
위·변조 확인), 제11조②(열람 관리대장)를 한 구조로 만족시킨다.

각 항목은 직전 항목의 해시를 품는다 — 중간 항목을 고치면 그 뒤 체인이 모두 깨진다.
ponytail: 로컬 체인이다. 외부 시점 증명은 체인 머리 해시를 GOONO 시점인증(허가형 블록체인)에 앵커링해서 얻는다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ACTIONS = ("create", "modify", "delete", "sign", "view", "access_denied")
GENESIS = "0" * 64


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class AuditChain:
    entries: list[dict] = field(default_factory=list)

    def append(self, actor: str, action: str, record_id: str, content: str = "",
               reason: str = "", meaning: str = "", now: Optional[datetime] = None) -> dict:
        """항목 추가. 시각은 시스템이 찍는다(`now` 는 테스트용)."""
        if action not in ACTIONS:
            raise ValueError(f"알 수 없는 행위: {action}")
        if not actor:
            raise ValueError("행위자 없음 — 입력 책임자를 식별해야 한다(58.130(e))")
        if action in ("modify", "delete") and not reason:
            raise ValueError("변경·삭제에는 사유가 필요하다(58.130(e), 58.185(c))")
        if action == "sign" and not meaning:
            raise ValueError("서명에는 의미(작성·검토·승인 등)가 필요하다(11.50(a)(3))")
        # 허용된 단계 순서(11.10(f)): 없는 기록은 수정·서명·삭제 불가, 있는 기록은 다시 생성 불가
        exists = self.current(record_id) is not None
        if action == "create" and exists:
            raise ValueError(f"이미 있는 기록: {record_id} — 변경은 modify(사유 포함)로")
        if action in ("modify", "sign", "delete") and not exists:
            raise ValueError(f"없는 기록에 {action}: {record_id}")
        e = {
            "seq": len(self.entries),
            "ts": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
            "actor": actor, "action": action, "record_id": record_id,
            # 내용 자체를 남긴다 — 변경이 이전 값을 가리지 않는다(11.10(e)). 열람(view)은 내용 없음.
            "content": content if action not in ("view", "access_denied") else "",
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest() if content else "",
            "reason": reason, "meaning": meaning,
            "prev": self.entries[-1]["hash"] if self.entries else GENESIS,
        }
        e["hash"] = _sha(e)
        self.entries.append(e)
        return e

    def verify(self) -> list[str]:
        """체인 전체 재계산. 문제 목록(비면 무결)."""
        out, prev = [], GENESIS
        for i, e in enumerate(self.entries):
            body = {k: v for k, v in e.items() if k != "hash"}
            if e.get("seq") != i:
                out.append(f"#{i} 순번 불일치")
            if e.get("prev") != prev:
                out.append(f"#{i} 이전 해시 불일치 — 앞 항목이 바뀌었거나 빠졌다")
            if e.get("content") and hashlib.sha256(e["content"].encode("utf-8")).hexdigest() != e.get("content_sha256"):
                out.append(f"#{i} 내용 해시 불일치 — 기록 내용 변조")
            if _sha(body) != e.get("hash"):
                out.append(f"#{i} 항목 해시 불일치 — 항목 변조")
            prev = e.get("hash")
        return out

    def current(self, record_id: str) -> Optional[str]:
        """기록의 현재 내용(마지막 create/modify). 삭제됐으면 None."""
        cur = None
        for e in self.entries:
            if e["record_id"] == record_id and e["action"] in ("create", "modify"):
                cur = e["content"]
            elif e["record_id"] == record_id and e["action"] == "delete":
                cur = None
        return cur

    def signatures(self, record_id: str, content: str) -> list[dict]:
        """이 **내용**에 유효한 서명들(11.70). 서명 뒤 내용이 바뀌면 그 서명은 새 내용에 붙지 않는다."""
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return [e for e in self.entries
                if e["record_id"] == record_id and e["action"] == "sign" and e["content_sha256"] == h]

    def head(self) -> str:
        """체인 머리 해시 — 외부 시점인증에 앵커링할 값."""
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def dump(self, path: Path) -> None:
        Path(path).write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in self.entries) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AuditChain":
        return cls([json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()])
