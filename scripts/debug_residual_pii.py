#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""비식별화 잔존 PII 진단 — 값을 노출하지 않고 **형태만** 본다.

숫자는 D, 한글은 K, 라틴은 A 로 마스킹해 패턴 모양만 확인한다.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from compliance_gateway.data.ocr.deidentify import RULES, audit, deidentify


def mask(s: str) -> str:
    s = re.sub(r"\d", "D", s)
    s = re.sub(r"[가-힣]", "K", s)
    s = re.sub(r"[A-Za-z]", "A", s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--context", type=int, default=14)
    a = ap.parse_args()

    seen: dict[str, int] = {}
    for line in Path(a.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        left = audit(text)
        if not left:
            continue
        for name, pat, _prefix in RULES:
            if name not in left:
                continue
            for m in pat.finditer(text):
                raw = m.group(1) if m.groups() else m.group(0)
                # 이미 가명이면 건너뜀
                if re.match(r"^_[0-9A-F]{4}\b", text[m.end():m.end() + 8]):
                    continue
                s, e = max(0, m.start() - a.context), min(len(text), m.end() + a.context)
                shape = f"{name} | ...{mask(text[s:e])}..."
                seen[shape] = seen.get(shape, 0) + 1

    print("=" * 62)
    print(" 비식별화 잔존 패턴 (값 미노출 — D=숫자 K=한글 A=영문)")
    print("=" * 62)
    if not seen:
        print("\n잔존 없음 ✅")
        return
    for shape, c in sorted(seen.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  [{c:>3}회] {shape}")


if __name__ == "__main__":
    main()
