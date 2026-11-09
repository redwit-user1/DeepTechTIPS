#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""토크나이저 확장 필요성 측정 — 연구노트(한국어) 토큰 효율.

확장(한국어 어휘 추가)은 임베딩 재학습이 필요해 비싸다. **효율 손실이 클 때만** 한다.
측정: 한글 음절당 토큰, 어절당 토큰, 같은 의미 영어 대비 배수 대신 **한국어 특화 토크나이저 대비** 배수.

    python scripts/tokenizer_fertility.py ocr_goono_ocr_texts.csv --rows 3000 \
        --tok Qwen/Qwen3.8-27B --tok skt/A.X-4.0-Light

tokenizer.json 만 받는다(가중치 없음). 원문은 출력하지 않는다 — 비식별화 후 수치만.
"""

from __future__ import annotations

import argparse
import itertools
import re
import urllib.request
from pathlib import Path

from compliance_gateway.data.ocr.csv_source import iter_rows
from compliance_gateway.data.ocr.deidentify import deidentify

_HANGUL = re.compile(r"[가-힣]")
_KO_WORD = re.compile(r"[가-힣]+")


def load(repo: str, cache: Path):
    from tokenizers import Tokenizer

    p = cache / (repo.replace("/", "__") + ".json")
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/tokenizer.json", p)
    return Tokenizer.from_file(str(p))


def measure(tok, texts: list[str]) -> dict:
    syl = sum(len(_HANGUL.findall(t)) for t in texts)
    words = [w for t in texts for w in _KO_WORD.findall(t)]
    ntok = sum(len(tok.encode(t, add_special_tokens=False).ids) for t in texts)
    nchar = sum(len(t) for t in texts)
    # 한글 어절을 단독으로 넣었을 때 토큰 수 — 문맥 없이 어휘 적합도만
    wtok = sum(len(tok.encode(w, add_special_tokens=False).ids) for w in words)
    return {"tokens": ntok, "tok_per_char": ntok / nchar, "tok_per_syllable_word": wtok / max(1, syl),
            "tok_per_eojeol": wtok / max(1, len(words)), "vocab": tok.get_vocab_size()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--tok", action="append", required=True)
    ap.add_argument("--cache", type=Path, default=Path("data/tokenizers"))
    a = ap.parse_args()
    texts = [deidentify(r.text).text for r in itertools.islice(iter_rows(Path(a.csv_path)), a.rows)
             if len(_HANGUL.findall(r.text)) > 50]
    print(f"한국어 행 {len(texts)} · 문자 {sum(map(len, texts)):,}")
    base = None
    for repo in a.tok:
        m = measure(load(repo, a.cache), texts)
        base = base or m["tokens"]
        print(f"{repo:28s} 어휘 {m['vocab']:>7,} · 토큰 {m['tokens']:>9,} (첫 토크나이저 대비 {m['tokens'] / base:.2f}배)"
              f" · 문자당 {m['tok_per_char']:.3f} · 한글 음절당 {m['tok_per_syllable_word']:.3f} · 어절당 {m['tok_per_eojeol']:.2f}")


if __name__ == "__main__":
    main()
