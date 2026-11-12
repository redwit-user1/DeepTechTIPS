#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OCR 연구노트 데이터셋 프로파일러 — **내용을 노출하지 않고** 구조만 파악한다.

연구노트는 미공개 연구내용·개인정보·영업비밀을 포함할 수 있다.
이 스크립트는 **원문을 출력하지 않고** 통계·구조·품질 지표만 산출하므로,
결과를 그대로 공유해도 안전하다(단, 결과를 눈으로 확인한 뒤 공유할 것).

  python scripts/profile_ocr_dataset.py /path/to/dataset
  python scripts/profile_ocr_dataset.py /path/to/dataset --json profile.json
  python scripts/profile_ocr_dataset.py /path/to/dataset --sample-fields   # 필드명만(값 제외)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

TEXT_EXT = {".txt", ".json", ".jsonl", ".csv", ".tsv", ".xml", ".md", ".hocr", ".html"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DOC_EXT = {".pdf", ".docx", ".hwp", ".hwpx", ".xlsx", ".pptx"}

# 연구노트 구조 단서 (한국어 지침 용어)
NOTE_MARKERS = {
    "과제": r"과제\s*(?:명|번호|고유번호)",
    "연구자/서명": r"연구자|작성자|서명|署名",
    "점검자": r"점검자|입회자|확인자",
    "일자": r"실험\s*일|기록\s*일|작성\s*일|\d{4}[-./]\d{1,2}[-./]\d{1,2}",
    "실험목적": r"실험\s*목적|연구\s*목적|목\s*적",
    "재료·방법": r"재료\s*(?:및|와)?\s*방법|실험\s*방법|방\s*법",
    "결과": r"실험\s*결과|결\s*과",
    "고찰": r"고\s*찰|결\s*론|차기\s*계획",
}

# OCR 품질 열화 신호
OCR_NOISE = {
    "치환문자(�)": r"�",
    "고립자모": r"[ㄱ-ㅎㅏ-ㅣ]{2,}",
    # 실제 OCR 혼동은 한 토큰 안에 O/0 또는 l/1 이 **섞여** 나타난다.
    # 단순히 0 으로 시작하는 숫자(노트번호 0002 등)를 잡으면 오탐이 된다.
    "숫자-문자 혼동": r"\b(?=[0-9O]*O)(?=[0-9O]*[0-9])[0-9O]{3,}\b|\b(?=[1l]*l)(?=[1l]*1)[1l]{3,}\b",
    "과다공백": r"\s{5,}",
    "깨진괄호": r"[\(\[]\s*[\)\]]",
}

# 개인정보 위험 신호 — **탐지만 하고 값은 절대 출력하지 않는다**
PII_PATTERNS = {
    "주민등록번호형": r"\d{6}\s*[-–]\s*[1-4]\d{6}",
    "휴대전화": r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}",
    "이메일": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "한국인명(3자)": r"(?:연구자|작성자|점검자|성명)\s*[:：]?\s*[가-힣]{2,4}",
}

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def read_text(path: Path, limit: int = 200_000) -> str:
    try:
        if path.suffix.lower() in {".json", ".jsonl"}:
            raw = path.read_text(encoding="utf-8", errors="replace")[:limit]
            return raw
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def json_field_names(path: Path, cap: int = 50) -> set[str]:
    """JSON/JSONL 의 **키 이름만** 수집(값은 읽지 않는다)."""
    names: set[str] = set()
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            if path.suffix.lower() == ".jsonl":
                for i, line in enumerate(f):
                    if i >= 20:
                        break
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        names |= set(obj.keys())
            else:
                obj = json.load(f)
                if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                    names |= set(obj[0].keys())
                elif isinstance(obj, dict):
                    names |= set(obj.keys())
    except Exception:
        pass
    return set(list(names)[:cap])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="데이터셋 최상위 경로")
    ap.add_argument("--max-text-files", type=int, default=400, help="본문 분석 표본 수")
    ap.add_argument("--json", default=None, help="결과 JSON 저장 경로")
    ap.add_argument("--sample-fields", action="store_true",
                    help="JSON/JSONL 의 필드명 수집(값은 수집하지 않음)")
    a = ap.parse_args()

    root = Path(a.root).expanduser()
    if not root.exists():
        sys.exit(f"경로 없음: {root}")

    ext_count: Counter[str] = Counter()
    ext_bytes: Counter[str] = Counter()
    depth_dirs: Counter[int] = Counter()
    total_files = total_bytes = 0

    for dirpath, _dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        depth_dirs[depth] += 1
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            ext = p.suffix.lower() or "(확장자없음)"
            ext_count[ext] += 1
            ext_bytes[ext] += sz
            total_files += 1
            total_bytes += sz

    print("=" * 66)
    print(f" OCR 연구노트 데이터셋 프로파일: {root}")
    print("=" * 66)
    print(f"\n총 {total_files:,} 파일 / {human(total_bytes)}")
    print(f"디렉터리 깊이 분포: {dict(sorted(depth_dirs.items()))}")

    print("\n[확장자별]")
    for ext, cnt in ext_count.most_common(15):
        kind = ("텍스트" if ext in TEXT_EXT else "이미지" if ext in IMAGE_EXT
                else "문서" if ext in DOC_EXT else "기타")
        print(f"  {ext:14s} {cnt:>8,}개  {human(ext_bytes[ext]):>9s}  [{kind}]")

    # ---- 본문 표본 분석 ----
    text_files = [Path(dp) / fn
                  for dp, _dn, fns in os.walk(root) for fn in fns
                  if Path(fn).suffix.lower() in TEXT_EXT]
    sample = text_files[: a.max_text_files]
    print(f"\n[본문 표본 분석] 텍스트 파일 {len(text_files):,}개 중 {len(sample)}개 표본")

    if not sample:
        print("  텍스트 파일이 없습니다 — 이미지만 있다면 OCR 단계가 선행되어야 합니다.")
    else:
        marker_hits: Counter[str] = Counter()
        noise_hits: Counter[str] = Counter()
        pii_hits: Counter[str] = Counter()
        hangul = latin = chars = 0
        line_lens: list[int] = []
        fields: set[str] = set()

        for p in sample:
            t = read_text(p)
            if not t:
                continue
            chars += len(t)
            hangul += len(_HANGUL.findall(t))
            latin += len(_LATIN.findall(t))
            line_lens += [len(l) for l in t.splitlines()[:200]]
            for name, pat in NOTE_MARKERS.items():
                if re.search(pat, t):
                    marker_hits[name] += 1
            for name, pat in OCR_NOISE.items():
                if re.search(pat, t):
                    noise_hits[name] += 1
            for name, pat in PII_PATTERNS.items():
                n = len(re.findall(pat, t))
                if n:
                    pii_hits[name] += n     # 건수만, 값은 저장하지 않음
            if a.sample_fields and p.suffix.lower() in {".json", ".jsonl"}:
                fields |= json_field_names(p)

        n = len(sample)
        print(f"  분석 문자수: {chars:,}  (한글 {hangul:,} / 라틴 {latin:,})")
        ratio = hangul / max(1, hangul + latin)
        print(f"  한글 비율: {ratio:.1%}  → {'한국어 중심' if ratio > 0.5 else '영문 중심' if ratio < 0.2 else '혼합'}")
        if line_lens:
            line_lens.sort()
            print(f"  줄 길이 중앙값: {line_lens[len(line_lens)//2]}자")

        print("\n  [연구노트 구조 단서] (표본 대비 출현율)")
        for name in NOTE_MARKERS:
            c = marker_hits[name]
            bar = "█" * int(c / n * 20)
            print(f"    {name:12s} {c/n*100:5.1f}% {bar}")
        struct = sum(marker_hits.values()) / (n * len(NOTE_MARKERS))
        verdict = ("연구노트 구조가 뚜렷함" if struct > 0.5 else
                   "부분적 구조" if struct > 0.2 else "구조 단서 약함(자유 서술일 가능성)")
        print(f"    → 구조성 지수 {struct:.2f}: {verdict}")

        print("\n  [OCR 품질 신호] (높을수록 노이즈 많음)")
        for name in OCR_NOISE:
            print(f"    {name:16s} {noise_hits[name]/n*100:5.1f}%")

        print("\n  [개인정보 위험] ⚠️ 건수만 표시 — 값은 수집하지 않음")
        if pii_hits:
            for name, c in pii_hits.most_common():
                print(f"    {name:16s} {c:,}건 검출")
            print("    → 학습·공유 전 **비식별화 필수**")
        else:
            print("    검출된 패턴 없음(표본 기준). 전수 검사는 별도 필요.")

        if a.sample_fields and fields:
            print(f"\n  [JSON 필드명] {sorted(fields)}")

    if a.json:
        out = {
            "root": str(root), "total_files": total_files, "total_bytes": total_bytes,
            "ext_count": dict(ext_count.most_common(30)),
            "ext_bytes": dict(ext_bytes.most_common(30)),
        }
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {a.json} 저장 (원문 미포함)")

    print("\n" + "=" * 66)
    print(" 다음 단계는 위 결과에 따라 갈립니다:")
    print("  · 구조성 높음 + 한글 중심 → 연구노트 파서로 바로 수확")
    print("  · 이미지만 존재         → OCR 선행 필요")
    print("  · 개인정보 검출         → 비식별화 후 사용")
    print("=" * 66)


if __name__ == "__main__":
    main()
