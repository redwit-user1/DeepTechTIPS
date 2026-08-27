# -*- coding: utf-8 -*-
"""OCR 연구노트 → 평가셋/DPO 수확.

로컬 OCR 데이터셋을 **비식별화 → 파싱 → 무결성 검사 → 데이터셋** 으로 전환한다.

실행:
  # 1단계: 구조 파악(내용 노출 없음)
  python scripts/profile_ocr_dataset.py /path/to/ocr

  # 2단계: 수확 (기본은 dry-run — 무엇이 나올지만 보고)
  python -m compliance_gateway.data.ocr.harvest /path/to/ocr
  python -m compliance_gateway.data.ocr.harvest /path/to/ocr --write --out data/real/labnote
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from compliance_gateway.data.ocr.deidentify import audit, deidentify
from compliance_gateway.data.ocr.parse import parse_note
from compliance_gateway.integrity import check_lab_note

TEXT_EXT = {".txt", ".md", ".hocr"}
# 파싱이 이 수준 미만이면 연구노트로 보기 어렵다(규정 위반이 아니라 파싱 실패)
MIN_COMPLETENESS = 0.4


def iter_text_files(root: Path, limit: int | None = None):
    n = 0
    for dirpath, _d, filenames in os.walk(root):
        for fn in sorted(filenames):
            if Path(fn).suffix.lower() in TEXT_EXT:
                yield Path(dirpath) / fn
                n += 1
                if limit and n >= limit:
                    return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="data/real/labnote")
    ap.add_argument("--limit", type=int, default=None, help="처리할 최대 파일 수")
    ap.add_argument("--write", action="store_true",
                    help="실제로 파일을 쓴다(기본은 dry-run)")
    ap.add_argument("--salt", default=None, help="비식별화 salt(미지정 시 환경변수 GOONO_DEID_SALT)")
    a = ap.parse_args()

    salt = a.salt or os.getenv("GOONO_DEID_SALT") or "goono-deid-v1"
    if salt == "goono-deid-v1":
        print("⚠️  기본 salt 사용 중 — 배포 시 GOONO_DEID_SALT 를 설정하고 공개하지 마세요.\n")

    root = Path(a.root).expanduser()
    if not root.exists():
        raise SystemExit(f"경로 없음: {root}")

    parsed, skipped = [], 0
    deid_counts: Counter[str] = Counter()
    residual: Counter[str] = Counter()
    integ_scores: list[float] = []
    violated: Counter[str] = Counter()

    for path in iter_text_files(root, a.limit):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            skipped += 1
            continue

        d = deidentify(raw, salt=salt)
        deid_counts.update(d.counts)
        residual.update(audit(d.text))

        note = parse_note(d.text, source=str(path.relative_to(root)))
        if note.completeness < MIN_COMPLETENESS:
            skipped += 1
            continue

        rep = check_lab_note(note.text)
        integ_scores.append(rep.overall)
        for v in rep.violated:
            violated[v] += 1
        parsed.append({**note.to_json(), "integrity": rep.as_dict()})

    print("=" * 62)
    print(f" OCR 연구노트 수확 {'(DRY-RUN)' if not a.write else ''}")
    print("=" * 62)
    print(f"\n파싱 성공 {len(parsed):,}건 / 제외 {skipped:,}건(구조 미달·읽기 실패)")
    if deid_counts:
        print(f"\n[비식별화] {sum(deid_counts.values()):,}건 치환")
        for k, v in deid_counts.most_common():
            print(f"  {k:12s} {v:,}")
    print(f"[잔존 위험] {dict(residual) if residual else '없음'}")

    if integ_scores:
        avg = sum(integ_scores) / len(integ_scores)
        clean = sum(1 for s in integ_scores if s >= 0.999)
        print(f"\n[기록 무결성] 평균 {avg:.3f} / 완전 준수 {clean:,}건({clean/len(integ_scores)*100:.1f}%)")
        print("  ALCOA+ 속성별 위반 건수:")
        for k, v in violated.most_common():
            print(f"    {k:16s} {v:,}건")
        print("\n  ※ 실제 노트의 위반 분포다. 합성 데이터와 달리 정답 라벨이 없으므로,")
        print("     이 결과는 **검사기가 무엇을 지적하는지**를 보여줄 뿐 정확도가 아니다.")
        print("     정확도를 알려면 표본을 사람이 검수해 정답을 만들어야 한다.")

    if a.write and parsed:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        with (out / "notes.jsonl").open("w", encoding="utf-8") as f:
            for r in parsed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\n→ {out}/notes.jsonl ({len(parsed):,}건, 비식별화 완료)")
    elif parsed:
        print("\n(dry-run — 저장하려면 --write)")


if __name__ == "__main__":
    main()
