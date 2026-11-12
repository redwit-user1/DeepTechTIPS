"""대량 수집 — KCI(OAI-PMH) / Europe PMC(한국 소속 OA) / 식약처 민원인안내서.

    python scripts/harvest.py kci   --out DIR [--start 2000-01] [--end 2026-11]
    python scripts/harvest.py epmc  --out DIR --query '...' [--fulltext]
    python scripts/harvest.py mfds  --out DIR

전부 **이어받기** 가능 — 세션이 회수돼도 같은 명령을 다시 돌리면 끝난 곳부터 계속한다.
  kci : 월별 파일(YYYY-MM.jsonl). 끝난 달은 건너뛴다.
  epmc: cursor 파일. 전문 XML 은 파일이 있으면 건너뛴다.
  mfds: index.jsonl 에 있는 게시글은 건너뛴다.
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
import urllib.error
import urllib.parse
from datetime import date
from pathlib import Path

from compliance_gateway.data.korean.sources import KCIOAIAdapter, MFDSGuideAdapter, _get, _text, polite_get

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
COMMERCIAL_OK = {"cc by", "cc0"}   # 전문은 상업 이용 가능한 라이선스만 받는다


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def retry(fn, *a, tries: int = 5, **kw):
    """네트워크 오류·5xx·429 만 백오프 재시도. 4xx·robots 금지(RuntimeError)는 즉시 올린다."""
    for i in range(tries):
        try:
            return fn(*a, **kw)
        except urllib.error.HTTPError as e:
            if (e.code < 500 and e.code != 429) or i == tries - 1:
                raise
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if i == tries - 1:
                raise
            err = e
        log(f"재시도 {i + 1}/{tries}: {err}")
        time.sleep(10 * 2 ** i)


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}", f"{y:04d}-{m:02d}-01", f"{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}"
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def kci(out: Path, start: str, end: str, delay: float = 1.0) -> None:
    a = KCIOAIAdapter()
    out.mkdir(parents=True, exist_ok=True)
    for ym, frm, until in months(start, end):
        done = out / f"{ym}.jsonl"
        if done.exists():
            continue
        part = out / f"{ym}.jsonl.part"
        n, token = 0, ""
        with part.open("w", encoding="utf-8") as f:   # 달 단위 원자성: 중간에 죽으면 그 달만 다시
            while True:
                recs, token = a.parse(retry(polite_get, a.url(frm, until, token), timeout=60))
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += len(recs)
                time.sleep(delay)
                if not token:
                    break
        part.rename(done)
        log(f"kci {ym}: {n}")


def _slim(r: dict) -> dict:
    j = r.get("journalInfo") or {}
    return {
        "id": r.get("pmcid") or r.get("id"), "pmid": r.get("pmid", ""), "doi": r.get("doi", ""),
        "title": _text(r.get("title", "")), "abstract": _text(r.get("abstractText", "")),
        "authors": r.get("authorString", ""), "affiliation": r.get("affiliation", ""),
        "journal": (j.get("journal") or {}).get("title", ""), "year": r.get("pubYear", ""),
        "license": r.get("license", ""), "language": r.get("language", ""),
        "keywords": (r.get("keywordList") or {}).get("keyword", []),
    }


def epmc(out: Path, query: str, fulltext: bool, delay: float = 0.5) -> None:
    out.mkdir(parents=True, exist_ok=True)
    xml_dir = out / "xml"
    xml_dir.mkdir(exist_ok=True)
    cur_f = out / "cursor"
    cursor = cur_f.read_text().strip() if cur_f.exists() else "*"
    if cursor == "DONE":
        log("epmc: 이미 완료")
        return
    total = 0
    while True:
        q = {"query": query, "format": "json", "resultType": "core", "pageSize": "1000", "cursorMark": cursor}
        d = json.loads(retry(_get, f"{EPMC}/search?{urllib.parse.urlencode(q)}", timeout=120))
        rows = d.get("resultList", {}).get("result", [])
        # ponytail: 페이지 처리 후 cursor 를 저장 — 그 사이 죽으면 마지막 페이지가 meta 에 중복된다. 읽을 때 id 로 dedupe
        with (out / "meta.jsonl").open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(_slim(r), ensure_ascii=False) + "\n")
        got = 0
        for r in rows if fulltext else ():
            pmcid = r.get("pmcid")
            dst = xml_dir / f"{pmcid}.xml"
            if not pmcid or r.get("license") not in COMMERCIAL_OK or dst.exists():
                continue
            try:
                body = retry(_get, f"{EPMC}/{pmcid}/fullTextXML", timeout=120, binary=True)
            except Exception as e:   # 전문 없는 개별 건(404 등)은 건너뛴다
                log(f"전문 실패 {pmcid}: {e}")
                continue
            tmp = dst.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.rename(dst)
            got += 1
            time.sleep(delay)
        total += len(rows)
        nxt = d.get("nextCursorMark")
        log(f"epmc +{len(rows)} (누적 {total}/{d.get('hitCount')}) 전문 +{got}")
        if not rows or not nxt or nxt == cursor:
            cur_f.write_text("DONE")
            break
        cursor = nxt
        cur_f.write_text(cursor)
        time.sleep(delay)


def mfds(out: Path, delay: float = 2.0) -> None:
    a = MFDSGuideAdapter()
    files = out / "files"
    files.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    seen = {json.loads(l)["seq"] for l in index.open(encoding="utf-8")} if index.exists() else set()

    seqs: list[str] = []
    page = 1
    while True:   # 목록 전체를 먼저 훑는다(페이지당 10건)
        new = [s for s in a.parse_list(retry(polite_get, a.list_url(page))) if s not in seqs]
        time.sleep(delay)
        if not new:
            break
        seqs += new
        page += 1
    log(f"mfds 목록 {len(seqs)}건 (기존 {len(seen)})")

    for seq in seqs:
        if seq in seen:
            continue
        v = a.parse_view(retry(polite_get, a.view_url(seq)))
        time.sleep(delay)
        for f in v["files"]:
            ext = Path(f["name"]).suffix.lower() or ".bin"
            dst = files / f"{seq}_{f['file_seq']}{ext}"
            if not dst.exists():
                dst.write_bytes(retry(polite_get, a.down_url(seq, f["file_seq"]), timeout=120, binary=True))
                time.sleep(delay)
            f["path"] = str(dst.relative_to(out))
        with index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"seq": seq, **v, "url": a.view_url(seq)}, ensure_ascii=False) + "\n")
        log(f"mfds {seq}: {v['title'][:40]} ({len(v['files'])}파일)")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kci")
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--start", default="2000-01")
    k.add_argument("--end", default=date.today().strftime("%Y-%m"))
    e = sub.add_parser("epmc")
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--query", required=True)
    e.add_argument("--fulltext", action="store_true")
    m = sub.add_parser("mfds")
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    if a.cmd == "kci":
        kci(a.out, a.start, a.end)
    elif a.cmd == "epmc":
        epmc(a.out, a.query, a.fulltext)
    else:
        mfds(a.out)
    log("완료")


if __name__ == "__main__":
    sys.exit(main())
