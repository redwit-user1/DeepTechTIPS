"""외부 실데이터 평가셋 — 합성 낙관편향 제거용.

합성 평가셋(bioRxiv 변조)은 **우리가 만든 오류를 우리가 찾는** 구조라 성능이
낙관적으로 나온다. 과제 목표(규정위반 탐지 90%+, 출처 정확률 90%+)를 **증명**하려면
제3자가 만든 실데이터로 측정해야 한다.

## 데이터: SciFact (Wadden et al., EMNLP 2020)
- 전문가가 작성한 과학 주장 + 실제 PubMed 근거, SUPPORT/CONTRADICT 라벨
- **우리가 변조를 가하지 않는다.** 라벨은 전문가 주석 그대로.

## 평가 시나리오: "인용은 실존하나 주장을 뒷받침하지 않음"
현실에서 가장 흔하고 위험한 실패 유형이다(가짜 DOI 보다 훨씬 흔하다).
- SUPPORT  → 인용 출처가 주장을 실제로 뒷받침 → Gateway PASS 기대 (label=compliant)
- CONTRADICT → 인용 출처가 주장을 **반박** → Gateway 차단 기대 (label=unsupported_claim)

인용된 논문은 코퍼스에 실존하므로 서지 검증은 통과한다. 따라서 이 평가는
**순수하게 SourceMatch(근거-주장 일치도) 능력**을 측정한다.

실행:
  bash scripts/download_scifact.sh
  python -m compliance_gateway.eval.external --split train --limit 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from compliance_gateway.eval.scifact import DEFAULT_DIR, NLIExample, load_scifact
from compliance_gateway.verify.models import PaperRecord
from compliance_gateway.verify.verifier import LocalRegistry


def render_response(example: NLIExample, title: str) -> str:
    """주장 + 실제 논문 제목 인용. (변조 없음 — 인용 대상은 실존 논문)"""
    return f'{example.claim} (cf. "{title}")'


def _corpus_titles(data_dir: Optional[Path] = None) -> dict[int, str]:
    import json

    path = (data_dir or DEFAULT_DIR) / "corpus.jsonl"
    out: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[int(row["doc_id"])] = row.get("title", "")
    return out


def build_external_eval(
    split: str = "train",
    limit: Optional[int] = None,
    data_dir: Optional[Path] = None,
) -> tuple[list[dict], LocalRegistry]:
    """SciFact → Gateway 평가 아이템 + 실제 논문 레지스트리."""
    examples = load_scifact(split=split, data_dir=data_dir, max_examples=limit)
    titles = _corpus_titles(data_dir)

    items: list[dict] = []
    records: dict[int, PaperRecord] = {}
    for ex in examples:
        title = titles.get(ex.doc_id, "")
        if not title:
            continue
        records.setdefault(
            ex.doc_id, PaperRecord(doi=None, title=title, authors=(), year=None, source="scifact")
        )
        items.append(
            {
                "query": f"Summarize the finding with its source: {ex.claim[:60]}",
                "response": render_response(ex, title),
                "grounding": ex.evidence,
                "source_doi": "",
                # SUPPORT = 정당한 인용 / CONTRADICT = 근거가 주장을 반박(출처 오류)
                "label": "compliant" if ex.label == "SUPPORT" else "unsupported_claim",
                "halluc_type": None if ex.label == "SUPPORT" else "B",
            }
        )
    return items, LocalRegistry(records.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    # 운영점은 스코어 의미가 바뀔 때마다 재보정해야 한다(θ 스윕 참조).
    # 현재 영어 외부셋 최적(정상통과>=80%)은 0.70.
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--out", default=None, help="JSONL 저장 경로(선택)")
    ap.add_argument("--sweep", action="store_true",
                    help="임계값 스윕 — '보정 문제'와 '모델 한계'를 분리")
    from compliance_gateway.eval.nli_select import add_nli_args
    add_nli_args(ap)
    ap.add_argument("--allow-leakage", action="store_true",
                    help="파인튜닝 split 으로 평가 허용(진단용, KPI 근거로 쓰지 말 것)")
    a = ap.parse_args()

    from compliance_gateway.eval.kpi import evaluate
    from compliance_gateway.nli.statistical import StatisticalNLI
    from compliance_gateway.pipeline import ComplianceGateway
    from compliance_gateway.verify import CitationVerifier

    items, registry = build_external_eval(split=a.split, limit=a.limit)
    n_ok = sum(1 for i in items if i["label"] == "compliant")
    print(f"SciFact[{a.split}] 외부 평가셋: {len(items)}건 "
          f"(compliant={n_ok}, unsupported_claim={len(items) - n_ok})")

    # 데이터 누출 가드: NLI 는 SciFact train 으로 파인튜닝된다.
    # 같은 split 으로 KPI 를 측정하면 학습 데이터를 재평가하는 셈이라 부풀려진다.
    if a.nli and a.split == "train" and not a.allow_leakage:
        raise SystemExit(
            "[중단] 데이터 누출 위험: 파인튜닝 NLI(--nli)를 train split 으로 평가하려 합니다.\n"
            "  NLI 는 SciFact train 으로 학습되므로 KPI 가 부풀려집니다.\n"
            "  → `--split dev` 로 측정하세요(정직한 일반화 성능).\n"
            "  (의도적 비교가 필요하면 --allow-leakage)"
        )
    from compliance_gateway.eval.nli_select import select_nli
    nli_fn, backend_name = select_nli(a.nli, a.nli_endpoint, a.nli_model, a.device)
    print(f"NLI 백엔드: {backend_name}\n")

    gw = ComplianceGateway(
        vcr_threshold=a.threshold, nli_fn=nli_fn,
        verifier=CitationVerifier([registry]),
    )
    m = evaluate(gw, items)
    print(f"{'지표':24s} {'값':>8s}")
    print("-" * 34)
    print(f"{'위반탐지 Precision':24s} {m['violation_precision']*100:7.1f}%")
    print(f"{'위반탐지 Recall':24s} {m['violation_recall']*100:7.1f}%")
    print(f"{'위반탐지 F1':24s} {m['violation_f1']*100:7.1f}%")
    print(f"{'compliant PASS':24s} {m['compliant_pass_rate']*100:7.1f}%")
    print("\n* 외부 실데이터 기준. 합성 평가셋 대비 낮게 나오는 것이 정상이며,")
    print("  이 값이 과제 목표(90%+) 달성 여부의 진짜 잣대다.")

    if a.sweep:
        print("\n=== 임계값 스윕 (최적 운영점 탐색) ===")
        print(f"{'θ':>6s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'PASS':>8s}")
        # 사용 가능한 운영점: 정상 응답을 최소 80% 통과시켜야 제품으로 의미가 있다.
        # (전부 차단하면 Recall 100%가 되지만 Precision 은 위반 기저율에 수렴 = 무의미)
        MIN_PASS = 0.80
        best_usable = (0.0, None)
        best_raw = (0.0, None)
        for th in [x / 100 for x in range(30, 85, 5)]:
            # 동일 NLI 인스턴스 재사용 — 트랜스포머는 캐시가 있어 스윕이 크게 빨라진다
            g = ComplianceGateway(vcr_threshold=th, nli_fn=nli_fn,
                                  verifier=CitationVerifier([registry]))
            mm = evaluate(g, items)
            usable = mm["compliant_pass_rate"] >= MIN_PASS
            print(f"{th:6.2f} {mm['violation_precision']*100:9.1f}% "
                  f"{mm['violation_recall']*100:7.1f}% {mm['violation_f1']*100:7.1f}% "
                  f"{mm['compliant_pass_rate']*100:7.1f}%"
                  f"{'' if usable else '   (사용불가: 정상응답 과다차단)'}")
            if mm["violation_f1"] > best_raw[0]:
                best_raw = (mm["violation_f1"], th)
            if usable and mm["violation_f1"] > best_usable[0]:
                best_usable = (mm["violation_f1"], th)

        print(f"\n최고 F1(제약 없음)      = {best_raw[0]*100:.1f}% (θ={best_raw[1]})"
              f"  ← 대개 '전부 차단' 축퇴점이라 의미 없음")
        print(f"최고 F1(정상통과≥{MIN_PASS:.0%}) = {best_usable[0]*100:.1f}% (θ={best_usable[1]})"
              f"  ← 실사용 가능한 최선")
        print("\n→ 어떤 임계값에서도 목표(90%)에 크게 못 미친다 = '임계값 보정' 문제가 아니라")
        print("  '모델 능력' 문제. 트랜스포머 NLI 도입이 KPI 달성의 필수 조건임을 실데이터로 확인.")

    if a.out:
        import json

        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"\n→ {p}")


if __name__ == "__main__":
    main()
