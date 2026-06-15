"""SciFact 로더.

SciFact (Wadden et al., EMNLP 2020): 전문가 작성 과학 claim + 근거 abstract.
출처: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz  (CC BY-NC)

SourceMatch 관점으로 변환한다:
  premise   = 인용 출처(근거 문장)
  hypothesis= 응답의 주장(claim)
  label     = SUPPORT(근거가 주장을 뒷받침) / CONTRADICT(근거가 주장을 반박)

SUPPORT 는 높은 entailment, CONTRADICT 는 낮은 점수가 나와야 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_DIR = Path("data/raw/data")


@dataclass
class NLIExample:
    claim: str            # hypothesis (응답 주장)
    evidence: str         # premise (인용 출처 근거)
    label: str            # "SUPPORT" | "CONTRADICT"
    claim_id: int
    doc_id: int


def _load_corpus(corpus_path: Path) -> dict[int, list[str]]:
    corpus: dict[int, list[str]] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            corpus[int(row["doc_id"])] = row["abstract"]
    return corpus


def load_scifact(
    split: str = "train",
    data_dir: Optional[Path] = None,
    max_examples: Optional[int] = None,
) -> list[NLIExample]:
    """근거가 달린(SUPPORT/CONTRADICT) claim 만 NLIExample 로 반환.

    근거 없는(NEI) claim 은 SourceMatch 이분 벤치마크에서 제외한다.
    """
    data_dir = data_dir or DEFAULT_DIR
    corpus = _load_corpus(data_dir / "corpus.jsonl")
    claims_path = data_dir / f"claims_{split}.jsonl"

    examples: list[NLIExample] = []
    with claims_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            evidence = row.get("evidence") or {}
            if not evidence:
                continue
            for doc_id_str, rationales in evidence.items():
                doc_id = int(doc_id_str)
                abstract = corpus.get(doc_id)
                if not abstract:
                    continue
                for rat in rationales:
                    sent_idx = rat.get("sentences", [])
                    label = rat.get("label")
                    if label not in ("SUPPORT", "CONTRADICT"):
                        continue
                    evidence_text = " ".join(
                        abstract[i] for i in sent_idx if 0 <= i < len(abstract)
                    )
                    if not evidence_text.strip():
                        continue
                    examples.append(
                        NLIExample(
                            claim=row["claim"],
                            evidence=evidence_text,
                            label=label,
                            claim_id=int(row["id"]),
                            doc_id=doc_id,
                        )
                    )
                    if max_examples and len(examples) >= max_examples:
                        return examples
    return examples
