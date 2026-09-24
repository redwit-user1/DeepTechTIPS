"""NLI 파인튜닝 — SciFact 로 규정위반·출처 판정 정밀도 확보 (M1 최우선).

외부 실데이터 평가에서 통계 NLI 는 어떤 임계값에서도 Precision 41% 상한이었다
(docs/EVAL_EXTERNAL.md). 트랜스포머 NLI 도입이 KPI 달성의 필수 조건이며,
본 스크립트가 그 첫 단계다.

학습된 모델 주입:
    TransformerNLI(model_name="checkpoints/nli")  →  ComplianceGateway(nli_fn=...)

A100 실행:
  python -m compliance_gateway.train.nli_finetune --base deberta-mnli --output checkpoints/nli
  python -m compliance_gateway.eval.external --split dev --nli checkpoints/nli   # 재측정

GPU 없이 데이터·설정 검증:
  python -m compliance_gateway.train.nli_finetune --dry-run
"""

from __future__ import annotations

import argparse
from typing import Optional

from compliance_gateway.train.config import NLIFinetuneConfig

# SciFact 라벨 → 학습 라벨. id2label 을 반드시 저장해야 추론 시
# TransformerNLI 가 entailment 확률을 찾을 수 있다(미설정 시 LABEL_0/LABEL_1 로 저장됨).
LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1}
ID2LABEL = {0: "entailment", 1: "contradiction"}


def prepare_examples(split: str, data_dir=None) -> list:
    """SciFact → (premise=근거, hypothesis=주장, label) 목록."""
    from compliance_gateway.eval.scifact import load_scifact

    return load_scifact(split=split, data_dir=data_dir)


def dry_run(cfg: NLIFinetuneConfig) -> dict:
    """GPU 없이 데이터·설정을 검증한다(A100 에서 첫 실행 실패 방지)."""
    train = prepare_examples(cfg.train_split)
    dev = prepare_examples(cfg.eval_split) if cfg.eval_split else []
    counts = {"SUPPORT": 0, "CONTRADICT": 0}
    for e in train:
        counts[e.label] = counts.get(e.label, 0) + 1
    report = {
        "model_id": cfg.model_id(),
        "train_examples": len(train),
        "eval_examples": len(dev),
        "label_distribution": counts,
        "id2label": ID2LABEL,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "batch_size": cfg.per_device_batch_size,
    }
    assert train, "학습 예시가 비어 있음 — bash scripts/download_scifact.sh 먼저 실행"
    assert set(counts) <= set(LABEL2ID), f"예상치 못한 라벨: {set(counts)}"
    return report


def run(cfg: NLIFinetuneConfig) -> None:
    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(cfg.model_id())
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id(),
        num_labels=2,
        id2label=ID2LABEL,              # ← 추론 시 'entailment' 라벨 조회에 필수
        label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True,   # MNLI(3-class) → 2-class 헤드 교체 허용
    )

    def to_dataset(split: str) -> Optional[Dataset]:
        examples = prepare_examples(split)
        if not examples:
            return None
        ds = Dataset.from_dict({
            "premise": [e.evidence for e in examples],
            "hypothesis": [e.claim for e in examples],
            "labels": [LABEL2ID[e.label] for e in examples],
        })
        return ds.map(
            lambda b: tok(b["premise"], b["hypothesis"], truncation=True, max_length=512),
            batched=True, remove_columns=["premise", "hypothesis"],
        )

    train_ds = to_dataset(cfg.train_split)
    eval_ds = to_dataset(cfg.eval_split) if cfg.eval_split else None

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        acc = float((preds == labels).mean())
        # SUPPORT(0) 를 positive 로 본 F1
        tp = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 0) & (labels != 0)).sum())
        fn = int(((preds != 0) & (labels == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return {
            "accuracy": acc,
            "f1_support": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        }

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size * 2,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="accuracy",
        logging_steps=20,
        warmup_ratio=0.1,
        bf16=True,                       # A100
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tok,            # 최신 transformers (tokenizer= 는 deprecated)
        data_collator=DataCollatorWithPadding(tok),   # ← 가변 길이 배치에 필수
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )
    trainer.train()
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    if eval_ds is not None:
        print("[NLI] eval:", trainer.evaluate())
    print(f"[NLI] saved → {cfg.output_dir}")
    print(f"[NLI] 재측정: python -m compliance_gateway.eval.external "
          f"--split dev --nli {cfg.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deberta-mnli")
    ap.add_argument("--output", default="checkpoints/nli")
    ap.add_argument("--split", default="train")
    ap.add_argument("--eval-split", default="dev")
    ap.add_argument("--dry-run", action="store_true", help="GPU 없이 데이터·설정 검증")
    a = ap.parse_args()
    cfg = NLIFinetuneConfig(base_model=a.base, output_dir=a.output,
                            train_split=a.split, eval_split=a.eval_split)
    if a.dry_run:
        report = dry_run(cfg)
        print("=== dry-run 검증 통과 ===")
        for k, v in report.items():
            print(f"  {k}: {v}")
        return
    run(cfg)


if __name__ == "__main__":
    main()
