"""NLI 파인튜닝 — SciFact 로 규정위반·출처 판정 정밀도 확보.

통계 NLI(AUC 0.57)를 트랜스포머로 교체하는 M1 핵심 작업.
학습된 모델은 compliance_gateway.nli.transformer.TransformerNLI(model_name=출력경로)로
Gateway 에 주입 → SourceMatch/Halluc 즉시 업그레이드.

A100 실행:
  python -m compliance_gateway.train.nli_finetune --base deberta-mnli --output checkpoints/nli
"""

from __future__ import annotations

import argparse

from compliance_gateway.train.config import NLIFinetuneConfig


def run(cfg: NLIFinetuneConfig) -> None:
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from compliance_gateway.eval.scifact import load_scifact

    # SciFact → (premise=evidence, hypothesis=claim, label: 0=entail(SUPPORT) / 1=contradict)
    train = load_scifact(split=cfg.train_split)
    label2id = {"SUPPORT": 0, "CONTRADICT": 1}

    tok = AutoTokenizer.from_pretrained(cfg.model_id())
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_id(), num_labels=2)

    def encode(ex):
        enc = tok(ex.evidence, ex.claim, truncation=True, max_length=512)
        enc["labels"] = label2id[ex.label]
        return enc

    ds = [encode(e) for e in train]
    args = TrainingArguments(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs, learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.per_device_batch_size, logging_steps=20, save_strategy="epoch",
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tok)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[NLI] saved → {cfg.output_dir}  (주입: TransformerNLI(model_name='{cfg.output_dir}'))")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deberta-mnli")
    ap.add_argument("--output", default="checkpoints/nli")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    run(NLIFinetuneConfig(base_model=a.base, output_dir=a.output, train_split=a.split))


if __name__ == "__main__":
    main()
