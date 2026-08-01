"""DPO — VCR 정렬(출처 없는 생성 = 비준수) 학습 (TRL DPOTrainer).

A100 실행:
  python -m compliance_gateway.train.dpo --base exaone \
      --sft-adapter checkpoints/sft --dataset data/synth/dpo_pairs.jsonl \
      --output checkpoints/dpo --vcr-accept 0.7

vcr-accept>0: 합성 순환오류 방지 위해 VCR 통과분만 학습(RLAIF, 사업계획서 p.20).
"""

from __future__ import annotations

import argparse
import tempfile

from compliance_gateway.train.config import DPOConfig
from compliance_gateway.train.data_format import dpo_pairs_to_dpo, write_jsonl


def run(cfg: DPOConfig) -> None:
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    from compliance_gateway.train.loader import apply_lora, load_model_and_tokenizer

    # 채택 기준 적용 후 임시 파일로 기록
    records = dpo_pairs_to_dpo(cfg.dataset_path, vcr_accept_threshold=cfg.vcr_accept_threshold)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    write_jsonl(records, tmp.name)
    print(f"[DPO] accepted pairs: {len(records)} (threshold={cfg.vcr_accept_threshold})")

    model, tokenizer, unsloth_used = load_model_and_tokenizer(
        cfg.model_id(), max_seq_len=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit, use_unsloth=cfg.use_unsloth, bf16=cfg.bf16,
    )
    dataset = load_dataset("json", data_files=tmp.name, split="train")

    model = apply_lora(model, cfg.lora, unsloth_used, max_seq_len=cfg.max_seq_len)
    peft_config = None if unsloth_used else LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules), task_type="CAUSAL_LM",
    )
    args = TRLDPOConfig(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs, beta=cfg.beta,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
        max_length=cfg.max_seq_len, bf16=cfg.bf16, logging_steps=10, save_strategy="epoch",
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[DPO] saved → {cfg.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="exaone")
    ap.add_argument("--sft-adapter", default="checkpoints/sft")
    ap.add_argument("--dataset", default="data/synth/dpo_pairs.jsonl")
    ap.add_argument("--output", default="checkpoints/dpo")
    ap.add_argument("--vcr-accept", type=float, default=0.0)
    a = ap.parse_args()
    run(DPOConfig(base_model=a.base, sft_adapter=a.sft_adapter, dataset_path=a.dataset,
                  output_dir=a.output, vcr_accept_threshold=a.vcr_accept))


if __name__ == "__main__":
    main()
