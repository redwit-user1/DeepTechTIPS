"""DPO — VCR 정렬(출처 없는 생성 = 비준수) 학습 (TRL DPOTrainer).

A100 실행:
  python -m compliance_gateway.train.dpo --base qwen \
      --sft-adapter checkpoints/sft --dataset data/synth/dpo_pairs.jsonl \
      --output checkpoints/dpo --vcr-accept 0.7

vcr-accept>0: 합성 순환오류 방지 위해 VCR 통과분만 학습(RLAIF, 사업계획서 p.20).
"""

from __future__ import annotations

import argparse
import tempfile

from compliance_gateway.train.config import DPOConfig
from compliance_gateway.train.data_format import dpo_pairs_to_dpo, write_jsonl


def run(cfg: DPOConfig, max_steps: int = -1) -> None:
    from pathlib import Path

    from datasets import load_dataset
    from peft import LoraConfig
    from trl import DPOConfig as TRLDPOConfig
    from trl import DPOTrainer

    try:
        # sft.py 와 같은 우회: trl 융합 LM 헤드의 addmm(out_dtype) 를 NGC torch 가 bf16 입력과 함께 거부한다
        import trl.kernels.chunked_logprob as _cl
        _cl._MM_OUT_DTYPE = False
    except ImportError:
        pass

    from compliance_gateway.train.loader import apply_lora, load_model_and_tokenizer
    from compliance_gateway.train.sft import language_only, report_lora

    # 채택 기준 적용 후 임시 파일로 기록
    records = dpo_pairs_to_dpo(cfg.dataset_path, vcr_accept_threshold=cfg.vcr_accept_threshold)
    if not records:
        raise SystemExit(f"거부: 선호쌍이 0건이다 — {cfg.dataset_path}")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    write_jsonl(records, tmp.name)
    print(f"[DPO] accepted pairs: {len(records)} (threshold={cfg.vcr_accept_threshold})")

    model, tokenizer, unsloth_used = load_model_and_tokenizer(
        cfg.model_id(), max_seq_len=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit, use_unsloth=cfg.use_unsloth, bf16=cfg.bf16,
    )
    dataset = load_dataset("json", data_files=tmp.name, split="train")

    peft_config = None
    if cfg.sft_adapter and (Path(cfg.sft_adapter) / "adapter_config.json").exists():
        # SFT 어댑터에서 이어서 정렬한다. TRL 은 사전학습 어댑터가 실린 PeftModel 을 받으면 그 사본("ref")을
        # 기준 정책으로 쓴다 — 기준이 기반모델이 아니라 SFT 모델이 된다.
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.sft_adapter, is_trainable=True)
        print(f"[DPO] SFT 어댑터에서 이어서 학습: {cfg.sft_adapter}")
    else:
        print(f"[DPO] SFT 어댑터 없음({cfg.sft_adapter}) — 기반모델에 새 LoRA")
        model = apply_lora(model, cfg.lora, unsloth_used, max_seq_len=cfg.max_seq_len)
        peft_config = None if unsloth_used else LoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
            target_modules=language_only(cfg.lora.target_modules), task_type="CAUSAL_LM",
            bias=getattr(cfg.lora, "bias", "none"),
        )
    args = TRLDPOConfig(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs, max_steps=max_steps, beta=cfg.beta,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
        max_length=cfg.max_seq_len, bf16=cfg.bf16, logging_steps=5,
        save_strategy="no", report_to="none",
    )
    trainer = DPOTrainer(model=model, args=args, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=peft_config)
    report_lora(trainer.model)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[DPO] saved → {cfg.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="qwen")
    ap.add_argument("--sft-adapter", default="checkpoints/sft")
    ap.add_argument("--dataset", default="data/synth/dpo_pairs.jsonl")
    ap.add_argument("--output", default="checkpoints/dpo")
    ap.add_argument("--vcr-accept", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=-1, help="시험 학습용 스텝 상한(-1 = 에폭 기준)")
    a = ap.parse_args()
    cfg = DPOConfig(base_model=a.base, sft_adapter=a.sft_adapter, dataset_path=a.dataset,
                    output_dir=a.output, vcr_accept_threshold=a.vcr_accept)
    cfg.epochs = a.epochs or cfg.epochs
    cfg.lr = a.lr or cfg.lr
    cfg.per_device_batch_size = a.batch or cfg.per_device_batch_size
    cfg.grad_accum = a.grad_accum or cfg.grad_accum
    run(cfg, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
