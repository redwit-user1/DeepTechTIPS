"""SFT — 도메인 LoRA 지시 미세조정 (TRL SFTTrainer).

A100 실행:
  python -m compliance_gateway.train.sft --base qwen \
      --dataset data/synth/sft.jsonl --output checkpoints/sft

의존성(A100 환경): pip install "compliance-gateway[train]"
"""

from __future__ import annotations

import argparse

from compliance_gateway.train.config import SFTConfig


# Qwen3.5+ 는 비전 통합 모델이라 비전 타워에도 gate/up/down_proj 가 있다 — 이름만 주면 거기에도 LoRA 가 붙는다.
_VISION = ("visual", "vision")


def language_only(names) -> str:
    """PEFT 정규식(전체 일치): 비전 모듈을 뺀 언어 모델 투영층만."""
    return r"^(?!.*(?:" + "|".join(_VISION) + r")).*\.(?:" + "|".join(names) + r")$"


def report_lora(model) -> dict:
    """LoRA 가 붙은 모듈 수를 접미사별로 출력하고, 비전 타워에 붙었으면 멈춘다."""
    from collections import Counter

    hit = [n for n, _ in model.named_modules() if n.endswith("lora_A")]
    bad = [n for n in hit if any(v in n for v in _VISION)]
    by = Counter(n.rsplit(".", 2)[-2] for n in hit)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SFT] LoRA 모듈 {len(hit)} {dict(by)} · 학습 파라미터 {tr:,} / {tot:,} ({tr / tot:.3%})")
    assert not bad, f"비전 타워에 LoRA 부착: {bad[:3]}"
    return dict(by)


def run(cfg: SFTConfig, max_steps: int = -1) -> None:
    # 지연 임포트: torch/transformers/trl/peft 는 A100 환경에서만 필요.
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig as TRLSFTConfig
    from trl import SFTTrainer

    try:
        # ponytail: trl 1.17 융합 LM 헤드의 addmm(out_dtype=fp32) 를 NGC 25.11 torch 가 bf16 입력과 함께 거부한다
        # ("self and mat2 must have the same dtype") → trl 자체의 fp32 폴백 경로로. trl 이 고치면 지운다.
        import trl.kernels.chunked_logprob as _cl
        _cl._MM_OUT_DTYPE = False
    except ImportError:
        pass

    from compliance_gateway.train.loader import apply_lora, load_model_and_tokenizer

    model, tokenizer, unsloth_used = load_model_and_tokenizer(
        cfg.model_id(), max_seq_len=cfg.max_seq_len,
        load_in_4bit=cfg.load_in_4bit, use_unsloth=cfg.use_unsloth, bf16=cfg.bf16,
    )
    dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")
    dataset = dataset.select_columns(["messages"])          # 출처(source) 등 부가 필드는 학습에 안 쓴다

    # Unsloth 경로는 네이티브 PEFT 부착, HF 경로는 TRL 에 peft_config 전달
    model = apply_lora(model, cfg.lora, unsloth_used, max_seq_len=cfg.max_seq_len)
    peft_config = None if unsloth_used else LoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
        target_modules=language_only(cfg.lora.target_modules), task_type="CAUSAL_LM",
        bias=getattr(cfg.lora, "bias", "none"),
    )
    # trl 0.20+ 는 max_seq_length → max_length 로 이름이 바뀌었다
    seq_key = "max_length" if "max_length" in TRLSFTConfig.__dataclass_fields__ else "max_seq_length"
    args = TRLSFTConfig(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs, max_steps=max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.lr,
        bf16=cfg.bf16, logging_steps=5, save_strategy="no" if max_steps > 0 else "epoch",
        report_to="none", **{seq_key: cfg.max_seq_len},
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                         peft_config=peft_config, processing_class=tokenizer)
    report_lora(trainer.model)
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[SFT] saved → {cfg.output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="qwen")
    ap.add_argument("--dataset", default="data/synth/sft.jsonl")
    ap.add_argument("--output", default="checkpoints/sft")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="시험 학습용 스텝 상한(-1 = 에폭 기준)")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    a = ap.parse_args()
    cfg = SFTConfig(base_model=a.base, dataset_path=a.dataset, output_dir=a.output, epochs=a.epochs)
    cfg.per_device_batch_size = a.batch or cfg.per_device_batch_size
    cfg.grad_accum = a.grad_accum or cfg.grad_accum
    run(cfg, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
