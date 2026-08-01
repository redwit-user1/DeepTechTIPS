"""학습 설정 — SFT / DPO / GRPO / NLI 파인튜닝.

A100 2장(HF 접근 가능) 환경 기준 기본값. 순수 dataclass 라 GPU 없이도
설정 검증·직렬화가 가능하다. 실제 학습 실행은 train/{sft,dpo,nli_finetune}.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# 베이스 모델 후보 (모두 Apache-2.0, HF 배포). 사업계획서 p.16 기준.
BASE_MODELS = {
    "exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",   # 한국어 R&D 최적(우선)
    "qwen": "Qwen/Qwen2.5-7B-Instruct",                 # 수학·코드 강점, RLVR 실적
    "gemma": "google/gemma-2-9b-it",                    # 커뮤니티 풍부
}

# NLI 백엔드 후보 (규정위반·출처 KPI 직결)
NLI_MODELS = {
    "deberta-mnli": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "deberta-large": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # 7B 계열 어텐션+MLP 투영 (모델별 명칭은 sft/dpo에서 자동 해석)
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass
class SFTConfig:
    base_model: str = "exaone"
    dataset_path: str = "data/synth/sft.jsonl"
    output_dir: str = "checkpoints/sft"
    epochs: int = 2
    lr: float = 2e-4
    per_device_batch_size: int = 4
    grad_accum: int = 8
    max_seq_len: int = 2048
    bf16: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def model_id(self) -> str:
        return BASE_MODELS.get(self.base_model, self.base_model)


@dataclass
class DPOConfig:
    base_model: str = "exaone"
    sft_adapter: str = "checkpoints/sft"       # SFT LoRA 어댑터 경로
    dataset_path: str = "data/synth/dpo_pairs.jsonl"
    output_dir: str = "checkpoints/dpo"
    epochs: int = 1
    lr: float = 5e-6
    beta: float = 0.1                          # DPO 온도
    per_device_batch_size: int = 2
    grad_accum: int = 8
    max_seq_len: int = 2048
    bf16: bool = True
    # 채택 기준: VCR≥threshold 쌍만 학습(합성 순환오류 방지, 사업계획서 p.20)
    vcr_accept_threshold: float = 0.0          # 0=필터 없음, 0.7 권장(RLAIF)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def model_id(self) -> str:
        return BASE_MODELS.get(self.base_model, self.base_model)


@dataclass
class GRPOConfig:
    base_model: str = "exaone"
    dpo_adapter: str = "checkpoints/dpo"
    output_dir: str = "checkpoints/grpo"
    num_generations: int = 8                   # 질의당 N=8 샘플링
    lr: float = 1e-6
    beta: float = 0.04                         # KL 계수
    kl_rollback_threshold: float = 0.60        # VCR<0.60 시 자동 롤백
    per_device_batch_size: int = 1
    grad_accum: int = 16

    def model_id(self) -> str:
        return BASE_MODELS.get(self.base_model, self.base_model)


@dataclass
class NLIFinetuneConfig:
    base_model: str = "deberta-mnli"
    train_split: str = "train"                 # SciFact split
    output_dir: str = "checkpoints/nli"
    epochs: int = 3
    lr: float = 2e-5
    per_device_batch_size: int = 16

    def model_id(self) -> str:
        return NLI_MODELS.get(self.base_model, self.base_model)


def to_dict(cfg) -> dict:
    return asdict(cfg)
