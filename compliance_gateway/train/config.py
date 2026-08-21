"""학습 설정 — SFT / DPO / GRPO / NLI 파인튜닝.

**H100 80GB × 2** (KT Cloud AI Nexus) 기준 기본값. 순수 dataclass 라 GPU 없이도
설정 검증·직렬화가 가능하다. 실제 학습 실행은 train/{sft,dpo,grpo,nli_finetune}.py.

A100 대비 H100 변경점:
  - **FP8 지원**(Hopper Transformer Engine) → RL(GRPO) 메모리·속도 이득
  - 대역폭·연산 향상 → 배치/시퀀스 확대 여지
  - 7B LoRA 는 4-bit 불필요(80GB 여유). 4-bit 는 온프레미스 배포 목표일 뿐 학습 요건 아님
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# 베이스 모델 후보 (모두 Apache-2.0, HF 배포). 사업계획서 p.16 기준.
BASE_MODELS = {
    "exaone": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",   # 한국어 R&D 최적(우선)
    "qwen": "Qwen/Qwen2.5-7B-Instruct",                 # 수학·코드 강점, RLVR 실적
    "gemma": "google/gemma-2-9b-it",                    # 커뮤니티 풍부
}

# GPU 프로파일 — 환경에 맞춰 배치·정밀도 기본값을 잡는다.
GPU_PROFILES = {
    "h100": {"bf16": True, "fp8_capable": True, "sft_batch": 8, "dpo_batch": 4, "max_seq": 4096},
    "a100": {"bf16": True, "fp8_capable": False, "sft_batch": 4, "dpo_batch": 2, "max_seq": 2048},
}
DEFAULT_GPU = "h100"

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
    gpu: str = DEFAULT_GPU
    epochs: int = 2
    lr: float = 2e-4
    per_device_batch_size: int = 8      # H100 80GB 기준(A100 은 4)
    grad_accum: int = 4
    max_seq_len: int = 4096             # H100 여유 메모리 활용
    bf16: bool = True
    # Unsloth 백엔드: 학습 가속 + VRAM 절감(장문맥 3배 빠름/30% 절감).
    # 멀티GPU 는 torchrun/accelerate DDP 로 구동. → docs/UPSTREAM_TECH.md
    use_unsloth: bool = True
    load_in_4bit: bool = False                 # QLoRA(A100 80GB면 불필요)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def model_id(self) -> str:
        return BASE_MODELS.get(self.base_model, self.base_model)


@dataclass
class DPOConfig:
    base_model: str = "exaone"
    sft_adapter: str = "checkpoints/sft"       # SFT LoRA 어댑터 경로
    dataset_path: str = "data/synth/dpo_pairs.jsonl"
    output_dir: str = "checkpoints/dpo"
    gpu: str = DEFAULT_GPU
    epochs: int = 1
    lr: float = 5e-6
    beta: float = 0.1                          # DPO 온도
    per_device_batch_size: int = 4             # H100 80GB 기준(A100 은 2)
    grad_accum: int = 4
    max_seq_len: int = 4096
    bf16: bool = True
    use_unsloth: bool = True
    load_in_4bit: bool = False
    # 채택 기준: VCR≥threshold 쌍만 학습(합성 순환오류 방지, 사업계획서 p.20)
    vcr_accept_threshold: float = 0.0          # 0=필터 없음, 0.7 권장(RLAIF)
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def model_id(self) -> str:
        return BASE_MODELS.get(self.base_model, self.base_model)


@dataclass
class GRPOConfig:
    """GRPO(RLVR) 설정.

    기본값은 2026 커뮤니티 권장치(num_generations=8, temperature=0.8, beta=0.04)를
    따른다 — 사업계획서의 'N=8 샘플링'과도 일치. → docs/UPSTREAM_TECH.md
    """

    base_model: str = "exaone"
    dpo_adapter: str = "checkpoints/dpo"
    output_dir: str = "checkpoints/grpo"
    num_generations: int = 8                   # 질의당 N=8 (안정적 어드밴티지 최소값)
    temperature: float = 0.8
    lr: float = 1e-6
    beta: float = 0.04                         # KL 계수
    kl_rollback_threshold: float = 0.60        # VCR<0.60 시 자동 롤백
    gpu: str = DEFAULT_GPU
    # FP8: Hopper(H100+) 전용. RL 은 생성 비중이 커 FP8 이득이 크다.
    # A100 에서는 반드시 False(미지원 하드웨어).
    fp8: bool = True
    per_device_batch_size: int = 2             # H100 기준(A100 은 1)
    grad_accum: int = 8
    max_prompt_len: int = 2048
    max_completion_len: int = 1024
    # 보상 스케일: VCR 은 [0,1]. RLVR 권장은 [-1,1] → 선택적 재스케일.
    reward_rescale_to_signed: bool = True
    # vLLM 생성 가속 (A100 2장: server 모드로 생성/학습 GPU 분리 권장)
    use_vllm: bool = True
    vllm_mode: str = "server"                  # server | colocate
    vllm_gpu_memory_utilization: float = 0.30  # colocate 시 생성에 할당할 비율

    def model_id(self) -> str:
        return BASE_MODELS.get(self.base_model, self.base_model)


@dataclass
class NLIFinetuneConfig:
    base_model: str = "deberta-mnli"
    train_split: str = "train"                 # SciFact split
    eval_split: str = "dev"                    # 학습 검증(빈 문자열이면 비활성)
    output_dir: str = "checkpoints/nli"
    epochs: int = 3
    lr: float = 2e-5
    per_device_batch_size: int = 16

    def model_id(self) -> str:
        return NLI_MODELS.get(self.base_model, self.base_model)


def to_dict(cfg) -> dict:
    return asdict(cfg)


def validate(cfg) -> list[str]:
    """설정과 GPU 프로파일의 정합성 점검. 경고 목록을 반환한다."""
    warnings: list[str] = []
    profile = GPU_PROFILES.get(getattr(cfg, "gpu", DEFAULT_GPU))
    if profile is None:
        return [f"알 수 없는 GPU 프로파일: {getattr(cfg, 'gpu', None)}"]
    if getattr(cfg, "fp8", False) and not profile["fp8_capable"]:
        warnings.append(
            f"fp8=True 이지만 {cfg.gpu} 는 FP8 미지원 하드웨어입니다 → fp8=False 로 두세요."
        )
    if getattr(cfg, "load_in_4bit", False) and profile["fp8_capable"]:
        warnings.append(
            "H100 80GB 에서 7B 학습에 4-bit 는 불필요합니다(정확도만 손해). "
            "4-bit 는 온프레미스 배포 목표(M5)에서 사용하세요."
        )
    return warnings
