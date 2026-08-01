# 업스트림 기술 동향 조사 및 적용 (2026)

A100 2장 확보 시점에 최신 GitHub/Unsloth/TRL 동향을 조사해 본 과제에 적용한 기록.
**채택한 것 / 채택하지 않은 것 / 적용 결과(측정치)** 를 함께 남긴다.

---

## 1. Citation-Hallucination-Detection (3단계 인용 검증) — ✅ 채택, 최대 성과

**출처**: [Vikranth3140/Citation-Hallucination-Detection](https://github.com/Vikranth3140/Citation-Hallucination-Detection)

### 무엇을 배웠나
- **3단계 파이프라인**: exact lookup → fuzzy retrieval(BM25) → LLM verification
- **가중 메타데이터 스코어링**: title 60% + author 30% + year 10%, 임계값 ≥0.92
- **3-class 판정**: `valid` / `partially_valid` / `hallucinated`
- **서지 DB**: CrossRef(130M, 50 req/s), OpenAlex(250M, 100K/day), Semantic Scholar(200M, 100 req/s)

### 왜 우리에게 중요한가
기존 우리 구현은 `DOIResolver = Callable[[str], bool]` — **존재/부재 이분법**이었다.
이 구조는 치명적 사각지대가 있다: **"실존 DOI + 변조된 저자"** 는 DOI 가 실제로 존재하므로
그대로 통과한다. 이것은 ALCOA+ 의 **`Attributable`(귀속가능) 직접 위반**이며,
잘못된 연구자에게 성과를 귀속시키는 실제 연구부정 시나리오다.

### 적용
- 신규 모듈 `compliance_gateway/verify/`
  - `scoring.py` — 업스트림 가중치·임계값 채택(표준 라이브러리 `difflib` 만 사용, 무의존성)
  - `verifier.py` — 3단계 파이프라인 + `LocalRegistry`(오프라인/에어갭) + LLM 어댑터 훅
  - `backends.py` — CrossRef/OpenAlex 온라인 백엔드(운영·A100 환경용)
- `Halluc` 이 3-class 를 **가중 감점**으로 반영: VALID 0.0 / PARTIALLY_VALID 0.5 / HALLUCINATED 1.0
  - `UNVERIFIED`(조회 실패)는 감점하지 않음 — 네트워크 문제로 정상 인용을 벌하지 않기 위해
- **부수 버그 수정**: `extract_citations` 가 저자-연도와 DOI 를 별개 인용으로 분리하던 문제.
  같은 참고문헌 조각을 병합하지 않으면 대조할 메타데이터가 없어 **무조건 통과**했다.
  → 인접 조각 병합(`_MERGE_WINDOW`) 구현.
- 합성 파이프라인에 신규 변조 유형 2종 추가: `biblio_tamper`(저자 변조), `year_drift`(연도 변조)

### 측정 결과 (합성 평가셋 n=68)

| 유형 | 기존 binary resolver | 3-class verifier |
|---|---|---|
| biblio_tamper (실존 DOI + 변조 저자) | 9.1% | **100.0%** |
| year_drift (실존 DOI + 변조 연도) | 9.1% | **100.0%** |
| fake_doi / no_source / numeric_tamper | 100% | 100% |
| polarity_flip | 83.3% | 83.3% |
| **종합 Recall** | 0.632 | **0.982** |
| **종합 F1** | 0.766 | **0.982** |
| compliant PASS(오탐 역지표) | 90.9% | 90.9% (유지) |

> 오탐 증가 없이 재현율만 올랐다 — 규정위반 탐지 KPI(90%+)에 직접 기여.

---

## 2. Unsloth — ✅ 채택 (학습 가속·VRAM)

**출처**: [unsloth 문서/체인지로그](https://unsloth.ai/docs/new/changelog), [멀티GPU 문서](https://docs.unsloth.ai/basics/unsloth-multi-gpu-support)

### 무엇을 배웠나
- LoRA/QLoRA 학습 가속, **GRPO 1.3배 빠름**, **RL VRAM 대폭 절감**
- 장문맥 학습 3배 속도 / 30% 메모리 절감, FP8 RL 지원
- **멀티GPU**: 현재는 accelerate/DeepSpeed(DDP/FSDP) 경유 — `torchrun --nproc_per_node 2`

### 적용
- `train/loader.py` — **Unsloth 우선, 실패 시 HF 자동 폴백**
  (Unsloth 는 API 변화가 빠르므로 폴백 필수. 어느 경로인지 항상 로그 출력)
- `SFTConfig/DPOConfig` 에 `use_unsloth`, `load_in_4bit` 추가
- `use_gradient_checkpointing="unsloth"` 로 장문맥 메모리 절감

### 주의
A100 80GB 2장이면 7B LoRA 에 4-bit 는 불필요(`load_in_4bit=False` 기본).
4-bit 는 M5 온프레미스 양자화(GPTQ/AWQ) 단계의 배포용 목표이지 학습 요건이 아니다.

---

## 3. TRL GRPO + vLLM (RLVR) — ✅ 채택

**출처**: [vLLM TRL 문서](https://docs.vllm.ai/en/latest/training/trl/), [awesome-RLVR](https://github.com/opendilab/awesome-RLVR)

### 무엇을 배웠나
- **vLLM 2가지 모드**: `server`(전용 GPU 분리) / `colocate`(같은 GPU 공유)
- 2026 권장 하이퍼파라미터: `num_generations=8`(안정적 어드밴티지 최소값),
  `temperature=0.8`, `beta=0.04`(KL)
- 보상함수는 **스칼라 [-1, 1]** 반환 권장
- RLVR 정의: 학습된 보상모델이 아닌 **결정론적·검증가능** 보상

### 적용
- `train/grpo.py` 신규 — **VCR 을 보상함수로 사용**(`make_vcr_reward_fn`)
  - VCR 은 출처 존재/일치·ALCOA+·환각이 모두 기계 검증 가능 → **RLVR 요건 충족**
  - `reward_rescale_to_signed`: VCR[0,1] → [-1,1] 재스케일 옵션
- **A100 2장 최적 구성**: `vllm_mode="server"` — GPU0 생성(vLLM) / GPU1 학습으로 간섭 제거
- 권장 하이퍼파라미터를 기본값으로 반영 (`num_generations=8`, `temperature=0.8`, `beta=0.04`)
  → 사업계획서의 "질의당 N=8 샘플링" 과도 일치

---

## 4. 조사했으나 채택하지 않은 것

| 항목 | 판단 |
|---|---|
| FP8 GRPO | A100 은 FP8 미지원(H100+ 필요). 해당 없음 |
| MoE 파인튜닝 가속 | 본 과제 베이스는 dense 7B. 불필요 |
| verl / 대규모 RL 프레임워크 | 2 GPU 규모엔 과함. TRL+vLLM 로 충분 |
| Semantic Scholar 백엔드 | CrossRef+OpenAlex 로 커버리지 충분, API 키 관리 비용 회피 |

---

## 5. A100 환경 실행 요약

```bash
pip install -e ".[train]"
pip install unsloth vllm            # 가속(선택, 실패해도 HF 폴백)

# 멀티GPU SFT (DDP)
torchrun --nproc_per_node 2 -m compliance_gateway.train.sft --base exaone

# GRPO (2 GPU 분리: 생성/학습)
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
CUDA_VISIBLE_DEVICES=1 python -m compliance_gateway.train.grpo --vllm-mode server

# 온라인 서지 검증 활성화(운영)
python - <<'PY'
from compliance_gateway.verify import CitationVerifier, LocalRegistry
from compliance_gateway.verify.backends import CrossRefBackend, OpenAlexBackend
v = CitationVerifier([LocalRegistry.from_seed(), CrossRefBackend(), OpenAlexBackend()])
PY
```
