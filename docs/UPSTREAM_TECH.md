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


---

## 5. GitHub 스킬/리포 조사 (Unsloth·서빙) — 2차 조사

### 발견한 것

| 리소스 | 내용 | 우리에게 |
|---|---|---|
| [`unslothai/unsloth`](https://github.com/unslothai/unsloth) (74k★) | **실행+학습 통합 로컬 UI** 로 확장 | 서빙까지 커버 |
| [`TYH-labs/unsloth-buddy`](https://github.com/TYH-labs/unsloth-buddy) | 7단계 파인튜닝 라이프사이클 스킬(환경감지→학습→평가→GGUF 내보내기→서빙) | **gotcha 목록이 유용** |
| [`wshobson/agents`](https://github.com/wshobson/agents) `plugins/llm-finetuning/` | `grpo-rlvr-training`, `preference-optimization`, `vision-sft` 스킬 | **GRPO 레시피·점검 원칙 채택** |
| `unsloth-grpo` (cuba6112/skillfactory) | 다목적 보상함수·RLVR 패턴 | 보상 설계 참고 |
| Unsloth 공식 배포 문서 | `save_pretrained_merged` / `save_pretrained_gguf` / vLLM·Ollama·SGLang 서빙 | M5 배포 경로 |

> 여러 스킬 카탈로그(`K-Dense-AI/scientific-agent-skills` 34k★, `VoltAgent/awesome-agent-skills` 31k★)를
> 뒤졌으나 **Unsloth 전용 스킬은 위 두 곳뿐**이었다.

### 적용 1 — LoRA 설정이 Unsloth 고속 경로를 벗어나 있었다 (실제 성능 버그)

`lora_dropout=0` / `bias="none"` 이 Unsloth 최적화 조건이다.
0 이 아니면 **Unsloth 가 LoRA 행렬을 제외한 나머지 레이어만 패치**해 성능 손해가 난다.
우리 기본값은 `dropout=0.05` 였고 `bias` 는 아예 지정되지 않았다.

→ `LoRAConfig.dropout=0.0`, `bias="none"` 로 수정. `validate()` 가 위반 시 경고.

### 적용 2 — 보상함수 사전 점검 게이트 (`train/reward_check.py`)

GRPO 스킬의 핵심 원칙을 구현했다:
> "학습 전에 50~100개 샘플에 보상함수를 돌려 조용한 오정렬을 잡아라."

RL 은 보상이 틀려도 **조용히** 잘못된 목표를 최적화한다. GPU 시간을 쓰기 전에 4가지를 본다:
분리도(AUC) · 축퇴(분산) · 보상 해킹 취약성 · 구성요소 생존.

```bash
python -m compliance_gateway.train.reward_check --dataset kr_real
python -m compliance_gateway.train.reward_check --dataset mixed --samples 200
```

#### 게이트가 즉시 잡아낸 것 3가지

**(a) `source_exist` 회귀 — 영어 인용이 전부 0점이었다**
한국어 인용 병합을 넣으면서 부분문자열 매칭을 썼는데, 병합 인용 안의 마침표
(`"Alam et al."`)에서 문장이 쪼개져 매칭이 실패했다. 영어 합성셋 `source_exist` 가
**상수 0.0** 이었다(= 보상의 25%가 죽어 있었다).
→ **위치(span) 겹침 기준**으로 교체. 수정 후 mean 0.0 → **0.838**.

**(b) 단일 데이터셋으로는 VCR 4요소를 다 자극하지 못한다**

| 데이터셋 | AUC | source_exist | halluc | 판정 |
|---|---|---|---|---|
| kr_real | 0.715 | **상수 1.0** | **상수 0.0** | ❌ 보상 45% 낭비 |
| synth (bioRxiv) | 0.960 | σ=0.368 | σ=0.399 | ✅ |
| kr_synth | 1.000 | σ=0.376 | σ=0.342 | ✅ |
| **mixed** | 0.617 | σ=0.271 | σ=0.350 | ❌ (아래 참조) |

국내 실데이터는 **모든 항목이 실존 인용을 갖도록 설계**돼 있어(교차 귀속 오류만 다룸)
`source_exist`·`halluc` 가 상수가 된다. 데이터셋 결함이 아니라 설계상 속성이지만,
**이것만으로 GRPO 를 돌리면 보상 가중치의 45%가 학습 신호 없이 낭비된다.**

**(c) VCR 점수는 데이터셋·언어 간 비교가 불가능하다**
혼합하면 4요소는 모두 살아나지만 AUC 가 0.617 로 떨어진다.
`kr_real` **위반**(0.617)이 `synth` **준수**(0.584)보다 높기 때문이다.
→ 단일 전역 보상 스케일로는 다도메인 RL 이 성립하지 않는다.
임계값 이슈(영어 θ=0.70 vs 한국어 θ=0.94)와 **같은 현상이 보상 수준에서 재확인**된 것이다.
사업계획서의 **VCR v2 도메인별 가중치 자동 최적화**가 필요한 실증 근거.

### GRPO 하이퍼파라미터 — 출처별 권장치가 갈린다

| 출처 | lr | beta |
|---|---|---|
| 2026 커뮤니티 일반 권장 | — | 0.04 |
| `wshobson/agents` GRPO 레시피 | 5e-7 | 0.01 |
| **현재 우리 설정** | 1e-6 | 0.04 |

`num_generations ≥ 8`("floor, not a suggestion")과 vLLM 모드 선택은 양쪽이 일치한다.
lr/beta 는 실측으로 정할 문제라 **바꾸지 않고 기록만 남긴다**.

### 아직 적용하지 않은 것 (M5 대상)

- GGUF 내보내기(`save_pretrained_gguf`) → Ollama/llama.cpp/vLLM 서빙
- **gotcha**: 4-bit 로 로드한 모델은 GGUF 내보내기 실패 → FP16 으로 로드해야 함
- **gotcha**: 서빙 시 **학습 때와 같은 chat template** 을 써야 함(오류 1순위 원인)
- **gotcha**: GRPO 는 보상이 오르기까지 **≥300 스텝** 필요(정상 동작이니 조기 중단 금지)
