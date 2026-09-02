# H100 6개월 실행 계획 (KT Cloud AI Nexus)

**H100 80GB × 2** 확보. 7B 모델 기준 **컴퓨트는 병목이 아니다**
(LoRA/DPO 는 수 시간, GRPO 도 수일). 병목은 ① 데이터 ② 평가 체계 ③ 환경 접근성이다.

## 0. 먼저 환경부터 확인할 것

KT Cloud AI Nexus 는 두 서비스를 합친 플랫폼이라, **무엇이 프로비저닝됐는지에 따라
할 수 있는 일이 완전히 다르다.**

| 서비스 | 성격 | 우리 과제에서 |
|---|---|---|
| **AI Train** | 컨테이너 기반 GPU(H100), Jupyter/VSCode/SSH | **학습 가능** — M1~M5 전부 실행 |
| **AI Serv** | 추론 전용, GPU 슬라이싱(1장→0.2 단위 5개) | **학습 불가** — 서빙 엔드포인트로만 사용 |

> `/serving` URL 은 AI Serv 를 가리킬 가능성이 높다. 슬라이싱된 0.2 GPU 로는
> 7B 파인튜닝이 불가능하므로, 학습에는 **AI Train 컨테이너가 별도로 필요**하다.

```bash
python scripts/probe_env.py                                  # 자동 판정
python scripts/probe_env.py --endpoint http://<서빙>/v1      # 서빙도 함께 점검
```

이 스크립트가 GPU 유무·개수·FP8 지원·학습 스택·데이터·엔드포인트를 확인하고
**무엇을 실행하면 되는지** 알려준다.

## 이 세션을 H100 으로 옮기기 (teleport) — 가장 확실한 방법

현재 개발 세션은 Anthropic 관리형 VM 이라 **GPU 가 없고 KT Cloud 로 나가는 egress 도 막혀 있다**.
`--teleport` 로 이 세션을 H100 컨테이너로 옮기면 두 제약이 한 번에 풀린다
(대화 맥락·브랜치가 그대로 따라온다).

```bash
# AI Train(H100) 컨테이너의 터미널에서
npm install -g @anthropic-ai/claude-code     # 또는 공식 설치 스크립트
claude auth login                            # 이 세션과 같은 claude.ai 계정

git clone <이 저장소> && cd DeepTechTIPS
git checkout claude/busy-wright-11w7w4

claude --teleport                            # 세션 선택 → 이 세션 고르기
#   또는  claude --teleport <session-id>
```

teleport 요구사항(모두 충족 상태):
- 같은 claude.ai 계정 · 같은 저장소 체크아웃
- 작업 디렉터리 clean · **브랜치가 원격에 푸시돼 있을 것** → `claude/busy-wright-11w7w4` 푸시 완료 ✅

옮겨온 뒤 첫 실행:
```bash
bash scripts/setup_h100.sh        # 패키지·데이터·학습스택·진단 한 번에
bash scripts/run_m1_h100.sh       # M1 전체
```

> teleport 는 **web → terminal 단방향**이다. 옮긴 뒤의 작업은 로컬(H100)에만 남는다.
> 휴대폰/웹에서 계속 지켜보려면 옮긴 세션에서 `/remote-control` 을 켠다.

### 대안 — 이 세션에서 그대로 진행하고 싶다면

**네트워크 정책만 열기**: claude.ai 의 cloud environment 설정에서 network access 를
조정해 `ainexus.ktcloud.com` 등 KT Cloud 도메인을 허용하면, **서빙 엔드포인트 평가**는
지금 세션에서도 가능하다(단, GPU 가 없으므로 학습은 여전히 불가).
→ `docs/en/cloud-environments` 참고.

| 방법 | 학습 | 서빙 평가 | 비고 |
|---|---|---|---|
| **teleport → H100** | ✅ | ✅ | 권장. 맥락 유지 |
| 네트워크 정책만 허용 | ❌ | ✅ | GPU 없음 |
| 현행 유지(코드만 준비) | ❌ | ❌ | 사람이 옮겨 실행 |

### 경로 A — AI Train 이 있는 경우 (권장)
```bash
bash scripts/setup_h100.sh           # 패키지·데이터·학습스택·진단
bash scripts/run_m1_h100.sh          # 진단→기준선→NLI 파인튜닝→EN/KR 재측정
```

### 경로 B — AI Serv(서빙)만 있는 경우
학습은 못 하지만 **Gateway 는 Model-Agnostic 설계**라 서빙 모델을 그대로 붙여 평가할 수 있다.
```bash
python -m compliance_gateway.eval.external --split dev \
    --nli-endpoint http://<서빙>/v1 --nli-model <모델ID> --sweep
python -m compliance_gateway.eval.korean --real \
    --nli-endpoint http://<서빙>/v1 --nli-model <모델ID>
```
셋업은 `bash scripts/setup_h100.sh --eval-only` (학습 스택 생략).
인증키는 `OPENAI_API_KEY` 또는 `KT_API_KEY` 환경변수로 전달한다.
서빙 모델을 NLI 판정기로 쓰는 것은 로컬 파인튜닝 NLI 보다 비용·지연이 크지만,
**학습 환경 없이도 KPI 를 측정할 수 있는 현실적 대안**이다.

## A100 → H100 변경점

| 항목 | A100 | **H100** | 반영 |
|---|---|---|---|
| FP8 (Transformer Engine) | 미지원 | **지원** | `GRPOConfig.fp8=True` 기본 — RL 메모리·속도 이득 |
| SFT 배치 | 4 | **8** | `GPU_PROFILES` |
| DPO 배치 | 2 | **4** | 〃 |
| 최대 시퀀스 | 2048 | **4096** | 〃 |
| 4-bit 양자화 | 학습에 불필요 | 학습에 불필요 | M5 배포 목표일 뿐 학습 요건 아님 |

`train/config.py:validate()` 가 설정-하드웨어 불일치를 잡는다
(예: A100 에 fp8=True → 경고, H100 에 4-bit → 불필요 경고).

## 자원 배분 원칙

| 자원 | 상태 | 전략 |
|---|---|---|
| H100 80GB × 2 | 6개월 | GPU0=생성(vLLM) / GPU1=학습. FP8 RL 가능 |
| 베이스 모델 | HF 접근 ✅ | EXAONE 3.5 7.8B 우선, Qwen2.5 7B 대조 |
| GOONO 데이터 | 일부만 | 합성 파이프라인(bioRxiv) 대폭 확대로 보완 |
| KPI 평가 | 스캐폴드 완료 | 학습 전/후 동일 코드로 측정 → 개선 증명 |

## KPI ↔ 작업 ↔ 측정 매핑

| KPI (목표) | 담당 작업 | 측정 |
|---|---|---|
| 규정위반 탐지 정밀도 90%+ | NLI 파인튜닝 + Gateway | `eval/kpi.py` (precision/recall/F1) |
| 출처 정확률 90%+ | NLI + DOI resolver(DB 대조) | `eval/kpi.py`, SciFact AUC |
| 보고서 자동생성 정확도 88%+ | SLM SFT+DPO + RAG | 외부 전문가 블라인드(과제 30건) |
| 도메인 질의 정확도 2배+ | 도메인 LoRA | 벤치 3도메인×100 A/B |
| VCR 0.85+ | DPO→GRPO | VCR 자기평가 분포 |
| 양자화 보존율 80%+ | GPTQ/AWQ 4-bit | FP16 vs 4-bit 동일 벤치 |

## 6개월 타임라인

### M1 (1개월) — 인프라 + 측정 체계  ← 스캐폴드 완료
- [x] 학습 파이프라인 스캐폴드: `train/{config,data_format,sft,dpo,nli_finetune}.py`
- [x] KPI 측정 하니스: `eval/kpi.py`, 외부 EN(`eval/external.py`), 실데이터 KR(`eval/korean.py --real`)
- [x] 환경 진단(`scripts/probe_env.py`), 서빙 연동(`compliance_gateway/serving/`)
- [ ] **(H100) NLI 파인튜닝**: `python -m compliance_gateway.train.nli_finetune`
      → `TransformerNLI(model_name="checkpoints/nli")` 를 Gateway 에 주입 → KPI 재측정
- [ ] 합성 데이터 확대(수천 쌍) + FDA Warning Letters/CiteAudit 외부 평가셋 연동

### M2–3 (2개월) — Compliance-Aligned SLM
- [ ] 데이터: 합성 DPO 수만 + GOONO 일부 → SFT/DPO 데이터셋 빌드
      `python -m compliance_gateway.train.data_format` (SFT/DPO 변환)
- [ ] SFT: `python -m compliance_gateway.train.sft --base exaone`
- [ ] DPO: `python -m compliance_gateway.train.dpo --sft-adapter checkpoints/sft --vcr-accept 0.7`
- [ ] 목표: VCR 0.60 기준선, 규정위반 탐지·보고서 정확도 KPI 1차 달성

### M4 (1개월) — 에이전트 + RAG
- [ ] 탐색·기획 에이전트 v1 + LangGraph DAG (모든 출력 Gateway 경유)
- [ ] R&D 특화 RAG(구조 인식 청킹) + ScienceON/NTIS 연동
- [ ] 목표: 워크플로우 자동화율 KPI

### M5 (1개월) — RLVR + 온프레미스
- [ ] GRPO 전환: `train/grpo.py` (질의당 N=8, VCR 보상, 롤백) — VCR 0.7→0.85
- [ ] GPTQ/AWQ 4-bit 양자화, 단일 GPU(RTX 4090급) 추론 검증
- [ ] 목표: VCR 0.85+, 양자화 보존율 80%+

### M6 (1개월) — 통합·측정·산출
- [ ] 전 KPI 최종 측정(TTA V&V 대응 포맷), 온프레미스 번들
- [ ] 특허(VCR/Gateway) · 논문 초안(KISTI 공동)

## 셋업 상세

```bash
git clone <repo> && cd DeepTechTIPS
pip install -e ".[train]"                       # torch/transformers/trl/peft/bitsandbytes
bash scripts/download_scifact.sh                # NLI 학습·평가셋

# 1) NLI 파인튜닝 (M1 핵심 — 규정위반/출처 KPI)
python -m compliance_gateway.train.nli_finetune --base deberta-mnli --output checkpoints/nli
python -m compliance_gateway.eval.kpi           # 트랜스포머 NLI 주입 후 재측정

# 2) 데이터 빌드 → SLM SFT → DPO (M2-3)
python -m compliance_gateway.data.build_dpo     # 합성 DPO/SFT 생성
python -m compliance_gateway.train.sft  --base exaone --dataset data/synth/sft.jsonl
python -m compliance_gateway.train.dpo  --base exaone --sft-adapter checkpoints/sft --vcr-accept 0.7
```

## 리스크

| # | 리스크 | 대응 |
|---|---|---|
| R1 | 합성 데이터가 실제보다 쉬움(KPI 낙관 편향) | FDA Warning Letters/CiteAudit 등 **외부 실데이터** 평가셋 필수 |
| R2 | GOONO 데이터 일부만 → 도메인 편중 | 합성 카테고리·연도 다양화, 3도메인(공학/ICT/바이오) 균형 |
| R3 | 6개월 내 GRPO까지 무리 | DPO(VCR 0.7)를 확정 목표, GRPO는 스트레치 |
| R4 | polarity(결론 역전) 탐지 약점 | 트랜스포머 NLI로 M1에서 우선 해소, 재측정 |
