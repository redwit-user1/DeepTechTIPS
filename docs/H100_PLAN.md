# H100 6개월 실행 계획 (KT Cloud AI Nexus)

> **⚠️ 실제 할당 환경 (2026-11-09 실측) — 아래 본문의 가정과 다르다.**
>
> | 항목 | 문서 가정 | 실제 |
> |---|---|---|
> | GPU | H100 80GB × 2 | **H200 141GB × 2** (NIPA 고성능 컴퓨팅 지원사업, 프로젝트 `NIPA-H200`) |
> | 플랫폼 | AI Train / AI Serv 구분 | **Backend.AI 웹 UI** 컨테이너 세션 (인터랙티브·배치) |
> | 이미지 | 자유 | NGC PyTorch 25.11 / 25.03 / 24.09 만. `trl`·`vllm` 미포함, torch 는 `/etc/pip/constraint.txt` 로 고정 |
> | 접속 | teleport | 세션 앱 → **SSH/SFTP** (호스트 `proxy2`/`proxy3`·포트가 세션마다 바뀜, 첫 연결이 끊기면 아이콘 재클릭) |
> | 저장 | — | `/home/work` 49GB 스크래치(세션 종료 시 삭제). 영속 데이터는 NAS 폴더 `research-data-analysis`(연구데이터 분석 관리) |
> | 회수 | — | **GPU util 6시간 평균 1% 미만이면 세션 자동 회수** |
> | 외부망 | 차단 가정 | PyPI·HF·GitHub 열림, sudo 가능 |
>
> - HF 캐시는 스크래치에 둔다(NAS 는 NFSv3 라 HF filelock 이 멈춘다).
> - 계정이 레드윗 다른 과제와 공유다. **고객 연구노트 원문은 업로드하지 않는다**(이용 권한 미확인 — `docs/OCR_HARVEST.md`).

**H100 80GB × 2** 확보. 7B 모델 기준 **컴퓨트는 병목이 아니다**
(LoRA/DPO 는 수 시간, GRPO 도 수일). 병목은 ① 데이터 ② 평가 체계 ③ 환경 접근성이다.

## 0. 실행 절차 (실제 환경)

teleport·AI Train/AI Serv 경로는 **실제 환경과 달라 삭제했다**(2026-11-09). 실제 절차는
[`README.md` → GPU 실행](../README.md#gpu-실행-kt-cloud-ai-nexus--h200-141gb--2) 과 `scripts/ainexus_setup.sh`.
구 M1 스크립트(`setup_h100.sh`·`run_m1_h100.sh`)도 삭제했다 — SciFact 로 학습해 한국어로 재는 흐름은
전이되지 않음이 실측됐다(연구노트 2026-11-09 §6).

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
| 베이스 모델 | HF 접근 ✅ | Qwen3.8-27B (Apache 2.0), 경량 Qwen3.5-9B |
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
- [ ] SFT: `python -m compliance_gateway.train.sft --base qwen`
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
python -m compliance_gateway.train.sft  --base qwen --dataset data/synth/sft.jsonl
python -m compliance_gateway.train.dpo  --base qwen --sft-adapter checkpoints/sft --vcr-accept 0.7
```

## 리스크

| # | 리스크 | 대응 |
|---|---|---|
| R1 | 합성 데이터가 실제보다 쉬움(KPI 낙관 편향) | FDA Warning Letters/CiteAudit 등 **외부 실데이터** 평가셋 필수 |
| R2 | GOONO 데이터 일부만 → 도메인 편중 | 합성 카테고리·연도 다양화, 3도메인(공학/ICT/바이오) 균형 |
| R3 | 6개월 내 GRPO까지 무리 | DPO(VCR 0.7)를 확정 목표, GRPO는 스트레치 |
| R4 | polarity(결론 역전) 탐지 약점 | 트랜스포머 NLI로 M1에서 우선 해소, 재측정 |
