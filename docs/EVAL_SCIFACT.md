# SourceMatch / NLI 벤치마크 (SciFact)

VCR 의 `SourceMatch` / `Halluc` 가 의존하는 entailment 능력을 **실제 바이오 데이터**로 측정한다.
초기 집중 도메인(제약·바이오)과 정합한 SciFact 를 사용했다.

## 데이터

- **SciFact** (Wadden et al., EMNLP 2020) — 전문가 작성 과학 claim + PubMed 근거 abstract. 라이선스 CC BY-NC.
- 출처: `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz` (HuggingFace 차단 환경에서도 S3 접근 가능)
- 변환: `premise = 인용 근거`, `hypothesis = 주장(claim)`, `label ∈ {SUPPORT, CONTRADICT}`.
- 근거 없는(NEI) claim 은 이분 벤치마크에서 제외.

## 실행

```bash
python -m compliance_gateway.eval.benchmark --split train
python -m compliance_gateway.eval.benchmark --split dev --transformer  # HF/로컬 모델 접근 가능 시
```

## 결과 (train, n=957: SUPPORT 616 / CONTRADICT 341)

| 백엔드 | AUC | mean(SUPPORT) | mean(CONTRADICT) | 해석 |
|---|---|---|---|---|
| lexical (baseline) | **0.52** | 0.286 | 0.269 | 토큰 중첩 ≈ **랜덤**. SUPPORT/CONTRADICT 분리 불가 |
| statistical (v0.5) | **0.57** | 0.156 | 0.124 | 극성(부정·반의어) 처리로 +0.05. 여전히 부족 |
| transformer | (HF/로컬 필요) | — | — | 운영 목표 백엔드 |

> AUC = SUPPORT 점수가 CONTRADICT 보다 높을 확률(1.0=완벽, 0.5=랜덤).

## 핵심 결론

1. **순수 토큰 중첩(lexical)은 사실상 무력하다(AUC 0.52).** "근거가 주장을 *반박*하는" 경우를
   "뒷받침"과 구분하지 못한다 → 사업계획서가 지적한 **수치 변조·반대 결론형 환각**을 놓친다(R1 리스크 실증).
2. 극성 처리(statistical v0.5)는 의미 있는 개선(+0.05)이지만, 규정위반 탐지 목표(Precision 90%+)에는
   **사전학습 NLI 트랜스포머가 필수**임을 정량적으로 확인.
3. 본 환경은 네트워크 정책상 `huggingface.co` 가 차단(403)되어 트랜스포머 NLI 를 즉시 로드할 수 없다.
   `compliance_gateway/nli/transformer.py` 가 동일 시그니처로 준비되어 있어, HF 가 열린 환경 또는
   온프레미스(에어갭, 로컬 모델 경로) 배포에서 **함수 하나 주입으로 교체**된다:
   `ComplianceGateway(nli_fn=TransformerNLI(model_name=...))`.

## 다음 단계 (M1)

- HF 접근 가능 환경에서 과학/바이오 NLI 모델(예: SciFact 파인튜닝, DeBERTa-MNLI-FEVER-ANLI) 벤치마크 추가.
- SciFact dev/test 로 일반화 측정, 최적 임계값 θ 도메인별 보정(VCR v2).
- `Halluc` 유형 A/B 평가셋(CiteAudit/FalseCite) 연동.
