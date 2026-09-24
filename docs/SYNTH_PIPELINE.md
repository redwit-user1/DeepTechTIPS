# 합성 데이터 파이프라인 (bioRxiv → DPO)

GOONO 자체 R&D 데이터가 없는 단계에서, **실제 제약·바이오 논문**으로부터
DPO Preference 학습 데이터와 Gateway 평가셋을 자동 구축한다.

## 왜 이 방식인가

- 초기 도메인(제약·바이오)과 정합한 **실데이터** 사용 → 도메인 신뢰성.
- 사업계획서가 정의한 **복합기만형 환각 3유형**을 통제된 방식으로 생성 → Gateway/VCR 평가 가능.
- 생성 데이터를 **VCR 로 자기검증**(chosen > rejected) → 데이터 품질과 보상함수를 동시에 검증.
  AI 생성 데이터를 그대로 학습하면 순환오류가 나므로(사업계획서 p.20 지적),
  반드시 VCR 통과분만 채택하는 구조와 동일한 철학.

## 흐름

```
bioRxiv preprint (실데이터 시드)
  → 정량 주장 문장 추출            extract.claim_sentences()
  → 양성 y_w (출처+정확)            synth: "{claim} ({Author et al. (year)}; DOI: ...)"
  → 음성 y_l 4종                    tamper:
       no_source        무출처 (대조군)
       numeric_tamper   수치 변조 + 인용 유지        ← 유형 C
       fake_doi         실존 저자 + 형식만 유효한 DOI ← 유형 A
       polarity_flip    결론 방향 반의어 치환 + 인용
  → DPO Preference 쌍 + Gateway 평가 아이템 (JSONL)
  → VCR 자기검증 (VCR(y_w) > VCR(y_l) 승률)
```

## 데이터 출처

- bioRxiv (`pharmacology and toxicology`, 2024), CC BY 등 라이선스 표기.
- bioRxiv **MCP** 로 수집 후 시드 고정: `compliance_gateway/data/seed/biorxiv_pharma.json`
  (repo 코드의 api.biorxiv.org 직접 접근은 본 환경 네트워크 정책상 차단됨 → 시드로 재현).
- 시드 갱신: `python scripts/build_seed.py` (RECORDS 교체).

## 실행

```bash
python -m compliance_gateway.data.build_dpo
# → data/synth/dpo_pairs.jsonl, gateway_eval.jsonl + 콘솔 리포트
```

## 결과 (시드 4편, claim 11개)

```
DPO pairs=35  eval items=46

[VCR 자기검증] chosen > rejected 승률: 100.0%
  fake_doi        +0.2000
  numeric_tamper  +0.1979
  no_source       +0.1625
  polarity_flip   +0.0555   ← 최약(통계 NLI 한계)

[Gateway 결정]  compliant PASS 90.9% / 위반 차단 97.1%
```

## 해석

1. VCR/Gateway 가 **유형 A(가짜 DOI)·유형 C(수치변조)·무출처**를 안정적으로 거른다(마진 +0.16~0.20).
2. **극성 역전(결론 방향 조작)의 마진이 +0.06로 가장 약하다** — SciFact 벤치마크(AUC 0.57)와 동일하게,
   통계적 NLI 의 한계를 재확인. → **트랜스포머 NLI 도입 시 가장 큰 개선 기대 지점**.
3. 이 파이프라인은 버그도 잡아냈다(개발 중 발견·수정):
   - DOI 정규식이 끝의 `)` 까지 포착 → 진짜 DOI 가 resolver 실패 → compliant 오탐.
   - author-year 인용을 근거 본문 미등장으로 환각 처리 → 정상 인용 오탐.
   수정 후 승률 42.9% → 100.0%.

## 다음 단계

- 시드 확대(카테고리·연도 다양화)로 수천 쌍 규모 구축.
- HF 접근/온프레미스 환경에서 트랜스포머 NLI 주입 후 polarity_flip 마진 재측정.
- 채택 기준(VCR ≥ 0.7) 적용한 DPO 학습 → SLM 정렬(M3) 연결.
