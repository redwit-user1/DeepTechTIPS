# 공개 데이터셋 카탈로그

GOONO 자체 데이터(연구노트 70K)가 없는 단계에서 VCR/Gateway/SLM 을 부트스트랩하기 위한
공개 데이터셋 정리. 초기 집중 도메인은 **제약·바이오**.

> 환경 제약: 본 개발 환경은 네트워크 정책상 `huggingface.co` 가 차단(403)된다.
> S3·GitHub·PyPI 는 접근 가능. HF 의존 데이터/모델은 열린 환경 또는 온프레미스에서 사용.

## 모듈 ↔ 데이터셋 매핑 요약

| 모듈/용도 | 1순위 데이터셋 | 비고 |
|---|---|---|
| VCR `SourceMatch`/`Halluc` (NLI) | **SciFact** | 바이오, S3 직접 다운로드 가능 ✅ (연동 완료) |
| `Halluc` 유형 A/B (가짜 인용) | **CiteAudit**, **FalseCite** | bioRxiv 합성 파이프라인으로 자체 생성도 가능 ✅ |
| 출처바인딩 E2E | **LongCite**, **ALCE** | 문장 수준 attribution |
| SLM Instruction (영어) | **SciInstruct** | CC BY-4.0 |
| SLM Instruction (한국어) | **AI Hub 논문요약** | 18만 논문 + 70만 요약 |
| 한국어 R&D 원천 | **ScienceON / NTIS** | 신청 기반, 사업계획서 명시 연동처 |
| 규정/온톨로지 | **FDA Warning Letters**, eCFR | 규정위반 탐지·ALCOA+ 근거 |

---

## ① 과학 NLI / Claim Verification — VCR `SourceMatch`·`Halluc` 핵심

휴리스틱(AUC 0.52≈랜덤)을 실제 모델로 교체하기 위한 학습·평가셋. → `docs/EVAL_SCIFACT.md`

| 데이터셋 | 규모/내용 | 도메인 | 라이선스 | 접근 | 상태 |
|---|---|---|---|---|---|
| **SciFact** | 1.4K claim + 5.2K evidence abstract, SUPPORT/REFUTE/NEI + rationale | PubMed 바이오 | CC BY-NC | S3 ✅ | **연동 완료** |
| SciNLI | 컴퓨터언어학 NLI | NLP | 연구용 | HF/GitHub | 후보 |
| MedNLI | 의료 NLI(임상노트) | 의료 | MIMIC 신청 | PhysioNet | 후보 |
| MultiVerS | 전문서 맥락 claim verification | 과학 | Apache-2.0 | GitHub | 후보 |

- SciFact: `bash scripts/download_scifact.sh`
- 참고: [Fact or Fiction (SciFact, EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.609/) · [MultiVerS](https://arxiv.org/pdf/2112.01640)

## ② 인용 환각 벤치마크 — `Halluc` 유형 A/B/C

사업계획서가 정의한 복합기만형 환각(가짜 DOI·가짜 논문·수치변조) 평가셋.

| 데이터셋 | 규모 | 특징 |
|---|---|---|
| **CiteAudit** | real 3,586 + fake 2,500 인용 (OpenReview/Scholar/arXiv/bioRxiv) | 가짜 인용 taxonomy → 유형 A/B |
| **FalseCite** | 기만적 인용 유발 응답셋 | GPT-4o-mini 등 환각 증가 실증 |
| **CiteCheck** | 982 인용(물리) | metadata drift + 완전 날조 → 유형 C |
| **Field-Level Hallucination** | 50주제×8 양식, OpenAlex 검증 | 저자명 오류 최다 |

- 참고: [CiteAudit](https://arxiv.org/pdf/2602.23452) · [CiteCheck](https://arxiv.org/html/2605.27700v1) · [Reference hallucination detection](https://arxiv.org/html/2604.03173v1)
- **대안**: `compliance_gateway/data/`(bioRxiv 합성)로 동일 유형을 통제 생성 가능 → `docs/SYNTH_PIPELINE.md`

## ③ Attribution 생성·E2E Gateway 평가 — 출처바인딩(04단계)

| 데이터셋 | 내용 | 비고 |
|---|---|---|
| **ALCE** (ASQA/ELI5/QAMPARI) | attributed generation 표준 벤치 | fluency/correctness/citation quality |
| **LongCite** | 문장 수준 attribution | 출처바인딩 모듈과 1:1 |
| REASONS / CiteME | 과학 문헌 인용 생성·모호성 | |

- 참고: [LongCite](https://arxiv.org/pdf/2408.04568) · [ALCE/복합 attribution 평가](https://aclanthology.org/2025.acl-long.837.pdf) · [REASONS](https://arxiv.org/pdf/2405.02228)

## ④ SLM 도메인 Instruction 학습 (M3)

| 데이터셋 | 규모 | 언어 | 라이선스 | 접근 |
|---|---|---|---|---|
| **SciInstruct** (THUDM/SciGLM) | 254K (물리/화학/수학/증명) | 영어 | CC BY-4.0 | HF/GitHub |
| **AI Hub 논문요약** | 18만 논문 + 70만 요약 | 한국어 | 가입 | AI Hub |
| AI Hub instruction tuning | 도메인별 | 한국어 | 가입 | AI Hub |

- 참고: [SciInstruct/SciGLM](https://arxiv.org/abs/2401.07950) · [GitHub](https://github.com/THUDM/SciGLM) · [AI Hub 논문요약](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=90)

## ⑤ 한국어 R&D 원천 — RAG grounding · Instruction 추출 (신청 기반)

사업계획서 명시 연동처. 에이전트(③) 네이티브 연동 대상이기도 함.

| 출처 | 규모 | 접근 |
|---|---|---|
| **ScienceON** (KISTI) | 논문 186.8만 · 보고서 24.7만 · 특허 102만 | R&D데이터 신청 / Open API |
| **NTIS** | 국가R&D 통합 | 데이터구축 신청 |
| KIPRIS | 특허 | Open API |

- 참고: [ScienceON R&D데이터](https://scienceon.kisti.re.kr/srch/selectPORSrchFnct.do?cn=FNCT000125) · [NTIS 데이터구축](https://www.ntis.go.kr/ThMainRnddata.do)

## ⑥ 규제/컴플라이언스 텍스트 — 규정위반 탐지·ALCOA+ 온톨로지

| 출처 | 내용 | 접근 |
|---|---|---|
| **FDA Warning Letters** | 실제 위반 사례(Data Integrity 35%) | FDA 웹 공개(구조화 X → 스크래핑) |
| **eCFR 21 CFR Part 11** | 전자기록·서명 규정 원문 | ecfr.gov 공개 |
| ICH E6(R3)/GxP 가이드 | GLP/GMP 데이터 무결성 | 기관 공개 |

- 참고: [FDA Warning Letters & Data Integrity](https://www.certivo.io/blog/fda-warning-letters-data-integrity) · [21 CFR Part 11 & Annex 11 audit trail](https://intuitionlabs.ai/articles/audit-trail-requirements-ai-gxp-compliance)

---

## 라이선스 주의

- **SciFact = CC BY-NC**: 비상업 연구·평가용. 상용 학습 데이터로는 별도 검토.
- **SciInstruct = CC BY-4.0**: 출처 표기 시 상용 가능.
- **AI Hub**: 이용약관(국내 기관·목적 제한) 확인 필요.
- bioRxiv preprint: 논문별 라이선스 상이(cc_by / cc_no 등) → 시드에 `license` 필드 보존.
- 상용 배포 데이터는 **GOONO 자체 데이터(독점)** 중심으로 전환하는 것이 최종 방향.

## 권장 도입 순서

1. ✅ SciFact (NLI 벤치) — 완료
2. ✅ bioRxiv 합성 (DPO/유형 A·C) — 완료
3. CiteAudit/FalseCite (유형 A/B 외부 검증셋, HF 열린 환경)
4. SciInstruct + AI Hub (SLM Instruction, M3)
5. ScienceON/NTIS 신청 (한국어 R&D 원천, 본 과제 연동)
