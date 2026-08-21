# 데이터 확보 체크리스트 — 브라우저 에이전트(Aside 등) 활용

## 배경

이 프로젝트 내내 가장 큰 병목은 **모델도 컴퓨트도 아닌 네트워크 접근**이었다.
관리형 개발 VM 의 egress 정책상 아래가 전부 차단된다.

| 소스 | 상태 | 우리에게 왜 필요한가 |
|---|---|---|
| ScienceON (KISTI) | ❌ 차단 | **자연 한국어 R&D 문서** — 한국어 KPI 주장의 전제 |
| NTIS | ❌ 차단 | 국가R&D 과제·보고서 |
| KCI | ❌ 차단 | 국내 학술논문 |
| AI Hub | ❌ 차단 | 논문요약 18만건(한국어) — SLM Instruction |
| arXiv | ❌ 차단 | 국내 기관 소속 논문 |
| FDA Warning Letters | ❌ 차단 | 실제 규정위반 텍스트(규정위반 탐지 KPI) |
| HuggingFace | ❌ 차단 | 베이스 모델·NLI 모델 |
| KT Cloud AI Nexus 콘솔 | ❌ 차단 | 무엇이 프로비저닝됐는지 확인 |

**Aside**(AI 브라우저 + CLI/MCP) 같은 브라우저 에이전트를 쓰면, 사용자의 네트워크와
**이미 로그인된 세션**으로 이 사이트들에 접근할 수 있다.

> 전제: 브라우저 에이전트는 **Claude Code 와 같은 머신**에서 돌아야 한다.
> 관리형 VM 에는 브라우저를 설치할 수 없으므로 `claude --teleport` 로
> 로컬/H100 으로 세션을 옮긴 뒤 연결한다. → `docs/H100_PLAN.md`

## ⚠️ 원칙: 스크래핑이 아니라 "공식 경로 통과"에 쓴다

대량 스크래핑은 각 사이트 이용약관 위반 소지가 있고, 데이터 재배포 시 라이선스 문제가 된다.
브라우저 에이전트의 **진짜 가치는 사람이 해야 하는 인증·신청·키 발급 단계를 통과하는 것**이다.
그 다음은 공식 API/다운로드로 받는다. 우리 어댑터는 이미 그 형태로 준비돼 있다.

| 단계 | 브라우저 에이전트 | 이후 |
|---|---|---|
| 로그인·약관 동의·신청서 제출 | ✅ 적합 | — |
| **API 키 발급** | ✅ 적합 | 어댑터에 키 주입 → 공식 API 로 수집 |
| 공식 데이터셋 다운로드(로그인 필요) | ✅ 적합 | 파일 그대로 사용 |
| 페이지 대량 크롤링 | ❌ 지양 | 약관·라이선스 위반 소지 |

## 소스별 실행 항목

### 1. ScienceON (KISTI) — 최우선
자연 한국어 R&D 문서. **한국어 외부 평가셋 미확보**가 현재 가장 큰 공백이다.

1. https://scienceon.kisti.re.kr 로그인
2. OpenAPI 키 발급 (마이페이지 → OpenAPI 신청)
3. 키를 환경변수로 주입하면 기존 어댑터가 바로 동작:
   ```python
   from compliance_gateway.data.korean.sources import ScienceONAdapter
   ScienceONAdapter(api_key="<발급키>").search("연구데이터 관리", rows=50)
   ```
4. 필요 시 'R&D데이터 신청'으로 대량 데이터 별도 요청

### 2. AI Hub — 논문요약 18만건(한국어)
1. https://aihub.or.kr 회원가입·로그인
2. '논문 요약 데이터셋' 신청 및 약관 동의
3. 다운로드 → `data/raw/aihub/` 배치
4. 용도: SLM Instruction 학습(M2-3). 이용약관의 목적 제한 확인 필수

### 3. NTIS — 국가R&D 과제정보
1. https://www.ntis.go.kr 로그인 → 공개 API 신청
2. `NTISAdapter(api_key=...)` 에 주입

### 4. HuggingFace — 베이스/NLI 모델
학습 환경(H100)에서 직접 받는 게 정석. 브라우저 경유가 필요한 경우는
사내망 제약이 있을 때뿐이며, 그 경우 모델을 내려받아 로컬 경로로 지정한다:
```bash
python -m compliance_gateway.train.nli_finetune --base /local/path/to/model
```

### 5. FDA Warning Letters — 규정위반 실데이터
1. https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/compliance-actions-and-activities/warning-letters
2. 공개 자료이나 구조화 데이터셋은 없음 → 검색 결과를 CSV 로 내보내거나 개별 문서 확보
3. 용도: GLP/Part 11 조항 단위 위반 탐지 평가셋(현재 미연동)

### 6. KT Cloud AI Nexus 콘솔
1. https://www.ainexus.ktcloud.com 로그인
2. **확인할 것**: AI Train(학습 컨테이너) 프로비저닝 여부, H100 할당 형태,
   AI Serv 엔드포인트 URL 과 모델 ID, 인증키
3. 결과에 따라 `docs/H100_PLAN.md` 의 경로 A/B 결정

## 확보 후 검증

```bash
python scripts/probe_env.py                    # 소스별 사용 가능 여부 재확인
python -m compliance_gateway.eval.korean --real  # 한국어 실데이터 기준선(AUC 0.715) 대비
```

한국어 자연 문서를 확보하면 **`docs/DATASET_KR.md` 의 최대 한계**(텍스트가 영어,
자연 한국어 R&D 문서 미확보)가 해소되고, 그때 비로소 한국어 KPI 를 정직하게 주장할 수 있다.
