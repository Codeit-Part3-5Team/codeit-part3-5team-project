# 라우트 C 평가 설계 (2차 개정본)

작성: 김도혁 (DE, 라우트 C 추출+평가 담당)
대상: B2G 입찰 RFP RAG — 입찰메이트(Bidmate)
상태: 인프라 코딩 + fixture dry-run 직전 확정본

> 핵심 한 줄: 라우트 C를 청크 hit으로 평가하면 안 된다는 발견은 맞다. 항목 평가로 바꾸는 순간 "gold가 독립·완전한가", "대분류와 원자 항목을 섞지 않았는가", "발주기관 절차를 입찰자 의무로 오인하지 않았는가", "scope를 모델 분류에만 의존하지 않았는가"가 새 평가의 신뢰성을 결정한다. 그래서 V4는 진단용(diagnostic), 정식 검증은 V5 blind로 분리한다.

> 확정 상태: 아래 계약을 반영하고 DOC_068 fixture dry-run을 통과하면 matcher 구현을 시작할 수 있는 상태가 된다. (지금은 "구현해도 오판 안 함"이 아니라 "오판 막을 계약이 정의됨" 단계)

---

## 1. 라우트 C가 무엇인가

100건 공공조달 RFP에서, 사용자가 고른 문서 1건의 입찰 준수항목 체크리스트를 자동 추출.

- 입력: selected_doc_id
- 출력: ex.run(doc_id) → {manifest, items}
  - items[i].item: 의무사항 텍스트
  - items[i].primary_category: requirement / qualification / submission / scoring (모델 분류 결과 — 오분류 가능)
  - items[i].evidence: declared_chunk_id, quote, match_status

추출기: RFP 로드 → 윈도우 분할(청크 8개) → gpt-5-mini 추출 → quote 검증(scope) → sentinel → dedupe → completeness gate. evidence-first, 단위테스트 41 PASS.

---

## 2. 핵심 발견 — 청크 hit은 라우트 C 평가에 부적합

골든셋은 정답을 청크 위치로 라벨링한다. 검색(A/B)은 "정답 청크를 찾았나"라 적합하지만, 추출(C)에 적용하니 문서마다 40~100%로 출렁였다.

원인: 추출기는 요구사항을 정확히 뽑되, 골든이 지정한 "목차/색인 청크"가 아니라 "상세 본문 청크"에서 뽑는다.
- 실측(DOC_068 raw): SFR-004 "회원 운영"을 목차표(#10)가 아니라 상세 본문(#14)에서 추출 — 사용자 관리·로그인 신규 구축, 그룹별 접근권한 차등, 회원가입·변경·탈퇴 운영, 개인화 페이지·메뉴, 접속통계 확인을 개별 requirement로 분해. 내용 정확, 청크 번호만 다름.
- 검색은 청크 찾기, 추출은 내용 가공 → 청크 단위 채점은 category error.

→ 측정 단위를 청크에서 항목(expected_items)으로 전환.

---

## 3. 평가 데이터셋 두 단계 — V4(진단) / V5(정식)

### V4 — Diagnostic Benchmark
golden_dataset_v4.json (라우트 C 6문항 + expected_items + item_match_mode, 아인님 빌드 PR #80).

정식 test set이 아닌 이유:
- **독립성 부족**: expected_items 초안이 추출 결과와 원문을 함께 대조해 작성됨(제안서 명시). 모델 출력이 gold 설계에 영향 → 같은 모델 재평가 시 수치 부풀 수 있음.
- **완전성 부족**: expected가 RFP 모든 의무항목이 아니라 일부. Q018 원문 answer엔 자격 ④ASP.NET ⑤MS-SQL ⑥.NET프레임워크가 있으나 expected엔 없음 → 추출기가 정확히 뽑아도 extra로 깎임.

역할: 추출기 회귀·실패 패턴 진단, matcher·캐시·리포트 파이프라인 검증, extra 유형 분석.
주장 금지: 일반화 Coverage, 정식 Precision, RFP 전수 추출률.

### V5 — Formal Blind Evaluation
- 개발·프롬프트 개선에 안 쓴 새 RFP (development/evaluation 분리)
- requirement/qualification/submission/scoring 모두 포함, text·table 문서 혼합
- gold 작성자가 라우트 C 출력 안 보고 원문+answer로 라벨링
- gold freeze 후 prompt/schema/matcher 고정
- expected item마다 gold_evidence(chunk_id, quote) 부착

V4 라벨링 기준(mode 구분, 항목 입도)은 V5에 재사용 → 처음부터 다시 아님.

---

## 4. 지표 체계

### 4-1. 재현율(Recall) — 2축 분리, 합산 금지

**Diagnostic Expected-item Recovery Rate (원자, V4)** — Q015/Q018/Q019/Q020 (item mode)
```
Recovery numerator = 1개 이상 extracted와 match된 unique expected 수
Recovery = Recovery numerator / 전체 expected_items 수
```
V4에선 "현재 gold 기대 항목을 현재 추출기가 얼마나 회수하나". 정식 Atomic Coverage는 V5.

**Taxonomy Representation Rate (대분류, V4)** — Q016/Q017 (category mode)
```
Taxonomy Representation = 실질 하위항목으로 대표된 대분류 수 / 전체 expected 대분류 수
```
성공 기준: 해당 taxonomy(SFR 등)에 속하는 실질 수행·제출·자격·배점 항목이 1개 이상 verified evidence와 함께 추출됐을 때만 represented. **목차·개수·분류표 문장만으론 충족 금지.**
→ ★V4 taxonomy mapping 파일 필수(아래 4-6). 없으면 SFR/PER 대표성 판정이 matcher 감(感)이 됨.

→ 두 축 평균 안 냄. 주 집계 Macro-average(문항별 산술평균). Atomic+Taxonomy 합산 단일 Micro score 금지. 각 그룹 내 Micro는 문서별 항목 수 차이 보여주는 보조값으로만 병기 가능.

### 4-2. Review 점수 처리 (3값 동시 보고)
match/miss/review를 점수에서 몰래 빼거나 match로 억지 포함하지 않는다.
```
Confirmed Recovery     = match / total_expected            (보수적 하한)
Review Burden          = review / total_expected           (사람 검수 부담)
Unresolved Upper Bound = (match + review) / total_expected (낙관적 상한)
```
사람 검수 후 → Human-reviewed Recovery = final_match / total_expected. 발표·보고서엔 Human-reviewed만. V4는 위 3값을 diagnostic으로 병기.

### 4-3. 정밀도(Precision) — 이원화

**Expected-item Alignment Rate (참고용)**
```
Alignment numerator = 1개 이상 expected와 match된 unique extracted 수
Alignment = Alignment numerator / 전체 추출 항목 수
```
추출기 품질 아니라 진단셋 기준선 부합도. gold 불완전성 때문에 정식 Precision 아님.

**Semantic Item Precision (정식, V5 중심)** — 추출 항목이 아래를 모두 만족하는 비율. V5에서 사람 검수 병행.
1. item이 evidence quote의 의미를 보존하는가
2. item이 질문의 evaluation_scope에 속하는가
3. item의 행위 주체가 bidder_action 또는 bidder_evaluation_rule인가
4. buyer_internal_process / context_only가 아닌가

### 4-4. 근거 정합성 (기술 지표)

**Mechanical Quote-Source Match Rate (현 96~98%)** — 추출 quote가 허용 원문 범위에서 기계적으로 발견되는 비율. evidence-first 안정화 증거. semantic precision/hallucination rate와 동일시 금지.
- 포함: strict_verified / relocated_verified / cross_chunk_verified (source_scope_valid=True)
- 별도 보고: unverified_evidence / out_of_scope_source
- ★보고 시 manifest 동반 필수: 평가 문서 수, 실행 날짜, 모델명, prompt/schema version·hash, window size/overlap, strict/relocated/cross-chunk 비율, out_of_scope 건수. 발표 수치는 해당 manifest에 연결.

**Sentinel Flag Rate** — 조건·숫자·기한 손실 의심 후보 탐지율(오류율 아님).

**Buyer-process Contamination Rate (V4 진단값)**
```
= buyer_internal_process로 판정된 final item 수 / final item 총수
```
정식 Precision 아님. 라우트 C가 체크리스트에 발주기관 절차를 얼마나 섞는지 보는 품질 진단.

### 4-5. Extraction Density
문서당 dedupe 후 항목 총수. 중복 폭증 통제.

### 4-6. ★Actor / Applicability 4분류 (DOC_068 raw로 실증된 결함)

추출기가 발주기관 행동을 requirement로 뽑는 사례 확인(DOC_068 raw: "기술평가 위원회 구성하여 평가 실시", "우선협상대상자 선정은 득점자순"). quote-source match는 통과하나 입찰자 체크리스트론 부적합. 단순 이진(입찰자/발주기관)으로 묶으면 scoring이 깨지므로 4분류:

| 라벨 | 의미 | 예시 | scope 처리 |
|------|------|------|-----------|
| bidder_action | 제안자가 제출·충족·구현·운영 | 로그인 기능 구축, 보안서약서 제출 | in-scope |
| bidder_evaluation_rule | 발주기관이 평가하나 제안자에 직접 영향 | 기술평가 90점, 85% 미만 협상 제외 | in-scope (특히 scoring) |
| buyer_internal_process | 제안자가 행동으로 충족 불가한 내부 절차 | 평가위원회 구성, 내부 협상 순서 | out-of-scope |
| context_or_background | 사업방식·기관소개·일반 배경 | 제한경쟁·협상계약 방식 설명 | out-of-scope/review |

※ "기술평가 85% 이상"은 발주기관 평가 규칙이지만 제안자에게 중요한 scoring → bidder_evaluation_rule로 살림. "평가위원회 구성"은 buyer_internal_process로 제외. 둘을 같은 Invalid로 묶지 않는다.

### 4-7. Extra(미매칭 추출) — 내부 5분류 / 발표 4분류

expected 매칭 안 된 항목을 획일적 오답 처리 안 함. 내부 JSON enum 5개:

| 유형 | 예시 | 원인 | 대응 |
|------|------|------|------|
| Extra-Valid | gold에 없던 실제 제출 의무 | gold gap | V5 gold 보강 후보 (자동 흡수 금지) |
| Extra-Redundant | 같은 서류 두 번 분할 | dedupe/overlap | density 개선 |
| Extra-OutOfScope | 평가위원회 구성 | actor/scope | extractor scope 개선 후보 |
| Extra-Unsupported | quote가 item을 지지 안 함 | grounding/semantic 오류 | hallucination 후보 |
| Extra-Review | 일부 중복·scope 애매 | 판정 모호 | 사람 검수 |

★OutOfScope와 Unsupported를 Invalid로 합치면 "범위 오류"와 "근거 검증 실패"를 같은 오류로 봄(원인·처방 다름). DOC_068 평가위원회는 OutOfScope(근거는 있음). 발표용 표는 4개로 단순화 가능(Valid/Redundant/Invalid/Review)하되, 내부 enum은 5개.

1차 LLM 후보 분류(근거·사유) → 2차 사람 최종 확정. V4 59개는 사람 전수 가능.

---

## 5. criticality (V5 gold 구조 — 6분류)

expected item에 입찰 영향 구조 라벨링. LLM이 추측 아니라 원문 명시 영향도 기록(criticality_basis: explicit/inferred 동반).

| 라벨 | 의미 | 예시 |
|------|------|------|
| eligibility_gate | 미충족 시 자격 박탈·실격 명시 | 참가자격 미충족 입찰 불가, 공동수급 불가, 마감 후 무효 |
| mandatory_submission | 입찰 참여 단계 제출 의무 | 제안서 10부, 보안서약서, 위임장 |
| mandatory_delivery | 계약 이행 시 구현·제공·준수 | 로그인 기능 구축, 개인정보보호 준수 |
| scored_criterion | 평가 배점·가점 영향 | 기술평가 90점, 정량/정성 기준 |
| optional_or_preferred | 원문이 권장·우대·선택 명시 | 가점 요소, 추가 제안 |
| unknown | 원문만으론 영향도 불명확 | 모호한 일반 요구사항 |

★mandatory_submission 분리: 보안서약서·제안서·위임장은 계약 이행 산출물이 아니라 입찰 참여 제출 의무. eligibility_gate는 "미충족 시 실격"이 원문 명시된 경우만(단순 제출서류를 다 gate로 과분류 금지).

**Eligibility Gate Miss Count**: 필수 항목 누락은 평균에 묻지 않고 별도 경보. "일반 회수율 90%여도 필수 누락 0%"가 라우트 C 핵심 비즈니스 가치.

V4는 criticality 필드 구조만 준비, 정식 측정은 V5.

---

## 6. 평가 스크립트 (eval_route_c.py) 아키텍처

### 흐름
1. golden_dataset_v4.json에서 라우트 C 6문항 로드 (expected_items 있는 것)
2. doc_id로 ex.run() → 추출 항목 (artifact 캐시: 중복 DOC_001 제외 실제 5문서, 약 10분)
3. ★Scope projection (hard filter 아님):
   - raw_items 전체 보존 (ex.run output)
   - Recovery: 모든 raw item을 후보로 matcher가 expected 대응 탐색 (primary_category 하드 필터 안 함 — 모델 분류 오류가 Recovery miss로 둔갑 방지)
   - Extra 분석: 질문 scope + actor/applicability로 in_scope / out_of_scope / review 분리
   - out_of_scope는 삭제 말고 out_of_eval_scope_items로 저장
   - primary_category는 후보 우선순위·보조 단서로만
4. ★Pre-Matcher Soft-Flagging (하드 드롭 금지):
   - "우리원/평가위원회/발주기관" 등 키워드는 정규식으로 flag만 (flagged_actor_relevance=true)
   - 삭제 안 함. matcher에 "발주기관 행동 의심 플래그됨, 입찰자 의무인지 발주기관 절차인지 판정하라" 주의 전달
   - ("제안사는 발주기관 승인을 받아 구축" 같은 입찰자 의무가 키워드로 잘못 드롭되는 False Negative 방지)
5. LLM 판정 매칭(gpt-5-mini) — 프롬프트 역할 분리로 편향 일부 완화(통제 아님; 둘 다 gpt-5-mini라 오류 상관성 잔존):
   - 판정 프롬프트에서 '추출기/RAG' 맥락 제거, NLI 감사관 역할 격리
   - 감사 가능한 짧은 사유(근거 quote + 1~2문장 + decision). 내부 추론 전체 공개(CoT 강제) 안 함
6. ★양방향 순회:
   - Recovery/Taxonomy: expected 기준 순회 (놓친 것)
   - Alignment/Extra: extracted 기준 순회 (군더더기)
   - 단방향 루프 금지 (순회 방향 다름)
7. item_match_mode 분기 (item: 원자 / category: taxonomy 대표성, mapping 파일 참조)
8. 산출: 문항별 + 그룹별(Atomic/Taxonomy) Macro-average + Confirmed/Review/Upper 3값 + Extra 5분류 + Buyer-process Contamination + out_of_eval_scope 리포트

### Matcher 출력 계약 (3값)
```json
{
  "gold_id": "Q019-14",
  "expected_item": "보안서약서(별지7)",
  "candidate_item_index": 12,
  "decision": "match | miss | review",
  "confidence": "high | low",
  "actor_label": "bidder_action | bidder_evaluation_rule | buyer_internal_process | context_or_background",
  "reason": "판정 사유 1~2문장",
  "evidence_quote": "..."
}
```
- match: 추출 item이 expected 핵심 행위·대상·조건 충족 + verified evidence 직접 지지
- miss: 어떤 추출 item도 핵심 의무 미충족
- review: 부분 충족·조건 손실 가능·복수 분할·과도 일반화·evidence 불명확

### Cardinality (unique 규칙)
- 기본 expected ↔ extracted 1:1. 단 하나의 extracted가 여러 expected 핵심 조건을 실제 포함하면 복수 매칭 허용. 하나의 expected가 여러 extracted로 분할되면 합쳐서 required condition 충족하는지 review.
- 분자는 unique set: Recovery=unique expected, Alignment=unique extracted.
- N:M 추적은 Dict[expected_id, List[extracted_id]] 매핑 객체로. pair cache는 1:1 판정만, 최종 assignment는 run-level 집계에서(candidate pool 따라 달라지므로 캐시에 assignment 저장 금지).

### Decision Pair Cache (재현성, 정확성 보증 아님)
1:1 판정만 해시 캐싱. 키(scope/actor policy 바뀌면 자동 무효화):
```
gold_version | match_mode | matcher_model | matcher_prompt_hash | rubric_version |
evaluation_scope_version | actor_policy_version |
expected_id | normalized_extracted_item_hash | normalized_evidence_quote_hash
```

### 안 바뀌는 인프라 (지금 구현)
V4 loader / 6문항 filter / doc_id별 추출 artifact 캐시 / run manifest 저장 / scope projection / Decision Pair Cache / 결과 JSON / report skeleton / fixture test 러너 / V4 taxonomy mapping 로더.

### 아직 고정 안 함 (fixture dry-run 후)
matcher prompt 본문 / Item Precision 최종 공식 / extra 자동 판정 세부 규칙.

---

## 7. fixture dry-run (LLM 핑퐁 대신)

eval_rubric_fixtures.json — DOC_068에서 8~12개 항목을 골라 사람이 정답 라벨 고정. matcher 구현 전 scope/actor/extra 분류 로직을 코드 호출로 검증.

| 항목 | scope | actor/applicability | 기대 판정 |
|------|-------|--------------------|----------|
| 기술평가 90%, 가격평가 10% | scoring | bidder_evaluation_rule | in-scope |
| 기술점수 85% 미만 협상 제외 | scoring | bidder_evaluation_rule | in-scope |
| 평가위원회 구성하여 평가 실시 | requirement | buyer_internal_process | out-of-scope |
| 우선협상대상자 선정 순서(득점자순) | requirement | buyer_internal_process | out-of-scope |
| 로그인 기능 신규 구축 | requirement | bidder_action | in-scope |
| 사용자별 접근 권한 차등 | requirement | bidder_action | in-scope |
| 제안목적·핵심기술 요약 기술 | requirement | bidder_action | in-scope |
| 제한경쟁·협상계약 방식 | requirement | context_or_background | out-of-scope 또는 review |

검증 항목: scope projection이 정상인지, actor relevance가 scoring을 잘못 버리지 않는지, extra가 OutOfScope/Unsupported를 분리하는지, matcher가 애매 사례를 review로 보내는지.

---

## 8. 발표(중간) 표현 가이드

말해도 됨: evidence-first 파이프라인 완료, Pydantic thin/rich 분리, scope 검증·sentinel·completeness gate, 단위테스트 41 PASS, 5문서 실행, Mechanical Quote-Source Match Rate 96~98%(manifest 동반), DOC_068 상세 본문 항목 분해 사례.

말하면 안 됨: 라우트 C Coverage 100%, RFP 100% 추출, Precision 96~98%, Hallucination 수치, 59개 정식 성능, V4 정식 Coverage/Precision.

서사: 청크 hit이 아니라 항목 단위 평가가 필요함을 실측 확인 → V4 진단셋에서 회수·대표성·근거 정합성 점검 → 정식 Coverage·semantic precision은 개발과 분리된 V5 blind로 검증.

---

## 부록 — 외부 LLM 피드백 디버깅 사례 (참고)

초기에 외부 LLM 5건이 청크 hit 저하를 보고 "scope 협소·프롬프트 수정"을 만장일치 권고했으나, raw 출력 대조 결과 추출기는 정확했고 원인은 평가 방식(청크 hit)이었다. 프롬프트/스키마 0줄 변경. → 추측 처방 전 데이터로 원인 확정한 사례.
