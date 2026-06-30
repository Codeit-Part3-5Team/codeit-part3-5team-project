# 라우트 C 인계 메모 (업데이트: 키 연결 + 실측 완료)

## 현재 상태 (6/25 마감)
- 브랜치: feat/data-compliance-extractor (origin, 커밋 9f75abe)
- v2 파이프라인 완성 + gpt-5-mini 실측 연결 완료
- 단위테스트 41 PASS (verify8/dedupe6/sentinel6/gate6/v1 15)
- 보드 #65 In Progress

## 키
- 팀 OpenAI 키 .env에 설정됨 (OPENAI_API_KEY, BOM 없이 utf-8)
- 개인키 금지 — 팀 $20 한도/!usage 관리. .env는 gitignore됨
- gpt-5-mini는 temperature=1 고정(기본값), reasoning 모델

## 실측 결과 (DOC_001)
- 골든셋 정답 청크 100% hit (5,6,7,8,9,10)
- 검증율 97.5% (strict373/relocated10/cross1/unverified10)
- 문서당 ~2분 (16윈도우 병렬 max_workers=6)
- 남은 unverified 2.5% = 진짜 표현차이 → selective repair 영역

## gpt-5-mini 실전 제약 (수민님과 공통 이슈)
- reasoning 모델이라 느림 (윈도우당 30~90초) → 병렬 필수
- reasoning 토큰이 max를 먹어 출력잘림(LengthError) → max_completion_tokens=16000 + 방어
- reasoning_effort=low (minimal은 과추출)
- 100문서 전체는 비현실적(18시간) → 골든셋 5문서 샘플 검증

## 핵심 기술 결정 (실측으로 확정)
- 윈도우: 청크 8개 기준 (토큰16k는 타임아웃)
- 검증율 개선: PDF 추출 띄어쓰기 노이즈가 주범 → normalize 공백 완전제거 (75%→97.5%)
  ※ NLI 불필요했음 — 데이터 까보니 기호/공백 문제였음 (트렌드: mechanical fix 우선)
- relocated: ±1 → allowed 범위 전체 (scope 유지)
- LLM 노이즈 측정: raw 출력 고정 후 A/B 비교 (data/raw_llm_cache, gitignore)

## 다음 (6/26)
1. 골든셋 5문서 마저 실측 + 캐시 저장
2. selective repair (unverified/flag 항목만 재검증, item생성 금지)
3. 결과 artifact 저장 (캐시 — 재요청 0초, input_manifest_hash로 stale 무효화)
4. 팀미팅: 키 공유완료 보고 / #62 골든셋 파일정책(나 지목) / V4 시나리오(아인) / 재헌 인터페이스
5. PR 열기 (실측+개선 담아서)

## 인터페이스 (재헌님 연동)
- ex.run(doc_id, use_mock=False, client=client) → {manifest, items, raw_items, deduped_items}
- manifest: status(complete_verified/complete_with_review/partial/failed), windows, input_manifest_hash
- PM 제시 generate_checklist(query, doc_id, config) → {items, doc_id, extracted_count, validation_summary}와 거의 일치

## 절대 금지
- 청킹/FAISS 인덱스 변경
- NDA: 원문/골든셋/raw_llm_cache git 커밋
- 개인 OpenAI 키 사용
