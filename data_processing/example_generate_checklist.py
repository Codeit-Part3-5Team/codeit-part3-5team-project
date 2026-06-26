"""
라우트 C 추출기 연동 예시 (재헌님 generate_checklist 참고용)
- selected_doc_id 하나만 넘기면 체크리스트 항목 리스트가 나옵니다.
- 검증/중복제거는 추출기가 이미 끝낸 상태라, 꺼내서 형태만 맞추면 됩니다.
"""
from data_processing.compliance_extractor_v2 import ComplianceExtractorV2
from openai import OpenAI


def generate_checklist(doc_id: str, client=None, config=None) -> dict:
    """
    사용자가 고른 문서(doc_id)에서 입찰 준수 체크리스트를 추출해 반환.

    Returns:
        {
          "items": [ {text, category, evidence, flags}, ... ],
          "doc_id": str,
          "status": str,            # complete_verified / complete_with_review / partial / failed / doc_not_found
          "extracted_count": int,
        }
    """
    if client is None:
        client = OpenAI(timeout=180.0, max_retries=2)
    ex = ComplianceExtractorV2(max_workers=6)

    result = ex.run(doc_id, use_mock=False, client=client)

    # 문서 없음 처리
    if result.get("status") == "doc_not_found":
        return {"items": [], "doc_id": doc_id, "status": "doc_not_found", "extracted_count": 0}

    manifest = result["manifest"]
    items = result["items"]   # List[AuditChecklistItem], 검증·dedupe 완료

    checklist = []
    for it in items:
        checklist.append({
            "text": it.item,                       # 요구사항 내용
            "category": it.primary_category,       # requirement/qualification/submission/scoring
            "evidence": [
                {"chunk_id": ev.declared_chunk_id, "quote": ev.quote, "status": ev.match_status}
                for ev in it.evidence
            ],
            "flags": {                             # 검토 우선순위용 (빠졌을 수 있는 항목)
                "condition": it.condition_loss_risk,
                "numeric": it.numeric_mismatch_risk,
                "deadline": it.deadline_loss_risk,
            },
        })

    return {
        "items": checklist,
        "doc_id": doc_id,
        "status": manifest.status,
        "extracted_count": len(checklist),
    }


if __name__ == "__main__":
    # 사용 예시
    out = generate_checklist("DOC_001")
    print(f"status: {out['status']}, 추출 {out['extracted_count']}건")
    for c in out["items"][:3]:
        print(f"  [{c['category']}] {c['text'][:60]}")
