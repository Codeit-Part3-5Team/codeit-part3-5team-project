"""
route_c.py
라우트 C 노드 — 선택된 문서(selected_doc_id)에서 의무 요구사항을 추출해 체크리스트 생성.
'검색'이 아니라 '문서 전체 추출'이라는 점이 라우트 A와 다르다(질문이 아니라 doc_id로 동작).

알맹이는 추출기 generate_checklist(doc_id)를 호출만 한다.
주의: generate_checklist는 gpt-5-mini를 호출해 문서당 수 분이 걸리는 무거운 작업이다.
"""
from data_processing.example_generate_checklist import generate_checklist


def route_c_node(state) -> dict:
    """
    선택된 문서에서 의무 요구사항을 추출해 체크리스트를 생성한다(Compliance Validator).

    - selected_doc_id가 없으면 추출기를 부르지 않고 no_doc_selected로 단락한다
      (generate_checklist가 무거운 LLM 호출이라 불필요한 실행을 미리 막음).
    - 추출기 status(complete_verified / complete_with_review / partial / failed /
      doc_not_found)를 그대로 route_status에 실어, answer_generation이 분기하게 한다.

    Returns:
        dict: checklist_items, route_status
    """
    doc_id = state.get("selected_doc_id")
    # 문서 미선택: 추출기 호출 없이 단락
    if not doc_id:
        return {"checklist_items": [], "route_status": "no_doc_selected"}

    result = generate_checklist(doc_id)          # 도혁님 추출기(문서당 수 분 소요)
    return {
        "checklist_items": result.get("items", []),
        "route_status": result.get("status", "failed"),
    }