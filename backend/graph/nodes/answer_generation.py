"""
answer_generation.py
답변 생성 노드 — route_a의 청크든 route_c의 체크리스트든 받아 최종 답변을 만든다.
두 라우트의 합류 지점.

- route_a: 검색 Document 기반 RAG 답변(generate_answer, gpt-5-mini / 시나리오A면 Ollama)
- route_c: 추출 체크리스트를 평문 리스트로 렌더(render_checklist, MVP는 단순 렌더)
- 라우트 실행이 비정상(문서없음·미선택 등)이면 안내 문구로 단락 처리.

알맹이: generate_answer(기존 생성 함수) + render_checklist(C용, 본 노드에서 신설).
"""
from backend.generation.generator import generate_answer  # 기존 생성 함수

# route_c 정상 status (항목이 있는 경우). 그 외는 안내 문구로 단락 처리.
_ROUTE_C_OK = {"complete_verified", "complete_with_review", "partial"}


def render_checklist(items: list[dict]) -> str:
    """
    체크리스트 항목 리스트를 평문 답변으로 렌더한다(MVP: 단순 번호 나열).

    Args:
        items: [{text, category, evidence, flags}, ...]
    Returns:
        사람이 읽는 평문 체크리스트 문자열
    """
    if not items:
        return "해당 문서에서 추출된 체크리스트 항목이 없습니다."
    lines = ["다음은 해당 문서에서 추출한 준수 체크리스트입니다.\n"]
    for i, it in enumerate(items, 1):
        cat = it.get("category", "")
        text = it.get("text", "")
        lines.append(f"{i}. [{cat}] {text}")
    return "\n".join(lines)


def _status_message(status: str) -> str:
    """라우트 비정상 상태를 사용자 안내 문구로 변환."""
    return {
        "no_doc_selected": "체크리스트를 만들 문서를 먼저 선택해 주세요.",
        "doc_not_found": "선택하신 문서를 찾을 수 없습니다.",
        "failed": "체크리스트 추출에 실패했습니다. 잠시 후 다시 시도해 주세요.",
        "empty": "해당 문서에서 추출된 항목이 없습니다.",
    }.get(status, "처리 중 문제가 발생했습니다.")


def answer_generation_node(state) -> dict:
    """
    라우트 결과를 받아 최종 답변을 생성한다(두 라우트의 합류 지점).

    Returns:
        dict: answer, tokens_used
    """
    route = state["route"]
    config = state.get("config", {})

    if route == "route_a":
        # 검색 청크(Document) 기반 RAG 답변. 재구성 질문으로 답변(followup 일관성).
        query = state.get("rewritten_question") or state["question"]
        docs = state.get("docs", [])
        answer, tokens_used = generate_answer(
            query,
            docs,
            state.get("history", []),                       # 원본 history(OpenAI 형식) 그대로
            use_ollama=state.get("use_ollama", False),
            ollama_model=config.get("ollama_model", "llama3.2"),
            prompt_version=config.get("prompt_version", "system_v2"),
        )
        return {"answer": answer, "tokens_used": tokens_used}

    # route_c: 정상 status면 체크리스트 렌더, 아니면 안내 문구
    status = state.get("route_status", "failed")
    if status in _ROUTE_C_OK:
        answer = render_checklist(state.get("checklist_items", []))
    else:
        answer = _status_message(status)
    # route_c는 추출 단계에서 토큰을 쓰고, 본 노드는 생성 호출이 없으므로 0
    return {"answer": answer, "tokens_used": 0}