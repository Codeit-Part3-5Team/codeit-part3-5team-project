"""
route_a.py
라우트 A 노드 — 재구성 질문으로 검색해 청크(Document)를 회수한다(단순 RAG 경로).
생성은 공통 노드(answer_generation)에서 하므로, 여기서는 검색만 수행한다.

알맹이는 통합 파이프라인의 retrieve()를 호출만 한다.
"""
from backend.pipeline import retrieve   # 검색 흡수 경로(기존 통합 함수)


def route_a_node(state) -> dict:
    """
    재구성 질문으로 FAISS 검색을 수행해 Document를 회수한다.

    - 재구성 질문(rewritten_question)으로 검색해야 followup이 맞는 청크를 회수한다.
    - (결정2) Document를 통째로 state.docs에 싣고, str·dict 변환은 최종 get_ai_response에서.

    Returns:
        dict: docs(list[Document]), route_status
    """
    # 재구성 질문 우선, 없으면 원문
    query = state.get("rewritten_question") or state["question"]
    config = state.get("config", {})
    # retrieve(query, config): top_k 등은 config 안에서 꺼내 씀(시그니처 보정)
    docs = retrieve(query, config)
    status = "ok" if docs else "empty"
    return {"docs": docs, "route_status": status}