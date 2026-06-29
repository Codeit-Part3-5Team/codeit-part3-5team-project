"""
build.py
RFPilot 오케스트레이션 그래프 조립부.

[1단계] 노드는 전부 스텁(빈 껍데기)이다. 목적은 알맹이 연결 전에
"배선(START→...→END)이 에러 없이 한 바퀴 도는지"를 먼저 확인하는 것.
2단계부터 각 노드를 실제 함수(rewrite_query, retrieve, generate_answer 등)로 채운다.

흐름:
  START → question_analysis → routing →(조건부) route_a/route_c
        → answer_generation → self_check → END
"""
from langgraph.graph import StateGraph, START, END
from backend.graph.state import GraphState


# ===== 노드 스텁 (1단계: 흐름 확인용. 이후 실제 함수로 교체예정) =====

def question_analysis_node(state: GraphState) -> dict:
    # 이후 rewrite_query(+history 변환) 연결예정 — 지금은 원문 통과
    return {"rewritten_question": state["question"]}


def routing_node(state: GraphState) -> dict:
    # 이후 룰 기반 의도 분류 연결예정 — 지금은 무조건 route_a
    return {"route": "route_a"}


def route_a_node(state: GraphState) -> dict:
    # 이후 retrieve(query, config) 연결예정
    return {"docs": [], "route_status": "ok"}


def route_c_node(state: GraphState) -> dict:
    # 이후 generate_checklist(doc_id) 연결예정
    return {"checklist_items": [], "route_status": "ok"}


def answer_generation_node(state: GraphState) -> dict:
    # 이후 generate_answer / render_checklist 연결예정
    return {"answer": "[stub] 답변 자리", "tokens_used": 0}


def self_check_node(state: GraphState) -> dict:
    # 이후 룰 게이트(출처 누락·PII 노출) 연결예정
    return {"check_passed": True, "check_flags": []}


# ===== 조건부 분기 선택자 =====

def _route_selector(state: GraphState) -> str:
    """routing_node가 기록한 route 값을 조건부 엣지로 전달."""
    return state["route"]   # "route_a" 또는 "route_c"


# ===== 그래프 조립 =====

def build_graph():
    """
    오케스트레이션 그래프를 조립해 컴파일된 앱을 반환한다.

    Returns:
        컴파일된 LangGraph 앱(invoke 가능)
    """
    g = StateGraph(GraphState)

    # 노드 등록
    g.add_node("question_analysis", question_analysis_node)
    g.add_node("routing", routing_node)
    g.add_node("route_a", route_a_node)
    g.add_node("route_c", route_c_node)
    g.add_node("answer_generation", answer_generation_node)
    g.add_node("self_check", self_check_node)

    # 순서 엣지
    g.add_edge(START, "question_analysis")
    g.add_edge("question_analysis", "routing")

    # 조건부 분기: routing 결과(route_a/route_c)로 갈림
    g.add_conditional_edges(
        "routing",
        _route_selector,
        {"route_a": "route_a", "route_c": "route_c"},
    )

    # 두 라우트 모두 답변 생성으로 합류
    g.add_edge("route_a", "answer_generation")
    g.add_edge("route_c", "answer_generation")

    # 답변 생성 → 검증 → 종료
    g.add_edge("answer_generation", "self_check")
    g.add_edge("self_check", END)

    return g.compile()


# 직접 실행 시 흐름 확인 (1단계 검증)
# 실행: (루트에서) python -m backend.graph.build
if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({
        "question": "테스트 질문입니다",
        "history": [],
        "config": {"top_k": 5},
    })
    print("=== invoke 성공: 그래프가 끝까지 흘렀습니다 ===")
    for k, v in result.items():
        print(f"  {k}: {v}")