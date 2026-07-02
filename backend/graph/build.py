"""
build.py
RFPilot 오케스트레이션 그래프 조립부.

각 노드는 실제 함수(rewrite_query, retrieve, generate_answer 등)에 연결돼 있다.
route_a(검색) 경로에는 grade(검색결과 평가) → re_retrieve(재검색) 루프가 걸려,
검색이 부실하면 스스로 다시 검색하는 agentic 동작을 한다(최대 2회).

흐름:
  START → question_analysis → routing →(조건부) route_a / route_c
  route_a → grade ─┬ sufficient/out_of_scope → answer_generation
                   └ insufficient(재시도 여유) → re_retrieve → grade ↺
  route_c → answer_generation
  answer_generation → self_check → END
"""
from dotenv import load_dotenv
load_dotenv()   # .env의 OPENAI_API_KEY 등을 환경변수로 로드 (직접 실행 진입점이라 명시적으로 읽음)

from langgraph.graph import StateGraph, START, END
from backend.graph.state import GraphState
from backend.graph.nodes.question_analysis import question_analysis_node
from backend.graph.nodes.routing import routing_node
from backend.graph.nodes.route_a import route_a_node
from backend.graph.nodes.route_c import route_c_node
from backend.graph.nodes.grade import grade_node                # 검색결과 평가(재검색 루프 판단)
from backend.graph.nodes.re_retrieve import re_retrieve_node    # 재검색(루프 동작)
from backend.graph.nodes.answer_generation import answer_generation_node
from backend.graph.nodes.self_check import self_check_node

# 재검색 최대 횟수. grade가 insufficient여도 이 한도를 넘으면 더 돌지 않고 생성으로 나간다.
_MAX_RETRY = 2

# ===== 조건부 분기 선택자 =====

def _route_selector(state: GraphState) -> str:
    """routing_node가 기록한 route 값을 조건부 엣지로 전달."""
    return state["route"]   # "route_a" 또는 "route_c"


def _grade_selector(state: GraphState) -> str:
    """
    grade 판정 + 재시도 한도로 다음 행선지를 정한다.
      - insufficient & 재시도 여유 있음 → re_retrieve(루프)
      - 그 외(sufficient / out_of_scope / 한도 초과) → answer_generation
    """
    grade = state.get("grade", "sufficient")
    retry_count = state.get("retry_count", 0)
    if grade == "insufficient" and retry_count < _MAX_RETRY:
        return "re_retrieve"
    return "answer_generation"


def _mode_selector(state: GraphState) -> str:
    """
    retriever_type으로 route_a 다음 행선지를 정한다(naive/agentic 토글).
      - agentic_rag → grade(재검색 루프 수행)
      - naive_rag(기본) → grade 건너뛰고 바로 answer_generation
    config는 state로 주입된다(그래프 컴파일은 인자 없이 하므로 state에서 읽음).
    """
    config = state.get("config", {})
    if config.get("retriever_type") == "agentic_rag":
        return "grade"
    return "answer_generation"


# ===== 그래프 조립 =====


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
    g.add_node("grade", grade_node)                # 검색결과 평가(route_a 전용)
    g.add_node("re_retrieve", re_retrieve_node)    # 재검색(루프 동작)
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

    # route_a(검색) 다음: retriever_type에 따라 분기(naive/agentic 토글).
    #   agentic_rag → grade(재검색 루프)로. naive_rag → grade 건너뛰고 바로 생성.
    g.add_conditional_edges(
        "route_a",
        _mode_selector,
        {"grade": "grade", "answer_generation": "answer_generation"},
    )
    # grade 판정으로 분기: 부실하고 재시도 여유 있으면 재검색 루프, 아니면 생성
    g.add_conditional_edges(
        "grade",
        _grade_selector,
        {"re_retrieve": "re_retrieve", "answer_generation": "answer_generation"},
    )

    # 재검색 후 다시 grade로 돌아가 재판정(루프백). retry_count 한도로 무한루프 차단.
    g.add_edge("re_retrieve", "grade")

    # route_c는 grade 없이 바로 답변 생성으로 합류
    g.add_edge("route_c", "answer_generation")

    # 답변 생성 → 검증 → 종료
    g.add_edge("answer_generation", "self_check")
    g.add_edge("self_check", END)

    return g.compile()


# 직접 실행 시 흐름 확인
# 실행: (루트에서) python -m backend.graph.build
if __name__ == "__main__":
    app = build_graph()

    # followup rewriting + route_a 검색까지 가볍게 확인 (route_c 실제 추출은 호출 안 함)
    result = app.invoke({
        "question": "그럼 이 사업 입찰 마감일은 언제야?",
        "history": [
            {"role": "user", "content": "한국수자원공사 건설통합시스템 고도화 사업 예산이 얼마야?"},
            {"role": "assistant", "content": "5억원입니다"},
        ],
        "config": {"top_k": 5},
    })
    print("원본 질문    :", result["question"])
    print("재구성 질문  :", result["rewritten_question"])
    print("route        :", result["route"])
    print("grade        :", result.get("grade"), "| retry_count:", result.get("retry_count"))
    print("docs 개수    :", len(result.get("docs", [])))
    print("answer       :", result["answer"][:120])
    print("check_passed :", result["check_passed"], "| flags:", result["check_flags"])