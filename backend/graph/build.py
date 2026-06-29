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
from dotenv import load_dotenv
load_dotenv()   # .env의 OPENAI_API_KEY 등을 환경변수로 로드 (직접 실행 진입점이라 명시적으로 읽음)

from langgraph.graph import StateGraph, START, END
from backend.graph.state import GraphState
# 2단계: 실제 노드 연결 (question_analysis, routing)
from backend.graph.nodes.question_analysis import question_analysis_node
from backend.graph.nodes.routing import routing_node
# 3단계: route_a 실제 연결 (retrieve)
from backend.graph.nodes.route_a import route_a_node

# ===== 노드 스텁 (1단계: 흐름 확인용. 이후 실제 함수로 교체예정) =====

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


# 직접 실행 시 흐름 확인
# 실행: (루트에서) python -m backend.graph.build
if __name__ == "__main__":
    app = build_graph()

    # followup 테스트: "이 사업"이 앞 대화의 실제 사업명으로 풀리는지 확인
    result = app.invoke({
        "question": "그럼 이 사업 입찰 마감일은 언제야?",
        "history": [
            {"role": "user", "content": "한국수자원공사 건설통합시스템 고도화 사업 예산이 얼마야?"},
            {"role": "assistant", "content": "5억원입니다"},
        ],
        "config": {"top_k": 5},
    })
    print("=== followup rewriting 테스트 ===")
    print("원본 질문    :", result["question"])
    print("재구성 질문  :", result["rewritten_question"])   # ← 여기가 핵심
    print("route        :", result["route"])
    print("docs 개수    :", len(result["docs"]))   # route_a가 검색한 청크 수