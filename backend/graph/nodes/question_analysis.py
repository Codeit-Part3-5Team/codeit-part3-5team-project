"""
question_analysis.py
질문 분석 노드 — history를 참고해 지시어("이 사업", "그 기관" 등)를 구체 명칭으로
해소(query rewriting)한다. followup 질문이 검색 단계에서 무관 청크를 회수하던
문제(평가 보고서 4.4)의 구조적 해법.

알맹이는 검색 측 기구현 함수 rewrite_query()를 호출만 한다.
"""
from backend.retrieval.query_rewriter import rewrite_query  # 검색 측 기구현 함수


def _to_qa_history(history: list[dict]) -> list[dict]:
    """
    대화 이력을 rewrite_query가 기대하는 {"question","answer"} 형식으로 변환한다.

    (결정1) state의 history는 OpenAI 형식 [{"role","content"}]으로 들어오므로,
    rewrite_query 호출 직전에만 이 형식으로 바꿔 넘긴다. 원본 history는 건드리지 않아
    뒤의 answer_generation(OpenAI 형식 사용)이 그대로 쓸 수 있다.
    이미 {"question","answer"} 형식으로 들어와도 그대로 통과시킨다(양쪽 입력 안전).

    Args:
        history: [{"role","content"}, ...] 또는 [{"question","answer"}, ...]
    Returns:
        [{"question","answer"}, ...]
    """
    if not history:
        return []
    # 이미 question/answer 형식이면 변환 없이 사용
    if "question" in history[0]:
        return history
    # OpenAI 형식(role/content) → user/assistant를 2개씩 묶어 question/answer 쌍 구성
    qa = []
    for i in range(0, len(history), 2):
        user_turn = history[i]
        asst_turn = history[i + 1] if i + 1 < len(history) else {"content": ""}
        qa.append({
            "question": user_turn.get("content", ""),
            "answer": asst_turn.get("content", ""),
        })
    return qa


def question_analysis_node(state) -> dict:
    """
    history를 참고해 질문의 지시어를 해소(query rewriting)한다.
    history가 없으면 rewrite_query가 원문을 그대로 반환하므로 비-followup도 안전하다.

    Returns:
        dict: rewritten_question 필드만 채워 반환(LangGraph가 state에 병합)
    """
    question = state["question"]
    history = state.get("history", [])
    qa_history = _to_qa_history(history)            # 결정1: 호출 직전에만 형식 변환
    rewritten = rewrite_query(question, qa_history)  # 지시어 해소된 단독 질문
    return {"rewritten_question": rewritten}