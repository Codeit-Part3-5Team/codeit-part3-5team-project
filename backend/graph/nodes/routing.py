"""
routing.py
1단 라우팅 노드 — 재구성 질문의 의도를 룰 기반으로 분류해 route_a / route_c 결정.
- 체크리스트/요구사항 전수/제출서류 목록 의도 → route_c
- 그 외 단순 조회 → route_a

이 노드는 route 문자열만 기록한다. 실제 분기는 build.py의 조건부 엣지가 수행한다.
"""

# 라우트 C(체크리스트 생성) 의도를 나타내는 키워드 (룰 기반 분류)
_ROUTE_C_KEYWORDS = [
    "체크리스트", "요구사항 전부", "요구사항 목록", "제출서류 목록",
    "필수 항목", "준비물", "빠짐없이", "전체 목록", "제출 서류", "준수 사항",
]


def routing_node(state) -> dict:
    """
    재구성 질문의 의도를 룰 기반으로 분류해 라우트를 결정한다.

    Returns:
        dict: route 필드("route_a" 또는 "route_c")
    """
    # 재구성 질문 우선, 없으면 원문 사용
    q = state.get("rewritten_question") or state["question"]
    # 의도 키워드가 하나라도 걸리면 라우트 C
    if any(kw in q for kw in _ROUTE_C_KEYWORDS):
        return {"route": "route_c"}
    return {"route": "route_a"}