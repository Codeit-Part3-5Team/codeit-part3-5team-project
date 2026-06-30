"""
re_retrieve.py
재검색 노드 — grade가 insufficient로 판정했을 때 호출되는 '다시 검색' 동작.
agentic 재검색 루프의 '재시도' 부분(루프 배선은 build.py의 조건부 엣지가 수행).

[임시 구현 안내]
지금은 기존 검색을 top_k만 늘려 다시 부르는 단순 재검색이다.
'1차와 다른 대안 검색 전략'은 검색 측(재헌님)이 retrieval 폴더에 구현 예정이며,
완성되면 아래 _temp_re_retrieve 호출을 그 함수로 교체한다(슬롯 교체).
교체 시 입출력 계약(아래 시그니처)은 그대로 유지하면 노드 수정 없이 꽂힌다.

재검색 결과 처리: 기존 docs를 '교체'한다(누적 아님).
  - 재검색 전제가 '1차가 부실'이므로, 부실한 1차를 안고 가면 재판정이 또 부실로 빠져 루프가
    안 끝난다. 더 나은 결과로 갈아끼우는 것이 의미에 맞다.
"""
from langchain_core.documents import Document
from backend.pipeline import retrieve   # 기존 검색(임시 재검색이 재활용)


# ── 재검색 슬롯(입출력 계약) ──────────────────────────────────────────────
# 재헌님 대안 검색 함수가 들어올 자리. 이 시그니처를 유지하면 노드 코드 변경 없이 교체된다.
#   re_retrieve_fn(query, prev_docs, attempt, config) -> list[Document]
#     query     : 원본(또는 재구성) 질문
#     prev_docs : 직전 검색에서 회수한 (부실한) Document 리스트 — 보완 검색 시 참고용
#     attempt   : 몇 번째 재검색인지(1, 2...) — 전략을 시도마다 바꿀 때 사용
#     config    : 설정 dict(top_k 등)
#     반환      : list[Document]  ← 기존 retrieve와 동일 형식(그래프에 그대로 꽂힘)
def _temp_re_retrieve(query: str, prev_docs: list, attempt: int, config: dict) -> list[Document]:
    """임시 재검색: top_k를 늘려 기존 검색을 다시 수행한다(대안 전략 도입 전 placeholder)."""
    base_top_k = config.get("top_k", 5)
    # 시도할수록 더 넓게: 2차=2배, 3차=3배 (재헌님 전략 들어오면 이 규칙 자체가 교체됨)
    widened_config = dict(config)
    widened_config["top_k"] = base_top_k * (attempt + 1)
    return retrieve(query, widened_config)


def re_retrieve_node(state) -> dict:
    """
    grade가 insufficient로 판정한 경우, 질문을 다시 검색해 docs를 교체한다.

    Returns:
        dict: docs(재검색 결과로 교체), route_status
    """
    query = state.get("rewritten_question") or state["question"]
    prev_docs = state.get("docs", [])
    attempt = state.get("retry_count", 1)   # grade에서 증가시킨 시도 횟수
    config = state.get("config", {})

    # 슬롯 호출: 지금은 임시(top_k 확대), 추후 재헌님 함수로 교체
    new_docs = _temp_re_retrieve(query, prev_docs, attempt, config)

    status = "ok" if new_docs else "empty"
    return {"docs": new_docs, "route_status": status}