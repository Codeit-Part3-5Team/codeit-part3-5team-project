"""
rag_subgraph.py
LangGraph 기반 V1 Agentic RAG 서브그래프

노드 흐름:
    rewrite → parse → retrieve → grade
                          ↑          |
                          └── retry ─┘ (최대 MAX_RETRY회)

    retry 시 동작:
        1회차: agency/project_name 필터 해제 후 재검색
        2회차(MAX_RETRY 초과): 강제 pass → END

State 필드:
    question           (in)  : 원본 사용자 질문
    history            (in)  : 대화 이력 [{"question": ..., "answer": ...}]
    rewritten_query    (mid) : rewrite 노드 출력
    metadata           (mid) : parse 노드 출력 (필터/정렬 조건)
    documents          (out) : 최종 검색 문서 리스트
    retrieval_attempts (mid) : 재시도 횟수 누적
    grade              (mid) : "pass" | "retry"

사용법:
    from rag_subgraph import run_rag

    docs = run_rag(
        question="입찰 마감일은 언제인가요?",
        history=[{"question": "...", "answer": "..."}],
    )
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from query_rewriter import rewrite_query
from query_parser import extract_metadata
from retriever import get_retriever, load_vectorstore, re_retrieve_fn
from config import MMR_K
from openai import OpenAI

# ── 설정 ─────────────────────────────────────────────────────────────────────

MAX_RETRY    = 2   # grade="retry" 최대 횟수 (초과 시 강제 pass)

_openai_client = OpenAI()
_vectorstore   = None


def _get_vectorstore():
    """벡터스토어 지연 로드 (싱글턴)"""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question:            str            # 원본 사용자 질문
    history:             list[dict]     # 대화 이력
    rewritten_query:     str            # rewrite 노드 출력
    metadata:            dict           # parse 노드 출력
    documents:           list[Document] # 최종 문서
    retrieval_attempts:  int            # 재시도 횟수 (0부터 시작)
    grade:               str            # "pass" | "retry"
    agency_skipped:      bool           # agency 필터 자동 스킵 여부


# ── 노드 ──────────────────────────────────────────────────────────────────────

def rewrite_node(state: RAGState) -> dict:
    """
    followup 질문을 독립 질문으로 재구성합니다.

    history가 없거나 지시어가 없으면 원문 그대로 반환됩니다.
    """
    rewritten = rewrite_query(
        question=state["question"],
        history=state.get("history", []),
    )
    print(f"[rewrite] '{state['question']}' → '{rewritten}'")
    return {"rewritten_query": rewritten}


def parse_node(state: RAGState) -> dict:
    """
    rewritten_query에서 메타데이터 필터/정렬 조건을 추출합니다.

    추출 필드: agency, project_name, budget_min, budget_max,
               date_field, date_min, date_max, sort_by, sort_order
    """
    meta = extract_metadata(state["rewritten_query"])
    print(f"[parse]   필터: {meta}")
    return {"metadata": meta}


def retrieve_node(state: RAGState) -> dict:
    """
    FAISS MMR 검색을 수행합니다.

    재시도(retrieval_attempts > 0)에서는 agency/project_name 필터를 해제해
    검색 범위를 넓힙니다.
    """
    attempts = state.get("retrieval_attempts", 0)
    meta = dict(state.get("metadata", {}))  # 복사 (원본 보존)

    if attempts > 0:
        meta["agency"]       = None
        meta["project_name"] = None
        print(f"[retrieve] 재시도 {attempts}회 — agency/project 필터 해제")

    vs = _get_vectorstore()
    docs, agency_skipped = re_retrieve_fn(
        query=state["rewritten_query"],
        vectorstore=vs,
        agency=meta.get("agency"),
        project_name=meta.get("project_name"),
        budget_min=meta.get("budget_min"),
        budget_max=meta.get("budget_max"),
        date_field=meta.get("date_field"),
        date_min=meta.get("date_min"),
        date_max=meta.get("date_max"),
        sort_by=meta.get("sort_by"),
        sort_order=meta.get("sort_order"),
        k=MMR_K,
    )
    return {
        "documents":          docs,
        "retrieval_attempts": attempts + 1,
        "agency_skipped":     agency_skipped,
    }


# ── Grader 프롬프트 ───────────────────────────────────────────────────────────

_GRADE_SYSTEM = """\
당신은 공공 입찰 RFP 검색 시스템의 문서 관련성 평가자입니다.
주어진 질문과 검색된 문서들을 보고, 질문에 답변하기에 충분한 관련 문서가 있는지 판단하세요.

판단 기준:
- 질문에서 요구하는 정보(예산, 기관명, 사업명, 날짜 등)가 문서에 포함되어 있으면 → "pass"
- 관련 문서가 없거나 전혀 다른 주제의 문서만 검색된 경우 → "retry"

"pass" 또는 "retry" 중 하나만 출력하세요.
"""


def grade_node(state: RAGState) -> dict:
    """
    검색 결과 품질을 LLM으로 평가합니다.

    반환:
        grade="pass"  → END (최종 문서 확정)
        grade="retry" → retrieve 재시도 (필터 해제)

    최대 재시도 횟수(MAX_RETRY) 초과 시 강제로 "pass" 처리합니다.
    비용 절감을 위해 상위 3개 문서만 평가에 사용합니다.
    """
    attempts = state.get("retrieval_attempts", 1)
    docs     = state.get("documents", [])

    # 최대 재시도 초과 → 강제 종료
    if attempts > MAX_RETRY:
        print(f"[grade]   최대 재시도({MAX_RETRY}회) 초과 → 강제 pass")
        return {"grade": "pass"}

    # 문서 없음 → 즉시 retry
    if not docs:
        print("[grade]   문서 없음 → retry")
        return {"grade": "retry"}

    # 상위 3개만 grading (비용 절감)
    docs_text = "\n\n".join(
        f"[문서 {i+1}]\n{doc.page_content[:300]}"
        for i, doc in enumerate(docs[:3])
    )
    user_content = f"질문: {state['rewritten_query']}\n\n{docs_text}"

    response = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _GRADE_SYSTEM},
            {"role": "user",   "content": user_content},
        ],
        max_completion_tokens=10,
    )
    raw   = (response.choices[0].message.content or "").strip().lower()
    grade = "pass" if "pass" in raw else "retry"
    print(f"[grade]   시도 {attempts}회 → LLM='{raw}' → grade={grade}")
    return {"grade": grade}


# ── 조건부 엣지 ───────────────────────────────────────────────────────────────

def _route_after_grade(state: RAGState) -> str:
    """grade 결과에 따라 다음 노드를 결정합니다."""
    return "retrieve" if state.get("grade") == "retry" else END


# ── 서브그래프 빌드 ───────────────────────────────────────────────────────────

def build_rag_subgraph():
    """
    V1 Agentic RAG 서브그래프를 빌드하고 컴파일합니다.

    Returns:
        컴파일된 LangGraph CompiledGraph
    """
    graph = StateGraph(RAGState)

    graph.add_node("rewrite",  rewrite_node)
    graph.add_node("parse",    parse_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade",    grade_node)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite",  "parse")
    graph.add_edge("parse",    "retrieve")
    graph.add_edge("retrieve", "grade")

    graph.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"retrieve": "retrieve", END: END},
    )

    return graph.compile()


# ── 싱글턴 서브그래프 ─────────────────────────────────────────────────────────

rag_subgraph = build_rag_subgraph()


# ── 편의 함수 ─────────────────────────────────────────────────────────────────

def run_rag(
    question: str,
    history: list[dict] | None = None,
) -> list[Document]:
    """
    RAG 서브그래프를 실행하고 최종 검색 문서를 반환합니다.

    Args:
        question : 사용자 질문
        history  : [{"question": ..., "answer": ...}] 형태의 대화 이력

    Returns:
        list[Document]: 최종 검색 문서 (rerank_score 포함)
    """
    initial_state: RAGState = {
        "question":           question,
        "history":            history or [],
        "rewritten_query":    "",
        "metadata":           {},
        "documents":          [],
        "retrieval_attempts": 0,
        "grade":              "",
        "agency_skipped":     False,
    }
    result = rag_subgraph.invoke(initial_state)
    return result["documents"]


# ── 동작 확인 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 테스트 1: followup 질문 (rewrite 필요)
    history = [
        {
            "question": "국민연금공단 IT 인프라 고도화 사업의 예산은 얼마인가요?",
            "answer":   "약 5억 원입니다.",
        }
    ]
    question = "그럼 입찰 마감일은 언제인가요?"

    print("=" * 60)
    print(f"[테스트 1] followup 질문")
    print(f"  질문: {question}")
    print(f"  이력: {history[0]['question'][:40]}...")
    print("=" * 60)

    docs = run_rag(question, history)

    print(f"\n최종 문서: {len(docs)}개")
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        score = m.get("rerank_score", m.get("score", 0.0))
        print(f"\n  [{i}] {m.get('agency_normalized', 'N/A')} | {m.get('project_name', 'N/A')[:30]}")
        print(f"       score={score:.4f} | bid_end={m.get('bid_end', 'N/A')}")
        print(f"       {doc.page_content[:80]}...")
