"""
multi_query.py
Multi-Query Retrieval

하나의 질문을 LLM으로 여러 관점의 질문으로 재표현한 뒤,
각각 검색하고 결과를 합산합니다 (중복 제거, 점수 기반 정렬).

사용법:
    from multi_query import multi_query_retrieve

    results = multi_query_retrieve(query, vectorstore, n_queries=3, k=5)
"""

import json
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

_SYSTEM_PROMPT = """\
당신은 공공 입찰 RFP 검색 시스템의 쿼리 확장기입니다.
사용자 질문을 다양한 관점으로 재표현한 질문 {n}개를 생성하세요.

규칙:
- 원래 질문의 의도를 유지하면서 다른 표현·관점으로 재표현하세요.
- 각 질문은 독립적으로 검색에 사용됩니다.
- JSON 배열로만 출력하세요. 예: ["질문1", "질문2", "질문3"]
- 설명 없이 JSON만 출력하세요.
"""


def generate_queries(question: str, n_queries: int = 3) -> list[str]:
    """
    LLM으로 원래 질문을 n개의 다양한 표현으로 재생성합니다.

    Args:
        question : 원래 사용자 질문
        n_queries: 생성할 질문 수

    Returns:
        재표현된 질문 리스트 (원래 질문 포함)
    """
    prompt = _SYSTEM_PROMPT.format(n=n_queries)
    user_content = f"질문: {question}"

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=4000,
    )

    raw = (response.choices[0].message.content or "").strip()

    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        queries = json.loads(raw)
        if not isinstance(queries, list):
            raise ValueError("리스트가 아님")
        # 원래 질문을 맨 앞에 포함
        all_queries = [question] + [q for q in queries if q != question]
        print(f"[multi_query] 생성된 쿼리 {len(all_queries)}개: {all_queries}")
        return all_queries
    except (json.JSONDecodeError, ValueError):
        print(f"[multi_query] 쿼리 생성 실패 — 원래 질문만 사용: {raw!r}")
        return [question]


def _rrf_merge(
    ranked_lists: list[list[Document]],
    k: int = 5,
    rrf_k: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion으로 여러 ranked list를 합산합니다.

    Args:
        ranked_lists : 각 쿼리의 검색 결과 리스트 (순위 있음)
        k            : 최종 반환 문서 수
        rrf_k        : RRF 상수 (기본값 60)

    Returns:
        RRF 점수 기준 상위 k개 Document 리스트
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc.metadata.get("doc_id", "")
            if not doc_id:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            doc_map[doc_id] = doc

    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:k]
    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        doc.metadata["rrf_score"] = rrf_scores[doc_id]
        results.append(doc)
    return results


def multi_query_retrieve(
    query: str,
    vectorstore: FAISS,
    agency: str | None = None,
    project_name: str | None = None,
    n_queries: int = 3,
    k: int = 5,
    fetch_k: int = 100,
    lambda_mult: float = 0.95,
) -> list[Document]:
    """
    Multi-Query Retrieval: 여러 쿼리로 검색 후 RRF로 합산.

    Args:
        query        : 원래 사용자 질문
        vectorstore  : FAISS 벡터스토어
        agency       : 기관명 필터 (선택)
        project_name : 사업명 필터 (선택)
        n_queries    : 생성할 추가 쿼리 수
        k            : 최종 반환 문서 수
        fetch_k      : MMR 후보 풀 크기
        lambda_mult  : MMR 관련성 가중치

    Returns:
        RRF 합산 후 상위 k개 Document 리스트
    """
    from retriever import get_retriever

    queries = generate_queries(query, n_queries=n_queries)

    ranked_lists = []
    for q in queries:
        results = get_retriever(
            q, vectorstore,
            agency=agency,
            project_name=project_name,
            k=fetch_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        ranked_lists.append(results)

    merged = _rrf_merge(ranked_lists, k=k)
    print(f"[multi_query] {len(queries)}개 쿼리 RRF 합산 → top {len(merged)}개 반환")
    return merged
