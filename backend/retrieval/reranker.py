"""
reranker.py
Flashrank 기반 Reranker (ONNX, PyTorch 불필요)

MMR 검색 후보를 query-document 쌍 점수로 재정렬합니다.
모델: ms-marco-MiniLM-L-12-v2 (flashrank 기본 모델)

사용법:
    from reranker import rerank

    docs = get_retriever(query, vs, k=20)   # 후보 넉넉히
    reranked = rerank(query, docs, top_k=5) # 상위 5개만 반환
"""

from functools import lru_cache
from langchain_core.documents import Document


@lru_cache(maxsize=1)
def _load_ranker():
    from flashrank import Ranker
    ranker = Ranker()
    print("[reranker] Flashrank 모델 로드 완료")
    return ranker


def rerank(
    query: str,
    docs: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Flashrank로 문서를 재점수 매겨 상위 top_k개를 반환합니다.

    Args:
        query : 사용자 질문
        docs  : MMR 검색 후보 Document 리스트
        top_k : 반환할 문서 수

    Returns:
        재정렬된 상위 top_k개 Document 리스트
        (각 doc.metadata["rerank_score"]에 재정렬 점수 포함)
    """
    if not docs:
        return docs

    from flashrank import RerankRequest

    ranker = _load_ranker()

    passages = [
        {"id": i, "text": doc.page_content}
        for i, doc in enumerate(docs)
    ]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    # 점수 기준 내림차순 정렬 후 top_k 반환
    scored = sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]

    reranked_docs = []
    for r in scored:
        doc = docs[r["id"]]
        doc.metadata["rerank_score"] = float(r["score"])
        reranked_docs.append(doc)

    print(f"[reranker] {len(docs)}개 → top {len(reranked_docs)}개 재정렬 완료")
    return reranked_docs
