"""
reranker.py
BGE Reranker 기반 Reranker (한국어 지원)

MMR 검색 후보를 query-document 쌍 점수로 재정렬합니다.
모델: BAAI/bge-reranker-large (한국어 포함 다국어 지원)

사용법:
    from reranker import rerank

    docs = get_retriever(query, vs, k=20)   # 후보 넉넉히
    reranked = rerank(query, docs, top_k=5) # 상위 5개만 반환
"""

from functools import lru_cache
from langchain_core.documents import Document

MODEL_NAME = "BAAI/bge-reranker-large"


@lru_cache(maxsize=1)
def _load_ranker():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(MODEL_NAME)
    print(f"[reranker] {MODEL_NAME} 모델 로드 완료")
    return model


def rerank(
    query: str,
    docs: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    BGE CrossEncoder로 문서를 재점수 매겨 상위 top_k개를 반환합니다.

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

    model = _load_ranker()

    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    reranked_docs = []
    for idx, score in scored:
        doc = docs[idx]
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    print(f"[reranker] {len(docs)}개 → top {len(reranked_docs)}개 재정렬 완료")
    return reranked_docs
