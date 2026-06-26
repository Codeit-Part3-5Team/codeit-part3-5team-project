"""
reranker.py
Cross-encoder 기반 Reranker

MMR 검색 후보를 query-document 쌍 점수로 재정렬합니다.
모델: cross-encoder/ms-marco-MiniLM-L-6-v2 (경량, 한국어 일부 지원)

사용법:
    from reranker import rerank

    docs = get_retriever(query, vs, k=20)   # 후보 넉넉히
    reranked = rerank(query, docs, top_k=5) # 상위 5개만 반환
"""

from functools import lru_cache
from langchain_core.documents import Document

# 모델은 최초 호출 시 한 번만 로드 (이후 캐시)
@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length=512,
        device = "cpu",
    )
    print("[reranker] Cross-encoder 모델 로드 완료: ms-marco-MiniLM-L-6-v2")
    return model


def rerank(
    query: str,
    docs: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Cross-encoder로 문서를 재점수 매겨 상위 top_k개를 반환합니다.

    Args:
        query : 사용자 질문
        docs  : MMR 검색 후보 Document 리스트
        top_k : 반환할 문서 수

    Returns:
        재정렬된 상위 top_k개 Document 리스트
        (각 doc.metadata["rerank_score"]에 cross-encoder 점수 포함)
    """
    if not docs:
        return docs

    model = _load_model()

    # (query, passage) 쌍 생성
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    # 점수 부착 후 내림차순 정렬
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)

    reranked = sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True)
    result = reranked[:top_k]

    print(f"[reranker] {len(docs)}개 → top {len(result)}개 재정렬 완료")
    return result
