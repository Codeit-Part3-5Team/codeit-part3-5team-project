"""
embedder.py
OpenAI 임베딩 모델 로드 및 캐싱

모델: text-embedding-3-small (1536차원)
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from config import EMBEDDING_MODEL

load_dotenv()

def get_embeddings() -> OpenAIEmbeddings:
    """
    OpenAI 임베딩 모델 반환 (캐싱으로 반복 초기화 방지)

    Returns:
            OpenAIEmbeddings: text-embedding-3-small 모델
    """
    return OpenAIEmbeddings(
        model = EMBEDDING_MODEL,
        openai_api_key = os.getenv("OPENAI_API_KEY")
    )

# 모듈 라벨 캐싱 (같은 세션 내 재사용)
_embeddings = None

def get_cached_embeddings() -> OpenAIEmbeddings:
    """
    임베딩 모델을 캐싱하여 반환 (API 중복 호출 방지)

    Returns:
        OpenAIEmbeddings: 캐싱된 임베딩 모델
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


if __name__ == "__main__":
    embeddings = get_cached_embeddings()
    test = embeddings.embed_query("국민연금공단 예산이 얼마야?")
    print(f"임베딩 모델: {EMBEDDING_MODEL}")
    print(f"벡터 차원: {len(test)}")
    print("embedder.py 정상 동작!")