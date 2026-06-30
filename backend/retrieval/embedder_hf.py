"""
embedder_hf.py
HuggingFace 임베딩 모델 로드 및 캐싱

모델: BAAI/bge-m3 (1024차원, 한국어 포함 다국어 지원)

기존 embedder.py(OpenAI text-embedding-3-small)와 인터페이스 동일.
FAISS 인덱스를 새로 빌드할 때 이 파일을 사용하세요.

⚠️  인덱스는 임베딩 모델에 종속됩니다.
    embedder_hf.py로 빌드한 인덱스는 embedder.py와 혼용 불가.
"""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "BAAI/bge-m3"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    BAAI/bge-m3 임베딩 모델 반환 (lru_cache로 반복 초기화 방지)

    Returns:
        HuggingFaceEmbeddings: bge-m3 임베딩 모델
    """
    print(f"[embedder_hf] {MODEL_NAME} 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # bge 계열은 normalize 권장
    )
    print(f"[embedder_hf] {MODEL_NAME} 로드 완료")
    return embeddings


def get_cached_embeddings() -> HuggingFaceEmbeddings:
    """
    임베딩 모델을 캐싱하여 반환 (기존 embedder.py와 동일한 인터페이스)

    Returns:
        HuggingFaceEmbeddings: 캐싱된 bge-m3 임베딩 모델
    """
    return get_embeddings()


if __name__ == "__main__":
    embeddings = get_cached_embeddings()
    test = embeddings.embed_query("국민연금공단 예산이 얼마야?")
    print(f"임베딩 모델: {MODEL_NAME}")
    print(f"벡터 차원: {len(test)}")
    print("embedder_hf.py 정상 동작!")
