"""
pipeline.py
RAG 파이프라인 메인 조립부.
검색(retrieval) → 대화이력 정리 → 답변 생성을 하나로 묶어 get_ai_response로 제공한다.

핵심 함수:
    get_ai_response(query, history, config, ...) : 외부 인터페이스(서비스/평가 진입점)
    retrieve(query, config)                      : retriever_type에 따라 검색 수행
"""
from utils.config import load_config  # config.yaml 읽기
# 지금까지 만든 모듈들을 하나로 조립하는 메인 파이프라인
from backend.generation.generator import generate_answer  # 답변 생성
from backend.generation.memory import trim_history        # 대화 이력 자르기
from langchain_core.documents import Document
import os
import sys
import time

# retrieval 모듈은 로컬 import(from embedder, from config 등)를 전제로 작성돼 있어서,
# 해당 폴더를 import 경로에 추가해 retrieval 코드 수정 없이 그대로 불러온다.
# (재헌님 retrieval 코드는 건드리지 않고 생성 측에서 경로로 흡수)
_RETRIEVAL_DIR = os.path.join(os.path.dirname(__file__), "retrieval")
if _RETRIEVAL_DIR not in sys.path:
    sys.path.insert(0, _RETRIEVAL_DIR)

# retrieval 진입점: 벡터스토어 로드 + 검색 함수
from retriever import load_vectorstore, get_retriever

# 임시 검색 함수 (mock)
# retrieval이 아직 없어서, 실제 검색결과(get_retriever)와 같은 v2 청크 구조를
# 흉내낸 Document 리스트를 가짜로 반환한다. metadata 형식이 실제와 같으므로,
# 이후 이 함수 호출만 실제 retriever로 교체하면 출처/형식이 그대로 맞물린다.
def mock_retrieve(query: str, top_k: int = 5) -> list[Document]:
    return [
        Document(
            page_content="[사업개요] 사업명: 국민연금공단 이러닝시스템 고도화\n사업금액(예산): 540,000,000원\n사업기간: 계약 체결일로부터 6개월",
            metadata={
                "doc_id": "DOC_001",
                "file_name": "국민연금공단_이러닝시스템 고도화.hwp",   # 출처 표기에 사용
                "section": "메타요약",                                  # 출처 표기에 사용
                "content_type": "meta_summary",
                "page": None,                  # hwp라 페이지 개념 없음 → 출처에 미사용
                "budget_amount": 540000000,
                "agency_normalized": "국민연금공단",
                "score": 0.92,                 # retriever가 부착하는 유사도(0~1)
            },
        ),
        Document(
            page_content="사업 수행 기간은 계약 체결일로부터 6개월로 한다.",
            metadata={
                "doc_id": "DOC_001",
                "file_name": "국민연금공단_이러닝시스템 고도화.hwp",
                "section": "Ⅱ. 사업개요 — 사업기간",
                "content_type": "text",
                "page": None,
                "score": 0.85,
            },
        ),
    ]


# 벡터스토어(FAISS)는 로드 비용이 크므로 모듈 레벨에서 1회만 로드해 재사용한다.
# (요청마다 load_vectorstore 호출하면 인덱스+임베딩 재초기화로 느려짐)
_vectorstore = None

def get_vectorstore():
    """
    FAISS 벡터스토어를 1회만 로드해 캐싱 후 반환한다.

    Returns:
        FAISS: 로드된 벡터스토어 (retrieval 모듈의 load_vectorstore 사용)
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()  # data/faiss_index에서 로드
    return _vectorstore



# retriever_type에 따라 검색 방식 선택
def retrieve(query: str, config: dict) -> list[Document]:
    """
    retriever_type에 따라 검색을 수행한다.

    Args:
        query : 사용자 질문
        config: 설정 dict (retriever_type, top_k 등)

    Returns:
        list[Document]: 검색된 청크 리스트 (metadata에 score 포함)
    """
    retriever_type = config.get("retriever_type", "naive_rag")
    top_k = config.get("top_k", 5)

    vs = get_vectorstore()  # 캐싱된 벡터스토어 사용

    if retriever_type == "agentic_rag":
        # agentic 미구현 — 현재는 naive와 동일하게 MMR 검색 사용 (이후 agentic 오면 교체예정)
        return get_retriever(query, vs, k=top_k)
    else:  # naive_rag (기본)
        return get_retriever(query, vs, k=top_k)


# use_ollama: True면 시나리오 A(Ollama), False면 B(gpt). 평가에서 모델 비교용
def get_ai_response(query: str, history: list[dict] = None, config: dict = None,
                    max_history: int = None, use_ollama: bool = False) -> dict:
    start = time.time()
    history = history or []
    config = config or load_config()   # config 없으면 config.yaml에서 로드

    # 1) 대화 이력 자르기
    #    우선순위: 프론트가 보낸 max_history > config 값 > 기본 10
    if max_history is None:
        max_history = config.get("max_history", 10)
    trimmed_history = trim_history(history, max_history)

    # 2) 검색 (지금은 mock, 이후 retriever로 교체)
    #    config로 naive/agentic 분기는 추후 추가 예정 (지금은 자리만)
    docs = retrieve(query, config)     # retriever_type에 따라 분기

    # 3) 답변 생성 — use_ollama로 시나리오 A/B 선택, Ollama 모델은 config에서 읽음
    ollama_model = config.get("ollama_model", "llama3.2")
    # 프롬프트 버전도 config에서 읽어 주입 (평가 시 v1/v2 전환용, 기본 v2)
    prompt_version = config.get("prompt_version", "system_v2")
    answer, tokens_used = generate_answer(query, docs, trimmed_history,
                                          use_ollama=use_ollama, ollama_model=ollama_model,
                                          prompt_version=prompt_version)
    
    # 출처는 file_name + section으로 표기 (page는 hwp라 null이므로 미사용)
    # doc_id는 화면 표기엔 안 쓰지만 추적/디버깅용으로 함께 보관
    sources = [
        {
            "doc_id": d.metadata.get("doc_id"),
            "file_name": d.metadata.get("file_name"),
            "section": d.metadata.get("section"),
            "score": d.metadata.get("score"),
        }
        for d in docs
    ]
    retrieved_chunks = [d.page_content for d in docs]
    elapsed_sec = round(time.time() - start, 2)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved_chunks,
        "elapsed_sec": elapsed_sec,
        "tokens_used": tokens_used,
    }


# 직접 실행 시 전체 파이프라인 테스트
# 실행: (루트에서) python -m backend.pipeline
if __name__ == "__main__":
    # 시나리오 B (gpt) 테스트
    result = get_ai_response("이 사업 담당자 휴대폰 번호 알려줘", history=[], config={"top_k": 5})
    print("=== get_ai_response 결과 (gpt) ===")
    print("answer:", result["answer"])
    print("sources:", result["sources"])
    print("chunks 수:", len(result["retrieved_chunks"]))
    print("elapsed_sec:", result["elapsed_sec"])
    print("tokens_used:", result["tokens_used"])

    # 시나리오 A (Ollama) 테스트 — use_ollama=True, 모델은 config의 ollama_model
    result_ollama = get_ai_response("이 사업 예산이랑 기간 알려줘", history=[], use_ollama=True)
    print("\n=== get_ai_response 결과 (Ollama) ===")
    print("answer:", result_ollama["answer"])
    print("elapsed_sec:", result_ollama["elapsed_sec"])
    print("tokens_used:", result_ollama["tokens_used"])