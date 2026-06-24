"""
retriever.py
FAISS 벡터스토어 기반 Retriever

검색 전략:
    1. 메타데이터 필터링 (agency_normalized / project_name)
       - 1단계: exact match (정규화 기관명)
       - 2단계: 별칭 매핑 (데이터 내 agency_aliases + 하드코딩 보조사전)
       - 3단계: rapidfuzz 유사도 매칭 (threshold 80)
    2. MMR (Maximal Marginal Relevance) 검색 — 다양성 + 관련성 균형

메타데이터 스키마 참고 (chunks_v1_enriched.json):
    필터 사용 필드 : agency_normalized, project_name
    별칭 매핑 필드 : agency_aliases (list[str], 청크별 제공)
    그 외 enriched: announcement_date, bid_start, bid_end,
                    announcement_no, announcement_round, budget_amount

주요 함수:
    build_vectorstore(docs)   : Document 리스트 → FAISS 벡터스토어 생성 & 저장
    load_vectorstore()        : 저장된 FAISS 인덱스 로드
    get_retriever(query, ...) : 메타데이터 필터링 + MMR 검색 결과 반환
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rapidfuzz import fuzz, process

from embedder import get_cached_embeddings

# 모든 설정값은 config.yaml(팀 공용) + .env(개인)에서 옴 -> config.py가 통합 제공
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    MMR_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    FUZZY_THRESHOLD
)

# agency 보조 별칭 사전
# 데이터의 agency_aliases로 못 잡는 약칭만 보조로 관리 (약칭 → 정규화 기관명)
AGENCY_ALIASES: dict[str, str] = {
    "국연": "국민연금공단",
    "국민연금": "국민연금공단",
    "건보": "국민건강보험공단",
    "건강보험": "국민건강보험공단",
    "근복": "근로복지공단",
    # 필요 시 추가
}


# 벡터스토어 빌드 / 로드

def build_vectorstore(docs: list[Document]) -> FAISS:
    """
    Document 리스트로 FAISS 벡터스토어를 생성하고 로컬에 저장합니다.

    Args:
        docs: LangChain Document 리스트 (loader.py 출력)

    Returns:
        FAISS: 생성된 벡터스토어
    """
    embeddings = get_cached_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"[retriever] 벡터스토어 저장 완료: {FAISS_INDEX_PATH} ({len(docs)}개 청크)")
    return vectorstore


def load_vectorstore() -> FAISS:
    """
    로컬에 저장된 FAISS 인덱스를 로드합니다.

    Returns:
        FAISS: 로드된 벡터스토어
    """
    embeddings = get_cached_embeddings()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[retriever] 벡터스토어 로드 완료: {FAISS_INDEX_PATH}")
    return vectorstore


# 메타데이터 필터링

def _build_alias_index(docs: list[Document]) -> dict[str, str]:
    """
    Document들의 agency_aliases를 모아 (별칭 → 정규화 기관명) 역인덱스를 만듭니다.
    데이터 제공 별칭 위에 하드코딩 보조사전(AGENCY_ALIASES)을 덮어씁니다.

    Args:
        docs: 별칭을 수집할 Document 리스트

    Returns:
        별칭 문자열 → agency_normalized 매핑
    """
    index: dict[str, str] = {}
    for d in docs:
        normalized = d.metadata.get("agency_normalized")
        if not normalized:
            continue
        for alias in d.metadata.get("agency_aliases", []) or []:
            index[alias] = normalized
    # 보조 사전이 우선
    index.update(AGENCY_ALIASES)
    return index


def _resolve_agency(
    query_agency: str,
    candidates: list[str],
    alias_index: dict[str, str],
) -> str | None:
    """
    3단계 로직으로 agency 쿼리를 정규화 기관명(agency_normalized)으로 매핑합니다.

    1단계: exact match (정규화 기관명 후보)
    2단계: 별칭 매핑 (데이터 agency_aliases + 보조사전)
    3단계: rapidfuzz 유사도 매칭

    Args:
        query_agency: 사용자가 입력한 기관명 (혹은 약칭)
        candidates  : 후보 내 실제 agency_normalized 값 목록
        alias_index : 별칭 → agency_normalized 역인덱스

    Returns:
        매핑된 정규화 기관명 | None (매칭 실패)
    """
    # 1단계: exact
    if query_agency in candidates:
        return query_agency

    # 2단계: alias
    resolved = alias_index.get(query_agency)
    if resolved and resolved in candidates:
        return resolved

    # 3단계: fuzzy
    result = process.extractOne(
        query_agency,
        candidates,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=FUZZY_THRESHOLD,
    )
    return result[0] if result else None


def _filter_docs(
    docs: list[Document],
    agency: str | None = None,
    project_name: str | None = None,
) -> list[Document]:
    """
    Document 리스트를 agency_normalized / project_name 기준으로 필터링합니다.
    agency는 3단계 매핑 로직 적용, project_name은 부분 문자열 매칭.

    Args:
        docs        : 필터링할 Document 리스트
        agency      : 기관명 필터 (없으면 미적용)
        project_name: 사업명 필터 (없으면 미적용)

    Returns:
        필터링된 Document 리스트
    """
    filtered = docs

    if agency:
        all_agencies = list({
            d.metadata.get("agency_normalized", "")
            for d in filtered if d.metadata.get("agency_normalized")
        })
        alias_index = _build_alias_index(filtered)
        resolved = _resolve_agency(agency, all_agencies, alias_index)
        if resolved:
            filtered = [d for d in filtered if d.metadata.get("agency_normalized") == resolved]
            print(f"[retriever] agency 필터: '{agency}' → '{resolved}' ({len(filtered)}개)")
        else:
            print(f"[retriever] agency 필터: '{agency}' 매칭 실패 — 필터 미적용")

    if project_name:
        filtered = [
            d for d in filtered
            if project_name in d.metadata.get("project_name", "")
        ]
        print(f"[retriever] project_name 필터: '{project_name}' → {len(filtered)}개")

    return filtered


# 메인 검색 함수

def get_retriever(
    query: str,
    vectorstore: FAISS,
    agency: str | None = None,
    project_name: str | None = None,
    k: int = MMR_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
) -> list[Document]:
    """
    메타데이터 필터링 + MMR 검색을 수행합니다.

    Args:
        query       : 사용자 질문
        vectorstore : FAISS 벡터스토어
        agency      : 기관명 필터 (선택)
        project_name: 사업명 필터 (선택)
        k           : 최종 반환 문서 수
        fetch_k     : MMR 후보 풀 크기
        lambda_mult : MMR 관련성 가중치 (1=관련성만, 0=다양성만)

    Returns:
        list[Document]: 검색된 문서 리스트
    """
    # MMR 후보 풀 검색
    query_vec = get_cached_embeddings().embed_query(query)
    docs_and_scores = vectorstore.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k = fetch_k,
        fetch_k = fetch_k *2,
        lambda_mult = lambda_mult
    )

    # L2 거리 -> 유사도 (0~1, 높을수록 유사)로 변환해 각 청크 메타데이터에 부착
    candidates: list[Document] = []
    for doc, distance in docs_and_scores:
        doc.metadata["score"] = 1.0 / (1.0 + float(distance))
        candidates.append(doc)

    # 메타데이터 필터링
    if agency or project_name:
        candidates = _filter_docs(candidates, agency=agency, project_name=project_name)

    # 필터 후 k개 반환
    results = candidates[:k]
    print(f"[retriever] 검색 완료: '{query}' → {len(results)}개 문서 반환")
    return results


# 동작 확인

if __name__ == "__main__":
    import sys
    from loader import load_chunks

    # 벡터스토어 로드 (없으면 빌드)
    if os.path.exists(FAISS_INDEX_PATH):
        vs = load_vectorstore()
    else:
        json_path = sys.argv[1] if len(sys.argv) > 1 else CHUNKS_PATH
        docs = load_chunks(json_path)
        vs = build_vectorstore(docs)

    # 테스트 검색
    test_query   = "한영대학교 특성화 사업 예산은 얼마야?"
    test_agency  = "한영대"     # alias(agency_aliases) 테스트
    results = get_retriever(test_query, vs, agency=test_agency)

    print(f"\n[테스트 결과] 질문: {test_query}")
    for i, doc in enumerate(results, 1):
        m = doc.metadata
        print(f"\n--- 문서 {i} ---")
        print(f"  score             : {m.get('score', 0.0):.4f} (유사도 0~1, 높을수록 유사 / 값이 없을 때는 0.0000 출력)")
        print(f"  agency_normalized : {m.get('agency_normalized', 'N/A')}")
        print(f"  project_name      : {m.get('project_name', 'N/A')}")
        print(f"  doc_id            : {m.get('doc_id', 'N/A')}")
        print(f"  file_name         : {m.get('file_name', 'N/A')}")
        print(f"  content_type      : {m.get('content_type', 'N/A')}")
        print(f"  section           : {m.get('section', 'N/A')}")
        print(f"  budget_amount     : {m.get('budget_amount', 'N/A')}")
        print(f"  announcement_date : {m.get('announcement_date', 'N/A')}")
        print(f"  bid_start         : {m.get('bid_start', 'N/A')}")
        print(f"  bid_end           : {m.get('bid_end', 'N/A')}")
        print(f"  announcement_no   : {m.get('announcement_no', 'N/A')}")
        print(f"  announcement_round: {m.get('announcement_round', 'N/A')}")
        print(f"  content           : {doc.page_content[:120]}...")