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
from reranker import rerank

# ── 설정 ──────────────────────────────────────────────────────────────────────
# 모든 설정값은 config.yaml(팀 공용) + .env(개인/비밀)에서 옴 → config.py가 통합 제공.
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    MMR_K,
    MMR_FETCH_K,
    MMR_LAMBDA,
    FUZZY_THRESHOLD,
)

# ── agency 보조 별칭 사전 ─────────────────────────────────────────────────────
# 데이터의 agency_aliases로 못 잡는 약칭만 보조로 관리 (약칭 → 정규화 기관명)
AGENCY_ALIASES: dict[str, str] = {
    "국연": "국민연금공단",
    "국민연금": "국민연금공단",
    "건보": "국민건강보험공단",
    "건강보험": "국민건강보험공단",
    "근복": "근로복지공단",
    # 필요 시 추가
}


# ── 벡터스토어 빌드 / 로드 ────────────────────────────────────────────────────

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


# ── 메타데이터 필터링 ─────────────────────────────────────────────────────────

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
    budget_min: int | None = None,
    budget_max: int | None = None,
    date_field: str | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
) -> list[Document]:
    """
    Document 리스트를 메타데이터 조건으로 필터링합니다.

    Args:
        docs        : 필터링할 Document 리스트
        agency      : 기관명 필터 (3단계 매핑 적용)
        project_name: 사업명 필터 (부분 문자열 매칭)
        budget_min  : 예산 하한 (원 단위 정수, 없으면 미적용)
        budget_max  : 예산 상한 (원 단위 정수, 없으면 미적용)
        date_field  : 날짜 필터 기준 필드 (announcement_date | bid_start | bid_end)
        date_min    : 날짜 하한 (YYYY-MM-DD, 없으면 미적용)
        date_max    : 날짜 상한 (YYYY-MM-DD, 없으면 미적용)

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

    if budget_min is not None or budget_max is not None:
        before = len(filtered)
        def _budget_ok(d: Document) -> bool:
            raw = d.metadata.get("budget_amount")
            if raw is None:
                return False
            try:
                amount = int(raw)
            except (ValueError, TypeError):
                return False
            if budget_min is not None and amount < budget_min:
                return False
            if budget_max is not None and amount > budget_max:
                return False
            return True
        filtered = [d for d in filtered if _budget_ok(d)]
        print(f"[retriever] budget 필터: min={budget_min} max={budget_max} → {before}개 → {len(filtered)}개")

    if date_field and (date_min or date_max):
        before = len(filtered)
        def _date_ok(d: Document) -> bool:
            raw = d.metadata.get(date_field)
            if not raw:
                return False
            date_str = str(raw)[:10]   # YYYY-MM-DD 앞 10자만 비교
            if date_min and date_str < date_min:
                return False
            if date_max and date_str > date_max:
                return False
            return True
        filtered = [d for d in filtered if _date_ok(d)]
        print(f"[retriever] date 필터: {date_field} [{date_min}, {date_max}] → {before}개 → {len(filtered)}개")

    return filtered


# ── 메인 검색 함수 ────────────────────────────────────────────────────────────

def get_retriever(
    query: str,
    vectorstore: FAISS,
    agency: str | None = None,
    project_name: str | None = None,
    budget_min: int | None = None,
    budget_max: int | None = None,
    date_field: str | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    sort_by: str | None = None,
    sort_order: str | None = None,
    k: int = MMR_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    use_rerank: bool = False,
    rerank_top_n: int = 100,
) -> list[Document]:
    """
    메타데이터 필터링 + MMR 검색을 수행합니다.

    Args:
        query        : 사용자 질문
        vectorstore  : FAISS 벡터스토어
        agency       : 기관명 필터 (선택)
        project_name : 사업명 필터 (선택)
        budget_min   : 예산 하한 (원 단위 정수, 선택)
        budget_max   : 예산 상한 (원 단위 정수, 선택)
        date_field   : 날짜 필터 기준 필드 (announcement_date | bid_start | bid_end, 선택)
        date_min     : 날짜 하한 (YYYY-MM-DD, 선택)
        date_max     : 날짜 상한 (YYYY-MM-DD, 선택)
        sort_by      : 정렬 기준 필드 (budget_amount | announcement_date | bid_end, 선택)
        sort_order   : 정렬 방향 (asc | desc, 선택)
        k            : 최종 반환 문서 수
        fetch_k      : MMR 후보 풀 크기
        lambda_mult  : MMR 관련성 가중치 (1=관련성만, 0=다양성만)
        use_rerank   : True면 cross-encoder로 재정렬 후 top-k 반환
        rerank_top_n : reranking 입력 후보 수 (필터 후 상위 N개만 rerank)

    Returns:
        list[Document]: 검색된 문서 리스트
                        (각 doc.metadata["score"]에 유사도 점수 포함)

    Note:
        score는 FAISS L2 거리를 유사도로 변환한 값입니다(0~1, **높을수록 유사**).
        변환식: similarity = 1 / (1 + L2_distance)
    """
    # MMR 후보 풀 검색 (score 포함 → score는 벡터로만 받을 수 있어 쿼리를 먼저 임베딩)
    query_vec = get_cached_embeddings().embed_query(query)
    docs_and_scores = vectorstore.max_marginal_relevance_search_with_score_by_vector(
        query_vec,
        k=fetch_k,
        fetch_k=fetch_k * 2,
        lambda_mult=lambda_mult,
    )

    # L2 거리 → 유사도(0~1, 높을수록 유사)로 변환해 각 청크 메타데이터에 부착
    candidates: list[Document] = []
    for doc, distance in docs_and_scores:
        doc.metadata["score"] = 1.0 / (1.0 + float(distance))
        candidates.append(doc)

    # 메타데이터 필터링
    need_filter = any([agency, project_name,
                       budget_min is not None, budget_max is not None,
                       date_field and (date_min or date_max)])
    if need_filter:
        candidates = _filter_docs(
            candidates,
            agency=agency,
            project_name=project_name,
            budget_min=budget_min,
            budget_max=budget_max,
            date_field=date_field,
            date_min=date_min,
            date_max=date_max,
        )

    # 정렬 (sort_by 지정 시 score 기반 MMR 순서 대신 메타데이터 기준으로 재정렬)
    if sort_by and sort_order:
        reverse = (sort_order == "desc")
        is_date_field = sort_by in ("announcement_date", "bid_start", "bid_end")
        def _sort_key(d: Document):
            val = d.metadata.get(sort_by)
            if is_date_field:
                # None은 항상 끝으로 (asc: "9999", desc: "")
                return str(val)[:10] if val is not None else ("" if reverse else "9999-99-99")
            # 숫자 필드
            try:
                return int(val) if val is not None else (float('-inf') if reverse else float('inf'))
            except (ValueError, TypeError):
                return float('-inf') if reverse else float('inf')
        candidates = sorted(candidates, key=_sort_key, reverse=reverse)
        print(f"[retriever] 정렬: {sort_by} {sort_order}")

    # Reranking (cross-encoder)
    if use_rerank and candidates:
        top_n = candidates[:rerank_top_n]
        candidates = rerank(query, top_n, top_k=k)
    else:
        candidates = candidates[:k]

    print(f"[retriever] 검색 완료: '{query}' → {len(candidates)}개 문서 반환")
    return candidates


# ── agency 필터 스킵 재검색 ───────────────────────────────────────────────────

def re_retrieve_fn(
    query: str,
    vectorstore: FAISS,
    agency: str | None = None,
    project_name: str | None = None,
    budget_min: int | None = None,
    budget_max: int | None = None,
    date_field: str | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    sort_by: str | None = None,
    sort_order: str | None = None,
    k: int = MMR_K,
    fetch_k: int = MMR_FETCH_K,
    lambda_mult: float = MMR_LAMBDA,
    use_rerank: bool = False,
    rerank_top_n: int = 100,
) -> tuple[list[Document], bool]:
    """
    agency 필터로 검색 후 결과가 없으면 agency 필터를 스킵하고 재검색합니다.

    agency가 지정됐으나 결과가 0개인 경우(인덱스에 해당 기관 청크 없음 등)
    agency=None으로 재검색해 빈 결과를 방지합니다.

    Args:
        query        : 검색 쿼리
        vectorstore  : FAISS 벡터스토어
        agency       : 기관명 필터 (선택)
        project_name : 사업명 필터 (선택)
        budget_min   : 예산 하한 (선택)
        budget_max   : 예산 상한 (선택)
        date_field   : 날짜 필터 기준 필드 (선택)
        date_min     : 날짜 하한 YYYY-MM-DD (선택)
        date_max     : 날짜 상한 YYYY-MM-DD (선택)
        sort_by      : 정렬 기준 필드 (선택)
        sort_order   : 정렬 방향 asc|desc (선택)
        k            : 최종 반환 문서 수
        fetch_k      : MMR 후보 풀 크기
        lambda_mult  : MMR 관련성 가중치
        use_rerank   : cross-encoder 재정렬 여부
        rerank_top_n : reranking 입력 후보 수

    Returns:
        (docs, agency_skipped)
            docs           : 검색된 Document 리스트
            agency_skipped : agency 필터를 스킵하고 재검색했으면 True
    """
    common_kwargs = dict(
        project_name=project_name,
        budget_min=budget_min,
        budget_max=budget_max,
        date_field=date_field,
        date_min=date_min,
        date_max=date_max,
        sort_by=sort_by,
        sort_order=sort_order,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        use_rerank=use_rerank,
        rerank_top_n=rerank_top_n,
    )

    docs = get_retriever(query, vectorstore, agency=agency, **common_kwargs)

    # agency 필터로 결과가 없으면 자동 스킵 후 재검색
    if not docs and agency:
        print(f"[retriever] agency='{agency}' 결과 없음 → agency 필터 스킵 재검색")
        docs = get_retriever(query, vectorstore, agency=None, **common_kwargs)
        return docs, True

    return docs, False


# ── 동작 확인 ─────────────────────────────────────────────────────────────────

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
