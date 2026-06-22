"""
export_chunks.py
retriever 결과(청크)를 JSON으로 떨궈서 generator 담당에게 넘기는 스크립트.

용도:
    generator 개발/테스트 단계에서 generator(수민님)이 retriever 스택(데이터·인덱스·API키·빌드)
    없이 결과 청크만 받아 작업할 수 있도록, retriever(나)가 미리 검색을 돌려 JSON으로 저장.

주의:
    검색은 질문마다 실행되므로, 이 JSON은 아래 QUERIES에 적은 질문들에 대한
    결과만 담깁니다. 최종 통합 단계(임의의 사용자 질문)에서는 JSON이 아니라
    get_retriever() 함수를 직접 호출하는 방식으로 합쳐야 합니다.

실행:
    cd backend/retrieval
    python export_chunks.py            # → sample_chunks.json 생성
"""

import json
from retriever import load_vectorstore, get_retriever

# ── 내보낼 질문 목록 (query, agency, project_name) ────────────────────────────
# agency / project_name은 필터 안 쓸 거면 None
QUERIES: list[tuple[str, str | None, str | None]] = [
    ("한영대 특성화 사업 예산 얼마야?", "한영대", None),
    ("입찰 마감일이 언제야?",            "한영대", None),
    ("제안요청 주요 내용 알려줘",         "한영대", None),
    # 필요한 만큼 추가
]

OUTPUT_PATH = "sample_chunks.json"


def chunk_to_dict(c) -> dict:
    """LangChain Document → JSON 직렬화 가능한 dict."""
    meta = dict(c.metadata)
    if "score" in meta and isinstance(meta["score"], float):
        meta["score"] = round(meta["score"],4)
    return {
        "page_content": c.page_content,
        "metadata": meta,   # score 포함 모든 메타 필드 (전부 JSON-safe)
    }


def main() -> None:
    vs = load_vectorstore()

    results = []
    for query, agency, project_name in QUERIES:
        chunks = get_retriever(
            query,
            vs,
            agency=agency,
            project_name=project_name,
            k=5,
        )
        results.append({
            "query": query,
            "agency": agency,
            "project_name": project_name,
            "chunks": [chunk_to_dict(c) for c in chunks],
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[export] {len(results)}개 질문 결과 저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
