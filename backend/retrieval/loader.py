"""
loader.py
데이터팀이 생성한 chunks_v1.json을 LangChain Document 리스트로 변환

청크 스키마 (데이터팀 chunks_v1.json 기준):
    page_content: str
    metadata:
        doc_id        : str
        file_name     : str
        section       : str
        content_type  : "text" | "table" | "meta_summary"
        chunk_index   : int
        page          : None  (hwp 특성상 nullable)
        token_count   : int
        has_image_budget : bool
        pii_masked    : bool | None
        budget_amount : int | None  (meta_summary 청크만 존재)
"""

import os
import json
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()
# 데이터 경로는 .env(CHUNKS_PATH)로 주입 - 하드코딩 지양
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/processed/chunks_v1_enriched.json")


def load_chunks(json_path: str) -> list[Document]:
    """
    chunks_v1.json을 읽어 LangChain Document 리스트로 변환

    Args:
        json_path (str): chunks_v1.json 파일 경로

    Returns:
        list[Document]: LangChain Document 리스트
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in data]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/chunks_v1.json"
    docs = load_chunks(path)

    # 요약 출력
    n_meta  = sum(1 for d in docs if d.metadata["content_type"] == "meta_summary")
    n_table = sum(1 for d in docs if d.metadata["content_type"] == "table")
    n_text  = sum(1 for d in docs if d.metadata["content_type"] == "text")

    print(f"총 청크: {len(docs)}")
    print(f"  meta_summary : {n_meta}")
    print(f"  table        : {n_table}")
    print(f"  text         : {n_text}")
    print(f"\n[샘플 1번째 청크]")
    print(f"  content_type : {docs[0].metadata['content_type']}")
    print(f"  doc_id       : {docs[0].metadata['doc_id']}")
    print(f"  page_content : {docs[0].page_content[:100]}...")