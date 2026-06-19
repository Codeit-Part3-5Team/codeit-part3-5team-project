# call_gpt: 단계 2-1에서 만든 gpt-5-mini 호출 함수
from backend.generation.llm_client import call_gpt
# Document: LangChain 표준 문서 객체 (재헌님이 검색 결과를 이 형식으로 넘겨줌)
from langchain_core.documents import Document
import os

# system_v1.txt 경로 (이 파일 기준 상대경로 → 절대경로. 하드코딩 방지)
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


# 시스템 프롬프트(역할+규칙) 파일을 읽어옴
def load_system_prompt(filename: str = "system_v1.txt") -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    # UTF-8 명시 (한글 깨짐/cp949 에러 방지)
    with open(path, encoding="utf-8") as f:
        return f.read()


# 검색된 Document 리스트 → LLM에 넘길 context 텍스트로 변환
# 각 청크 앞에 출처 정보(doc_id, page)를 붙여서 LLM이 출처를 인지하도록 함
def build_context(docs: list[Document]) -> str:
    blocks = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id", "알수없음")
        page = doc.metadata.get("page", "?")
        source_info = f"[문서ID: {doc_id} | p.{page}]"
        blocks.append(f"{source_info}\n{doc.page_content}")
    # 청크끼리 구분선으로 분리
    return "\n\n---\n\n".join(blocks)


# 질문 + 검색결과 → gpt-5-mini 답변 생성
# 반환: (답변 텍스트, 사용된 토큰 수)
def generate_answer(question: str, docs: list[Document]) -> tuple[str, int]:
    system_prompt = load_system_prompt()       # 역할+규칙
    context = build_context(docs)               # 검색결과 → context

    # 사용자 메시지: 문서 내용 + 질문을 한 덩어리로 구성
    user_content = (
        f"### 문서 내용:\n{context}\n\n"
        f"### 질문:\n{question}\n\n"
        f"### 답변:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    answer, tokens_used = call_gpt(messages)
    return answer, tokens_used


# 직접 실행 시 mock Document로 테스트
# 실행: (backend 폴더에서) python -m generation.generator
if __name__ == "__main__":
    # 재헌님 예상 형식(Document + doc_id/page/score)으로 mock 검색결과 생성
    mock_docs = [
        Document(
            page_content="본 사업은 국민연금공단의 이러닝시스템 고도화 사업이며, 총 사업예산은 540,000,000원이다.",
            metadata={"doc_id": "DOC_001", "page": 1, "score": 0.92},
        ),
        Document(
            page_content="사업 수행 기간은 계약 체결일로부터 6개월로 한다.",
            metadata={"doc_id": "DOC_001", "page": 3, "score": 0.85},
        ),
    ]

    # 1) 문서에 있는 질문 → 정상 답변 + 출처 기대
    print("=== 테스트1: 문서에 있는 질문 ===")
    answer, tokens = generate_answer("이 사업의 예산이 얼마야?", mock_docs)
    print(answer)
    print(f"(토큰: {tokens})\n")

    # 2) 문서에 없는 질문 → "찾을 수 없습니다" 거부 기대
    print("=== 테스트2: 문서에 없는 질문 ===")
    answer, tokens = generate_answer("이 사업의 담당자 전화번호는?", mock_docs)
    print(answer)
    print(f"(토큰: {tokens})")