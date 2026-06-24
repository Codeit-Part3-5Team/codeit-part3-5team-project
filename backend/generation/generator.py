# call_gpt: gpt-5-mini 호출 / call_ollama: Ollama(시나리오 A) 호출
from backend.generation.llm_client import call_gpt, call_ollama
# Document: LangChain 표준 문서 객체 (재헌님이 검색 결과를 이 형식으로 넘겨줌)
from langchain_core.documents import Document
import os

# system_v1.txt 경로 (이 파일 기준 상대경로 → 절대경로. 하드코딩 방지)
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


# 시스템 프롬프트(역할+규칙) 파일을 읽어옴
def load_system_prompt(filename: str = "system_v2.txt") -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    # UTF-8 명시 (한글 깨짐/cp949 에러 방지)
    with open(path, encoding="utf-8") as f:
        return f.read()


# 검색된 Document 리스트 → LLM에 넘길 context 텍스트로 변환
# 각 청크 앞에 출처 정보(file_name, section)를 붙여서 LLM이 출처를 인지하도록 함
# (page는 hwp 특성상 null이라 출처에 쓰지 않음)
def build_context(docs: list[Document]) -> str:
    blocks = []
    for doc in docs:
        file_name = doc.metadata.get("file_name", "알수없음")
        section = doc.metadata.get("section", "")
        source_info = f"[출처: {file_name} | {section}]"
        blocks.append(f"{source_info}\n{doc.page_content}")
    # 청크끼리 구분선으로 분리
    return "\n\n---\n\n".join(blocks)


# use_ollama: True면 Ollama(시나리오 A), False면 gpt-5-mini(시나리오 B)
# ollama_model: Ollama 사용 시 모델명 (평가에서 후보 바꿔가며 지정)
def generate_answer(question: str, docs: list[Document], history: list[dict] = None,
                    use_ollama: bool = False, ollama_model: str = "llama3.2",
                    prompt_version: str = "system_v2") -> tuple[str, int]:
    # prompt_version으로 시스템 프롬프트 파일 선택 (config에서 주입, 평가 시 v1/v2 전환)
    system_prompt = load_system_prompt(f"{prompt_version}.txt")
    context = build_context(docs)               # 검색결과 → context
    history = history or []                      # history 없으면 빈 리스트로

    # 사용자 메시지: 문서 내용 + 질문을 한 덩어리로 구성
    user_content = (
        f"### 문서 내용:\n{context}\n\n"
        f"### 질문:\n{question}\n\n"
        f"### 답변:"
    )

    # messages 구성: system → 이전 대화이력 → 이번 질문 순서로 쌓음
    # 이전 대화를 system과 이번 질문 사이에 넣어야 모델이 맥락을 이어받음
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)                     # trim된 이전 대화 끼워넣기
    messages.append({"role": "user", "content": user_content})

    # 모델 선택: 시나리오 A(Ollama) vs B(gpt) — 입출력 형식이 같아 함수만 분기
    if use_ollama:
        answer, tokens_used = call_ollama(messages, model=ollama_model)
    else:
        answer, tokens_used = call_gpt(messages)
    return answer, tokens_used


# 직접 실행 시 mock Document로 테스트
# 실행: (루트에서) python -m backend.generation.generator
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