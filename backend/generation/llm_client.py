# OpenAI: gpt-5-mini 호출용 공식 клиент
from openai import OpenAI
# ChatOllama: GCP 서버의 Ollama 모델 호출용 (langchain-ollama)
from langchain_ollama import ChatOllama
# load_dotenv: .env에서 OPENAI_API_KEY, OLLAMA_BASE_URL 불러오기
from dotenv import load_dotenv
import os

# .env 로드 (OPENAI_API_KEY, OLLAMA_BASE_URL을 환경변수로 올림)
load_dotenv()


# gpt-5-mini 호출 함수
# messages: [{"role": "system"/"user"/"assistant", "content": "..."}] 형식의 대화 메시지
# 반환: (답변 텍스트, 사용된 토큰 수) 튜플
def call_gpt(messages: list[dict], model: str = "gpt-5-mini") -> tuple[str, int]:
    # API 키로 클라이언트 생성 (.env의 OPENAI_API_KEY를 자동으로 읽음)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 채팅 완성 요청
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # 답변 본문 추출
    answer = response.choices[0].message.content
    # 토큰 사용량 추출 (인터페이스의 tokens_used 채울 값)
    tokens_used = response.usage.total_tokens

    return answer, tokens_used


# Ollama 호출 함수 (시나리오 A용)
# messages: call_gpt와 동일한 [{"role": ..., "content": ...}] 형식
# model: Ollama 모델명 (예: "llama3.2", "qwen3:8b"). 없으면 기본값 사용
# 반환: (답변 텍스트, 토큰 수) — call_gpt와 입출력 형식 통일
def call_ollama(messages: list[dict], model: str = "llama3.2") -> tuple[str, int]:
    # GCP Ollama 서버 주소는 .env의 OLLAMA_BASE_URL에서 읽음 (하드코딩 방지)
    base_url = os.getenv("OLLAMA_BASE_URL")

    # ChatOllama 클라이언트 생성 (GCP 서버의 모델을 호출)
    client = ChatOllama(model=model, base_url=base_url)

    # 호출: messages(dict 리스트)를 그대로 넘기면 LangChain이 처리
    response = client.invoke(messages)

    # 답변 본문 추출 (LangChain 응답은 .content에 텍스트가 들어있음)
    answer = response.content

    # 토큰 수: Ollama 응답 메타데이터에서 추출 시도 (없으면 0)
    # gpt와 달리 Ollama는 토큰 정보를 항상 주지는 않아서 안전하게 처리
    usage = response.usage_metadata or {}
    tokens_used = usage.get("total_tokens", 0)

    return answer, tokens_used


# 이 파일 직접 실행 시 gpt-5-mini가 실제로 응답하는지 테스트
# 실행: (루트에서) python -m backend.generation.llm_client
if __name__ == "__main__":
    test_messages = [
        {"role": "system", "content": "당신은 친절한 한국어 도우미입니다."},
        {"role": "user", "content": "한 문장으로 자기소개 해줘."},
    ]
    answer, tokens = call_gpt(test_messages)
    print("=== gpt-5-mini 응답 ===")
    print(answer)
    print(f"\n토큰 사용량: {tokens}")

    # Ollama(시나리오 A) 호출 테스트 — GCP 서버의 모델로 한국어 답변 확인
    print("\n=== Ollama 응답 ===")
    ollama_answer, ollama_tokens = call_ollama(test_messages)
    print(ollama_answer)
    print(f"\n토큰 사용량: {ollama_tokens}")


    