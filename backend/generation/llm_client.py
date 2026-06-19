# OpenAI: gpt-5-mini 호출용 공식 클라이언트
from openai import OpenAI
# load_dotenv: backend/.env에서 OPENAI_API_KEY 불러오기
from dotenv import load_dotenv
import os

# .env 로드 (OPENAI_API_KEY를 환경변수로 올림)
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


# Ollama 호출 함수 (시나리오 A용 - 지금은 자리만, 단계 6에서 구현)
def call_ollama(messages: list[dict], model: str = None) -> tuple[str, int]:
    raise NotImplementedError("Ollama 경로는 단계 6(시나리오 A)에서 구현 예정")


# 이 파일 직접 실행 시 gpt-5-mini가 실제로 응답하는지 테스트
# 실행: (backend 폴더에서) python -m generation.llm_client
if __name__ == "__main__":
    test_messages = [
        {"role": "system", "content": "당신은 친절한 한국어 도우미입니다."},
        {"role": "user", "content": "한 문장으로 자기소개 해줘."},
    ]
    answer, tokens = call_gpt(test_messages)
    print("=== gpt-5-mini 응답 ===")
    print(answer)
    print(f"\n토큰 사용량: {tokens}")


    