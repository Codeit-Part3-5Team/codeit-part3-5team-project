"""
query_rewriter.py
대화 이력(history)을 참고해 followup 질문을 독립 질문으로 재구성합니다.

사용법:
    from query_rewriter import rewrite_query

    rewritten = rewrite_query(
        question="그럼 이 사업의 입찰 마감일은 언제인가요?",
        history=[
            {"question": "한국원자력연구원 선량평가시스템 고도화 사업의 예산은 얼마인가요?",
             "answer": "46,600,000원"}
        ]
    )
    # → "한국원자력연구원 선량평가시스템 고도화 사업의 입찰 마감일은 언제인가요?"
"""

from openai import OpenAI

client = OpenAI()

_SYSTEM_PROMPT = """\
당신은 검색 쿼리 전처리 전문가입니다.
사용자의 대화 이력과 현재 질문을 보고, 지시어(이 사업, 그 기관, 해당 사업 등)를 \
구체적인 명칭으로 교체하여 문맥 없이도 이해 가능한 단독 질문으로 재작성하세요.

규칙:
- 지시어가 없거나 이미 독립적인 질문이면 원문 그대로 반환하세요.
- 재작성된 질문만 출력하세요. 설명이나 부가 문장은 쓰지 마세요.
- 질문의 의도와 형식(어미 등)은 최대한 유지하세요.
"""


def rewrite_query(
    question: str,
    history: list[dict],
    use_system_prompt: bool = True,
) -> str:
    """
    history를 참고해 question을 독립 질문으로 재구성합니다.

    Args:
        question          : 현재 사용자 질문
        history           : [{"question": ..., "answer": ...}, ...] 형태의 직전 대화 이력
        use_system_prompt : False면 시스템 프롬프트 없이 호출

    Returns:
        재구성된 질문 (지시어 없으면 원문 그대로)
    """
    if not history:
        return question

    history_text = "\n".join(
        f"Q: {h['question']}\nA: {h['answer']}" for h in history
    )
    user_content = f"[대화 이력]\n{history_text}\n\n[현재 질문]\n{question}"

    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": _SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=messages,
        max_completion_tokens=256,
    )
    rewritten = response.choices[0].message.content.strip()
    return rewritten
