"""
grade.py
검색결과 평가 노드 — route_a가 회수한 청크가 질문에 답하기 충분한지 LLM이 판정한다.
agentic 재검색 루프의 '판단' 부분(루프 자체는 build.py의 조건부 엣지가 수행).

판정 3종 (state.grade에 기록):
  sufficient    : 청크로 답변 충분 → answer_generation으로
  insufficient  : 부족하나 재검색 여지 있음 → re_retrieve 루프로
  out_of_scope  : RFP 범위 밖(거부 대상) → 재검색 말고 answer_generation(거부)으로

비용 설계(하이브리드 1차):
  - docs가 0개면 LLM을 부르지 않고 룰로 즉시 out_of_scope 처리(빈 검색에 LLM 낭비 방지).
  - 청크가 있을 때만 LLM 판정을 호출한다.
알맹이는 기존 call_gpt()를 재활용한다(새 LLM 호출 함수 만들지 않음).
"""
import json
from backend.generation.llm_client import call_gpt   # 기존 gpt-5-mini 호출 함수 재활용

# grade 전용 판정 프롬프트. JSON 한 줄만 출력하도록 강제(파싱 안정성).
_GRADE_SYSTEM = """너는 RFP(제안요청서) 질의응답 시스템의 검색결과 평가자다.
사용자 질문과 검색된 문서 청크를 보고, 이 청크들로 질문에 답할 수 있는지 판정하라.

판정값은 다음 셋 중 하나다:
- "sufficient"   : 청크 안에 질문의 답이 될 근거가 있다.
- "insufficient" : 질문은 RFP에서 답할 수 있는 종류인데, 청크에 근거가 부족하거나 무관하다(재검색하면 나아질 수 있음).
- "out_of_scope" : 질문 자체가 RFP 문서로 답할 수 없는 범위(개인정보 요구, 일반상식, 문서와 무관한 주제 등)다.

반드시 아래 JSON 한 줄만 출력하라. 다른 말 금지.
{"grade": "sufficient" | "insufficient" | "out_of_scope"}"""


def _docs_to_text(docs: list, max_chars: int = 2000) -> str:
    """청크 본문을 판정용 텍스트로 합친다(너무 길면 잘라 비용 절약)."""
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[청크 {i}] {d.page_content}")
    text = "\n".join(parts)
    return text[:max_chars]


def grade_node(state) -> dict:
    """
    검색결과(docs)가 질문에 충분한지 LLM으로 판정하고, 재시도 횟수를 증가시킨다.

    - docs가 비었으면 LLM 없이 out_of_scope로 단락(하이브리드: 빈 검색은 룰로 처리).
    - JSON 파싱 실패 시 sufficient로 폴백(재검색 루프를 타지 않는 안전한 쪽).

    Returns:
        dict: grade(판정값), retry_count(증가된 시도 횟수)
    """
    # 현재까지의 재검색 시도 횟수(없으면 0). 이 노드를 지날 때마다 1 증가.
    retry_count = state.get("retry_count", 0) + 1

    docs = state.get("docs", [])
    # 빈 검색결과: LLM 부르지 않고 즉시 범위밖 처리(비용 절약)
    if not docs:
        return {"grade": "out_of_scope", "retry_count": retry_count}

    # 재구성 질문 우선(followup 일관성), 없으면 원문
    question = state.get("rewritten_question") or state["question"]
    context = _docs_to_text(docs)

    messages = [
        {"role": "system", "content": _GRADE_SYSTEM},
        {"role": "user", "content": f"### 질문:\n{question}\n\n### 검색된 청크:\n{context}"},
    ]

    answer, _ = call_gpt(messages)   # 판정 토큰은 누적 토큰에 미포함(생성 토큰과 구분)

    # JSON 파싱: 실패하면 sufficient로 폴백(루프 안 도는 안전한 기본값)
    try:
        grade = json.loads(answer.strip()).get("grade", "sufficient")
        if grade not in ("sufficient", "insufficient", "out_of_scope"):
            grade = "sufficient"
    except (json.JSONDecodeError, AttributeError):
        grade = "sufficient"

    return {"grade": grade, "retry_count": retry_count}