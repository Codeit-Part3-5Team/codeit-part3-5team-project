"""
query_parser.py
사용자 질문에서 메타데이터 필터 값(agency, project_name)을 추출합니다.

사용법:
    from query_parser import extract_metadata

    meta = extract_metadata("한국수자원공사에서 발주한 사업을 알려주세요.")
    # → {"agency": "한국수자원공사", "project_name": None}

    meta = extract_metadata("건설통합시스템(CMS) 고도화 사업의 예산은 얼마인가요?")
    # → {"agency": None, "project_name": "건설통합시스템(CMS) 고도화"}
"""

import json
from openai import OpenAI

client = OpenAI()

_SYSTEM_PROMPT = """\
당신은 공공 입찰 RFP 검색 시스템의 쿼리 분석기입니다.
사용자 질문에서 아래 두 가지 필터 값을 추출하세요.

- agency      : 발주기관명 (예: 한국수자원공사, 대검찰청). 질문에 없으면 null.
- project_name: 사업명 (예: 건설통합시스템(CMS) 고도화). 질문에 없으면 null.

규칙:
- 반드시 JSON 형식으로만 출력하세요: {"agency": "...", "project_name": "..."}
- 값이 없으면 null을 사용하세요.
- 기관명은 질문에 나온 표현 그대로 추출하세요 (정규화 불필요).
- "모든 사업", "가장 큰", "몇 건" 같은 집계성 질문은 agency/project_name 모두 null.
"""


def extract_metadata(question: str) -> dict[str, str | None]:
    """
    질문에서 agency, project_name을 추출합니다.

    Args:
        question: 사용자 질문

    Returns:
        {"agency": str | None, "project_name": str | None}
    """
    user_content = f"""{_SYSTEM_PROMPT}

질문: {question}"""

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=4000,
    )

    choice = response.choices[0]
    print(f"[query_parser] finish_reason: {choice.finish_reason!r}")
    raw = choice.message.content
    print(f"[query_parser] raw response: {raw!r}")
    raw = (raw or "").strip()

    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        return {
            "agency": result.get("agency") or None,
            "project_name": result.get("project_name") or None,
        }
    except json.JSONDecodeError:
        print(f"[query_parser] JSON 파싱 실패: {raw!r} — 필터 미적용")
        return {"agency": None, "project_name": None}
