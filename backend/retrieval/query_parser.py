"""
query_parser.py
사용자 질문에서 메타데이터 필터/정렬 조건을 추출합니다.

추출 필드:
    agency        : 발주기관명
    project_name  : 사업명
    budget_min    : 예산 하한 (원 단위 정수)
    budget_max    : 예산 상한 (원 단위 정수)
    date_field    : 정렬/필터 기준 날짜 필드 (announcement_date | bid_end | bid_start)
    date_min      : 날짜 하한 (YYYY-MM-DD)
    date_max      : 날짜 상한 (YYYY-MM-DD)
    sort_by       : 정렬 기준 필드 (budget_amount | announcement_date | bid_end)
    sort_order    : 정렬 방향 (asc | desc)

사용법:
    from query_parser import extract_metadata

    extract_metadata("예산이 1억원 이상인 사업을 모두 알려주세요.")
    # → {"budget_min": 100000000, "sort_by": None, ...}

    extract_metadata("예산이 가장 큰 사업은?")
    # → {"sort_by": "budget_amount", "sort_order": "desc", ...}
"""

import json
from openai import OpenAI

client = OpenAI()

_SYSTEM_PROMPT = """\
당신은 공공 입찰 RFP 검색 시스템의 쿼리 분석기입니다.
사용자 질문을 분석해 아래 필드를 추출하고 JSON으로만 출력하세요.

추출 필드:
- agency       : 발주기관명. 없으면 null.
- project_name : 사업명. 없으면 null.
- budget_min   : 예산 하한 (원 단위 정수). 없으면 null. 예) "1억원 이상" → 100000000
- budget_max   : 예산 상한 (원 단위 정수). 없으면 null. 예) "5억원 이하" → 500000000
- date_field   : 날짜 필터 기준 필드. announcement_date | bid_end | bid_start | null
- date_min     : 날짜 하한 (YYYY-MM-DD). 없으면 null.
- date_max     : 날짜 상한 (YYYY-MM-DD). 없으면 null.
- sort_by      : 정렬 기준. budget_amount | announcement_date | bid_end | null
- sort_order   : 정렬 방향. asc | desc | null. "가장 큰/높은/늦은" → desc, "가장 작은/낮은/먼저" → asc

규칙:
- JSON만 출력하세요. 설명 없이.
- 값이 없으면 null.
- 기관명/사업명은 질문에 나온 표현 그대로.
- 금액 단위 변환: 1억=100000000, 5억=500000000, 10억=1000000000

예시:
질문: "예산이 1억원 이상인 사업을 모두 알려주세요."
출력: {"agency": null, "project_name": null, "budget_min": 100000000, "budget_max": null, "date_field": null, "date_min": null, "date_max": null, "sort_by": null, "sort_order": null}

질문: "예산이 가장 큰 사업은 무엇인가요?"
출력: {"agency": null, "project_name": null, "budget_min": null, "budget_max": null, "date_field": null, "date_min": null, "date_max": null, "sort_by": "budget_amount", "sort_order": "desc"}

질문: "가장 먼저 공개된 사업은 무엇인가요?"
출력: {"agency": null, "project_name": null, "budget_min": null, "budget_max": null, "date_field": "announcement_date", "date_min": null, "date_max": null, "sort_by": "announcement_date", "sort_order": "asc"}
"""

_EMPTY_META = {
    "agency": None,
    "project_name": None,
    "budget_min": None,
    "budget_max": None,
    "date_field": None,
    "date_min": None,
    "date_max": None,
    "sort_by": None,
    "sort_order": None,
}


def extract_metadata(question: str) -> dict:
    """
    질문에서 필터/정렬 조건을 추출합니다.

    Args:
        question: 사용자 질문

    Returns:
        {agency, project_name, budget_min, budget_max,
         date_field, date_min, date_max, sort_by, sort_order}
    """
    user_content = f"{_SYSTEM_PROMPT}\n\n질문: {question}"

    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": user_content}],
        max_completion_tokens=4000,
    )

    raw = (response.choices[0].message.content or "").strip()

    # 마크다운 코드블록 제거
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        return {
            "agency":        result.get("agency") or None,
            "project_name":  result.get("project_name") or None,
            "budget_min":    result.get("budget_min"),
            "budget_max":    result.get("budget_max"),
            "date_field":    result.get("date_field") or None,
            "date_min":      result.get("date_min") or None,
            "date_max":      result.get("date_max") or None,
            "sort_by":       result.get("sort_by") or None,
            "sort_order":    result.get("sort_order") or None,
        }
    except json.JSONDecodeError:
        print(f"[query_parser] JSON 파싱 실패: {raw!r} — 필터 미적용")
        return dict(_EMPTY_META)
