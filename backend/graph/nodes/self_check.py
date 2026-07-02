"""
self_check.py
검증 노드 — 생성된 답변을 내보내기 전 마지막으로 게이트한다.
룰 기반 점검(LLM 자기검증 없음). 플래그가 걸리면 답변을 안전 문구로 '통째 교체'하고
출처(docs)도 비운다. 이 항목들은 재생성으로 고쳐지지 않으므로(같은 docs면 또 샘),
재생성 루프가 아니라 교체로 처리한다. self_check는 프롬프트가 못 막은 예외를 잡는 최후 안전망.

점검 항목:
  ① source_missing    : 답변이 [출처...]를 인용했는데 실제 검색결과(docs)가 비었음
  ② possible_pii_leak : 답변에 휴대폰/주민번호 형식이 노출됨(v3 거부 정책의 안전망)
  ③ stale_source      : "찾을 수 없습니다" 류 거부 답변인데 [출처...]가 붙음(모순) — 출처 표기·sources 제거

플래그 우선순위: PII 노출이 source_missing보다 위험하므로 PII 문구를 우선 적용한다.
①② 통과 답변에만 ③을 적용한다(①②는 통째 교체라 출처가 이미 사라짐).
"""
import re

# 플래그별 교체 문구. 플래그가 걸리면 답변을 이 문구로 통째 교체한다.
_PII_MESSAGE = "해당 정보는 개인정보에 해당하여 제공할 수 없습니다."
_SOURCE_MISSING_MESSAGE = "제공된 문서에서 해당 정보의 근거를 찾을 수 없습니다."

# ③ 거부 답변 판별용 문구. 이런 답변에 [출처...]가 붙으면 모순이므로 출처를 뗀다.
_REFUSAL_MARKERS = ["찾을 수 없습니다", "제공할 수 없습니다", "지원하지 않습니다"]


def _looks_like_pii_leak(answer: str) -> bool:
    """답변에 마스킹 대상(휴대폰/주민번호 형식)이 노출됐는지 룰 검사."""
    # 휴대폰: 010/011/016~019-xxxx-xxxx
    if re.search(r"01[016789]-?\d{3,4}-?\d{4}", answer):
        return True
    # 주민등록번호: 6자리-7자리
    if re.search(r"\d{6}-?\d{7}", answer):
        return True
    return False


def _strip_source_tag(answer: str) -> str:
    """답변에서 [출처: ...] 표기를 제거하고 뒤따르는 공백·빈 줄을 정리한다."""
    # [출처: 로 시작해 줄 끝(또는 문자열 끝)까지 제거.
    # 출처 안에 [표: ...] 같은 중첩 대괄호가 있어 non-greedy로는 덜 잘리므로,
    # '[출처:'부터 그 줄 끝까지' 통째로 지운다(출처는 답변 맨 끝 줄 규칙).
    stripped = re.sub(r"\[출처:[^\n]*", "", answer)
    return stripped.strip()


def self_check_node(state) -> dict:
    """
    생성된 답변을 룰 기반으로 점검한다. 플래그가 걸리면 답변을 안전 문구로 통째 교체하고
    출처(docs)도 비운다(근거 없는 답에 출처가 남는 모순 방지).
    ①② 미해당이라도 ③(거부 답변 + 출처 모순)이면 출처 줄만 제거한다.

    Returns:
        dict: check_passed, check_flags. 플래그 시 answer(교체/정리분)·docs 포함.
    """
    answer = state.get("answer", "")
    flags = []

    # ① 근거 없는 출처 인용: [출처 표기는 있는데 검색결과(docs)가 비었음
    #    (결정2로 state엔 sources 대신 docs를 싣으므로 docs 기준으로 검사)
    if "[출처" in answer and not state.get("docs"):
        flags.append("source_missing")

    # ② PII 노출: 휴대폰/주민번호 형식이 답변에 보이면 플래그
    if _looks_like_pii_leak(answer):
        flags.append("possible_pii_leak")

    # ①② 해당: 안전 문구로 통째 교체. PII가 더 위험하므로 우선 적용.
    if flags:
        if "possible_pii_leak" in flags:
            safe_answer = _PII_MESSAGE
        else:
            safe_answer = _SOURCE_MISSING_MESSAGE
        # 답변 교체 + docs 비우기(근거 없는 답에 출처가 따라나가지 않도록)
        return {
            "check_passed": False,
            "check_flags": flags,
            "answer": safe_answer,
            "docs": [],
        }

    #    거부 문구는 답변 첫 줄에서만 검사한다(집계형 답변의 하위 필드 "…찾을 수 없습니다"를
    #    전체 거부로 오판해 정상 답변의 docs를 비우는 것을 방지).
    first_line = answer.strip().split("\n", 1)[0]
    is_refusal = any(m in first_line for m in _REFUSAL_MARKERS)
    if is_refusal and "[출처" in answer:
        return {
            "check_passed": True,           # 답변 자체는 정상(거부가 옳음)
            "check_flags": ["stale_source"],
            "answer": _strip_source_tag(answer),
            "docs": [],
        }

    # 아무 항목도 해당 없음: 답변 그대로 통과
    return {"check_passed": True, "check_flags": []}