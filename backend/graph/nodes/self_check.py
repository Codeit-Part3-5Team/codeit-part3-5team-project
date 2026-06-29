"""
self_check.py
검증 노드 — 생성된 답변을 내보내기 전 마지막으로 게이트한다.
MVP는 가벼운 룰 기반 점검(LLM 자기검증 없음). 통과 여부와 걸린 항목만 기록하며,
답변을 막거나 고치지는 않는다(추후 '플래그 시 재생성 루프'로 확장 가능).

점검 항목:
  ① source_missing    : 답변이 [출처...]를 인용했는데 실제 검색결과(docs)가 비었음
  ② possible_pii_leak : 답변에 휴대폰/주민번호 형식이 노출됨(v3 거부 정책의 안전망)
"""
import re


def _looks_like_pii_leak(answer: str) -> bool:
    """답변에 마스킹 대상(휴대폰/주민번호 형식)이 노출됐는지 룰 검사."""
    # 휴대폰: 010/011/016~019-xxxx-xxxx
    if re.search(r"01[016789]-?\d{3,4}-?\d{4}", answer):
        return True
    # 주민등록번호: 6자리-7자리
    if re.search(r"\d{6}-?\d{7}", answer):
        return True
    return False


def self_check_node(state) -> dict:
    """
    생성된 답변을 룰 기반으로 점검해 플래그를 기록한다.

    Returns:
        dict: check_passed(통과 여부), check_flags(걸린 항목 목록)
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

    return {"check_passed": len(flags) == 0, "check_flags": flags}