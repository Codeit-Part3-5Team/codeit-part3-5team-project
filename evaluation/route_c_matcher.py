# evaluation/route_c_matcher.py
# 라우트 C matcher — expected_item ↔ extracted_item LLM 판정
#
# 설계: evaluation/route_c_eval_design.md (2차 개정본)
# 안전장치:
#   1. 파싱 실패 → review_parse_error로 1회 재시도, 그래도 실패 시 사람
#   2. scope 2-pass (1차 scope 카테고리, 못 찾으면 2차 전체 재탐색 — 하드 필터 아님)
#   3. Decision Pair Cache (prompt_hash 포함, 깨진 판정 재사용 방지)
#   4. confidence 강제, low + 애매 → review (match로 안 밀어넣음)
#   5. review 비율 30%↑ 시 경고 (matcher/expected 부적절 신호)
#   6. fixture 자가검증 (실전 전 matcher가 사람 정답 맞히는지)
#
# 판정 권한: LLM은 1차 후보 판정만. 최종 확정(특히 review/extra)은 사람.

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

# pydantic은 추출기가 이미 쓰므로 환경에 있음
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# matcher 출력 스키마 (structured output 강제 — 형식 깨짐 방지)
# ─────────────────────────────────────────────────────────────
class MatchVerdict(BaseModel):
    decision: str = Field(description="match | miss | review")
    actor_label: str = Field(
        description="bidder_action | bidder_evaluation_rule | "
                    "buyer_internal_process | context_or_background")
    confidence: str = Field(description="high | low")
    reason: str = Field(description="판정 사유 1~2문장")


# ─────────────────────────────────────────────────────────────
# matcher 프롬프트 — rubric. 이 문자열이 바뀌면 캐시 무효화돼야 함
# ─────────────────────────────────────────────────────────────
MATCHER_RUBRIC = """너는 두 항목이 같은 입찰 의무/배점/자격을 가리키는지 검증하는 냉정한 감사관이다.
추출기나 RAG의 맥락은 무시하고, 오직 두 텍스트의 의미만 비교한다.

[입력]
- expected: 골든셋이 정의한 기대 항목
- extracted: 시스템이 추출한 항목
- evidence: extracted의 근거 인용문(원문)

[판정 decision — 셋 중 하나]
- match: extracted가 expected의 핵심 대상·배점·조건을 충족한다.
  * 복합 항목(예: "정량평가 20점: 실적6/신인6/조직6/경영2")은 세부까지 extracted에 드러나야 match.
  * 큰 틀(헤드라인)만 언급하고 세부가 없으면 match 아님 → review.
- miss: 두 항목이 서로 관련 없다.
- review: 부분 충족 / 헤드라인만 언급 / 조건·수치 일부 손실 / 판단 애매.
  확신이 없으면 match로 밀어넣지 말고 review로 보낸다.

[판정 actor_label — extracted의 행위 주체]
- bidder_action: 제안자·수행사가 제출·구현·운영·충족해야 하는 것
- bidder_evaluation_rule: 발주기관이 평가하지만 제안자에 직접 영향(배점·합격선·평가기준)
- buyer_internal_process: 제안자가 충족할 수 없는 발주기관 내부 절차(평가위원회 구성, 협상 순서, 점수 계산방식)
- context_or_background: 사업방식·기관소개·일반 배경 설명

[confidence]
- high: 판정이 명확
- low: 애매하거나 정보 부족

반드시 JSON으로만 답한다."""

RUBRIC_VERSION = "v1.0"
MATCHER_MODEL = "gpt-5-mini"
MATCHER_PROMPT_HASH = hashlib.sha256(MATCHER_RUBRIC.encode("utf-8")).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────
# Decision Pair Cache
# ─────────────────────────────────────────────────────────────
class DecisionPairCache:
    def __init__(self, path: Path = ROOT / "data" / "eval_cache" / "decision_pairs.json"):
        self.path = path
        self.cache: Dict[str, dict] = self._load()
        self._dirty = 0

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"  [경고] decision cache 손상 — 새로 시작")
        return {}

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        self._dirty = 0

    @staticmethod
    def _norm(t: str) -> str:
        return "".join(t.split()).lower()

    def _key(self, expected_id, extracted_item, evidence_quote, match_mode) -> str:
        raw = "|".join([
            "v4", match_mode, MATCHER_MODEL, MATCHER_PROMPT_HASH, RUBRIC_VERSION,
            expected_id, self._norm(extracted_item), self._norm(evidence_quote),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, expected_id, extracted_item, evidence_quote, match_mode):
        return self.cache.get(self._key(expected_id, extracted_item, evidence_quote, match_mode))

    def set(self, expected_id, extracted_item, evidence_quote, match_mode, verdict: dict):
        self.cache[self._key(expected_id, extracted_item, evidence_quote, match_mode)] = verdict
        self._dirty += 1
        if self._dirty >= 10:   # 10건마다 flush (I/O 절충 + 중단 방어)
            self.flush()


# ─────────────────────────────────────────────────────────────
# LLM 판정 (안전장치 1: 파싱 실패 1회 재시도 → review_parse_error)
# ─────────────────────────────────────────────────────────────
def call_matcher(client, expected_item: str, extracted_item: str,
                 evidence_quote: str) -> dict:
    user = (f"expected: {expected_item}\n"
            f"extracted: {extracted_item}\n"
            f"evidence: {evidence_quote}")
    for attempt in range(2):  # 최대 2회 (파싱 실패 시 1회 재시도)
        try:
            completion = client.beta.chat.completions.parse(
                model=MATCHER_MODEL,
                messages=[{"role": "system", "content": MATCHER_RUBRIC},
                          {"role": "user", "content": user}],
                response_format=MatchVerdict,
                max_completion_tokens=4000,
                reasoning_effort="low",
            )
            v = completion.choices[0].message.parsed
            if v is None:
                continue  # 재시도
            # 값 검증 — 허용 범위 밖이면 review로 안전 폴백
            dec = v.decision if v.decision in ("match", "miss", "review") else "review"
            actor = v.actor_label if v.actor_label in (
                "bidder_action", "bidder_evaluation_rule",
                "buyer_internal_process", "context_or_background") else "context_or_background"
            return {"decision": dec, "actor_label": actor,
                    "confidence": v.confidence if v.confidence in ("high", "low") else "low",
                    "reason": v.reason or ""}
        except Exception as e:
            if attempt == 0:
                continue  # 1회 재시도
            return {"decision": "review", "actor_label": "context_or_background",
                    "confidence": "low",
                    "reason": f"review_parse_error: {type(e).__name__}"}
    # 두 번 다 parsed None
    return {"decision": "review", "actor_label": "context_or_background",
            "confidence": "low", "reason": "review_parse_error: parsed_none"}


# ─────────────────────────────────────────────────────────────
# fixture 자가검증 (안전장치 6) — 실전 전 matcher가 사람 정답 맞히나
# ─────────────────────────────────────────────────────────────
# actor_label → scope 귀결 (in_scope / out_of_scope)
ACTOR_TO_SCOPE = {
    "bidder_action": "in_scope",
    "bidder_evaluation_rule": "in_scope",
    "buyer_internal_process": "out_of_scope",
    "context_or_background": "out_of_scope",
}


def validate_matcher_with_fixtures(client) -> bool:
    """
    matcher가 사람이 고정한 정답을 맞히는지 검증.
    판정 기준: actor_label 정확 매칭이 아니라 scope 귀결(in/out) 일치.
    actor 라벨이 경계에서 갈려도 in_scope/out_of_scope 결론이 같으면 정답으로 인정
    (예: "제한경쟁·협상계약"은 context냐 buyer_internal_process냐 애매하나 둘 다 out_of_scope).
    """
    fx_path = ROOT / "evaluation" / "fixtures" / "eval_rubric_fixtures.json"
    fx = json.load(open(fx_path, encoding="utf-8"))
    items = fx["fixtures"]
    print(f"=== matcher fixture 자가검증 ({len(items)}개) ===")
    print("기준: scope 귀결(in/out) 일치 — actor 라벨이 갈려도 결론 같으면 정답\n")
    correct = 0
    for i, it in enumerate(items):
        v = call_matcher(client, expected_item="(actor 판정 전용)",
                         extracted_item=it["item"], evidence_quote=it["item"])
        got_actor = v["actor_label"]
        got_scope = ACTOR_TO_SCOPE.get(got_actor, "out_of_scope")
        want_scope = it["expected_scope"]
        allowed = it.get("allowed_actor_labels", [it["actor_label"]])
        # scope 귀결이 맞으면 정답 (actor 라벨 정확 일치는 부가 정보)
        ok = (got_scope == want_scope)
        exact = got_actor in allowed
        if ok:
            correct += 1
        if ok and exact:
            mark = "OK"
        elif ok and not exact:
            mark = f"OK(scope일치, 라벨은 {got_actor}/허용{allowed})"
        else:
            mark = f"MISS (got_scope={got_scope}, want={want_scope})"
        print(f"  [{i}] actor={got_actor:24} scope={got_scope:12} {mark}")
        print(f"       {it['item'][:50]}")
    acc = correct / len(items)
    print(f"\nscope 귀결 정확도: {correct}/{len(items)} ({acc:.0%})")
    passed = acc >= 0.75  # 75% 이상 통과 (8개 중 6개)
    print(f"{'[PASS]' if passed else '[FAIL]'} matcher 자가검증 "
          f"({'실전 투입 OK' if passed else 'prompt 재조정 필요'})")
    return passed


if __name__ == "__main__":
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
    print(f"matcher prompt hash: {MATCHER_PROMPT_HASH}\n")
    validate_matcher_with_fixtures(client)
