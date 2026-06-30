# evaluation/eval_q020.py
# Q020 실전 평가 — matcher 파이프라인이 끝까지 도는지 검증하는 단일 질문 러너
#
# 흐름:
#   1. Q020 expected(4개) + DOC_004 추출 캐시 로드
#   2. scope projection: scoring 카테고리를 1차 후보로 (2-pass: 못 찾으면 전체 재탐색)
#   3. expected 기준 순회 → 각 expected를 후보들과 LLM 판정 (Decision Pair Cache)
#   4. unique 집계: 한 expected에 여러 extracted가 match돼도 recovered=1
#   5. Confirmed / Review / Upper 3값
#   6. 매칭 안 된 추출 → actor 기반 Extra 분류
#   7. 사람 확인용 리포트 출력
#
# 이 단계의 목적: match/miss/review 판정 품질을 사람 눈으로 검증 (자동 채점 신뢰 전)

import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))

from route_c_matcher import (
    call_matcher, DecisionPairCache, ACTOR_TO_SCOPE, MATCHER_PROMPT_HASH
)

GOLDEN_PATH = ROOT / "data" / "processed" / "golden_dataset_v4.json"
EXTRACT_DIR = ROOT / "data" / "eval_cache" / "extractions"
REPORT_DIR = ROOT / "evaluation" / "reports"

# scope projection: 질문 q_subtype → 1차 후보 카테고리 (하드 필터 아님, 우선순위)
SCOPE_PRIMARY = {
    "scoring": ["scoring"],
    "requirement": ["requirement"],
    "qualification": ["qualification", "submission"],
}


def load_q020():
    g = json.load(open(GOLDEN_PATH, encoding="utf-8"))
    return next(q for q in g if q["id"] == "Q020")


def load_extraction(doc_id):
    return json.load(open(EXTRACT_DIR / f"{doc_id}_extracted.json", encoding="utf-8"))


def evaluate_question(client, q, verbose=True):
    qid = q["id"]
    doc_id = q["answer_doc_id"]
    mode = q.get("item_match_mode", "item")
    expected = q["expected_items"]
    subtype = q.get("q_subtype")

    ext = load_extraction(doc_id)
    all_items = ext["items"]  # [{item, primary_category, evidence:[{quote,...}]}]

    # 1차 후보: scope 카테고리
    primary_cats = SCOPE_PRIMARY.get(subtype, [subtype])
    primary_pool = [(i, it) for i, it in enumerate(all_items)
                    if it["primary_category"] in primary_cats]
    full_pool = list(enumerate(all_items))

    if verbose:
        print(f"=== {qid} [{subtype}] {doc_id} mode={mode} ===")
        print(f"expected: {len(expected)}개 | 추출 전체: {len(all_items)} "
              f"| 1차 후보({primary_cats}): {len(primary_pool)}\n")

    cache = DecisionPairCache()

    def quote_of(it):
        evs = it.get("evidence", [])
        return evs[0]["quote"] if evs else ""

    def judge(exp_id, exp_text, pool):
        """expected 하나를 후보 pool과 판정. (matched_indices, reviews) 반환"""
        matched, reviews = [], []
        for idx, it in pool:
            cached = cache.get(exp_id, it["item"], quote_of(it), mode)
            if cached:
                v = cached
            else:
                v = call_matcher(client, exp_text, it["item"], quote_of(it))
                cache.set(exp_id, it["item"], quote_of(it), mode, v)
            if v["decision"] == "match":
                matched.append((idx, it, v))
            elif v["decision"] == "review":
                reviews.append((idx, it, v))
        return matched, reviews

    # expected 기준 순회 (2-pass: 1차 후보 → 못 찾으면 전체)
    results = []
    matched_ext_indices = set()
    for ei, exp_text in enumerate(expected):
        exp_id = f"{qid}-E{ei}"
        matched, reviews = judge(exp_id, exp_text, primary_pool)
        pass2 = False
        if not matched:  # 1차에서 match 없으면 전체에서 재탐색
            extra_pool = [(i, it) for i, it in full_pool
                          if (i, it) not in primary_pool and it["primary_category"] not in primary_cats]
            m2, r2 = judge(exp_id, exp_text, extra_pool)
            if m2:
                matched, pass2 = m2, True
            reviews += r2
        for idx, it, v in matched:
            matched_ext_indices.add(idx)
        status = "match" if matched else ("review" if reviews else "miss")
        results.append({
            "exp_id": exp_id, "exp_text": exp_text, "status": status,
            "n_matched": len(matched), "n_review": len(reviews),
            "matched": [(idx, it["item"]) for idx, it, v in matched],
            "reviews": [(idx, it["item"]) for idx, it, v in reviews],
            "pass2": pass2,
        })
    cache.flush()

    # ── 집계 (unique 기준) ──
    n_exp = len(expected)
    confirmed = sum(1 for r in results if r["status"] == "match")
    review_cnt = sum(1 for r in results if r["status"] == "review")
    miss_cnt = sum(1 for r in results if r["status"] == "miss")

    confirmed_recovery = confirmed / n_exp
    review_burden = review_cnt / n_exp
    unresolved_upper = (confirmed + review_cnt) / n_exp

    # ── 리포트 ──
    print("─" * 64)
    print("【 expected 기준 — Recovery 】")
    for r in results:
        tag = {"match": "✓ MATCH", "review": "△ REVIEW", "miss": "✗ MISS"}[r["status"]]
        p2 = " (2-pass)" if r["pass2"] else ""
        print(f"  {tag}{p2}  {r['exp_text'][:50]}")
        for idx, item in r["matched"][:3]:
            print(f"          match← S{idx}: {item[:45]}")
        for idx, item in r["reviews"][:2]:
            print(f"          review← S{idx}: {item[:45]}")
    print()
    print(f"  Confirmed Recovery : {confirmed}/{n_exp} = {confirmed_recovery:.0%}")
    print(f"  Review Burden      : {review_cnt}/{n_exp} = {review_burden:.0%}")
    print(f"  Unresolved Upper   : {confirmed+review_cnt}/{n_exp} = {unresolved_upper:.0%}")

    # review 비율 경고
    if review_burden > 0.30:
        print(f"  [경고] review 비율 {review_burden:.0%} > 30% — matcher/expected 재점검 필요")

    print("\n" + "─" * 64)
    print(f"【 매칭 정보 】 expected와 match된 unique 추출: {len(matched_ext_indices)}개")
    print(f"  (전체 추출 {len(all_items)} 중 — 나머지는 Extra 분류 대상, 다음 단계)")

    # 결과 저장
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "qid": qid, "doc_id": doc_id, "mode": mode,
        "prompt_hash": MATCHER_PROMPT_HASH,
        "n_expected": n_exp,
        "confirmed_recovery": confirmed_recovery,
        "review_burden": review_burden,
        "unresolved_upper": unresolved_upper,
        "results": results,
        "matched_unique_extracted": len(matched_ext_indices),
    }
    rpt = REPORT_DIR / f"{qid}_report.json"
    json.dump(out, open(rpt, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n리포트 저장: {rpt}")
    return out


if __name__ == "__main__":
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()
    q = load_q020()
    evaluate_question(client, q)
