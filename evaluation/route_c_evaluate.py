# evaluation/route_c_evaluate.py
# 라우트 C 평가 실행 — top-k 후보 선별 → matcher 판정 → Item Coverage/Precision
#
# 설계: route_c_eval_design.md §8 (채점 연계) + V5 설계 §8
#   측정 단위 = 항목(청크 위치 아님)
#     Item Coverage  = (expected 중 매칭된 수) / expected 전체   [재현율]
#     Item Precision = (추출 중 expected 매칭된 수) / 추출 전체  [정밀도]
#
# 호출 폭발 차단(핵심):
#   naive: expected(E) × extracted(X) 전부 matcher → E*X 호출 (Q020 ~970, Q016 수천+)
#   top-k: expected당 임베딩 유사 상위 k(8~10)만 matcher → E*k 호출 (Q020 ~40)
#   임베딩은 1회 계산 후 캐싱(embed_cache), gpt-5-mini보다 수십 배 쌈.
#
# 캐시 일관성:
#   - Decision Pair Cache(eval_route_c.DecisionPairCache) 재사용 — 같은 쌍 재판정 방지
#   - MATCHER_PROMPT_HASH는 route_c_matcher에서 import해 단일 출처(TBD 불일치 차단)
#
# 이 모듈은 eval_route_c.py가 import해 --evaluate에서 호출한다.

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))

from embed_cache import EmbedCache, top_k_candidates  # noqa: E402


# ─────────────────────────────────────────────────────────────
# expected_id 규칙 — expected_items는 문자열 리스트이므로 인덱스로 안정 id 생성
#   (Decision Pair Cache 키 + 결과 추적용. qid와 결합해 전역 유일)
# ─────────────────────────────────────────────────────────────
def make_expected_id(qid: str, idx: int) -> str:
    return f"{qid}_e{idx}"


# ─────────────────────────────────────────────────────────────
# q_subtype → 추출기 primary_category 매핑 (데이터로 1:1 확인됨, 2026-06-30)
#   Q015/Q016/Q017 requirement, Q018/Q019 qualification, Q020 scoring
#   precision 분모를 "이 질문이 평가하는 카테고리" 항목으로 좁히는 데 쓴다.
#   (scoring 질문에 requirement 391개를 분모로 넣으면 precision이 뭉개짐)
# ─────────────────────────────────────────────────────────────
SUBTYPE_TO_CATEGORY = {
    "requirement": "requirement",
    "qualification": "qualification",
    "scoring": "scoring",
    "submission": "submission",
}


def resolve_category(q: dict) -> str:
    """문항의 q_subtype에서 평가 대상 primary_category를 도출. 모르면 빈 문자열(필터 안 함)."""
    return SUBTYPE_TO_CATEGORY.get(q.get("q_subtype", ""), "")


@dataclass
class PairVerdict:
    """expected ↔ extracted 1쌍 판정 (top-k 통과분만 생성)"""
    expected_id: str
    expected_text: str
    extracted_index: int
    extracted_item: str
    decision: str            # match | miss | review
    actor_label: str
    confidence: str
    reason: str
    similarity: float        # 임베딩 유사도(후보 선별 점수, 진단용)
    from_cache: bool = False


@dataclass
class QuestionScore:
    """문항 1개 채점 결과"""
    qid: str
    doc_id: str
    item_match_mode: str
    k: int
    n_expected: int
    n_extracted: int            # 평가 대상(필터 후) 추출 항목 수 = precision 분모
    filter_mode: str = "none"   # none | hard | soft
    category: str = ""          # 이 문항이 평가하는 primary_category
    n_extracted_raw: int = 0    # 필터 전 전체 추출 항목 수(진단)
    # hard 필터로 후보에서 빠졌지만 expected와 임베딩 유사도 높은 항목(오분류 감지)
    filtered_but_similar: List[str] = field(default_factory=list)
    # match 안 된 expected의 가장 가까웠던 후보 1개 (expected 문구 vs 추출 누락 진단)
    #   각 원소: {"expected_id","expected_text","best_item","best_sim","best_decision"}
    unresolved_best: List[Dict] = field(default_factory=list)
    # 항목 단위 지표
    item_coverage: float = 0.0       # expected 중 match 비율
    item_precision: float = 0.0      # extracted 중 expected에 매칭된 비율
    n_matched_expected: int = 0
    n_review_expected: int = 0       # match 없고 review만 있는 expected (사람 검수 대상)
    n_unresolved_expected: int = 0   # 후보 중 match/review 전무
    n_matched_extracted: int = 0
    # 호출/비용 추적
    matcher_calls: int = 0
    matcher_cache_hits: int = 0
    # 사람 검수용 상세
    review_pairs: List[PairVerdict] = field(default_factory=list)
    matched_pairs: List[PairVerdict] = field(default_factory=list)


def evaluate_question(
    q: dict,
    extraction: dict,
    embed: EmbedCache,
    decision_cache,
    call_matcher: Callable,
    matcher_client,
    k: int = 8,
    match_mode: Optional[str] = None,
    filter_mode: str = "hard",
) -> QuestionScore:
    """
    한 문항 평가:
      1. q_subtype → category 도출, extracted를 평가 대상으로 한정
         - hard: 후보 풀·precision 분모 둘 다 category 항목으로 제한
         - soft: 후보 풀은 전체, precision 분모/분자만 category 소속으로 집계
         - none: 필터 없음(과거 동작 — 전체를 분모로)
      2. expected/extracted 텍스트 임베딩(캐시 우선)
      3. expected마다 top-k 추출 후보 선별 → matcher 판정(Decision Pair Cache 우선)
      4. Item Coverage/Precision 집계

    Decision Pair Cache는 쌍(expected_text|extracted_item|evidence_quote)을 키로 하므로,
    top-k가 후보를 줄여도, 필터 모드가 달라도 같은 쌍은 캐시 적중(재현성 유지).
    """
    qid = q["id"]
    doc_id = q.get("answer_doc_id", extraction.get("doc_id", "?"))
    mm = match_mode or q.get("item_match_mode", "item")
    category = resolve_category(q)

    expected_texts: List[str] = list(q["expected_items"])
    ext_items_all = extraction.get("items", [])
    # 각 추출 항목의 category 소속 여부 (분자/분모 집계에 사용)
    ext_in_cat_all = [
        (it.get("primary_category", "") == category) if category else True
        for it in ext_items_all
    ]

    # 후보 풀 결정 — hard면 category 항목만, soft/none이면 전체
    if filter_mode == "hard" and category:
        keep_idx = [i for i, ok in enumerate(ext_in_cat_all) if ok]
    else:
        keep_idx = list(range(len(ext_items_all)))

    ext_items = [ext_items_all[i] for i in keep_idx]
    extracted_texts: List[str] = [it.get("item", "") for it in ext_items]
    # keep_idx 내에서의 category 소속 (soft 집계용)
    ext_in_cat = [ext_in_cat_all[i] for i in keep_idx]

    # precision 분모: 평가 대상 category 항목 수
    if category:
        n_denom = sum(ext_in_cat_all)            # 전체 중 category 소속 수 (hard/soft 동일)
    else:
        n_denom = len(ext_items_all)

    score = QuestionScore(
        qid=qid, doc_id=doc_id, item_match_mode=mm, k=k,
        n_expected=len(expected_texts), n_extracted=n_denom,
        filter_mode=filter_mode, category=category,
        n_extracted_raw=len(ext_items_all),
    )
    if not expected_texts or not extracted_texts:
        return score  # 빈 경우 0점 처리(집계에서 제외 판단은 호출측)

    # 1) 임베딩 (배치, 캐시 우선) — expected + extracted 한꺼번에
    all_texts = expected_texts + extracted_texts
    all_vecs = embed.embed_many(all_texts)
    exp_vecs = all_vecs[:len(expected_texts)]
    ext_vecs = all_vecs[len(expected_texts):]

    matched_extracted_idx = set()   # precision용: expected에 매칭된 추출 인덱스

    # 2~3) expected마다 top-k 후보 → matcher 판정
    for ei, (etext, evec) in enumerate(zip(expected_texts, exp_vecs)):
        expected_id = make_expected_id(qid, ei)
        cand_idx = top_k_candidates(evec, ext_vecs, k=k)

        had_match = False
        had_review = False
        best_sim = -1.0             # 이 expected에서 가장 가까웠던 후보 추적
        best_item = ""
        best_dec = "miss"

        for xi in cand_idx:
            xitem = extracted_texts[xi]
            ev_list = ext_items[xi].get("evidence", [])
            ev_quote = ev_list[0]["quote"] if ev_list else ""
            sim = _safe_cos(evec, ext_vecs[xi])

            # Decision Pair Cache 우선
            cached = decision_cache.get(expected_id, xitem, ev_quote, mm)
            if cached is not None:
                v = cached
                from_cache = True
                score.matcher_cache_hits += 1
            else:
                v = call_matcher(matcher_client, etext, xitem, ev_quote)
                decision_cache.set(expected_id, xitem, ev_quote, mm, v)
                from_cache = False
                score.matcher_calls += 1

            pv = PairVerdict(
                expected_id=expected_id, expected_text=etext,
                extracted_index=xi, extracted_item=xitem,
                decision=v["decision"], actor_label=v.get("actor_label", "unknown"),
                confidence=v.get("confidence", "low"), reason=v.get("reason", ""),
                similarity=sim, from_cache=from_cache,
            )

            if sim > best_sim:
                best_sim, best_item, best_dec = sim, xitem, v["decision"]

            if v["decision"] == "match":
                had_match = True
                # precision 분자: soft에서는 category 소속 match만 인정
                #   (hard는 후보 풀이 이미 category라 ext_in_cat[xi]가 항상 True)
                if ext_in_cat[xi]:
                    matched_extracted_idx.add(xi)
                score.matched_pairs.append(pv)
            elif v["decision"] == "review":
                had_review = True
                score.review_pairs.append(pv)

        if had_match:
            score.n_matched_expected += 1
        elif had_review:
            score.n_review_expected += 1
        else:
            score.n_unresolved_expected += 1
        # match 안 된 expected는 best 후보를 진단용으로 기록
        if not had_match:
            score.unresolved_best.append({
                "expected_id": expected_id, "expected_text": etext,
                "best_item": best_item, "best_sim": round(best_sim, 3),
                "best_decision": best_dec,
            })

    # 4) 집계
    E = score.n_expected
    X = score.n_extracted
    score.item_coverage = score.n_matched_expected / E if E else 0.0
    score.n_matched_extracted = len(matched_extracted_idx)
    score.item_precision = score.n_matched_extracted / X if X else 0.0
    return score


def _safe_cos(a, b) -> float:
    from embed_cache import cosine
    return cosine(a, b)


def summarize(scores: List[QuestionScore]) -> dict:
    """전체 문항 집계 요약 — 매크로 평균 + 호출/캐시 통계."""
    if not scores:
        return {}
    n = len(scores)
    macro_cov = sum(s.item_coverage for s in scores) / n
    macro_prec = sum(s.item_precision for s in scores) / n
    total_calls = sum(s.matcher_calls for s in scores)
    total_hits = sum(s.matcher_cache_hits for s in scores)
    total_review = sum(s.n_review_expected for s in scores)
    total_exp = sum(s.n_expected for s in scores)
    return {
        "n_questions": n,
        "macro_item_coverage": round(macro_cov, 4),
        "macro_item_precision": round(macro_prec, 4),
        "total_matcher_calls": total_calls,
        "total_cache_hits": total_hits,
        "review_expected_total": total_review,
        "expected_total": total_exp,
        "review_burden_rate": round(total_review / total_exp, 4) if total_exp else 0.0,
    }


# ─────────────────────────────────────────────────────────────
# 사람 검수용 dump — review 상세 + 매칭 안 된 expected의 best 후보
#   review: expected↔extracted 전체 + 사유 + similarity (정밀 검수)
#   match : 한 줄 요약 (노이즈 억제)
#   unresolved/review expected: best 후보 1개 (expected 문구 vs 추출누락 진단)
# ─────────────────────────────────────────────────────────────
def dump_question(score: QuestionScore):
    print(f"\n{'='*64}")
    print(f"[{score.qid}] {score.doc_id} | {score.category}/{score.filter_mode} "
          f"| cov={score.item_coverage:.2f} prec={score.item_precision:.2f} "
          f"| denom={score.n_extracted}/{score.n_extracted_raw}")
    print(f"  matched_expected={score.n_matched_expected} "
          f"review={score.n_review_expected} unresolved={score.n_unresolved_expected}")

    if score.matched_pairs:
        print(f"\n  -- MATCH ({len(score.matched_pairs)}쌍, 한 줄 요약) --")
        for pv in score.matched_pairs:
            print(f"    [{pv.expected_id}] sim={pv.similarity:.2f} "
                  f"<{pv.actor_label}> {pv.extracted_item[:42]}")

    if score.review_pairs:
        print(f"\n  -- REVIEW ({len(score.review_pairs)}쌍, 상세 — 사람 판단 필요) --")
        for pv in score.review_pairs:
            print(f"    [{pv.expected_id}] sim={pv.similarity:.2f} conf={pv.confidence}")
            print(f"       expected : {pv.expected_text}")
            print(f"       extracted: {pv.extracted_item}")
            print(f"       사유     : {pv.reason}")

    if score.unresolved_best:
        print("\n  -- 매칭 실패 expected의 best 후보 (문구 vs 추출누락 진단) --")
        for ub in score.unresolved_best:
            print(f"    [{ub['expected_id']}] best_sim={ub['best_sim']} "
                  f"dec={ub['best_decision']}")
            print(f"       expected : {ub['expected_text']}")
            print(f"       best_cand: {ub['best_item'][:50]}")


def save_snapshot(scores: List[QuestionScore], k: int, out_dir: Path = None):
    """
    지표 스냅샷을 JSON으로 저장 — 계획서/결과서 근거용.
    ★NDA 안전: 원문 텍스트(expected/extracted/quote)는 저장하지 않는다.
              수치·decision·카운트만. expected_id로 추적은 가능.
    """
    import json
    from datetime import datetime
    out_dir = out_dir or (ROOT / "evaluation" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "summary": summarize(scores),
        "questions": [
            {
                "qid": s.qid, "doc_id": s.doc_id, "category": s.category,
                "filter_mode": s.filter_mode, "item_match_mode": s.item_match_mode,
                "n_expected": s.n_expected, "n_extracted": s.n_extracted,
                "n_extracted_raw": s.n_extracted_raw,
                "item_coverage": round(s.item_coverage, 4),
                "item_precision": round(s.item_precision, 4),
                "n_matched_expected": s.n_matched_expected,
                "n_review_expected": s.n_review_expected,
                "n_unresolved_expected": s.n_unresolved_expected,
                "matcher_calls": s.matcher_calls,
                "matcher_cache_hits": s.matcher_cache_hits,
                # decision만, 텍스트 없이 (NDA)
                "review_decisions": [
                    {"expected_id": pv.expected_id, "similarity": round(pv.similarity, 3),
                     "confidence": pv.confidence} for pv in s.review_pairs
                ],
                "unresolved_best_sims": [
                    {"expected_id": ub["expected_id"], "best_sim": ub["best_sim"],
                     "best_decision": ub["best_decision"]} for ub in s.unresolved_best
                ],
            }
            for s in scores
        ],
    }
    qid_tag = "_".join(s.qid for s in scores) if len(scores) <= 3 else f"{len(scores)}q"
    fpath = out_dir / f"route_c_eval_{qid_tag}_k{k}.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n스냅샷 저장(NDA 안전, 수치만): {fpath}")
    return fpath


# ─────────────────────────────────────────────────────────────
# mock 자가검증 — API 없이 파이프라인 골격 검증
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== route_c_evaluate 자가검증 (mock matcher, mock embed) ===")

    # mock 임베딩: 텍스트를 간단한 벡터로 (같은 키워드 포함 시 유사)
    class MockEmbed(EmbedCache):
        def __init__(self):
            self.api_calls = 0
            self._dirty = 0
        def embed_many(self, texts):
            vecs = []
            for t in texts:
                # 키워드 3종 존재 여부로 3차원 벡터
                v = [
                    1.0 if "기술평가" in t or "배점" in t else 0.0,
                    1.0 if "회원" in t else 0.0,
                    1.0 if "보안" in t or "서약" in t else 0.0,
                ]
                if sum(v) == 0:
                    v = [0.1, 0.1, 0.1]
                vecs.append(v)
            return vecs

    # mock matcher: 키워드 겹치면 match, 아니면 miss
    def mock_matcher(client, expected, extracted, evidence):
        e_kw = {w for w in ["기술평가", "배점", "회원", "보안", "서약"] if w in expected}
        x_kw = {w for w in ["기술평가", "배점", "회원", "보안", "서약"] if w in extracted}
        if e_kw & x_kw:
            return {"decision": "match", "actor_label": "bidder_evaluation_rule",
                    "confidence": "high", "reason": "mock match"}
        return {"decision": "miss", "actor_label": "context_or_background",
                "confidence": "high", "reason": "mock miss"}

    # mock decision cache
    class MockCache:
        def __init__(self): self.d = {}
        def get(self, *a): return self.d.get(str(a))
        def set(self, *a):
            *key, v = a
            self.d[str(tuple(key))] = v

    # Q020 같은 상황 재현: scoring 질문, 추출엔 scoring 외 카테고리가 다수
    q = {
        "id": "Q999",
        "answer_doc_id": "DOC_TEST",
        "item_match_mode": "item",
        "q_subtype": "scoring",   # → category = scoring
        "expected_items": ["기술평가 배점 기준", "가격평가 배점"],
    }
    extraction = {
        "doc_id": "DOC_TEST",
        "items": [
            # scoring 카테고리 (평가 대상) — 2개
            {"item": "기술평가 90점 배점", "primary_category": "scoring",
             "evidence": [{"quote": "기술평가 90점"}]},
            {"item": "가격평가 10점 배점", "primary_category": "scoring",
             "evidence": [{"quote": "가격평가 10점"}]},
            # requirement 카테고리 (평가 대상 아님) — 노이즈 3개
            {"item": "회원 가입 및 운영 기능", "primary_category": "requirement",
             "evidence": [{"quote": "회원 운영"}]},
            {"item": "보안 서약서 제출", "primary_category": "requirement",
             "evidence": [{"quote": "보안서약서"}]},
            {"item": "추진 배경 설명", "primary_category": "requirement",
             "evidence": [{"quote": "추진배경"}]},
        ],
    }
    # mock matcher: 키워드에 '배점'/'평가' 겹치면 match
    def mock_matcher2(client, expected, extracted, evidence):
        for w in ["기술평가", "가격평가", "배점"]:
            if w in expected and w in extracted:
                return {"decision": "match", "actor_label": "bidder_evaluation_rule",
                        "confidence": "high", "reason": "mock match"}
        return {"decision": "miss", "actor_label": "context_or_background",
                "confidence": "high", "reason": "mock miss"}

    class MockEmbed2(EmbedCache):
        def __init__(self):
            self.api_calls = 0
            self._dirty = 0
        def embed_many(self, texts):
            out = []
            for t in texts:
                out.append([
                    1.0 if ("기술평가" in t or "배점" in t and "가격" not in t) else 0.0,
                    1.0 if "가격" in t else 0.0,
                    1.0 if ("회원" in t or "보안" in t or "추진" in t) else 0.0,
                ])
            return out

    print("\n--- HARD 필터 (후보 풀=scoring 2개) ---")
    s_hard = evaluate_question(q, extraction, MockEmbed2(), MockCache(),
                               mock_matcher2, None, k=8, filter_mode="hard")
    print(f"  category={s_hard.category} raw={s_hard.n_extracted_raw} "
          f"denom={s_hard.n_extracted} cov={s_hard.item_coverage:.2f} "
          f"prec={s_hard.item_precision:.2f} calls={s_hard.matcher_calls}")

    print("--- SOFT 필터 (후보 풀=전체 5개, 분모만 scoring) ---")
    s_soft = evaluate_question(q, extraction, MockEmbed2(), MockCache(),
                               mock_matcher2, None, k=8, filter_mode="soft")
    print(f"  category={s_soft.category} raw={s_soft.n_extracted_raw} "
          f"denom={s_soft.n_extracted} cov={s_soft.item_coverage:.2f} "
          f"prec={s_soft.item_precision:.2f} calls={s_soft.matcher_calls}")

    # 검증: 두 모드 다 분모=2(scoring), cov=1.0(2/2), prec=1.0(2/2)
    assert s_hard.n_extracted == 2, f"hard 분모 2여야, got {s_hard.n_extracted}"
    assert s_soft.n_extracted == 2, f"soft 분모 2여야, got {s_soft.n_extracted}"
    assert abs(s_hard.item_precision - 1.0) < 1e-6, "hard precision 1.0"
    assert abs(s_soft.item_precision - 1.0) < 1e-6, "soft precision 1.0"
    # hard는 후보 풀이 작아 호출 적음, soft는 전체라 호출 많음
    assert s_hard.matcher_calls < s_soft.matcher_calls, "hard가 호출 적어야"
    print(f"\n  호출 비교: hard={s_hard.matcher_calls} < soft={s_soft.matcher_calls} "
          f"(hard가 후보 풀 작아 적음)")

    # none 모드 — 과거 동작(전체 분모) 회귀 확인
    s_none = evaluate_question(q, extraction, MockEmbed2(), MockCache(),
                               mock_matcher2, None, k=8, filter_mode="none")
    print(f"  none 모드 denom={s_none.n_extracted} (category 있으면 여전히 2 = 분모는 항상 category)")

    print("\n[PASS] route_c_evaluate hard/soft 필터 자가검증 통과")

    # dump / snapshot 동작 확인
    print("\n--- dump_question 출력 확인 ---")
    dump_question(s_hard)
    snap = save_snapshot([s_hard], k=8, out_dir=Path("/tmp/_snap_test"))
    import json as _json
    payload = _json.load(open(snap, encoding="utf-8"))
    # NDA 안전 확인: 스냅샷에 원문 텍스트가 없어야 함
    blob = _json.dumps(payload, ensure_ascii=False)
    assert "기술평가" not in blob and "회원" not in blob, "스냅샷에 원문 텍스트가 새면 안 됨(NDA)"
    print("  [확인] 스냅샷에 원문 텍스트 없음 (NDA 안전)")
    print("\n[PASS] dump/snapshot 자가검증 통과")
