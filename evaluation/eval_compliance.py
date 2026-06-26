# evaluation/eval_compliance.py
# 라우트 C 평가 수집기 — quote-source 일치율 + 골든셋 source coverage
# 주의: quote 일치율 ≠ 환각률. quote가 원문에 있어도 조건 왜곡 가능 → 정성 감사는 별도(사람).
# 실행: (venv) python evaluation/eval_compliance.py
#   (키 없는 현재: Mock 추출 결과로 측정기 동작 확인. 키 확보 후 실제 추출에 그대로 사용.)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_processing"))
from compliance_extractor import ChecklistGenerator, GOLDEN_COORDS, cid, normalize


def evaluate_doc(gen, doc_id, use_mock=True):
    """한 문서의 추출 결과를 평가. 3개 지표 반환."""
    result = gen.generate_checklist(doc_id, use_mock=use_mock)
    items = result["items"]

    # ── 지표 1: quote-source 일치율 ──────────────────────────────
    # 추출 항목의 evidence quote가 (검증 후) verified=True인 비율
    total_ev = 0
    verified_ev = 0
    boundary_fixed = 0
    for it in items:
        for ev in it.get("evidence", []):
            total_ev += 1
            if ev.get("verified"):
                verified_ev += 1
            if ev.get("boundary_match"):
                boundary_fixed += 1
    match_rate = verified_ev / total_ev if total_ev else 0.0

    # ── 지표 2: 골든셋 source coverage ───────────────────────────
    # 이 문서의 골든셋 정답 청크 중, 추출 항목이 근거로 잡은 비율
    golden_chunks = set()
    for qid, coords in GOLDEN_COORDS.items():
        for d, idx in coords:
            if d == doc_id:
                golden_chunks.add(cid(d, idx))
    extracted_sources = set()
    for it in items:
        for ev in it.get("evidence", []):
            extracted_sources.add(ev.get("chunk_id", ""))
    hit = golden_chunks & extracted_sources
    golden_hit_rate = len(hit) / len(golden_chunks) if golden_chunks else None

    # ── 지표 3: 카테고리 분포 (어떤 유형이 빠졌나 점검) ───────────
    from collections import Counter
    cat_dist = Counter(it.get("category", "unknown") for it in items)

    return {
        "doc_id": doc_id,
        "item_count": len(items),
        "evidence_count": total_ev,
        "quote_match_rate": match_rate,
        "boundary_fixed": boundary_fixed,
        "golden_chunks": len(golden_chunks),
        "golden_hit": len(hit),
        "golden_hit_rate": golden_hit_rate,
        "category_dist": dict(cat_dist),
        "parse_status": [w.get("status") for w in result["windows"]],
    }

# ── PM 인터페이스: generate_checklist (재헌님 LangGraph 노드 연동용) ──
# PM 디코 제시 시그니처: generate_checklist(query, doc_id) -> dict

if __name__ == "__main__":
    gen = ChecklistGenerator()
    test_docs = ["DOC_001", "DOC_038", "DOC_068", "DOC_062", "DOC_004"]

    print("=" * 76)
    print("라우트 C 평가 수집기 (현재 Mock — 키 확보 후 실제 추출에 그대로 적용)")
    print("=" * 76)
    print("주의: quote 일치율은 '환각 차단' 지표일 뿐. 조건 왜곡·항목 누락은 정성 감사(사람)로 별도 확인.\n")

    agg_match = []
    for doc_id in test_docs:
        r = evaluate_doc(gen, doc_id, use_mock=True)
        ghr = f"{r['golden_hit_rate']*100:.0f}%" if r['golden_hit_rate'] is not None else "N/A"
        print(f"[{r['doc_id']}]")
        print(f"  추출 항목: {r['item_count']} | evidence: {r['evidence_count']}")
        print(f"  quote 일치율: {r['quote_match_rate']*100:.1f}% (경계수정 {r['boundary_fixed']}건)")
        print(f"  골든셋 source hit: {r['golden_hit']}/{r['golden_chunks']} ({ghr})")
        print(f"  카테고리 분포: {r['category_dist']}")
        print(f"  파싱 상태: {r['parse_status']}")
        print()
        agg_match.append(r['quote_match_rate'])

    avg = sum(agg_match) / len(agg_match) if agg_match else 0
    print("=" * 76)
    print(f"전체 평균 quote 일치율: {avg*100:.1f}%")
    print("=" * 76)
    print("\n[키 확보 후 할 일]")
    print("  1. use_mock=False로 실제 gpt-5-mini 추출 → 위 지표 자동 측정")
    print("  2. 골든셋 5개 문서는 정성 감사: 누락/조건왜곡/원문밖 항목 3개 기준 사람이 직접 확인")

if __name__ == "__main__":
    gen = ChecklistGenerator()
    test_docs = ["DOC_001", "DOC_038", "DOC_068", "DOC_062", "DOC_004"]

    print("=" * 76)
    print("라우트 C 평가 수집기 (현재 Mock — 키 확보 후 실제 추출에 그대로 적용)")
    print("=" * 76)
    print("주의: quote 일치율은 '환각 차단' 지표일 뿐. 조건 왜곡·항목 누락은 정성 감사(사람)로 별도 확인.\n")

    agg_match = []
    for doc_id in test_docs:
        r = evaluate_doc(gen, doc_id, use_mock=True)
        ghr = f"{r['golden_hit_rate']*100:.0f}%" if r['golden_hit_rate'] is not None else "N/A"
        print(f"[{r['doc_id']}]")
        print(f"  추출 항목: {r['item_count']} | evidence: {r['evidence_count']}")
        print(f"  quote 일치율: {r['quote_match_rate']*100:.1f}% (경계수정 {r['boundary_fixed']}건)")
        print(f"  골든셋 source hit: {r['golden_hit']}/{r['golden_chunks']} ({ghr})")
        print(f"  카테고리 분포: {r['category_dist']}")
        print(f"  파싱 상태: {r['parse_status']}")
        print()
        agg_match.append(r['quote_match_rate'])

    avg = sum(agg_match) / len(agg_match) if agg_match else 0
    print("=" * 76)
    print(f"전체 평균 quote 일치율: {avg*100:.1f}%")
    print("=" * 76)
    print("\n[키 확보 후 할 일]")
    print("  1. use_mock=False로 실제 gpt-5-mini 추출 → 위 지표 자동 측정")
    print("  2. 골든셋 5개 문서는 정성 감사: 누락/조건왜곡/원문밖 항목 3개 기준 사람이 직접 확인")
    print("  3. quote 일치율 낮으면 프롬프트 'quote는 원문 그대로' 강조 보완")
