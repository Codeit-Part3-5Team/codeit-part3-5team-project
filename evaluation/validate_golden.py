# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 골든셋 라벨 정합성 검증 (DE: 도혁)
#  실행 : python evaluation/validate_golden.py
#  목적 : 평가용 골든셋(golden_dataset.json)의 정답 청크 라벨이
#         실제 enriched 청크와 정합하는지 검증
#         = 청크 제작자(DE)가 평가 데이터 품질을 보증 (GIGO 방지)
#
#  ※ 역할 경계:
#     - 골든셋 작성 = 아인님
#     - 평가 실행(retrieval/answer) = 수민님
#     - 라벨 정합성 보증(청크 실재·정답 정합) = 도혁(청크 만든 사람)
#
#  검증 항목:
#    [1] 라벨 존재  : answer_chunk_labels가 전부 실재 청크인지 (누락 0)
#    [2] 정답 정합  : auto 생성 정답이 청크 메타와 일치하는지 (예산·기관·마감·공고번호)
#    [3] refusal   : 답변거부 문항의 라벨 빈 값 정상 + PII 거부가 마스킹과 연결되는지
# =====================================================================
import os
import re
import sys
import json
from collections import Counter, defaultdict

GOLDEN = "data/processed/golden_dataset.json"
FINAL  = "data/processed/chunks_v1_enriched.json"


def num(s):
    d = re.sub(r"[^0-9]", "", str(s))
    return int(d) if d else None


def load():
    g = json.load(open(GOLDEN, encoding="utf-8"))
    e = json.load(open(FINAL, encoding="utf-8"))
    return g, e


def build_meta(chunks):
    """doc_id별 메타요약 청크(chunk_index=-1) 메타 인덱싱."""
    meta = {}
    keys = set()
    for ch in chunks:
        m = ch["metadata"]
        keys.add((m["doc_id"], m["chunk_index"]))
        if m["chunk_index"] == -1:
            meta[m["doc_id"]] = {
                "budget": m.get("budget_amount"),
                "agency": m.get("agency_normalized") or m.get("agency"),
                "bid_end": m.get("bid_end"),
                "ann_no": m.get("announcement_no"),
            }
    return meta, keys


def check_label_existence(golden, chunk_keys):
    """[1] 모든 라벨이 실재 청크인지."""
    total, missing = 0, []
    for q in golden:
        for lab in q.get("answer_chunk_labels", []):
            total += 1
            if (lab[0], lab[1]) not in chunk_keys:
                missing.append((q["id"], q["category"], tuple(lab)))
    return total, missing


def check_answer_consistency(golden, meta):
    """[2] auto 정답이 청크 메타와 일치하는지 (메타 대조 가능한 subtype만)."""
    results = defaultdict(lambda: {"ok": 0, "fail": []})
    for q in golden:
        if q.get("source") != "auto":
            continue
        st = q.get("q_subtype", "")
        did = q.get("answer_doc_id")
        # 메타 대조는 단일 문서(str) fact 문항만 — multi_doc(list)/refusal(None) 스킵
        if not isinstance(did, str):
            continue
        m = meta.get(did, {})
        ans = q.get("answer", "")

        if st == "fact_budget_amount":
            ok = (num(ans) == m.get("budget"))
        elif st == "fact_agency":
            ag = m.get("agency", "") or ""
            ok = bool(ag) and (ag.replace(" ", "") in ans.replace(" ", "")
                               or ans.replace(" ", "") in ag.replace(" ", ""))
        elif st == "fact_bid_end":
            ok = (re.sub(r"[^0-9]", "", ans)[:8] ==
                  re.sub(r"[^0-9]", "", str(m.get("bid_end")))[:8])
        elif st == "fact_announcement_no":
            ok = (num(ans) == num(m.get("ann_no")))
        else:
            continue  # 메타 대조 불가 subtype (synthesis/comparison 등)은 스킵

        if ok:
            results[st]["ok"] += 1
        else:
            results[st]["fail"].append((q["id"], did, ans, m))
    return dict(results)


def check_refusal(golden):
    """[3] refusal 문항: 라벨 빈 값 정상 + PII 거부 식별."""
    refusals = [q for q in golden if q["category"] == "refusal"]
    bad_label = [q["id"] for q in refusals if q.get("answer_chunk_labels")]
    pii_refusals = [(q["id"], q.get("q_subtype")) for q in refusals
                    if "pii" in q.get("q_subtype", "")]
    return len(refusals), bad_label, pii_refusals


def main():
    golden, chunks = load()
    meta, chunk_keys = build_meta(chunks)

    print("=" * 60)
    print("  골든셋 라벨 정합성 검증 (DE 데이터 보증)")
    print("=" * 60)
    print(f"  골든셋 {len(golden)}문항 | enriched 청크 {len(chunk_keys)}개")
    print(f"  카테고리: {dict(Counter(q['category'] for q in golden))}")

    # [1] 라벨 존재
    total, missing = check_label_existence(golden, chunk_keys)
    print(f"\n[1] 라벨 존재 검증")
    print(f"  총 라벨 {total}개 | 존재하지 않는 라벨: {len(missing)}건")
    if missing:
        for qid, cat, key in missing[:10]:
            print(f"    ✗ {qid} ({cat}): {key}")

    # [2] 정답 정합
    print(f"\n[2] auto 정답 메타 대조")
    cons = check_answer_consistency(golden, meta)
    cons_fail = 0
    for st, r in sorted(cons.items()):
        cons_fail += len(r["fail"])
        mark = "✓" if not r["fail"] else "✗"
        print(f"  {mark} {st}: 일치 {r['ok']} / 불일치 {len(r['fail'])}")
        for qid, did, ans, m in r["fail"][:3]:
            print(f"      {qid} {did}: 정답='{ans}' vs 메타={m}")

    # [3] refusal
    print(f"\n[3] refusal 설계 검증")
    n_ref, bad_label, pii_ref = check_refusal(golden)
    print(f"  refusal {n_ref}건 | 라벨 비어야 정상 → 위반: {len(bad_label)}건")
    print(f"  PII 거부 문항(마스킹 연결): {pii_ref}")

    # 종합
    print("\n" + "=" * 60)
    status = "PASS" if (not missing and cons_fail == 0 and not bad_label) else "FAIL"
    print(f"  종합 정합성: {status}")
    print(f"  - 라벨 누락 {len(missing)} / 정답 불일치 {cons_fail} / refusal 위반 {len(bad_label)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
