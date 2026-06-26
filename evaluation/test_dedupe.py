import sys
sys.path.insert(0, "data_processing")
from models import dedupe_items, AuditChecklistItem, AuditEvidence

passed = 0; failed = 0
def check(name, cond):
    global passed, failed
    print(("  [PASS] " if cond else "  [FAIL] ") + name)
    if cond: passed += 1
    else: failed += 1

def mk_item(item, cat, quote, chunk_ids):
    return AuditChecklistItem(
        item=item, primary_category=cat,
        evidence=[AuditEvidence(
            declared_chunk_id=chunk_ids[0], resolved_chunk_ids=chunk_ids,
            quote=quote, match_status="strict_verified",
            source_scope_valid=True, origin_window_id=0)])

print("=" * 72)
print("dedupe_items 단위 테스트")
print("=" * 72)

# [1] 완전중복 제거
print("\n[1] 완전중복 (item+quote+source 동일) → 1개")
items = [
    mk_item("협정서 제출", "submission", "협정서를 제출하여야 한다", ["DOC_001#6"]),
    mk_item("협정서 제출", "submission", "협정서를 제출하여야 한다", ["DOC_001#6"]),
]
r = dedupe_items(items)
check("2 → 1", len(r) == 1)

# [2] overlap 병합 (같은 item+quote, source 인접 #6/#7, category 다름)
print("\n[2] overlap 병합 (#6/#7 인접, category 다름) → 1개 + candidates")
items = [
    mk_item("협정서 제출", "submission", "협정서를 제출하여야 한다", ["DOC_001#6"]),
    mk_item("협정서 제출", "qualification", "협정서를 제출하여야 한다", ["DOC_001#7"]),
]
r = dedupe_items(items)
check("2 → 1 (인접이라 병합)", len(r) == 1)
if r:
    check("category_candidates에 둘 다", set([r[0].primary_category] + r[0].category_candidates) >= {"submission", "qualification"})
    check("category_conflict True", r[0].category_conflict is True)

# [3] boilerplate 유지 (같은 item+quote, source 멀리 #6/#91)
print("\n[3] boilerplate (#6/#91 멀리 떨어짐) → 2개 유지 (안 합침)")
items = [
    mk_item("기한까지 제출", "submission", "지정된 기한까지 제출하여야 한다", ["DOC_001#6"]),
    mk_item("기한까지 제출", "submission", "지정된 기한까지 제출하여야 한다", ["DOC_001#91"]),
]
r = dedupe_items(items)
check("2 → 2 (멀어서 유지)", len(r) == 2)

# [4] 다른 item 유지 (조건 다름)
print("\n[4] 다른 item (조건 다름) → 2개 유지")
items = [
    mk_item("공동수급 시 협정서 제출", "submission", "공동수급체 구성 시 협정서 제출", ["DOC_001#6"]),
    mk_item("대표사 원본 제출", "submission", "대표사는 원본을 제출", ["DOC_001#6"]),
]
r = dedupe_items(items)
check("2 → 2 (다른 항목)", len(r) == 2)

print("\n" + "=" * 72)
print(f"결과: {passed} PASS / {failed} FAIL")
print("=" * 72)
