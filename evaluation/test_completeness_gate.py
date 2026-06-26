import sys
sys.path.insert(0, "data_processing")
from models import AuditChecklistItem, AuditEvidence, dedupe_items

passed = 0; failed = 0
def check(name, cond):
    global passed, failed
    print(("  [PASS] " if cond else "  [FAIL] ") + name)
    if cond: passed += 1
    else: failed += 1

# gate 판정 로직만 떼어서 테스트 (run의 핵심부 재현)
def decide_status(items, windows_total, windows_failed):
    windows_completed = windows_total - windows_failed
    verified = {"strict_verified", "relocated_verified", "cross_chunk_verified"}
    needs_review = False
    for it in items:
        if any(ev.match_status not in verified for ev in it.evidence):
            needs_review = True
        if it.condition_loss_risk or it.numeric_mismatch_risk or it.deadline_loss_risk:
            needs_review = True
        if not it.evidence:
            needs_review = True
    if windows_completed == 0:
        return "failed"
    elif windows_failed > 0:
        return "partial"
    elif needs_review:
        return "complete_with_review"
    return "complete_verified"

def mk(status="strict_verified", cond=False):
    it = AuditChecklistItem(
        item="x", primary_category="submission",
        evidence=[AuditEvidence(
            declared_chunk_id="DOC_001#6", resolved_chunk_ids=["DOC_001#6"],
            quote="q", match_status=status, source_scope_valid=True, origin_window_id=0)])
    it.condition_loss_risk = cond
    return it

print("=" * 72)
print("completeness gate 분기 테스트")
print("=" * 72)

# [1] 전부 정상 → complete_verified
print("\n[1] 전 윈도우 성공 + verified → complete_verified")
check("complete_verified", decide_status([mk()], 4, 0) == "complete_verified")

# [2] 윈도우 일부 실패 → partial
print("\n[2] 윈도우 1개 실패 → partial")
check("partial", decide_status([mk()], 4, 1) == "partial")

# [3] 윈도우 성공인데 unverified evidence → complete_with_review
print("\n[3] unverified evidence 있음 → complete_with_review")
check("complete_with_review", decide_status([mk(status="unverified_evidence")], 4, 0) == "complete_with_review")

# [4] 윈도우 성공인데 sentinel flag → complete_with_review
print("\n[4] sentinel flag 켜짐 → complete_with_review")
check("complete_with_review", decide_status([mk(cond=True)], 4, 0) == "complete_with_review")

# [5] out_of_scope evidence → complete_with_review
print("\n[5] out_of_scope evidence → complete_with_review")
check("complete_with_review", decide_status([mk(status="out_of_scope_source")], 4, 0) == "complete_with_review")

# [6] 전 윈도우 실패 → failed
print("\n[6] 전 윈도우 실패 → failed")
check("failed", decide_status([], 4, 4) == "failed")

print("\n" + "=" * 72)
print(f"결과: {passed} PASS / {failed} FAIL")
print("=" * 72)
