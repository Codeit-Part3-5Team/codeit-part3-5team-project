import sys
sys.path.insert(0, "data_processing")
from models import run_sentinels, AuditChecklistItem, AuditEvidence

passed = 0; failed = 0
def check(name, cond):
    global passed, failed
    print(("  [PASS] " if cond else "  [FAIL] ") + name)
    if cond: passed += 1
    else: failed += 1

def mk(item, quote):
    return AuditChecklistItem(
        item=item, primary_category="submission",
        evidence=[AuditEvidence(
            declared_chunk_id="DOC_001#6", resolved_chunk_ids=["DOC_001#6"],
            quote=quote, match_status="strict_verified",
            source_scope_valid=True, origin_window_id=0)])

print("=" * 72)
print("run_sentinels 단위 테스트")
print("=" * 72)

# [1] condition-loss: quote에 "경우", item엔 없음
print("\n[1] 조건 누락 (quote='~경우', item엔 조건 없음)")
it = run_sentinels(mk("협정서를 제출해야 함", "공동수급체를 구성하는 경우 협정서를 제출하여야 한다"))
check("condition_loss_risk True", it.condition_loss_risk is True)

# [2] 정상: 조건 보존
print("\n[2] 조건 보존 (둘 다 '경우')")
it = run_sentinels(mk("공동수급 경우 협정서 제출", "공동수급체를 구성하는 경우 협정서를 제출"))
check("condition_loss_risk False", it.condition_loss_risk is False)

# [3] numeric-mismatch: quote 80/20, item엔 없음
print("\n[3] 숫자 누락 (quote=80점/20점, item엔 숫자 없음)")
it = run_sentinels(mk("기술과 가격을 평가함", "기술능력평가 80점 가격평가 20점"))
check("numeric_mismatch_risk True", it.numeric_mismatch_risk is True)

# [4] 숫자 보존
print("\n[4] 숫자 보존 (둘 다 80/20)")
it = run_sentinels(mk("기술 80점 가격 20점", "기술능력평가 80점 가격평가 20점"))
check("numeric_mismatch_risk False", it.numeric_mismatch_risk is False)

# [5] deadline-loss: quote 18시, item엔 날짜만
print("\n[5] 기한 누락 (quote='18시까지', item엔 시간 없음)")
it = run_sentinels(mk("제안서를 7월 1일 제출", "제안서는 7월 1일 18시까지 제출"))
check("deadline_loss_risk True", it.deadline_loss_risk is True)

# [6] out_of_scope quote는 sentinel 근거 안 됨
print("\n[6] out_of_scope evidence는 sentinel 무시")
it = AuditChecklistItem(
    item="협정서 제출", primary_category="submission",
    evidence=[AuditEvidence(
        declared_chunk_id="DOC_001#91", resolved_chunk_ids=["DOC_001#91"],
        quote="공동수급체를 구성하는 경우 제출", match_status="out_of_scope_source",
        source_scope_valid=False, origin_window_id=0)])
it = run_sentinels(it)
check("out_of_scope는 flag 안 켬", it.condition_loss_risk is False)

print("\n" + "=" * 72)
print(f"결과: {passed} PASS / {failed} FAIL")
print("=" * 72)
