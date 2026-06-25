import sys
sys.path.insert(0, "data_processing")
from models import build_window_context, verify_evidence, normalize_for_quote_match
from compliance_extractor import ChecklistGenerator

gen = ChecklistGenerator()
chunks = gen.assemble_document("DOC_001")
all_windows = gen.build_windows(chunks)

passed = 0
failed = 0
def check(name, cond):
    global passed, failed
    print(("  [PASS] " if cond else "  [FAIL] ") + name)
    if cond: passed += 1
    else: failed += 1

def cid(c):
    return c["metadata"]["doc_id"] + "#" + str(c["metadata"]["chunk_index"])

print("=" * 72)
print("verify_evidence 단위 테스트 (scope / position / cross-chunk)")
print("=" * 72)

# 윈도우0 context
w0 = all_windows[0]
ctx0 = build_window_context(0, w0, chunks)

# [1] strict: 윈도우0 첫 청크 실제 quote
print("\n[1] strict 매칭")
c0 = w0[0]
q0 = c0["page_content"].strip().split("\n")[0][:30]
ev = verify_evidence(cid(c0), q0, ctx0)
check("strict_verified", ev.match_status == "strict_verified")
check("scope_valid True", ev.source_scope_valid is True)

# [2] 환각
print("\n[2] 환각 (원문에 없음)")
ev = verify_evidence(cid(c0), "절대존재하지않는가짜요구사항문장입니다", ctx0)
check("unverified_evidence", ev.match_status == "unverified_evidence")
check("scope_valid False", ev.source_scope_valid is False)

# [3] out_of_scope: quote는 문서에 있지만 윈도우0 범위 밖 청크에서
print("\n[3] out_of_scope_source (문서엔 있으나 윈도우 범위 밖)")
# 윈도우0에 없는 뒷쪽 청크 하나 찾기
allowed0 = ctx0["allowed_chunk_ids"]
outside = None
for c in chunks:
    if cid(c) not in allowed0:
        outside = c
        break
# outside의 실제 quote를, 출처는 outside라고 선언 (정직한 출처지만 범위 밖)
q_out = outside["page_content"].strip().split("\n")[0][:30]
ev = verify_evidence(cid(outside), q_out, ctx0)
check("out_of_scope_source 또는 강등됨", ev.match_status in ("out_of_scope_source", "out_of_scope_cross_chunk"))
check("scope_valid False (범위밖)", ev.source_scope_valid is False)
print("     실제:", ev.match_status, "| resolved:", ev.resolved_chunk_ids)

# [4] position 기반 이웃: 윈도우0 두번째 청크 quote를 첫 청크 출처로 선언
print("\n[4] relocated (출처 틀렸지만 인접에서 발견 → 범위 내)")
if len(w0) >= 2:
    c1 = w0[1]
    n0 = normalize_for_quote_match(c0["page_content"])
    # c1에만 있는 문장 찾기
    relq = None
    for line in c1["page_content"].split("\n"):
        s = line.strip()
        if len(s) >= 15 and normalize_for_quote_match(s) not in n0:
            relq = s[:30]; break
    if relq:
        # 출처를 c0으로 잘못 선언, 실제 quote는 c1
        ev = verify_evidence(cid(c0), relq, ctx0)
        check("relocated 또는 strict (범위내 발견)", ev.match_status in ("relocated_verified", "strict_verified", "cross_chunk_verified"))
        check("scope_valid True (둘 다 윈도우 내)", ev.source_scope_valid is True)
        print("     실제:", ev.match_status, "| resolved:", ev.resolved_chunk_ids)
    else:
        print("     (c1 전용 문장 못 찾음 — skip)")

print("\n" + "=" * 72)
print(f"결과: {passed} PASS / {failed} FAIL")
print("=" * 72)
