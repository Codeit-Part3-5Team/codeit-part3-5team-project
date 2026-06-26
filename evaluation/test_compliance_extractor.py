# evaluation/test_compliance_extractor.py
# 라우트 C quote 검증기 단위 테스트
# 목적: 검증기가 (1) 정상 통과 (2) 환각 차단 (3) 경계 걸침 출처수정 을 실제로 하는지 증명
# 실행: (venv) python evaluation/test_compliance_extractor.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_processing"))
from compliance_extractor import ChecklistGenerator, cid, normalize

gen = ChecklistGenerator()

# 실제 청크에서 fixture 추출 (DOC_001 #6 = Q018 자격 정답 청크) ──
c6 = gen.by_key[("DOC_001", 6)]
c7 = gen.by_key[("DOC_001", 7)]
real_quote_6 = c6["page_content"].strip().split("\n")[0][:30]   # #6에 실제 있는 문장
# #7에만 있고 #6엔 없는 문장을 찾아 경계 테스트 fixture로 사용
n6 = normalize(c6["page_content"])
real_quote_7 = None
for line in c7["page_content"].split("\n"):
    s = line.strip()
    if len(s) >= 15 and normalize(s) not in n6:
        real_quote_7 = s[:30]
        break
assert real_quote_7, "경계 테스트용 #7 전용 문장을 못 찾음"

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    mark = "PASS" if condition else "FAIL"
    if condition: passed += 1
    else: failed += 1
    print(f"  [{mark}] {name}")

print("=" * 72)
print("quote 검증기 단위 테스트 — 3종 시나리오")
print("=" * 72)

# ── 시나리오 1: 정상 — quote가 선언된 청크에 실제 존재 ───────────
print("\n[1] 정상 항목 (quote가 원문에 그대로 있음)")
item_normal = {
    "category": "qualification",
    "item": "정상 항목",
    "evidence": [{"chunk_id": cid("DOC_001", 6), "quote": real_quote_6}],
}
vit, ok = gen.verify_quote(item_normal, "DOC_001")
check("전체 통과(ok=True)", ok is True)
check("evidence verified=True", vit["evidence"][0]["verified"] is True)
check("출처 수정 안 됨(boundary_match=False)", vit["evidence"][0].get("boundary_match") is False)

# ── 시나리오 2: 환각 — quote가 원문 어디에도 없음 ───────────────
print("\n[2] 환각 항목 (원문에 없는 문장을 지어냄)")
item_hallucination = {
    "category": "submission",
    "item": "환각 항목",
    "evidence": [{"chunk_id": cid("DOC_001", 6),
                  "quote": "이것은 원문에 절대 존재하지 않는 가짜 요구사항 문장입니다"}],
}
vit, ok = gen.verify_quote(item_hallucination, "DOC_001")
check("전체 탈락(ok=False)", ok is False)
check("evidence verified=False", vit["evidence"][0]["verified"] is False)

# ── 시나리오 3: 경계 걸침 — quote는 #7에 있는데 출처를 #6이라 잘못 선언 ──
print("\n[3] 경계 걸침 (출처를 #6이라 했지만 실제 quote는 #7에 있음)")
item_boundary = {
    "category": "requirement",
    "item": "경계 항목",
    "evidence": [{"chunk_id": cid("DOC_001", 6), "quote": real_quote_7}],  # 출처는 6, quote는 7것
}
vit, ok = gen.verify_quote(item_boundary, "DOC_001")
check("±1 fallback으로 통과(ok=True)", ok is True)
check("출처가 실제 위치 #7로 수정됨", vit["evidence"][0]["chunk_id"] == cid("DOC_001", 7))
check("boundary_match=True 기록", vit["evidence"][0].get("boundary_match") is True)

# ── 시나리오 4: dedupe — 완전중복만 제거, 조건 다른 건 유지 ──────
print("\n[4] dedupe (완전중복 제거 / 조건 다르면 유지)")
items = [
    {"category": "submission", "item": "협정서 제출",
     "evidence": [{"chunk_id": cid("DOC_001", 6), "quote": "공동수급 협정서 제출"}]},
    {"category": "submission", "item": "협정서 제출",   # 위와 완전중복
     "evidence": [{"chunk_id": cid("DOC_001", 6), "quote": "공동수급 협정서 제출"}]},
    {"category": "submission", "item": "협정서 원본 제출",  # quote 다름 → 유지돼야
     "evidence": [{"chunk_id": cid("DOC_001", 6), "quote": "대표사는 협정서 원본 제출"}]},
]
deduped = gen.dedupe(items)
check("완전중복 1개 제거 (3→2)", len(deduped) == 2)

# ── 시나리오 5: JSON 파싱 (정상/마크다운펜스/깨짐/객체래핑) ──────
print("\n[5] JSON 파싱 안전장치")
items, status = gen.parse_llm_output('[{"category":"requirement","item":"A","evidence":[{"chunk_id":"DOC_001#5","quote":"x"}]}]')
check("정상 JSON → ok + 1개", status == "ok" and len(items) == 1)

items, status = gen.parse_llm_output('```json\n[{"item":"B","evidence":[]}]\n```')
check("마크다운 펜스 제거 후 파싱", status == "ok" and len(items) == 1)

items, status = gen.parse_llm_output('[{"item":"C","evidence":[]}, {"item":"D"')
check("깨진 JSON → parse_fail (빈 리스트)", status == "parse_fail" and len(items) == 0)

items, status = gen.parse_llm_output('{"items":[{"item":"E","evidence":[]}]}')
check("객체 래핑 → items 추출", len(items) == 1)

items, status = gen.parse_llm_output('[{"category":"x"}, {"item":"F","evidence":[]}]')
check("필수필드 없는 항목 제외 (2→1)", len(items) == 1)

items, status = gen.parse_llm_output('')
check("빈 출력 → parse_fail", status == "parse_fail")

# ── 결과 ────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print(f"결과: {passed} PASS / {failed} FAIL")
print("=" * 72)
if failed == 0:
    print("검증기 정상 작동: 환각 차단 + 경계 출처수정 + 완전중복만 제거 확인됨.")
else:
    print("일부 실패 — 검증기 로직 점검 필요.")