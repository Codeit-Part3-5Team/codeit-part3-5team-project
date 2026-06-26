# evaluation/explore_section_meta.py
# 라우트 C 1단계-B — section 메타가 의무/자격/제출/요구사항 구조를 얼마나 깔끔하게 담는지 탐색
# 목적: page_content 키워드 노이즈(71%)를 section 필터로 우회 가능한지 데이터로 판단
# 실행: (venv) python evaluation/explore_section_meta.py  (프로젝트 루트에서)

import json
from collections import Counter, defaultdict
from pathlib import Path

CHUNK_PATH = Path("data/processed/chunks_v1_enriched.json")

with open(CHUNK_PATH, encoding="utf-8") as f:
    chunks = json.load(f)

total = len(chunks)
print(f"총 청크 수: {total:,}\n")

# ── 0. section 필드 존재/결측 현황 ──────────────────────────────
has_section = sum(1 for c in chunks if c["metadata"].get("section"))
none_section = total - has_section
print("=" * 72)
print("section 필드 존재 현황")
print("=" * 72)
print(f"  section 있음: {has_section:,} ({has_section/total*100:.1f}%)")
print(f"  section 없음/빈값: {none_section:,} ({none_section/total*100:.1f}%)")

# content_type별 section 보유율
ct_section = defaultdict(lambda: [0, 0])  # ct → [section있음, 전체]
for c in chunks:
    ct = c["metadata"]["content_type"]
    ct_section[ct][1] += 1
    if c["metadata"].get("section"):
        ct_section[ct][0] += 1
print("\n  content_type별 section 보유율:")
for ct, (h, t) in ct_section.items():
    print(f"    {ct:14s} | {h:5,}/{t:5,} ({h/t*100:5.1f}%)")

# ── 1. section 고유값 개수 + 빈도 상위 ──────────────────────────
section_counter = Counter(
    c["metadata"].get("section", "(없음)") for c in chunks
)
print("\n" + "=" * 72)
print(f"section 고유값 개수: {len(section_counter):,}")
print("=" * 72)
print("\n빈도 상위 30개 section:")
for sec, cnt in section_counter.most_common(30):
    sec_disp = sec[:50] if sec else "(빈값)"
    print(f"  {cnt:5,} | {sec_disp}")

# ── 2. 라우트 C 관련 키워드를 'section명'에서 검색 ───────────────
# page_content가 아니라 section 필드 자체에 구조 키워드가 있는지
SECTION_KEYWORDS = {
    "자격": ["자격", "참가자격", "참가 자격"],
    "제출": ["제출", "구비서류", "첨부", "서류"],
    "요구사항": ["요구사항", "요구 사항"],
    "평가배점": ["평가", "배점", "심사"],
    "의무준수": ["의무", "준수", "유의사항", "지침"],
}
print("\n" + "=" * 72)
print("section명에 라우트C 구조 키워드가 포함된 청크 수 (page_content 아님!)")
print("=" * 72)
sec_kw_hit = defaultdict(set)  # 카테고리 → 청크 idx 집합
sec_kw_docs = defaultdict(set)
sec_kw_examples = defaultdict(set)  # 카테고리 → 실제 매칭된 section명 샘플
for idx, c in enumerate(chunks):
    sec = c["metadata"].get("section", "") or ""
    doc_id = c["metadata"]["doc_id"]
    for cat, kws in SECTION_KEYWORDS.items():
        for kw in kws:
            if kw in sec:
                sec_kw_hit[cat].add(idx)
                sec_kw_docs[cat].add(doc_id)
                if len(sec_kw_examples[cat]) < 8:
                    sec_kw_examples[cat].add(sec[:50])
                break
for cat in SECTION_KEYWORDS:
    n = len(sec_kw_hit[cat])
    nd = len(sec_kw_docs[cat])
    print(f"\n  [{cat}] section 매칭: {n:,} 청크 ({n/total*100:.1f}%) | {nd}/100 문서")
    for ex in sorted(sec_kw_examples[cat]):
        print(f"      · {ex}")

# 합집합 (자격+제출+의무 = C핵심 section)
core_cats = ["자격", "제출", "의무준수"]
core_union = set()
for cat in core_cats:
    core_union |= sec_kw_hit[cat]
print(f"\n  C핵심 section(자격∪제출∪의무준수): {len(core_union):,} 청크 ({len(core_union)/total*100:.1f}%)")
print(f"  → page_content 키워드 방식(71.4%) 대비 비교")

# ── 3. 골든셋 정답 문서에서 section 분포 확인 ────────────────────
GOLDEN = {
    "Q015": ("DOC_001", "requirement"), "Q016": ("DOC_038", "requirement"),
    "Q017": ("DOC_068", "requirement"), "Q018": ("DOC_001", "qualification"),
    "Q019": ("DOC_062", "qualification"), "Q020": ("DOC_004", "scoring"),
}
print("\n" + "=" * 72)
print("골든셋 정답 문서의 section 목록 (실제 추출 대상이 어떤 section에 있나)")
print("=" * 72)
for q, (doc, subtype) in GOLDEN.items():
    doc_secs = Counter(
        c["metadata"].get("section", "(없음)")
        for c in chunks if c["metadata"]["doc_id"] == doc
    )
    print(f"\n  {q} ({doc}, {subtype}) — section {len(doc_secs)}종:")
    for sec, cnt in doc_secs.most_common(12):
        print(f"      {cnt:3} | {sec[:55]}")

print("\n탐색 완료.")