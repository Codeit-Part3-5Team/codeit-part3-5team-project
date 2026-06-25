# evaluation/explore_compliance_keywords.py
# 라우트 C 1단계 — RFP 청크에서 의무/제출/자격/요구사항 키워드 현황 데이터 탐색
# 목적: 추출기 만들기 전에, 이 키워드들이 실제로 얼마나/어디에/어떤 형태로 있는지 파악
# 실행: (venv) python evaluation/explore_compliance_keywords.py  (프로젝트 루트에서)

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

CHUNK_PATH = Path("data/processed/chunks_v1_enriched.json")

# ── 키워드 셋 (4그룹) ──────────────────────────────────────────
KEYWORD_GROUPS = {
    "G1_의무필수": ["필수", "반드시", "의무", "하여야", "해야",
                   "할 것", "포함하여야", "갖추어야", "준수"],
    "G2_제출서류": ["제출서류", "구비서류", "첨부서류", "제출하여야", "제출"],
    "G3_자격요건": ["참가자격", "자격요건", "자격", "면허", "실적", "등록"],
    "G4_요구배점": ["요구사항", "평가항목", "배점", "가점"],  # 회색지대 — 분리 집계
}

# ── 로드 ────────────────────────────────────────────────────────
with open(CHUNK_PATH, encoding="utf-8") as f:
    chunks = json.load(f)

print(f"총 청크 수: {len(chunks):,}")
ct_dist = Counter(c["metadata"]["content_type"] for c in chunks)
print(f"content_type 분포: {dict(ct_dist)}\n")

# ── 집계 ────────────────────────────────────────────────────────
kw_chunk_count = defaultdict(int)          # 키워드 → 포함 청크 수
kw_ct = defaultdict(Counter)               # 키워드 → content_type Counter
kw_docs = defaultdict(set)                 # 키워드 → 출현 doc_id 집합
kw_samples = defaultdict(list)             # 키워드 → 샘플 문장 (최대 3개)
group_hit_chunks = defaultdict(set)        # 그룹 → 매칭 청크 인덱스 (중복 제거)

for idx, c in enumerate(chunks):
    text = c["page_content"]
    ct = c["metadata"]["content_type"]
    doc_id = c["metadata"]["doc_id"]

    for group, kws in KEYWORD_GROUPS.items():
        for kw in kws:
            if kw in text:
                kw_chunk_count[kw] += 1
                kw_ct[kw][ct] += 1
                kw_docs[kw].add(doc_id)
                group_hit_chunks[group].add(idx)
                if len(kw_samples[kw]) < 3:
                    for sent in re.split(r"[.\n]", text):
                        if kw in sent:
                            s = sent.strip()[:80]
                            if s and s not in kw_samples[kw]:
                                kw_samples[kw].append(s)
                                break

# ── 출력 1: 키워드별 빈도 (그룹 순) ──────────────────────────────
print("=" * 72)
print("키워드별 출현 현황 (청크 수 / 문서 수 / content_type 분포)")
print("=" * 72)
for group, kws in KEYWORD_GROUPS.items():
    print(f"\n── {group} ──")
    sorted_kws = sorted(kws, key=lambda k: kw_chunk_count[k], reverse=True)
    for kw in sorted_kws:
        cnt = kw_chunk_count[kw]
        ndocs = len(kw_docs[kw])
        ct_str = ", ".join(f"{k}:{v}" for k, v in kw_ct[kw].most_common())
        print(f"  {kw:10s} | 청크 {cnt:5,} | 문서 {ndocs:3}/100 | [{ct_str}]")

# ── 출력 2: 그룹별 커버리지 (중복 제거) ──────────────────────────
print("\n" + "=" * 72)
print("그룹별 커버리지 (그룹 내 키워드 하나라도 포함한 청크 수, 중복 제거)")
print("=" * 72)
total = len(chunks)
for group in KEYWORD_GROUPS:
    n = len(group_hit_chunks[group])
    print(f"  {group:14s} | {n:6,} 청크 ({n/total*100:5.1f}%)")

# C 핵심(G1+G2+G3) 합집합 vs 전체
core = group_hit_chunks["G1_의무필수"] | group_hit_chunks["G2_제출서류"] | group_hit_chunks["G3_자격요건"]
print(f"\n  C핵심(G1∪G2∪G3) | {len(core):6,} 청크 ({len(core)/total*100:5.1f}%)")

# ── 출력 3: 샘플 문장 (키워드가 실제 어떤 형태로 쓰이는지) ────────
print("\n" + "=" * 72)
print("샘플 문장 (키워드별 최대 3개 — 실제 문맥 확인용)")
print("=" * 72)
for group, kws in KEYWORD_GROUPS.items():
    print(f"\n── {group} ──")
    for kw in kws:
        if kw_samples[kw]:
            print(f"  [{kw}]")
            for s in kw_samples[kw]:
                print(f"     · {s}")

# ── 출력 4: 골든셋 정답 문서에서의 분포 (Q015~Q020 영역 교차 확인) ──
GOLDEN_DOCS = {
    "Q015": "DOC_001", "Q016": "DOC_038", "Q017": "DOC_068",
    "Q018": "DOC_001", "Q019": "DOC_062", "Q020": "DOC_004",
}
print("\n" + "=" * 72)
print("골든셋 정답 문서별 C핵심 키워드 청크 수 (추출 대상 영역 사전 점검)")
print("=" * 72)
core_kws = KEYWORD_GROUPS["G1_의무필수"] + KEYWORD_GROUPS["G2_제출서류"] + KEYWORD_GROUPS["G3_자격요건"]
for q, doc in GOLDEN_DOCS.items():
    doc_chunks = [c for c in chunks if c["metadata"]["doc_id"] == doc]
    hit = sum(1 for c in doc_chunks if any(kw in c["page_content"] for kw in core_kws))
    print(f"  {q} ({doc}) | 전체 {len(doc_chunks):3} 청크 중 C핵심 키워드 포함 {hit:3}")

print("\n탐색 완료.")