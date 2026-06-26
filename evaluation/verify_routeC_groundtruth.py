# evaluation/verify_routeC_groundtruth.py
# 라우트 C 0단계 — 골든셋 6문항 정답 청크 실물 검증 (설계 분기점)
# 골든셋 파일 버전 의존 제거: C문항 6개 좌표를 v3 검증값으로 직접 명시
# 실행: (venv) python evaluation/verify_routeC_groundtruth.py

import json
from pathlib import Path
from collections import Counter

CHUNK_PATH = Path("data/processed/chunks_v1_enriched.json")

with open(CHUNK_PATH, encoding="utf-8") as f:
    chunks = json.load(f)

by_key = {}
for c in chunks:
    k = (c["metadata"]["doc_id"], c["metadata"]["chunk_index"])
    by_key[k] = c

# 청크 메타 키 확인 (table_id 존재 여부)
print("=" * 72)
print("0. 청크 메타데이터 키 목록 (table_id 존재 여부)")
print("=" * 72)
meta_keys = set()
for c in chunks:
    if c["metadata"]["content_type"] == "table":
        meta_keys.update(c["metadata"].keys())
        if len(meta_keys) > 20:
            break
print("  메타 키:", sorted(meta_keys))
table_keys = [k for k in meta_keys if "table" in k.lower()]
print(f"  -> 'table' 포함 키: {table_keys}")

# C문항 6개 — v3 검증 좌표 직접 명시
c_items = [
    {"id": "Q015", "q_subtype": "requirement",   "question": "한영대 주요 요구사항",
     "answer_chunk_labels": [["DOC_001", 5]]},
    {"id": "Q016", "q_subtype": "requirement",   "question": "정읍체육센터 주요 요구사항",
     "answer_chunk_labels": [["DOC_038", 5], ["DOC_038", 6], ["DOC_038", 9], ["DOC_038", 12], ["DOC_038", 91]]},
    {"id": "Q017", "q_subtype": "requirement",   "question": "NCIC 주요 요구사항",
     "answer_chunk_labels": [["DOC_068", 8], ["DOC_068", 40], ["DOC_068", 9], ["DOC_068", 10], ["DOC_068", 11]]},
    {"id": "Q018", "q_subtype": "qualification", "question": "한영대 입찰 자격 제출서류",
     "answer_chunk_labels": [["DOC_001", 6], ["DOC_001", 7], ["DOC_001", 8], ["DOC_001", 9], ["DOC_001", 10]]},
    {"id": "Q019", "q_subtype": "qualification", "question": "평택시 입찰 자격 제출서류",
     "answer_chunk_labels": [["DOC_062", 2], ["DOC_062", 3], ["DOC_062", 65]]},
    {"id": "Q020", "q_subtype": "scoring",       "question": "도시계획위 배점 기준",
     "answer_chunk_labels": [["DOC_004", 101], ["DOC_004", 103], ["DOC_004", 104]]},
]

print("\n" + "=" * 72)
print("1. 골든셋 6문항 정답 청크 실물 검증")
print("=" * 72)
for g in c_items:
    labels = g["answer_chunk_labels"]
    print(f"\n{'-'*72}")
    print(f"[{g['id']}] {g['q_subtype']} | 정답 청크 {len(labels)}개 | {g['question']}")
    indices = []
    for doc_id, idx in labels:
        c = by_key.get((doc_id, idx))
        if c is None:
            print(f"  WARN ({doc_id}, {idx}) 청크 못 찾음")
            continue
        m = c["metadata"]
        ct = m["content_type"]
        sec = m.get("section", "")
        indices.append(idx)
        tid = "-"
        for k in m:
            if "table" in k.lower() and "id" in k.lower():
                tid = m[k]; break
        print(f"  [{doc_id} #{idx}] type={ct:5} | table_id={tid} | sec={sec[:45]}")
    if len(indices) > 1:
        srt = sorted(indices)
        gaps = [srt[i+1]-srt[i] for i in range(len(srt)-1)]
        print(f"  -> chunk_index 정렬: {srt} | 연속={all(x==1 for x in gaps)} | 간격={gaps}")
    cts = [by_key[(d,i)]["metadata"]["content_type"] for d,i in labels if (d,i) in by_key]
    print(f"  -> content_type 분포: {dict(Counter(cts))}")

print("\n" + "=" * 72)
print("2. 정답 청크 page_content 실물 (표 깨짐 여부 눈검증)")
print("=" * 72)
for g in c_items:
    doc_id, idx = g["answer_chunk_labels"][0]
    c = by_key.get((doc_id, idx))
    if c is None:
        continue
    print(f"\n[{g['id']}] {doc_id} #{idx} ({c['metadata']['content_type']}) sec: {c['metadata'].get('section','')[:50]}")
    print("  " + "-"*68)
    for line in c["page_content"][:500].split("\n"):
        print(f"  | {line}")
    if len(c["page_content"]) > 500:
        print(f"  | ... (총 {len(c['page_content'])}자)")

print("\n검증 완료.")