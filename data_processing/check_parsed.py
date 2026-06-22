"""parsed_documents.json 검증"""
import json
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "data" / "processed" / "parsed_documents.json"
print("파일 크기:", f"{p.stat().st_size/1024/1024:.1f}MB")
docs = json.loads(p.read_text(encoding="utf-8"))
print("문서 수:", len(docs))
print("키:", list(docs[0].keys()))
print()

# 처음 3건
for d in docs[:3]:
    print(f"{d['doc_id']} | {d['char_count']:,}자 | {d['file_name'][:30]}")
    print(f"  본문앞: {d['text'][:70]}")
    print()

# 폴백 2건 (아세안, MILE) 정제 상태 확인
print("=== 폴백 2건 정제 확인 ===")
for d in docs:
    if "아세안" in d["file_name"] or "MILE" in d["file_name"]:
        print(f"[폴백] {d['doc_id']} | {d['char_count']:,}자")
        print(f"  본문앞: {d['text'][:70]}")
        print()

# 전체 통계
lens = [d["char_count"] for d in docs]
print("=== 전체 ===")
print(f"평균 {sum(lens)//len(lens):,}자 / 최소 {min(lens):,} / 최대 {max(lens):,}")
# 빈 문서 체크
empty = [d["doc_id"] for d in docs if d["char_count"] < 100]
print(f"100자 미만 문서: {empty if empty else '없음'}")