# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 청크 품질 검증 게이트 (DE: 도혁)
#  실행 : python evaluation/check_chunks.py
#  목적 : 청킹 산출물(chunks_v1 / enriched)의 품질을 전수 검사
#         청킹 로직을 어떻게 바꾸든 "100문서 다 정상인지" 한 번에 검증
#         = 청크 버그(중복·과세분화·PUA·빈청크)를 자동 차단하는 게이트
#
#  검사 항목 (FAIL 기준):
#    [1] doc 내 중복        : 같은 문서에서 동일 page_content 반복 (0이어야 PASS)
#    [2] 문서당 청크 이상치  : 평균 대비 과다 (과세분화 탐지)
#    [3] 표 과세분화        : 표 행 수 대비 청크 과다
#    [4] PUA 깨진문자       : 검색 노이즈 (0이어야 PASS)
#    [5] 빈/과소 청크       : page_content 빈값 또는 극단적 과소
#    [6] 토큰 상한 위반      : max_tokens 초과 청크
# =====================================================================
import os
import re
import sys
import json
import statistics
from collections import Counter, defaultdict

TARGET = sys.argv[1] if len(sys.argv) > 1 else "data/processed/chunks_v1.json"
MAX_TOKENS = 1024
MIN_TOKENS = 5          # 이보다 적으면 과소(의미 없는 파편)
OUTLIER_STD = 2.5       # 평균 + N*std 초과 시 청크수 이상치


def get_text(c):
    return c.get("page_content", c.get("text", ""))


def main():
    chunks = json.load(open(TARGET, encoding="utf-8"))
    print("=" * 60)
    print(f"  청크 품질 검증 게이트 — {os.path.basename(TARGET)}")
    print("=" * 60)
    print(f"  총 청크: {len(chunks)}")

    # [1] doc 내 중복
    keyed = Counter((c["metadata"]["doc_id"], get_text(c)) for c in chunks)
    dup = sum(v - 1 for v in keyed.values() if v > 1)
    dup_docs = sorted({k[0] for k, v in keyed.items() if v > 1})
    print(f"\n[1] doc 내 중복 청크: {dup}건"
          + (f" | 문서: {dup_docs[:10]}" if dup else " ✓"))

    # [2] 문서당 청크 수 이상치
    per_doc = Counter(c["metadata"]["doc_id"] for c in chunks)
    counts = list(per_doc.values())
    mean, std = statistics.mean(counts), (statistics.stdev(counts) if len(counts) > 1 else 0)
    threshold = mean + OUTLIER_STD * std
    outliers = [(d, n) for d, n in per_doc.most_common() if n > threshold]
    print(f"\n[2] 문서당 청크: 평균 {mean:.0f} ± {std:.0f} (이상치 기준 >{threshold:.0f})")
    print(f"  이상치 문서: {len(outliers)}개"
          + (f" → {outliers[:5]}" if outliers else " ✓"))

    # [3] 표 과세분화 (표 청크 비율이 비정상적으로 높은 문서)
    over_table = []
    for d, n in per_doc.items():
        tbl = sum(1 for c in chunks
                  if c["metadata"]["doc_id"] == d
                  and c["metadata"].get("content_type") == "table")
        if n > threshold and tbl / max(n, 1) > 0.8:   # 이상치이면서 표가 80%+
            over_table.append((d, tbl, n))
    print(f"\n[3] 표 과세분화 의심: {len(over_table)}개"
          + (f" → {[(d, f'{t}/{n}') for d, t, n in over_table[:5]]}" if over_table else " ✓"))

    # [4] PUA 깨진문자
    pua_chunks = [c for c in chunks if re.search(r"[\uE000-\uF8FF]", get_text(c))]
    print(f"\n[4] PUA 깨진문자 청크: {len(pua_chunks)}건"
          + ("" if pua_chunks else " ✓"))

    # [5] 빈/과소 청크
    empty = [c for c in chunks if not get_text(c).strip()]
    tiny = [c for c in chunks
            if 0 < c["metadata"].get("token_count", 0) < MIN_TOKENS]
    print(f"\n[5] 빈 청크: {len(empty)}건 | 과소 청크(<{MIN_TOKENS}토큰): {len(tiny)}건"
          + (" ✓" if not empty and not tiny else ""))

    # [6] 토큰 상한 위반
    over_tok = [c for c in chunks if c["metadata"].get("token_count", 0) > MAX_TOKENS]
    print(f"\n[6] 토큰 상한(>{MAX_TOKENS}) 위반: {len(over_tok)}건"
          + ("" if over_tok else " ✓"))

    # 종합 판정
    print("\n" + "=" * 60)
    fails = []
    if dup: fails.append(f"중복 {dup}")
    if outliers: fails.append(f"청크이상치 {len(outliers)}문서")
    if over_table: fails.append(f"표과세분화 {len(over_table)}문서")
    if pua_chunks: fails.append(f"PUA {len(pua_chunks)}")
    if empty: fails.append(f"빈청크 {len(empty)}")
    if over_tok: fails.append(f"토큰초과 {len(over_tok)}")
    status = "PASS" if not fails else "FAIL"
    print(f"  종합 판정: {status}")
    if fails:
        print(f"  실패 항목: {', '.join(fails)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
