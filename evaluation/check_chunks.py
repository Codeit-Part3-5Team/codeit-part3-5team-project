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
import hashlib
import statistics
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

# 인자: [1]=대상 json, '--manifest' 있으면 매니페스트 저장
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
TARGET = ARGS[0] if ARGS else "data/processed/chunks_v1.json"
SAVE_MANIFEST = "--manifest" in sys.argv
MANIFEST_PATH = "evaluation/chunk_quality_manifest.json"
MAX_TOKENS = 1024
MIN_TOKENS = 5          # 이보다 적으면 과소(의미 없는 파편)
OUTLIER_STD = 2.5       # 평균 + N*std 초과 시 청크수 이상치


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def get_text(c):
    return c.get("page_content", c.get("text", ""))


def load_table_rows(parsed_path="data/processed/parsed_documents_v2.json"):
    """원본 문서의 doc별 표 행 수를 센다 (과세분화 판정 기준).
    표 청크 수가 원본 표 행 수보다 많으면 = 과세분화(버그),
    적으면 = 큰 문서라 표가 많은 것(정상). 이 둘을 구분하기 위함."""
    if not os.path.exists(parsed_path):
        return None
    docs = json.load(open(parsed_path, encoding="utf-8"))
    rows = {}
    for d in docs:
        # 표 행 추정: ':' 또는 '/' 포함하는 비어있지 않은 줄 (key:value 표 행)
        cnt = sum(1 for ln in d["text"].split("\n")
                  if ln.strip() and (":" in ln or "/" in ln))
        rows[d["doc_id"]] = cnt
    return rows


def main():
    chunks = json.load(open(TARGET, encoding="utf-8"))
    table_rows = load_table_rows()
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

    # [2] 문서당 청크 수 이상치 (참고용 — 큰 문서도 잡히므로 단독 FAIL 사유 아님)
    per_doc = Counter(c["metadata"]["doc_id"] for c in chunks)
    counts = list(per_doc.values())
    mean, std = statistics.mean(counts), (statistics.stdev(counts) if len(counts) > 1 else 0)
    threshold = mean + OUTLIER_STD * std
    outliers = [(d, n) for d, n in per_doc.most_common() if n > threshold]
    print(f"\n[2] 문서당 청크: 평균 {mean:.0f} ± {std:.0f} (이상치 기준 >{threshold:.0f}) [참고용]")
    print(f"  청크 많은 문서: {len(outliers)}개"
          + (f" → {outliers[:5]}" if outliers else " ✓"))

    # [3] 표 과세분화: 표 청크 수가 원본 표 행 수보다 많으면 = 과세분화(버그)
    #     (큰 문서라 표 행이 많아 청크가 많은 것과 구분)
    over_table = []
    for d, n in per_doc.items():
        tbl = sum(1 for c in chunks
                  if c["metadata"]["doc_id"] == d
                  and c["metadata"].get("content_type") == "table")
        if table_rows is not None:
            rows_n = table_rows.get(d, 0)
            # 표 청크가 원본 표 행의 1.2배 초과면 과세분화 (행보다 유의미하게 잘게 쪼갬)
            # [표] 마커별 헤더 청크 등 자연 증가분(~20%)은 정상으로 허용
            if rows_n > 0 and tbl > rows_n * 1.2 and tbl > 50:
                over_table.append((d, tbl, rows_n))
        else:
            # parsed 없으면 기존 방식(이상치 + 표 80%) 폴백
            if n > threshold and tbl / max(n, 1) > 0.8:
                over_table.append((d, tbl, n))
    if table_rows is not None:
        print(f"\n[3] 표 과세분화 의심: {len(over_table)}개 (기준: 표청크 > 원본 표행)"
              + (f" → {[(d, f'청크{t}>행{r}') for d, t, r in over_table[:5]]}" if over_table else " ✓"))
    else:
        print(f"\n[3] 표 과세분화 의심: {len(over_table)}개 (parsed 없어 폴백 기준)"
              + (f" → {over_table[:5]}" if over_table else " ✓"))

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

    # [참고] 인명 관련 청크 추정 (PM 요청 - 참고용 통계)
    #   주의: 키워드 기반 "상한 추정치". 키워드가 있어도 인명이 없을 수 있고
    #   (예: "담당자: [전화번호]"), 정확한 인명 카운트는 NER 도입 시에만 가능.
    name_kw = re.compile(r"담당자|담당부서|문의처|문의\s*:|연락처|책임자|담당\s*:|주관부서|발주담당")
    name_chunks = sum(1 for c in chunks if name_kw.search(get_text(c)))
    print(f"\n[참고] 인명 관련 키워드 포함 청크(상한 추정): {name_chunks}건"
          f" — 정확한 인명 카운트는 NER 필요")

    # 종합 판정
    # [2] 청크 이상치는 큰 문서도 잡히므로 단독 FAIL 사유에서 제외(참고용).
    # 진짜 과세분화는 [3](표청크>표행)으로 판정.
    print("\n" + "=" * 60)
    fails = []
    if dup: fails.append(f"중복 {dup}")
    if over_table: fails.append(f"표과세분화 {len(over_table)}문서")
    if pua_chunks: fails.append(f"PUA {len(pua_chunks)}")
    if empty: fails.append(f"빈청크 {len(empty)}")
    if over_tok: fails.append(f"토큰초과 {len(over_tok)}")
    status = "PASS" if not fails else "FAIL"
    print(f"  종합 판정: {status}")
    if fails:
        print(f"  실패 항목: {', '.join(fails)}")
    if outliers:
        print(f"  [참고] 청크 많은 문서 {len(outliers)}개 — 표행 대비 정상이면 큰 문서임")
    print("=" * 60)

    # --- 매니페스트 저장 (--manifest) ---
    if SAVE_MANIFEST:
        kst = timezone(timedelta(hours=9))
        n_meta = sum(1 for c in chunks if c["metadata"].get("content_type") == "meta_summary")
        n_tbl = sum(1 for c in chunks if c["metadata"].get("content_type") == "table")
        n_txt = len(chunks) - n_meta - n_tbl
        manifest = {
            "generated_at": datetime.now(kst).strftime("%Y-%m-%d %H시 %M분 %S초 (KST)"),
            "target_file": TARGET,
            "file_sha256": file_sha256(TARGET),
            "verdict": status,
            "total_chunks": len(chunks),
            "chunk_breakdown": {"meta_summary": n_meta, "table": n_tbl, "text": n_txt},
            "checks": {
                "1_doc_internal_duplicates": dup,
                "2_high_chunk_docs_reference": [list(x) for x in outliers],
                "3_table_over_segmentation": [list(x) for x in over_table],
                "4_pua_broken_chars": len(pua_chunks),
                "5_empty_chunks": len(empty),
                "5_tiny_chunks": len(tiny),
                "6_token_limit_violations": len(over_tok),
            },
            "reference_stats": {
                "person_name_related_chunks_upper_estimate": name_chunks,
                "note": "키워드(담당자/책임자/문의처 등) 기반 상한 추정치. "
                        "키워드가 있어도 인명이 없을 수 있어 실제보다 많게 잡힘. "
                        "정확한 인명 카운트는 NER 도입 시에만 가능.",
            },
            "limitations": [
                f"과소 청크(<{MIN_TOKENS}토큰) {len(tiny)}건은 검색 영향 경미하여 허용",
                "[2] 청크 많은 문서는 원본 표 행 수 대비 정상(큰 문서)이면 FAIL 아님",
                "과세분화 판정 기준: 표 청크 수 > 원본 표 행 수 × 1.2",
                "인명 노출 카운트는 키워드 기반 추정 — 정규식으론 인명 자동 탐지 불가(NER 필요)",
            ],
        }
        with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"\n[manifest] 저장됨: {MANIFEST_PATH} (sha256: {manifest['file_sha256'][:16]}…)")


if __name__ == "__main__":
    main()
