# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 3단계 마스킹 검증 스캐너 (DE: 도혁)
#  목적 : parsed → masked → final(enriched) 3단계로 제한 식별정보 추적
#
#  단계별 통과 조건 (LLM 핑퐁 반영 — 각 단계 목적이 다름):
#    1) parsed_documents_v2   : PII 후보 인벤토리 (★발견되는 게 정상)
#    2) masked_documents_v3   : 제한 식별정보 잔존 0건 (★FAIL 기준)
#    3) chunks_v1_enriched    : 청킹·직렬화 후 재유입 0건 (★FAIL 기준)
#                               + 모든 문자열 필드(metadata/file_name) 재귀 스캔
#
#  ※ "PII 0건"이 아니라 "마스킹 정책 v3의 제한 식별정보 잔존 0건"을 검증
# =====================================================================
import json
import os
import sys
from collections import defaultdict

# import 환경 독립 처리 (어디서 실행하든 mask_common 찾도록)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask_common import (
    scan_restricted, scan_emails_by_class, MASK_TOKEN_RE,
    LANDLINE_RE, normalize, POLICY_VERSION,
)

PARSED = "data/processed/parsed_documents_v2.json"
MASKED = "data/processed/masked_documents_v3.json"
FINAL  = "data/processed/chunks_v1_enriched.json"


def get_text(item):
    return item.get("text", item.get("page_content", ""))


def scan_all_string_fields(item):
    """문서/청크의 모든 문자열 필드(text + metadata + file_name) 재귀 스캔."""
    findings = defaultdict(list)

    def walk(obj, path):
        if isinstance(obj, str):
            r = scan_restricted(obj)
            for k, v in r.items():
                findings[f"{path}:{k}"].extend(v)
        elif isinstance(obj, dict):
            for kk, vv in obj.items():
                walk(vv, f"{path}.{kk}")
        elif isinstance(obj, list):
            for i, vv in enumerate(obj):
                walk(vv, f"{path}[{i}]")

    walk(item, "root")
    return dict(findings)


def stage1_parsed():
    """원본 PII 후보 인벤토리 (발견 = 정상)."""
    docs = json.load(open(PARSED, encoding="utf-8"))
    inventory = defaultdict(int)
    docs_with_pii = set()
    for d in docs:
        r = scan_restricted(get_text(d))
        for k, v in r.items():
            inventory[k] += len(v)
            docs_with_pii.add(d["doc_id"])
    return {"total_docs": len(docs), "pii_inventory": dict(inventory),
            "docs_with_pii_candidates": len(docs_with_pii)}


def stage2_masked():
    """마스킹 후 제한 식별정보 잔존 검사 (0이어야 PASS)."""
    docs = json.load(open(MASKED, encoding="utf-8"))
    residual = defaultdict(list)
    for d in docs:
        r = scan_restricted(get_text(d))
        for k, v in r.items():
            residual[k].extend([(d["doc_id"], x) for x in v])
    # 이메일 분류 집계
    email_class = defaultdict(int)
    for d in docs:
        ec = scan_emails_by_class(get_text(d))
        for k, v in ec.items():
            email_class[k] += len(v)
    return {"total_docs": len(docs),
            "restricted_residual": {k: len(v) for k, v in residual.items()},
            "residual_detail": {k: v[:5] for k, v in residual.items()},
            "email_classification": dict(email_class)}


def stage3_final():
    """최종 청크 — 본문+metadata 전 필드 재유입 검사 (0이어야 PASS)."""
    chunks = json.load(open(FINAL, encoding="utf-8"))
    residual = defaultdict(list)
    token_count = 0
    for ch in chunks:
        f = scan_all_string_fields(ch)
        for k, v in f.items():
            residual[k].extend(v)
        token_count += len(MASK_TOKEN_RE.findall(get_text(ch)))
    return {"total_chunks": len(chunks),
            "restricted_residual_fields": {k: len(v) for k, v in residual.items()},
            "mask_tokens_present": token_count}


def main():
    print("=" * 60)
    print(f"  마스킹 정책 {POLICY_VERSION} — 3단계 제한 식별정보 검증")
    print("=" * 60)

    print("\n[1단계] parsed_v2 — PII 후보 인벤토리 (발견=정상)")
    s1 = stage1_parsed()
    print(f"  문서 {s1['total_docs']}건 | PII 후보 보유 문서 {s1['docs_with_pii_candidates']}건")
    print(f"  인벤토리: {s1['pii_inventory']}")

    print("\n[2단계] masked_v3 — 제한 식별정보 잔존 (0이어야 PASS)")
    s2 = stage2_masked()
    print(f"  문서 {s2['total_docs']}건")
    print(f"  잔존: {s2['restricted_residual'] or '없음 (0건)'}")
    print(f"  이메일 분류: {s2['email_classification']}")
    if s2["residual_detail"]:
        print(f"  잔존 상세: {s2['residual_detail']}")

    print("\n[3단계] enriched — 청킹·직렬화 후 재유입 (0이어야 PASS)")
    s3 = stage3_final()
    print(f"  청크 {s3['total_chunks']}건 | 마스킹 토큰 {s3['mask_tokens_present']}개 보존")
    print(f"  재유입(전 필드): {s3['restricted_residual_fields'] or '없음 (0건)'}")

    # 종합 판정
    print("\n" + "=" * 60)
    s2_fail = sum(s2["restricted_residual"].values())
    s3_fail = sum(s3["restricted_residual_fields"].values())
    status = "PASS" if (s2_fail == 0 and s3_fail == 0) else "FAIL"
    print(f"  종합 판정: {status}")
    print(f"  - masked 잔존: {s2_fail}건 / final 재유입: {s3_fail}건")
    print(f"  - unknown 이메일(검토대상): {s2['email_classification'].get('unknown', 0)}건")
    print("=" * 60)

    return {"stage1": s1, "stage2": s2, "stage3": s3, "status": status}


if __name__ == "__main__":
    main()
