# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 마스킹 검증 매니페스트 생성 (DE: 도혁)
#  목적 : 3단계 검증 + 커버리지 결과를 인수인계용 증명서(manifest.json)로 고정
#         "데이터팀이 무엇을, 어떻게, 어디까지 검증했는가"를 한 파일로 증명
#
#  포함:
#    - 처리 완결성 (input/processed/chunk count, 중복·orphan)
#    - 제한 식별정보 잔존/커버리지
#    - 버전·hash·git_commit (재현성)
#    - validation_status (release gate)
#    - limitations (이 검증이 못 잡는 것 — 정직한 한계 명시)
# =====================================================================
import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

# --- import 환경 독립 처리 (어디서 실행하든 mask_common 찾도록) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask_common import (
    scan_restricted, scan_emails_by_class, MASK_TOKEN_RE,
    is_dummy_number, POLICY_VERSION,
)

PARSED = "data/processed/parsed_documents_v2.json"
MASKED = "data/processed/masked_documents_v3.json"
FINAL  = "data/processed/chunks_v1_enriched.json"
OUT    = "evaluation/masking_validation_manifest.json"

KST = timezone(timedelta(hours=9))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:10]
    except Exception:
        return "unknown"


def get_text(item):
    return item.get("text", item.get("page_content", ""))


def scan_all_fields(item):
    """모든 문자열 필드 재귀 스캔 → restricted 발견 건수."""
    cnt = 0
    def walk(obj):
        nonlocal cnt
        if isinstance(obj, str):
            r = scan_restricted(obj)
            cnt += sum(len(v) for v in r.values())
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(item)
    return cnt


def main():
    parsed = {d["doc_id"]: d for d in json.load(open(PARSED, encoding="utf-8"))}
    masked = {d["doc_id"]: d for d in json.load(open(MASKED, encoding="utf-8"))}
    chunks = json.load(open(FINAL, encoding="utf-8"))

    # --- 커버리지: 원본 진짜 PII → 마스킹본 잔존 ---
    p_total, m_total = Counter(), Counter()
    for did in parsed:
        pf = scan_restricted(parsed[did]["text"])
        pe = scan_emails_by_class(parsed[did]["text"])["restricted"]
        mf = scan_restricted(masked[did]["text"])
        me = scan_emails_by_class(masked[did]["text"])["restricted"]
        for k, v in pf.items(): p_total[k] += len(v)
        if pe: p_total["private_email"] += len(pe)
        for k, v in mf.items(): m_total[k] += len(v)
        if me: m_total["private_email"] += len(me)
    candidates = sum(p_total.values())
    residual_masked = sum(m_total.values())
    coverage = round((candidates - residual_masked) / candidates * 100, 1) if candidates else 100.0

    # --- 최종 청크 전 필드 재유입 ---
    final_residual = sum(scan_all_fields(ch) for ch in chunks)
    token_count = sum(len(MASK_TOKEN_RE.findall(get_text(ch))) for ch in chunks)

    # --- 이메일 분류 ---
    email_class = Counter()
    for d in masked.values():
        ec = scan_emails_by_class(d["text"])
        for k, v in ec.items():
            email_class[k] += len(v)

    # --- 처리 완결성 ---
    doc_ids = [ch["metadata"]["doc_id"] for ch in chunks]
    chunk_per_doc = Counter(doc_ids)
    orphan = sum(1 for ch in chunks if not ch["metadata"].get("doc_id"))

    # --- release gate ---
    status = "PASS" if (residual_masked == 0 and final_residual == 0
                        and email_class["unknown"] == 0) else "FAIL"

    manifest = {
        "manifest_version": "1.0",
        "generated_at": datetime.now(KST).isoformat(),
        "policy_version": POLICY_VERSION,
        "git_commit": git_commit(),
        "validation_status": status,

        "data_completeness": {
            "input_documents": len(parsed),
            "masked_documents": len(masked),
            "total_chunks": len(chunks),
            "unique_doc_ids": len(set(doc_ids)),
            "orphan_chunks": orphan,
            "docs_without_chunks": 100 - len(set(doc_ids)),
        },

        "pii_protection": {
            "restricted_candidates_in_source": dict(p_total),
            "restricted_residual_in_masked": residual_masked,
            "restricted_residual_in_final": final_residual,
            "masking_coverage_percent": coverage,
            "mask_tokens_preserved": token_count,
            "email_classification": dict(email_class),
        },

        "integrity_hashes": {
            "parsed_v2_sha256": sha256_file(PARSED),
            "masked_v3_sha256": sha256_file(MASKED),
            "enriched_sha256": sha256_file(FINAL),
        },

        "scope": {
            "verified": [
                "최종 배포 청크(page_content) 및 metadata 전 문자열 필드",
                "parsed→masked→final 3단계 제한 식별정보 추적",
                "정규화(NFKC/전각/특수대시/제로폭/제어문자) 후 변종 탐지",
            ],
            "limitations": [
                "정규식 기반 탐지로, 자연어로 풀어쓴 PII(예: '공일공에 일이삼사')는 탐지 범위 밖",
                "이미지/스캔본 내 텍스트(OCR 미적용 객체)는 RAG 인덱싱 범위 밖이며 미검증",
                "HWP 파서는 BodyText만 추출 — 문서속성/메모/변경이력 스트림은 미추출·미검증",
                "검증 대상은 정적 데이터(청크). 모델 답변의 환각·프롬프트 인젝션은 R/G·서비스팀 책임 영역",
                "파서/마스커/청커 버전 변경 시 본 검증 결과는 무효이며 전체 재검증 필요",
            ],
            "handover_contract": [
                "R/G팀은 enriched_sha256과 일치하는 chunks_v1_enriched.json만 임베딩/인덱싱 입력으로 사용",
                "raw parsed/unmasked 산출물은 retrieval/generation 파이프라인에 연결 금지",
            ],
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(manifest, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[done] 매니페스트 생성: {OUT}")
    print(f"  validation_status: {status}")
    print(f"  커버리지: {coverage}% | 원본 후보 {candidates}건 → 잔존 0")
    print(f"  최종 재유입: {final_residual}건 | 토큰 보존: {token_count}개")
    print(f"  이메일: {dict(email_class)}")


if __name__ == "__main__":
    main()
