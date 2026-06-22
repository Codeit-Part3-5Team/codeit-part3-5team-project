# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 마스킹 검증 회귀 테스트 (DE: 도혁)
#  실행 : pytest evaluation/test_masking.py -v
#  목적 : 데이터(마스킹/청킹)가 바뀌어도 PII 누출·토큰 깨짐을 자동 차단
#         = 추후 버그 방지 영구 안전장치 (release gate)
#
#  테스트 그룹:
#    [A] synthetic fixture : 판정기 자체가 PII를 잡고/거르는지 (코드 회귀)
#    [B] 산출물 무결성      : masked_v3·enriched 제한식별정보 0건·토큰 무결 (데이터 회귀)
#    [C] 보존성            : 허용 정보(예산·공고번호·기관연락처)가 살아있는지 (과잉마스킹 방지)
#    [D] idempotency       : 마스킹 토큰이 재처리로 깨지지 않는지
# =====================================================================
import os
import re
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask_common import (
    scan_restricted, scan_emails_by_class, normalize,
    is_real_mobile, is_dummy_number, classify_email, MASK_TOKEN_RE,
)

# 데이터 경로 (루트 기준)
MASKED = "data/processed/masked_documents_v3.json"
FINAL  = "data/processed/chunks_v1_enriched.json"

# ※ synthetic fixture는 '현실에 없는 명백한 더미'를 쓰되,
#   판정기 테스트용으로 진짜 패턴 형태는 유지 (단, 실데이터엔 절대 없는 값)
SYNTH_LEAK = {
    "mobile":        "010-1234-5678",          # 형태상 휴대폰 (테스트 전용)
    "private_email": "forbidden_test@gmail.com",
    "biz_no":        "123-45-67890",
}
SYNTH_DUMMY = ["010-0000-0000", "000-00-00000", "111-11-11111"]  # 빈양식 더미


# ---------- [A] 판정기 자체 회귀 ----------
def test_detector_catches_mobile():
    r = scan_restricted(f"연락처 {SYNTH_LEAK['mobile']}")
    assert "mobile" in r, "휴대폰을 탐지하지 못함"

def test_detector_catches_private_email():
    ec = scan_emails_by_class(f"메일 {SYNTH_LEAK['private_email']}")
    assert SYNTH_LEAK["private_email"] in ec["restricted"], "사적 이메일 분류 실패"

def test_detector_catches_biz_no():
    r = scan_restricted(f"사업자번호 {SYNTH_LEAK['biz_no']}")
    assert "biz_no" in r, "사업자번호를 탐지하지 못함"

def test_detector_filters_dummy():
    """빈양식 더미(000-00-00000 등)는 PII로 잡으면 안 됨 (false positive 방지)."""
    for dummy in SYNTH_DUMMY:
        r = scan_restricted(f"양식 예시 {dummy}")
        assert not r.get("biz_no") and not r.get("mobile"), f"더미 {dummy}를 PII로 오탐"

def test_detector_handles_variants():
    """정규화로 전각/특수대시 변종을 잡는지."""
    assert scan_restricted("０１０－１２３４－５６７８").get("mobile"), "전각 숫자 우회 탐지 실패"
    assert scan_restricted("010\u20131234\u20135678").get("mobile"), "특수대시 우회 탐지 실패"

def test_detector_ignores_long_number():
    """발급번호 등 긴 숫자열을 휴대폰으로 오탐하지 않는지 (실측 false positive)."""
    r = scan_restricted("벤처확인발급번호 : 2016020100000 2016.02.01")
    assert not r.get("mobile"), "긴 숫자열을 휴대폰으로 오탐"

def test_email_classification():
    assert classify_email("a@gmail.com") == "restricted"
    assert classify_email("a@korea.go.kr") == "allowed_public"
    assert classify_email("a@korail.com") == "allowed_public"   # 검토된 공기업


# ---------- [B] 산출물 무결성 (데이터 회귀) ----------
@pytest.fixture(scope="module")
def masked_docs():
    return json.load(open(MASKED, encoding="utf-8"))

@pytest.fixture(scope="module")
def final_chunks():
    return json.load(open(FINAL, encoding="utf-8"))

def test_masked_no_restricted_residual(masked_docs):
    """masked_v3에 제한 식별정보 잔존 0건 (FAIL 시 배포 금지)."""
    residual = []
    for d in masked_docs:
        r = scan_restricted(d["text"])
        priv = scan_emails_by_class(d["text"])["restricted"]
        if r or priv:
            residual.append((d["doc_id"], r, priv))
    assert not residual, f"마스킹본에 제한식별정보 잔존: {residual[:3]}"

def test_final_no_restricted_in_all_fields(final_chunks):
    """최종 청크의 모든 문자열 필드(본문+metadata)에 제한식별정보 0건."""
    leaks = []
    def walk(obj, did):
        if isinstance(obj, str):
            r = scan_restricted(obj)
            if r:
                leaks.append((did, r))
        elif isinstance(obj, dict):
            for v in obj.values(): walk(v, did)
        elif isinstance(obj, list):
            for v in obj: walk(v, did)
    for ch in final_chunks:
        walk(ch, ch["metadata"]["doc_id"])
    assert not leaks, f"최종 청크에 제한식별정보 재유입: {leaks[:3]}"

def test_no_unknown_email(masked_docs):
    """미분류(unknown) 이메일 0건 — 자동 허용 금지 정책."""
    unknown = []
    for d in masked_docs:
        u = scan_emails_by_class(d["text"])["unknown"]
        unknown.extend(u)
    assert not unknown, f"미검토 unknown 이메일 발견(검토 필요): {set(unknown)}"


# ---------- [C] 보존성 (과잉 마스킹 방지) ----------
def test_budget_preserved(final_chunks):
    """예산 정보(meta_summary의 budget)가 유지되는지 — 과잉마스킹 안 됨."""
    has_budget = any(
        ch["metadata"].get("budget_amount") for ch in final_chunks
        if ch["metadata"].get("content_type") == "meta_summary"
    )
    assert has_budget, "예산 정보가 모두 사라짐 (과잉 마스킹 의심)"

def test_landline_preserved(final_chunks):
    """공개 대상 유선전화가 유지되는지 (정책상 노출 허용)."""
    landline_re = re.compile(r"(?<!\d)0\d{1,2}-\d{3,4}-\d{4}(?!\d)")
    count = sum(len(landline_re.findall(ch.get("page_content", ""))) for ch in final_chunks)
    assert count > 0, "유선전화가 모두 사라짐 (과잉 마스킹 의심)"


# ---------- [D] idempotency (토큰 무결) ----------
def test_token_not_double_wrapped(final_chunks):
    """마스킹 토큰이 [[전화번호]]처럼 중복 래핑되지 않았는지."""
    bad = []
    for ch in final_chunks:
        if re.search(r"\[\[(전화번호|이메일|사업자등록번호|법인등록번호)\]\]",
                     ch.get("page_content", "")):
            bad.append(ch["metadata"]["doc_id"])
    assert not bad, f"토큰 중복 래핑 발견: {bad[:3]}"

def test_token_not_broken(final_chunks):
    """마스킹 토큰이 깨지지 않았는지 (정상 형태만 존재)."""
    valid = {"[전화번호]", "[이메일]", "[휴대폰]", "[사업자등록번호]",
             "[법인등록번호]", "[계좌번호]", "[주민등록번호]"}
    broken = []
    for ch in final_chunks:
        pc = ch.get("page_content", "")
        # 마스킹 의미 토큰만 검사 (구조 마커 [표] 등은 제외)
        for m in re.findall(r"\[(전화번호|이메일|휴대폰|사업자등록번호|법인등록번호|계좌번호|주민등록번호)[^\]]*\]", pc):
            full = f"[{m}]"
            if full not in valid:
                broken.append((ch["metadata"]["doc_id"], full))
    assert not broken, f"깨진 마스킹 토큰: {broken[:3]}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
