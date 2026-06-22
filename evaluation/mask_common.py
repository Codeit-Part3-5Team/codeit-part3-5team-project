# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 마스킹 검증 공통 모듈 (DE: 도혁)
#  역할 : 정규화 함수 + PII 탐지 패턴 + 정책 3분류
#         (마스킹기·검증기가 '정규화 함수'만 공유, 탐지 패턴은 검증 전용으로 더 공격적)
#
#  설계 근거 (3차 LLM 핑퐁 + 실측 반영):
#    - 전각/특수대시/제로폭/제어문자 정규화 후 탐지 (우회 차단)
#    - "전체 공백 제거" 금지 → 숫자 사이 제한된 separator만 허용 (false positive 방지)
#      (실측: 발급번호 2016020100000 이 010 휴대폰으로 오탐되는 것 확인 → 경계·길이 검사 필수)
#    - 정책 3분류: restricted(마스킹) / allowed_public(허용) / unknown(검토)
# =====================================================================
import re
import unicodedata

# --- 정책 버전 ---
POLICY_VERSION = "v3"

# --- 정책 3분류 ---------------------------------------------------------
# restricted : 반드시 마스킹돼야 하는 식별정보 (잔존 시 FAIL)
# allowed_public : 정책상 노출 허용 (전수 검토된 기관 공개 연락처)
# unknown : 자동 허용 금지, 검토 대상

# 사적 이메일 도메인 (restricted)
RESTRICTED_EMAIL_DOMAINS = {
    "gmail.com", "naver.com", "daum.net", "hanmail.net",
    "nate.com", "kakao.com", "outlook.com", "hotmail.com", "yahoo.com",
}

# 허용 기관 도메인 (allowed_public) — suffix가 아니라 검토된 도메인 단위 관리
# ※ 실데이터 67건 이메일 전수 분석 결과를 여기에 고정 (회귀 방지)
ALLOWED_PUBLIC_EMAIL_SUFFIXES = (".go.kr", ".re.kr", ".ac.kr", ".or.kr", ".kr")

# 전수 검토 완료된 공기업/기관 도메인 (.com/.org 라 suffix로 못 잡지만 공식 도메인)
# ※ 검토 근거: korail.com=한국철도공사, kiria.org=한국로봇산업진흥원, 7luck.com=그랜드코리아레저(공기업)
APPROVED_PUBLIC_DOMAINS = {
    "korail.com",   # 한국철도공사
    "kiria.org",    # 한국로봇산업진흥원
    "7luck.com",    # 그랜드코리아레저(GKL) 공기업
}
# 단, suffix만으로 자동 허용하지 않고 분류용으로만 사용. 미검토 도메인은 unknown(review)로.


def normalize(text: str) -> str:
    """탐지 전 정규화: 전각→반각, 특수대시→'-', 제로폭 제거, 제어문자/PUA→공백.
    ※ 공백을 '제거'하지 않고 '치환'만 한다 (숫자 엉김 방지는 패턴 단계에서)."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    # 특수 대시류 → 표준 하이픈
    t = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", t)
    # 제로폭 문자 제거
    t = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", t)
    # Private Use Area → 공백 치환 (삭제 시 숫자 엉김 위험)
    t = re.sub(r"[\uE000-\uF8FF]", " ", t)
    # 기타 제어문자 → 공백
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    return t


# --- PII 탐지 패턴 (검증 전용, 공격적이되 경계·길이로 false positive 차단) ---
# 핵심: (?<!\d) 앞 숫자 없음 / (?!\d) 뒤 숫자 없음 → 긴 숫자열 일부 오탐 방지
SEP = r"[\s\-.]{0,2}"   # 숫자 그룹 사이 허용 separator (제한적)

PATTERNS = {
    # restricted (마스킹 대상)
    "mobile":   re.compile(r"(?<!\d)01[016789]" + SEP + r"\d{3,4}" + SEP + r"\d{4}(?!\d)"),
    "ssn":      re.compile(r"(?<!\d)\d{6}-[1-4]\d{6}(?!\d)"),           # 주민번호(뒷자리 1~4)
    "biz_no":   re.compile(r"(?<!\d)\d{3}-\d{2}-\d{5}(?!\d)"),          # 사업자등록번호
    "corp_no":  re.compile(r"(?<!\d)\d{6}-\d{7}(?!\d)"),               # 법인등록번호
    "account":  re.compile(r"(?<!\d)\d{2,6}-\d{2,6}-\d{2,6}-?\d{0,6}(?!\d)"),  # 계좌(느슨, 검토 플래그용)
}

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
LANDLINE_RE = re.compile(r"(?<!\d)0\d{1,2}-\d{3,4}-\d{4}(?!\d)")   # 유선전화(allowed)
MASK_TOKEN_RE = re.compile(r"\[(전화번호|이메일|휴대폰|주민등록번호|계좌번호|법인등록번호|사업자등록번호)\]")

# 휴대폰 검증: 패턴 매치 후 숫자만 추출해 정확히 11자리인지 재확인
def is_real_mobile(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return len(digits) == 11 and digits.startswith(("010", "011", "016", "017", "018", "019"))


def is_dummy_number(s: str) -> bool:
    """빈 양식/예시 더미 패턴 판별.
    - 전체가 단일 숫자 반복 (000-00-00000, 111-11-11111)
    - 휴대폰류: 앞 3자리(010 등) 제외한 가입자번호가 전부 0/단일숫자 (010-0000-0000)"""
    digits = re.sub(r"\D", "", s)
    if len(set(digits)) <= 1:
        return True
    # 11자리 휴대폰: 앞 3자리 떼고 나머지 8자리가 전부 같은 숫자면 더미
    if len(digits) == 11 and len(set(digits[3:])) <= 1:
        return True
    return False


def classify_email(email: str) -> str:
    """이메일을 restricted / allowed_public / unknown 으로 분류."""
    domain = email.split("@")[1].lower()
    if domain in RESTRICTED_EMAIL_DOMAINS:
        return "restricted"
    if domain in APPROVED_PUBLIC_DOMAINS:      # 전수 검토된 공기업 도메인
        return "allowed_public"
    if domain.endswith(ALLOWED_PUBLIC_EMAIL_SUFFIXES):
        return "allowed_public"
    return "unknown"


def scan_restricted(text: str):
    """텍스트에서 restricted(마스킹돼야 할) 식별정보 탐지.
    반환: {type: [matches]}  (false positive 거른 결과)"""
    norm = normalize(text)
    found = {}

    # 휴대폰 (11자리 재확인 + 빈양식 더미 제외)
    mob = [m.group() for m in PATTERNS["mobile"].finditer(norm)
           if is_real_mobile(m.group()) and not is_dummy_number(m.group())]
    if mob:
        found["mobile"] = mob

    # 주민번호 (뒷자리 1~4로 시작 — 법인번호와 구분)
    ssn = PATTERNS["ssn"].findall(norm)
    if ssn:
        found["ssn"] = ssn

    # 사업자등록번호 (XXX-XX-XXXXX) — 빈 양식 더미 제외
    biz = [b for b in PATTERNS["biz_no"].findall(norm) if not is_dummy_number(b)]
    if biz:
        found["biz_no"] = biz

    # 법인등록번호 (XXXXXX-XXXXXXX, 주민번호로 잡힌 것·더미 제외)
    corp = [c for c in PATTERNS["corp_no"].findall(norm)
            if c not in ssn and not is_dummy_number(c)]
    if corp:
        found["corp_no"] = corp

    # 사적 이메일
    priv = [e for e in EMAIL_RE.findall(norm) if classify_email(e) == "restricted"]
    if priv:
        found["private_email"] = priv

    return found


def scan_emails_by_class(text: str):
    """이메일을 분류별로 집계 (allowed/restricted/unknown)."""
    norm = normalize(text)
    result = {"restricted": [], "allowed_public": [], "unknown": []}
    for e in EMAIL_RE.findall(norm):
        result[classify_email(e)].append(e)
    return result


if __name__ == "__main__":
    # 정규화·탐지 자체 테스트
    tests = [
        ("정상 휴대폰", "연락처 010-1234-5678"),
        ("발급번호(오탐방지)", "벤처확인발급번호 : 2016020100000 2016.02.01"),
        ("특수대시 우회", "010\u20131234\u20135678"),
        ("전각 숫자", "０１０－１２３４－５６７８"),
        ("사적이메일", "담당 hong@gmail.com"),
        ("기관이메일", "문의 info@korea.go.kr"),
    ]
    for name, t in tests:
        r = scan_restricted(t)
        ec = scan_emails_by_class(t)
        print(f"[{name}] restricted={r} | email={ {k:v for k,v in ec.items() if v} }")
