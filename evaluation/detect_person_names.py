# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 담당자 인명 탐지 (규칙 1차 + NER 2차 하이브리드)
#  실행 : python evaluation/detect_person_names.py
#  목적 : RFP 담당자 인명을 실제로 탐지 (추정 카운트 → 실측 전환)
#
#  설계 (하이브리드):
#    [1차] 규칙 필터 — "이름+직위" / "담당자:이름" 패턴으로 인명 후보 추출
#          (전체 텍스트를 NER에 넣으면 느림 → 후보로 좁힘)
#    [2차] NER 검증 — 후보를 한국어 NER에 넣어 PER(인명) 태그 확인
#          → "고용노동"(기관)·"전략기획"(부서) 같은 규칙 오탐 제거
#
#  모델 : Leo97/KoELECTRA-small-v3-modu-ner (국립국어원 모두의말뭉치, 경량)
#         첫 실행 시 huggingface에서 다운로드(~50MB)
#
#  ※ 규칙 1차는 컨테이너에서 실데이터로 검증됨(v3).
#    NER 2차는 로컬(huggingface 접근 가능)에서 실행·검증 필요.
# =====================================================================
import os
import re
import sys
import json
import unicodedata
from collections import Counter, defaultdict

FINAL = "data/processed/chunks_v1_enriched.json"
NER_MODEL = "Leo97/KoELECTRA-small-v3-modu-ner"


# --- 정규화 (mask_common과 동일 기준) ---
def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D]", "-", t)
    t = re.sub(r"[\u200b\u200c\u200d\uFEFF]", "", t)
    t = re.sub(r"[\uE000-\uF8FF]", " ", t)
    return t


# --- [1차] 규칙 기반 인명 후보 추출 (컨테이너 검증 완료 v3) ---
# '연구원/연구관/연구사'는 기관명(한의학연구원) 오탐 주범이라 제외, 명확한 직위만
TITLES = ("사원", "대리", "주임", "계장", "과장", "차장", "부장", "팀장",
          "실장", "본부장", "주무관", "사무관", "서기관", "파트장",
          "센터장", "처장", "국장")
TITLE_RE = "|".join(sorted(TITLES, key=len, reverse=True))
LABEL_RE = (r"(?:담당자|문의처|책임자|사업담당자|구매담당자|과업담당자"
            r"|기술담당자|총괄담당자)")

# 동사/일반어 어미로 끝나는 오탐 차단
VERB_TAIL = re.compile(r"(하여|하며|으로|에서|지정|추진|위해|전에|통해|대해"
                       r"|관련|운영|관리|정보|과학|의학|기술|개발|노동|기획"
                       r"|전산|지원)$")
NOT_NAME = {
    "입찰", "계약", "문의", "담당", "관련", "기타", "부서", "과업", "제안",
    "사업", "구매", "성명", "이름", "직위", "소속", "연락", "내용", "평가",
    "심사", "제출", "접수", "공고", "위하여", "추진", "전에", "지정", "하며",
    "하여", "기초", "한의", "연구", "고용", "노동", "전략", "기획", "전산",
    "지원실", "공고문", "조달청", "안전", "신고자", "기술자로",
}
SURNAMES = set(
    "김이박최정강조윤장임한오서신권황안송전홍유고문양손배백허남심노하"
    "곽성차주우구라민지엄채원천방공현함변염여추도소석선설마길연위표명"
    "기반왕금옥육맹모"
)


def is_korean_name(s: str) -> bool:
    if not (2 <= len(s) <= 4):
        return False
    if not all("가" <= c <= "힣" for c in s):
        return False
    if s in NOT_NAME:
        return False
    if s[0] not in SURNAMES:
        return False
    if VERB_TAIL.search(s):
        return False
    return True


def rule_candidates(text: str):
    """규칙 1차: 인명 후보 + 주변 문맥(NER 입력용) 추출."""
    norm = normalize(text)
    cands = []  # (name, confidence, context)
    # high: 이름 + 직위
    for m in re.finditer(
        r"(?<![가-힣])([가-힣]{2,4})\s*(?:" + TITLE_RE + r")(?![가-힣])", norm
    ):
        nm = m.group(1)
        if is_korean_name(nm):
            s = max(0, m.start() - 20)
            e = min(len(norm), m.end() + 20)
            cands.append((nm, "high", norm[s:e]))
    # mid: 담당자 라벨 + 이름
    for m in re.finditer(
        LABEL_RE + r"\s*[:：]\s*([가-힣]{2,4})(?![가-힣])", norm
    ):
        nm = m.group(1)
        if is_korean_name(nm):
            s = max(0, m.start() - 5)
            e = min(len(norm), m.end() + 20)
            cands.append((nm, "mid", norm[s:e]))
    return cands


# --- [2차] NER 검증 ---
def load_ner():
    """한국어 NER 파이프라인 로드 (로컬 huggingface 접근 필요)."""
    from transformers import pipeline
    return pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")


def ner_is_person(ner, name: str, context: str) -> bool:
    """후보 이름이 문맥상 NER에서 PER(인명)로 태깅되는지 확인."""
    try:
        ents = ner(context)
    except Exception:
        return False
    for e in ents:
        grp = e.get("entity_group", "")
        word = e.get("word", "").replace(" ", "")
        # 모두의말뭉치 NER의 인명 태그는 'PS'(person) 계열
        if ("PS" in grp or "PER" in grp) and (name in word or word in name):
            return True
    return False


def main():
    if not os.path.exists(FINAL):
        print(f"[에러] {FINAL} 없음")
        sys.exit(1)

    chunks = json.load(open(FINAL, encoding="utf-8"))
    print(f"[1차] 규칙 기반 인명 후보 추출 중... (청크 {len(chunks)}개)")

    # 1차: 규칙 후보 수집
    raw = []  # (doc_id, name, conf, context)
    for c in chunks:
        for nm, conf, ctx in rule_candidates(c["page_content"]):
            raw.append((c["metadata"]["doc_id"], nm, conf, ctx))

    rule_names = Counter(r[1] for r in raw)
    print(f"  규칙 후보: 고유 {len(rule_names)}명 / 총 {len(raw)}건")

    # 2차: NER 검증
    print(f"\n[2차] NER 검증 중... (모델 로딩, 첫 실행 시 다운로드)")
    ner = load_ner()
    print("  모델 로드 완료. 후보 검증 중...")

    confirmed = []   # NER이 인명으로 확인
    rejected = []    # NER이 인명 아니라고 판정 (규칙 오탐)
    cache = {}       # (name, context) 중복 검증 방지
    for did, nm, conf, ctx in raw:
        key = (nm, ctx)
        if key not in cache:
            cache[key] = ner_is_person(ner, nm, ctx)
        if cache[key]:
            confirmed.append((did, nm, conf))
        else:
            rejected.append((did, nm, conf, ctx))

    conf_names = Counter(c[1] for c in confirmed)
    rej_names = Counter(r[1] for r in rejected)
    conf_docs = len(set(c[0] for c in confirmed))

    print("\n" + "=" * 60)
    print("  담당자 인명 탐지 결과 (규칙 + NER 하이브리드)")
    print("=" * 60)
    print(f"  NER 확인 인명: 고유 {len(conf_names)}명 / 총 {len(confirmed)}건"
          f" / {conf_docs}개 문서")
    print(f"  규칙 오탐(NER 제거): 고유 {len(rej_names)}명 / {len(rejected)}건")
    print(f"\n  [확인된 인명 상위]")
    for nm, cnt in conf_names.most_common(30):
        print(f"    {nm}: {cnt}건")
    print(f"\n  [NER이 거른 규칙 오탐 (인명 아님)]")
    for nm, cnt in rej_names.most_common(20):
        print(f"    {nm}: {cnt}건")

    # 결과 저장
    out = {
        "method": "rule_filter + NER verification",
        "ner_model": NER_MODEL,
        "rule_candidates_unique": len(rule_names),
        "ner_confirmed_unique": len(conf_names),
        "ner_confirmed_total": len(confirmed),
        "ner_confirmed_docs": conf_docs,
        "rule_false_positives_removed": len(rej_names),
        "confirmed_names": dict(conf_names),
        "removed_false_positives": dict(rej_names),
        "note": ("규칙으로 인명 후보를 좁힌 뒤 NER로 PER 태그 검증. "
                 "기관명/부서명 조각(고용노동, 전략기획 등) 오탐을 NER이 제거. "
                 "RFP 담당자는 정보공개법 9조1항6호 라목·마목상 직무상 공개대상."),
    }
    os.makedirs("evaluation", exist_ok=True)
    json.dump(out, open("evaluation/person_name_detection.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[저장] evaluation/person_name_detection.json")


if __name__ == "__main__":
    main()
