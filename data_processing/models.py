# data_processing/models.py
# 라우트 C 데이터 계약 (Pydantic v2) + quote 매칭 정규화 + 입력 fingerprint
# 설계 원칙:
#   - LLM 출력 모델(thin) / 내부 감사 모델(rich) 분리 → LLM 자가검증 환각 차단
#   - raw LLM 결과는 inplace 수정 안 함, validator가 별도 Audit 모델 생성
#   - normalize는 매칭용 1종만 (dedupe/sentinel용은 각 단계서 별도 구현)
#   - Pydantic은 경계(LLM출력/검증결과/manifest)만, 내부 계산은 dict

from typing import List, Dict, Optional, Literal
import unicodedata
import re
import json
import hashlib
from pydantic import BaseModel, Field, ConfigDict


# ══════════════════════════════════════════════════════════════
# A1. quote 매칭용 정규화 (매칭 전용 — dedupe/sentinel에 재사용 금지)
# ══════════════════════════════════════════════════════════════

def normalize_for_quote_match(text: str) -> str:
    """
    quote-원문 매칭 전용 정규화.
    NFKC + zero-width 제거 + 대시 통일 + 공백 축약(전체삭제 X, 가짜매칭 방지).
    ※ dedupe·숫자/조건 추출에는 절대 재사용하지 말 것 (의미 훼손).
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)   # zero-width 제거
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"[①-⓿❶-➓〇]", "", t)
    t = re.sub(r"[■-◿•◦·∙･〈-】「-』〔〕［］]", "", t)      # 대시 통일
    t = re.sub(r"\s+", "", t)   # 공백 완전제거 (PDF 추출 띄어쓰기 노이즈 대응 — 실측 75%->97%)
    return t.strip()


# ══════════════════════════════════════════════════════════════
# 입력 fingerprint (재현성 — ID만이 아니라 본문/section/type까지 해시)
# ══════════════════════════════════════════════════════════════

def compute_input_manifest_hash(ordered_chunks: List[dict]) -> str:
    """
    입력 청크의 재현성 fingerprint.
    chunk_id + content_sha256 + section + content_type 를 canonical JSON으로 해시.
    ID·순서만 해시하면 본문 변경을 못 잡으므로 본문 해시까지 포함.
    """
    payload = []
    for c in ordered_chunks:
        m = c["metadata"]
        content = c.get("page_content", "") or ""
        payload.append({
            "chunk_id": f"{m['doc_id']}#{m['chunk_index']}",
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "section": m.get("section", ""),
            "content_type": m.get("content_type", ""),
        })
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


CATEGORY = Literal["requirement", "qualification", "submission", "scoring"]


# ══════════════════════════════════════════════════════════════
# LLM 출력 모델 (thin — LLM은 "뭘 뽑았나"만. 검증값 생성 금지)
# ══════════════════════════════════════════════════════════════

class LLMEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_id: str = Field(description="요구사항이 발견된 청크 ID (예: DOC_001#12)")
    quote: str = Field(description="원문에서 토씨 하나 안 바꾸고 인용한 절")


class LLMChecklistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: CATEGORY
    item: str = Field(description="원문 기반 의무·준수 사항 요약")
    evidence: List[LLMEvidence]


class LLMExtractionResponse(BaseModel):
    """OpenAI Structured Outputs 주입용 최상위 (json_schema)."""
    model_config = ConfigDict(extra="forbid")
    items: List[LLMChecklistItem]


# ══════════════════════════════════════════════════════════════
# 내부 감사 모델 (rich — validator가 계산해 생성. LLM이 만들지 않음)
# ══════════════════════════════════════════════════════════════

MatchStatus = Literal[
    "strict_verified",        # 선언 청크 내 quote 완전 존재
    "relocated_verified",     # 인접 단일 청크에서 발견 → 출처 수정
    "cross_chunk_verified",   # 인접 두 청크 결합에서 발견 (단 범위 내)
    "unverified_evidence",    # 어디에도 없음 (환각 의심)
    "out_of_scope_source",    # 문서엔 있으나 윈도우 범위 밖
    "out_of_scope_cross_chunk",  # 결합 매칭인데 범위 밖 청크 사용
]


class AuditEvidence(BaseModel):
    """validator 후처리로 생성. raw LLM evidence는 따로 보존."""
    model_config = ConfigDict(extra="forbid")
    declared_chunk_id: str                              # LLM이 선언한 출처
    resolved_chunk_ids: List[str] = Field(default_factory=list)  # 실제 발견 위치
    quote: str
    match_status: MatchStatus
    source_scope_valid: bool                            # 윈도우 범위 내 정당한가
    origin_window_id: int


class AuditChecklistItem(BaseModel):
    """dedupe·sentinel·repair 관통 최종 모델."""
    model_config = ConfigDict(extra="forbid")
    item: str
    primary_category: CATEGORY
    category_candidates: List[str] = Field(default_factory=list)
    category_conflict: bool = False
    evidence: List[AuditEvidence] = Field(default_factory=list)
    # sentinel flags (자동삭제 금지 — 감사 우선순위용)
    condition_loss_risk: bool = False
    numeric_mismatch_risk: bool = False
    deadline_loss_risk: bool = False


# ══════════════════════════════════════════════════════════════
# 실행 manifest (재현성·완결성)
# ══════════════════════════════════════════════════════════════

RunStatus = Literal["complete_verified", "complete_with_review", "partial", "failed"]


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    status: RunStatus
    windows_total: int
    windows_completed: int
    windows_failed: int
    window_budget: int
    overlap_chunks: int
    input_manifest_hash: str                    # 입력 fingerprint (본문 포함)
    prompt_version: str = "v2"
    schema_version: str = "v2"
    model_name: str = "gpt-5-mini"
    model_snapshot: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# WindowContext (내부 계산용 — dict 기반, Pydantic 아님)
# ══════════════════════════════════════════════════════════════

def build_window_context(window_id, window_chunks, ordered_chunks):
    """윈도우 검증용 context. ordered_chunks=전체(이웃탐색), allowed=윈도우범위(스코프검증), position_map=위치(idx+1버그방지)."""
    def cid(c):
        m = c["metadata"]
        return f"{m['doc_id']}#{m['chunk_index']}"
    return {
        "window_id": window_id,
        "ordered_chunks": ordered_chunks,
        "position_map": {cid(c): pos for pos, c in enumerate(ordered_chunks)},
        "allowed_chunk_ids": {cid(c) for c in window_chunks},
    }


# models.py 맨 끝에 추가 (build_window_context 아래)

# ══════════════════════════════════════════════════════════════
# verify_evidence — quote 검증 + 출처 해석 + scope 검사
# LLMEvidence(raw) → AuditEvidence(검증결과) 생성. raw는 수정 안 함.
# ══════════════════════════════════════════════════════════════

def verify_evidence(llm_chunk_id: str, quote: str, ctx: dict) -> AuditEvidence:
    """
    LLM이 낸 (chunk_id, quote)를 검증해 AuditEvidence 생성.
    매칭 단계: strict → relocated(±1 position) → cross_chunk(결합).
    scope: resolved가 윈도우 allowed 범위 내일 때만 *_verified, 아니면 out_of_scope_*.
    """
    ordered = ctx["ordered_chunks"]
    pos_map = ctx["position_map"]
    allowed = ctx["allowed_chunk_ids"]
    window_id = ctx["window_id"]
    nq = normalize_for_quote_match(quote)

    def cid_at(pos):
        m = ordered[pos]["metadata"]
        return f"{m['doc_id']}#{m['chunk_index']}"

    def norm_content(pos):
        return normalize_for_quote_match(ordered[pos]["page_content"])

    # 기본 (실패) 응답 헬퍼
    def make(status, resolved):
        scope_ok = all(r in allowed for r in resolved) if resolved else False
        # 범위 밖이면 status 강등
        if resolved and not scope_ok:
            if status == "cross_chunk_verified":
                status = "out_of_scope_cross_chunk"
            elif status in ("strict_verified", "relocated_verified"):
                status = "out_of_scope_source"
        return AuditEvidence(
            declared_chunk_id=llm_chunk_id,
            resolved_chunk_ids=resolved,
            quote=quote,
            match_status=status,
            source_scope_valid=scope_ok,
            origin_window_id=window_id,
        )

    # quote 비었거나 chunk_id 모르면 즉시 unverified
    if not nq or llm_chunk_id not in pos_map:
        return AuditEvidence(
            declared_chunk_id=llm_chunk_id, resolved_chunk_ids=[], quote=quote,
            match_status="unverified_evidence", source_scope_valid=False,
            origin_window_id=window_id,
        )

    cur = pos_map[llm_chunk_id]

    # 1) strict: 선언된 청크 내부
    if nq in norm_content(cur):
        return make("strict_verified", [cid_at(cur)])

    # 2) relocated: position 기반 앞뒤 이웃 (idx+1 산술 아님)
    for delta in (-1, +1):
        npos = cur + delta
        if 0 <= npos < len(ordered) and nq in norm_content(npos):
            return make("relocated_verified", [cid_at(npos)])

    # 3) cross_chunk: 인접 두 청크 결합 (cur+다음)
    if cur + 1 < len(ordered):
        combined = norm_content(cur) + norm_content(cur + 1)
        if nq in combined:
            return make("cross_chunk_verified", [cid_at(cur), cid_at(cur + 1)])
    # 이전+현재 결합도 시도
    if cur - 1 >= 0:
        combined = norm_content(cur - 1) + norm_content(cur)
        if nq in combined:
            return make("cross_chunk_verified", [cid_at(cur - 1), cid_at(cur)])

    # 4) 어디에도 없음 → 환각 의심
    return make("unverified_evidence", [])

# ══════════════════════════════════════════════════════════════
# dedupe용 정규화 (quote매칭용과 분리 — 조사/숫자/조건 보존)
# ══════════════════════════════════════════════════════════════

def canonicalize_for_dedupe(text):
    """dedupe 비교용. quote매칭보다 보수적 — 숫자·조건·부정어 보존, 공백만 정리."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _resolved_set(item):
    """item의 모든 evidence가 가리키는 resolved chunk id 집합."""
    s = set()
    for ev in item.evidence:
        for r in ev.resolved_chunk_ids:
            s.add(r)
    return s


def _chunk_index(chunk_id):
    """DOC_001#7 → 7 (정수). 파싱 실패 시 None."""
    m = re.match(r".+#(\d+)$", chunk_id or "")
    return int(m.group(1)) if m else None


def _is_overlap_related(set_a, set_b):
    """두 resolved 집합이 인접/겹침 관계인지 (boilerplate 오병합 방지)."""
    if set_a & set_b:               # 겹침
        return True
    idx_a = {_chunk_index(x) for x in set_a if _chunk_index(x) is not None}
    idx_b = {_chunk_index(x) for x in set_b if _chunk_index(x) is not None}
    for a in idx_a:
        for b in idx_b:
            if abs(a - b) <= 1:     # 인접 (overlap=1)
                return True
    return False


def dedupe_items(items):
    """
    2단계 dedupe (AuditChecklistItem 리스트).
    1차: 완전중복 제거 (정규화 item + quote-source 쌍 동일).
    2차: 같은 item+quote인데 source가 인접/겹침이면 병합 (멀리 떨어진 boilerplate는 유지).
    category는 primary 유지 + candidates 누적.
    """
    # ── 1차: 완전중복 ──
    exact_seen = {}
    for it in items:
        pairs = frozenset(
            (canonicalize_for_dedupe(ev.quote), tuple(sorted(ev.resolved_chunk_ids)))
            for ev in it.evidence
        )
        key = (canonicalize_for_dedupe(it.item), pairs)
        if key not in exact_seen:
            exact_seen[key] = it
    stage1 = list(exact_seen.values())

    # ── 2차: overlap 병합 ──
    merged = []
    for it in stage1:
        it_norm = canonicalize_for_dedupe(it.item)
        it_quotes = frozenset(canonicalize_for_dedupe(ev.quote) for ev in it.evidence)
        it_sources = _resolved_set(it)

        target = None
        for m in merged:
            m_norm = canonicalize_for_dedupe(m.item)
            m_quotes = frozenset(canonicalize_for_dedupe(ev.quote) for ev in m.evidence)
            # 같은 item + 같은 quote집합 + source 인접/겹침일 때만 병합
            if it_norm == m_norm and it_quotes == m_quotes:
                if _is_overlap_related(it_sources, _resolved_set(m)):
                    target = m
                    break

        if target is None:
            merged.append(it)
        else:
            # category 병합 (primary 유지, candidates 누적)
            if it.primary_category not in target.category_candidates:
                target.category_candidates.append(it.primary_category)
            if target.primary_category != it.primary_category:
                target.category_conflict = True
            # evidence 누적 (중복 source 아닌 것만)
            existing = {(canonicalize_for_dedupe(e.quote), tuple(sorted(e.resolved_chunk_ids)))
                        for e in target.evidence}
            for ev in it.evidence:
                sig = (canonicalize_for_dedupe(ev.quote), tuple(sorted(ev.resolved_chunk_ids)))
                if sig not in existing:
                    target.evidence.append(ev)

    return merged


# ══════════════════════════════════════════════════════════════
# sentinel 3종 (조건/숫자/기한 왜곡 감지 — flag만, 자동삭제 금지)
# quote엔 있는데 item에서 빠진 위험을 문자열로 감지. 정성감사 우선순위용.
# ══════════════════════════════════════════════════════════════

_CONDITION_MARKERS = ["경우", "해당 시", "한해", "한하여", "때", "조건", "에 한", "한정"]
_TIME_MARKERS = ["시", "분", "오전", "오후", "까지", "마감", "이내", "이전", "전까지"]


def run_sentinels(item):
    """
    AuditChecklistItem에 sentinel flag 설정. verified evidence만 근거로 씀.
    - condition_loss_risk: quote에 조건어 있는데 item엔 없음
    - numeric_mismatch_risk: quote 숫자가 item에 다 안 들어감
    - deadline_loss_risk: quote에 시간어 있는데 item엔 없음
    flag만 켬 — 항목 삭제/수정 안 함.
    """
    import re as _re
    # verified evidence의 quote만 (out_of_scope/unverified 제외)
    valid_quotes = " ".join(
        ev.quote for ev in item.evidence
        if ev.match_status in ("strict_verified", "relocated_verified", "cross_chunk_verified")
    )
    if not valid_quotes:
        return item

    qn = canonicalize_for_dedupe(valid_quotes)
    itn = canonicalize_for_dedupe(item.item)

    # 1) condition-loss: quote에 조건어, item엔 없음
    for kw in _CONDITION_MARKERS:
        if kw in qn and kw not in itn:
            item.condition_loss_risk = True
            break

    # 2) numeric-mismatch: quote 숫자가 item에 다 포함 안 됨
    q_nums = set(_re.findall(r"\d+", qn))
    i_nums = set(_re.findall(r"\d+", itn))
    if q_nums and not q_nums.issubset(i_nums):
        item.numeric_mismatch_risk = True

    # 3) deadline-loss: quote에 시간어, item엔 없음
    for kw in _TIME_MARKERS:
        if kw in qn and kw not in itn:
            item.deadline_loss_risk = True
            break

    return item
