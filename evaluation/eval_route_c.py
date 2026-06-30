# evaluation/eval_route_c.py
# 라우트 C 평가 — V4 진단(diagnostic) 인프라
#
# 설계 근거: evaluation/route_c_eval_design.md (2차 개정본)
# 이 파일은 "안 바뀌는 인프라"만 구현한다:
#   - V4 loader (라우트 C 6문항 filter)
#   - 추출 artifact 캐시 (doc_id별 ex.run 결과 저장/재사용)
#   - scope projection (hard filter 아님 — out_of_scope 분리 저장)
#   - Decision Pair Cache (1:1 판정만, 키에 prompt/rubric/scope/actor version)
#   - 양방향 순회 집계 (Recovery: expected 기준 / Alignment: extracted 기준)
#   - report skeleton (Confirmed/Review/Upper 3값, Extra 5분류)
#   - fixture dry-run 러너
#
# 아직 구현 안 함(fixture dry-run 통과 후):
#   - _call_llm_matcher 본문 (LLM 판정 프롬프트)
#   - extra 자동 판정 세부 규칙
#
# 실행: (venv) python evaluation/eval_route_c.py --dry-run     # fixture 검증
#       (venv) python evaluation/eval_route_c.py --extract     # 추출만 (캐시 생성)
#       (venv) python evaluation/eval_route_c.py --evaluate    # 평가 (matcher 구현 후)

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import List, Literal, Optional, Dict
from dataclasses import dataclass, field, asdict

# ─────────────────────────────────────────────────────────────
# 경로 / 버전 상수
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "data" / "processed" / "golden_dataset_v4.json"
EXTRACT_CACHE_DIR = ROOT / "data" / "eval_cache" / "extractions"
DECISION_CACHE_PATH = ROOT / "data" / "eval_cache" / "decision_pairs.json"
FIXTURE_PATH = ROOT / "evaluation" / "fixtures" / "eval_rubric_fixtures.json"
REPORT_DIR = ROOT / "evaluation" / "reports"

# 버전 — 바뀌면 Decision Pair Cache 자동 무효화
GOLD_VERSION = "v4"
RUBRIC_VERSION = "v1.0"
MATCHER_MODEL = "gpt-5-mini"
EVALUATION_SCOPE_VERSION = "v1.0"
ACTOR_POLICY_VERSION = "v1.0"
# matcher prompt hash는 route_c_matcher에서 단일 출처로 가져온다
# (양쪽에 따로 두면 Decision Pair Cache 키가 어긋난다)
sys.path.insert(0, str(ROOT / "evaluation"))
from route_c_matcher import call_matcher, MATCHER_PROMPT_HASH  # noqa: E402

# ─────────────────────────────────────────────────────────────
# 평가 데이터 계약 (Pydantic 대신 dataclass — 의존성 최소화)
# ─────────────────────────────────────────────────────────────
MatchDecision = Literal["match", "miss", "review", "pending"]
ActorLabel = Literal[
    "bidder_action", "bidder_evaluation_rule",
    "buyer_internal_process", "context_or_background", "unknown"
]
ExtraType = Literal[
    "extra_valid", "extra_redundant", "extra_outofscope",
    "extra_unsupported", "extra_review", "none"
]


@dataclass
class MatchResult:
    """expected ↔ extracted 1:1 판정 결과 (Decision Pair Cache에 저장되는 단위)"""
    expected_id: str
    extracted_index: int
    decision: MatchDecision = "pending"
    confidence: str = ""
    actor_label: ActorLabel = "unknown"
    reason: str = ""
    evidence_quote: str = ""


@dataclass
class EvaluatedExtractedItem:
    """추출 항목 1개에 대한 평가 메타"""
    index: int
    item: str
    primary_category: str = ""
    in_eval_scope: bool = True            # scope projection 결과
    flagged_actor_relevance: bool = False  # pre-matcher soft flag
    actor_label: ActorLabel = "unknown"
    matched_expected_ids: List[str] = field(default_factory=list)  # N:M 매핑
    extra_type: ExtraType = "none"


@dataclass
class QuestionReport:
    """문항(질문) 1개 평가 리포트"""
    qid: str
    doc_id: str
    q_subtype: str
    item_match_mode: str
    n_expected: int
    n_extracted: int
    n_in_scope: int
    # Recovery (expected 기준) — Review 3값
    confirmed_recovery: float = 0.0
    review_burden: float = 0.0
    unresolved_upper: float = 0.0
    # Taxonomy (category mode일 때만)
    taxonomy_representation: Optional[float] = None
    # Precision 축 (참고)
    alignment_rate: float = 0.0
    # Extra 분포
    extra_counts: Dict[str, int] = field(default_factory=dict)
    out_of_eval_scope_items: List[str] = field(default_factory=list)
    extraction_density: int = 0


# ─────────────────────────────────────────────────────────────
# Decision Pair Cache — 1:1 판정만 저장, 키에 모든 버전 포함
# ─────────────────────────────────────────────────────────────
class DecisionPairCache:
    def __init__(self, path: Path = DECISION_CACHE_PATH):
        self.path = path
        self.cache: Dict[str, dict] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _norm(text: str) -> str:
        # 정규화: 공백 제거(PDF 노이즈 대응), 소문자
        return "".join(text.split()).lower()

    def _key(self, expected_id: str, extracted_item: str,
             evidence_quote: str, match_mode: str) -> str:
        raw = "|".join([
            GOLD_VERSION, match_mode, MATCHER_MODEL, MATCHER_PROMPT_HASH,
            RUBRIC_VERSION, EVALUATION_SCOPE_VERSION, ACTOR_POLICY_VERSION,
            expected_id,
            self._norm(extracted_item),
            self._norm(evidence_quote),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, expected_id, extracted_item, evidence_quote, match_mode):
        return self.cache.get(
            self._key(expected_id, extracted_item, evidence_quote, match_mode))

    def set(self, expected_id, extracted_item, evidence_quote, match_mode, decision: dict):
        self.cache[self._key(expected_id, extracted_item, evidence_quote, match_mode)] = decision
        self._save()


# ─────────────────────────────────────────────────────────────
# Loader — V4 골든셋에서 라우트 C 6문항만
# ─────────────────────────────────────────────────────────────
def load_route_c_questions(golden_path: Path = GOLDEN_PATH) -> List[dict]:
    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)
    qs = [q for q in data if "expected_items" in q]
    return qs


# ─────────────────────────────────────────────────────────────
# 추출 artifact 캐시 — doc_id별 ex.run 결과 저장/재사용
# ─────────────────────────────────────────────────────────────
def get_extraction(doc_id: str, force: bool = False) -> dict:
    """doc_id 추출 결과를 캐시에서 반환. 없으면 ex.run 실행 후 저장."""
    EXTRACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = EXTRACT_CACHE_DIR / f"{doc_id}_extracted.json"
    if cache_file.exists() and not force:
        try:
            with open(cache_file, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # 캐시가 깨졌으면 무시하고 재추출
            print(f"  [경고] 캐시 손상 — 재추출: {cache_file.name}")
    # 실제 추출 (문서당 약 2분). import는 여기서 — dry-run 시 불필요
    sys.path.insert(0, str(ROOT / "data_processing"))
    from compliance_extractor_v2 import ComplianceExtractorV2
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI()  # ★client를 만들어 넘겨야 병렬 윈도우가 호출됨 (None이면 전부 실패)
    ex = ComplianceExtractorV2()
    result = ex.run(doc_id, use_mock=False, client=client)
    items = [
        {"item": it.item, "primary_category": it.primary_category,
         "evidence": [{"quote": ev.quote, "chunk_id": ev.declared_chunk_id,
                       "match_status": ev.match_status} for ev in it.evidence]}
        for it in result.get("items", [])
    ]
    # manifest는 부가정보 — 직렬화 실패해도 items는 살린다
    try:
        manifest_dict = _manifest_to_dict(result.get("manifest"))
    except Exception as e:
        manifest_dict = {"_manifest_error": str(e)}
    out = {"doc_id": doc_id, "items": items, "manifest": manifest_dict}
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _manifest_to_dict(manifest) -> dict:
    if manifest is None:
        return {}
    if isinstance(manifest, dict):
        return manifest
    # Pydantic 모델이면 model_dump (FieldInfo 등 내부 메타 제외)
    if hasattr(manifest, "model_dump"):
        try:
            return manifest.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(manifest, "dict"):  # pydantic v1 호환
        try:
            return manifest.dict()
        except Exception:
            pass
    # 마지막 수단: 인스턴스 속성만, JSON 직렬화 가능한 값만
    out = {}
    for k in (vars(manifest) if hasattr(manifest, "__dict__") else []):
        if k.startswith("_"):
            continue
        v = getattr(manifest, k)
        try:
            json.dumps(v, ensure_ascii=False)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


# ─────────────────────────────────────────────────────────────
# Matcher — 아직 미구현 (fixture dry-run 통과 후)
# ─────────────────────────────────────────────────────────────
def _call_llm_matcher(expected_item: str, extracted_item: str,
                      evidence_quote: str, match_mode: str) -> dict:
    """gpt-5-mini NLI 감사관 판정. fixture dry-run 통과 후 구현."""
    raise NotImplementedError("matcher prompt 미확정 — fixture dry-run 먼저")


# ─────────────────────────────────────────────────────────────
# fixture dry-run — matcher 없이 분류 로직 골격 검증
# ─────────────────────────────────────────────────────────────
def run_fixture_dryrun() -> bool:
    """
    fixture의 사람 정답 라벨을 읽어, 분류 로직이 갖춰야 할 케이스를 점검.
    지금은 fixture 무결성 + 라벨 분포만 확인(분류 함수는 matcher 구현 후 연결).
    """
    if not FIXTURE_PATH.exists():
        print(f"[FAIL] fixture 없음: {FIXTURE_PATH}")
        return False
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        fx = json.load(f)
    items = fx["fixtures"]
    print(f"[fixture] {fx['doc_id']} / rubric {fx['rubric_version']} / {len(items)}개")
    print("-" * 60)

    valid_actor = {"bidder_action", "bidder_evaluation_rule",
                   "buyer_internal_process", "context_or_background", "unknown"}
    valid_judg = {"in_scope", "out_of_scope", "review"}
    ok = True
    from collections import Counter
    actor_dist, judg_dist = Counter(), Counter()
    for i, it in enumerate(items):
        a, j = it["actor_label"], it["expected_judgment"]
        actor_dist[a] += 1
        judg_dist[j] += 1
        bad = []
        if a not in valid_actor: bad.append(f"actor_label '{a}'")
        if j not in valid_judg: bad.append(f"judgment '{j}'")
        # 핵심 정합성: buyer_internal_process는 out_of_scope여야 함
        if a == "buyer_internal_process" and j != "out_of_scope":
            bad.append("buyer_internal_process인데 out_of_scope 아님")
        # bidder_evaluation_rule(scoring)은 in_scope여야 함
        if a == "bidder_evaluation_rule" and j != "in_scope":
            bad.append("bidder_evaluation_rule인데 in_scope 아님")
        status = "OK" if not bad else "FAIL: " + ", ".join(bad)
        if bad: ok = False
        print(f"  [{i}] {j:12} {a:24} {status}")
    print("-" * 60)
    print(f"actor 분포: {dict(actor_dist)}")
    print(f"judgment 분포: {dict(judg_dist)}")
    print(f"\n{'[PASS]' if ok else '[FAIL]'} fixture 정합성 검증")
    return ok


# ─────────────────────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────────────────────
def main():
    args = sys.argv[1:]
    if "--dry-run" in args:
        run_fixture_dryrun()
        return
    if "--extract" in args:
        qs = load_route_c_questions()
        doc_ids = sorted({q["answer_doc_id"] for q in qs})
        print(f"라우트 C {len(qs)}문항 / 고유 문서 {len(doc_ids)}개: {doc_ids}")
        for did in doc_ids:
            print(f"  추출 중: {did} ...")
            r = get_extraction(did)
            print(f"    items: {len(r['items'])}")
        return
    if "--evaluate" in args:
        # top-k 후보 선별 → matcher 판정 → Item Coverage/Precision
        k = 8
        filter_mode = "hard"   # hard | soft | none (q_subtype→category 필터)
        do_dump = "--dump" in args
        do_snapshot = "--snapshot" in args
        for a in args:
            if a.startswith("--k="):
                k = int(a.split("=")[1])
            if a.startswith("--filter="):
                filter_mode = a.split("=")[1]
        from route_c_evaluate import (evaluate_question, summarize,
                                     dump_question, save_snapshot)
        from embed_cache import EmbedCache
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        client = OpenAI()
        embed = EmbedCache(client=client)
        dcache = DecisionPairCache()

        qs = load_route_c_questions()
        only = [a.split("=")[1] for a in args if a.startswith("--qid=")]
        if only:
            qs = [q for q in qs if q["id"] in only]
        if not qs:
            print("해당 문항 없음:", only)
            return

        scores = []
        for q in qs:
            ext = get_extraction(q["answer_doc_id"])
            s = evaluate_question(q, ext, embed, dcache,
                                  call_matcher, client, k=k,
                                  filter_mode=filter_mode)
            scores.append(s)
            print(f"  {s.qid} [{s.category}/{s.filter_mode}] k={k} "
                  f"cov={s.item_coverage:.2f} prec={s.item_precision:.2f} "
                  f"denom={s.n_extracted}/{s.n_extracted_raw} "
                  f"calls={s.matcher_calls} hits={s.matcher_cache_hits} "
                  f"review={s.n_review_expected} "
                  f"unresolved={s.n_unresolved_expected}")
        if do_dump:
            for s in scores:
                dump_question(s)
        print(f"\n요약: {summarize(scores)}")
        print(f"임베딩 API 호출 텍스트 수: {embed.api_calls}")
        if do_snapshot:
            save_snapshot(scores, k=k)
        return
    # 기본: 로드 + 요약
    qs = load_route_c_questions()
    print(f"라우트 C 문항: {len(qs)}")
    for q in qs:
        print(f"  {q['id']} [{q.get('q_subtype')}] {q['answer_doc_id']} "
              f"mode={q.get('item_match_mode')} expected={len(q['expected_items'])}")


if __name__ == "__main__":
    main()
