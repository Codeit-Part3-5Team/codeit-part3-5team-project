# data_processing/compliance_extractor.py
# 라우트 C (Checklist Generator) — 추출기
# 설계: selected_doc_id → text+table 전체 로드 → 순서보존 윈도우(overlap) →
#       윈도우별 LLM 추출 → quote 검증(strict→±1 fallback→출처수정) → exact dedupe
# 키 없는 현 단계: LLM 호출은 Mock. 조립/윈도우/coverage가드/quote검증/dedupe 완성.
# 실행: (venv) python data_processing/compliance_extractor.py

import json
import re
from pathlib import Path
from collections import OrderedDict

CHUNK_PATH = Path("data/processed/chunks_v1_enriched.json")
WINDOW_TOKEN_BUDGET = 16000  # 윈도우당 토큰 예산. 실측: 호출수 72→26회로 감소, coverage 유지.
                             # 24k는 윈도우 과대로 품질 하락 우려(문헌). 키 확보 후 12k vs 16k 품질 재확인.
OVERLAP_CHUNKS = 1           # 윈도우 경계 겹침 (경계 누락 방지)

# 라우트 C 카테고리 (프롬프트로 LLM에 강제할 enum)
CATEGORIES = ["requirement", "qualification", "submission", "scoring"]

# 골든셋 6문항 정답 좌표 (coverage 가드용, v3 검증값)
GOLDEN_COORDS = {
    "Q015": [("DOC_001", 5)],
    "Q016": [("DOC_038", 5), ("DOC_038", 6), ("DOC_038", 9), ("DOC_038", 12), ("DOC_038", 91)],
    "Q017": [("DOC_068", 8), ("DOC_068", 40), ("DOC_068", 9), ("DOC_068", 10), ("DOC_068", 11)],
    "Q018": [("DOC_001", 6), ("DOC_001", 7), ("DOC_001", 8), ("DOC_001", 9), ("DOC_001", 10)],
    "Q019": [("DOC_062", 2), ("DOC_062", 3), ("DOC_062", 65)],
    "Q020": [("DOC_004", 101), ("DOC_004", 103), ("DOC_004", 104)],
}


def cid(doc_id, chunk_index):
    """청크 고유 식별자 (chunk_id 메타 없어서 조합으로 생성)."""
    return f"{doc_id}#{chunk_index}"


def normalize(text):
    """quote 매칭용 정규화: 공백 제거 (줄바꿈/공백 차이 무시)."""
    return re.sub(r"\s+", "", text or "")


class ChecklistGenerator:
    def __init__(self, chunk_path=CHUNK_PATH):
        with open(chunk_path, encoding="utf-8") as f:
            self.all_chunks = json.load(f)
        # (doc_id, chunk_index) → 청크 빠른 조회
        self.by_key = {}
        for c in self.all_chunks:
            m = c["metadata"]
            self.by_key[(m["doc_id"], m["chunk_index"])] = c

    # ── 1. 문서 조립 ────────────────────────────────────────────
    def assemble_document(self, doc_id):
        """doc_id의 text+table 청크 전체, 순서 정렬, meta_summary 제외. 아무것도 안 거름."""
        chunks = [
            c for c in self.all_chunks
            if c["metadata"]["doc_id"] == doc_id
            and c["metadata"]["content_type"] != "meta_summary"
        ]
        chunks.sort(key=lambda c: c["metadata"]["chunk_index"])
        return chunks

    # ── 2. 윈도우 분할 (토큰 예산 + overlap) ─────────────────────
    def build_windows(self, chunks, budget=WINDOW_TOKEN_BUDGET, overlap=OVERLAP_CHUNKS):
        """순서 보존 윈도우. 경계 1청크 overlap으로 경계 누락 방지."""
        windows = []
        i = 0
        n = len(chunks)
        while i < n:
            window = []
            tok = 0
            j = i
            while j < n:
                ct = chunks[j]["metadata"].get("token_count", 0) or 0
                # 윈도우가 비어있으면 토큰 초과해도 최소 1개는 넣음 (단일 청크 초과 대비)
                if window and tok + ct > budget:
                    break
                window.append(chunks[j])
                tok += ct
                j += 1
            windows.append(window)
            if j >= n:
                break
            # 다음 윈도우 시작을 overlap만큼 뒤로 (경계 겹침)
            i = max(j - overlap, i + 1)
        return windows

    # ── 3. coverage 가드 (골든셋 정답이 윈도우에 다 들어갔나) ────
    def verify_coverage(self, doc_id, windows):
        """이 문서의 골든셋 정답 청크가 윈도우 union에 전부 포함되는지 검증."""
        # 윈도우에 담긴 모든 청크 식별자
        covered = set()
        for w in windows:
            for c in w:
                m = c["metadata"]
                covered.add((m["doc_id"], m["chunk_index"]))
        # 이 문서 관련 골든셋 정답 좌표
        golden_here = []
        for qid, coords in GOLDEN_COORDS.items():
            for d, idx in coords:
                if d == doc_id:
                    golden_here.append((qid, d, idx))
        missing = [(qid, d, idx) for qid, d, idx in golden_here if (d, idx) not in covered]
        return {
            "golden_total": len(golden_here),
            "missing": missing,
            "ok": len(missing) == 0,
        }

    # ── 4. 프롬프트 구성 (LLM에 줄 입력) ─────────────────────────
    def build_prompt(self, window, doc_id):
        """윈도우 청크들을 LLM 추출 프롬프트로. 각 청크에 식별자/section 함께 제공."""
        blocks = []
        for c in window:
            m = c["metadata"]
            blocks.append(
                f"[{cid(doc_id, m['chunk_index'])}] (section: {m.get('section','')})\n{c['page_content']}"
            )
        context = "\n\n".join(blocks)
        system = (
            "당신은 정부 입찰 RFP에서 입찰 참여에 필요한 항목을 추출하는 도구입니다. "
            "아래 문서 조각에서 입찰자가 준수·충족·제출해야 하는 항목만 추출하세요.\n"
            "추출 대상: 요구사항(requirement), 자격(qualification), 제출서류(submission), 배점(scoring).\n"
            "규칙:\n"
            "1. 각 항목은 원문에 실제로 있는 내용만. 추측/일반지식 금지.\n"
            "2. 조건부 표현('~하는 경우', '해당 시', '~인 경우에 한해')은 절대 생략하지 말 것.\n"
            "3. category는 requirement/qualification/submission/scoring 중 하나.\n"
            "4. evidence.quote는 원문을 토씨 하나 안 바꾸고 그대로 인용하되, "
            "한 문장 또는 핵심 구절 1개로 짧게. 여러 문장을 이어 붙이지 말 것.\n"
            "5. 목차, 사업개요, 사업명/예산/기간, 일정표 같은 단순 안내 정보는 추출하지 말 것.\n"
            "6. 출력은 JSON 배열만. 마크다운/설명 금지.\n"
            "형식: [{\"category\":\"...\",\"item\":\"...\","
            "\"evidence\":[{\"chunk_id\":\"DOC_xxx#n\",\"quote\":\"...\"}]}]"
        )
        user = f"문서 조각:\n{context}\n\n위에서 항목을 추출하세요."
        return {"system": system, "user": user}

    # ── 5. 토큰 측정 ─────────────────────────────────────────────
    def estimate_tokens(self, window):
        """메타 token_count 합 (재계산 없이). 비용 예측용."""
        return sum(c["metadata"].get("token_count", 0) or 0 for c in window)

    # ── 6. LLM 추출 (오늘은 Mock — 키 확보 후 교체) ──────────────
    def extract(self, window, doc_id, use_mock=True, client=None):
        """LLM 호출 자리. Mock 또는 실제 gpt-5-mini. 반환: (items, status)."""
        if use_mock:
            return self._mock_extract(window, doc_id), "mock"
        # === 키 확보 후 ===
        prompt = self.build_prompt(window, doc_id)
        try:
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                response_format={"type": "json_object"},  # structured output
            )
            raw = resp.choices[0].message.content
        except Exception as e:
            return [], f"api_error: {e}"
        return self.parse_llm_output(raw)

    def parse_llm_output(self, raw):
        """
        LLM 출력을 strict 파싱. 정규식으로 억지로 살리지 않음.
        파싱 실패 = 부분 성공 위장 금지 → 빈 리스트 + review 상태 반환.
        반환: (items, status)  status: ok | parse_fail | schema_fail
        """
        if not raw or not raw.strip():
            return [], "parse_fail"
        # 마크다운 펜스만 제거 (이건 안전 — 내용 손실 없음)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
        # strict JSON 파싱
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # 정규식으로 배열만 뜯는 fallback은 안 씀 (누락 은폐 위험)
            return [], "parse_fail"
        # JSON 객체로 감싸진 경우 ({"items": [...]}) 배열 추출
        if isinstance(data, dict):
            data = data.get("items") or data.get("checklist") or []
        if not isinstance(data, list):
            return [], "schema_fail"
        # 스키마 최소 검증 (item + evidence 필수)
        valid = []
        for it in data:
            if not isinstance(it, dict):
                continue
            if "item" not in it or "evidence" not in it:
                continue
            if not isinstance(it.get("evidence"), list):
                continue
            # category 누락 시 unknown
            it.setdefault("category", "unknown")
            valid.append(it)
        status = "ok" if len(valid) == len(data) else "schema_fail"
        return valid, status

    def _mock_extract(self, window, doc_id):
        """Mock: 윈도우 첫 청크에서 quote를 실제로 떼어 정상 항목 1개 생성 (검증기 테스트용)."""
        if not window:
            return []
        c = window[0]
        m = c["metadata"]
        # 원문 첫 문장 일부를 실제 quote로 (검증 통과해야 정상)
        snippet = c["page_content"].strip().split("\n")[0][:40]
        return [{
            "category": "requirement",
            "item": f"(mock) {snippet}",
            "evidence": [{"chunk_id": cid(doc_id, m["chunk_index"]), "quote": snippet}],
        }]

    # ── 7. quote 검증 (strict → ±1 fallback → 출처 수정) ─────────
    def verify_quote(self, item, doc_id):
        """
        각 evidence의 quote가 선언된 chunk에 실제 있는지 검증.
        실패 시 ±1 청크에서 재시도하고, 발견되면 출처(chunk_id)를 실제 위치로 수정.
        반환: (검증된 item, 모든 evidence 통과 여부)
        """
        verified_evidence = []
        all_ok = True
        for ev in item.get("evidence", []):
            chunk_id = ev.get("chunk_id", "")
            quote = ev.get("quote", "")
            nq = normalize(quote)
            if not nq:
                all_ok = False
                continue
            # chunk_id 파싱 (DOC_001#5)
            mobj = re.match(r"(.+)#(\d+)$", chunk_id)
            if not mobj:
                all_ok = False
                continue
            d, idx = mobj.group(1), int(mobj.group(2))
            # 1) strict: 선언된 청크에서
            found_at = None
            c = self.by_key.get((d, idx))
            if c and nq in normalize(c["page_content"]):
                found_at = idx
            # 2) ±1 fallback: 앞뒤 청크에서 (경계 걸침 복구)
            if found_at is None:
                for delta in (-1, 1):
                    cc = self.by_key.get((d, idx + delta))
                    if cc and nq in normalize(cc["page_content"]):
                        found_at = idx + delta
                        break
            if found_at is None:
                # 그래도 못 찾음 → 환각 의심, 통과 못 함
                all_ok = False
                verified_evidence.append({**ev, "verified": False})
            else:
                # 발견 → 실제 위치로 출처 수정 (boundary_match 기록)
                verified_evidence.append({
                    "chunk_id": cid(d, found_at),
                    "quote": quote,
                    "verified": True,
                    "boundary_match": found_at != idx,
                })
        return {**item, "evidence": verified_evidence}, all_ok

    # ── 8. exact dedupe (완전중복만, 조건 손실 방지) ─────────────
    def dedupe(self, items):
        """category + normalized quote집합 + chunk집합이 모두 같은 완전중복만 제거."""
        seen = set()
        out = []
        for it in items:
            cat = it.get("category", "")
            quotes = frozenset(normalize(ev.get("quote", "")) for ev in it.get("evidence", []))
            chunks = frozenset(ev.get("chunk_id", "") for ev in it.get("evidence", []))
            key = (cat, quotes, chunks)
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    # ── 파이프라인 (오늘은 Mock 모드로 끝까지 동작) ──────────────
    def generate_checklist(self, doc_id, use_mock=True):
        chunks = self.assemble_document(doc_id)
        windows = self.build_windows(chunks)
        coverage = self.verify_coverage(doc_id, windows)

        raw_items = []
        window_stats = []
        for wi, w in enumerate(windows):
            toks = self.estimate_tokens(w)
            window_stats.append({"window": wi, "chunks": len(w), "tokens": toks})
            items, status = self.extract(w, doc_id, use_mock=use_mock)
            window_stats[-1]["status"] = status
            raw_items.extend(items)

        # quote 검증
        verified_items = []
        verify_pass = 0
        for it in raw_items:
            vit, ok = self.verify_quote(it, doc_id)
            if ok:
                verify_pass += 1
            verified_items.append(vit)

        deduped = self.dedupe(verified_items)

        return {
            "doc_id": doc_id,
            "windows": window_stats,
            "window_count": len(windows),
            "total_tokens": sum(w["tokens"] for w in window_stats),
            "coverage": coverage,
            "raw_item_count": len(raw_items),
            "verified_pass": verify_pass,
            "deduped_count": len(deduped),
            "items": deduped,
        }


# ── 실행: 골든셋 문서 5개로 조립/윈도우/coverage/토큰 점검 ──────
# 우리 설계: doc_id 필수(없으면 거부), query는 범위 필터(선택, MVP에선 미사용)
_GENERATOR = None  # 싱글톤 (청크 재로딩 방지)


def generate_checklist(query: str = None, doc_id: str = None, use_mock: bool = True) -> dict:
    """
    라우트 C 진입점. 대상 RFP(doc_id)의 체크리스트 생성.

    Args:
        query: 사용자 질문 (범위 필터용 — 예: "자격요건만". MVP에선 미사용, V2 확장)
        doc_id: 대상 RFP 문서 ID (필수). 없으면 거부.
        use_mock: True면 Mock, False면 gpt-5-mini (키 필요)

    Returns:
        {
          "status": "ok" | "no_doc_selected" | "doc_not_found",
          "doc_id": str,
          "items": [{category, item, evidence:[{chunk_id, quote, verified}]}],
          "summary": {item_count, quote_match_rate, ...},
          "message": str (status가 ok 아닐 때 안내)
        }
    """
    global _GENERATOR

    # 가드 1: doc_id 필수 (selected_doc_id 없으면 거부 — 설계 원칙)
    if not doc_id:
        return {
            "status": "no_doc_selected",
            "doc_id": None,
            "items": [],
            "message": "체크리스트를 생성할 RFP 공고를 먼저 선택해 주세요.",
        }

    # 싱글톤 초기화 (최초 1회만 청크 로드)
    if _GENERATOR is None:
        _GENERATOR = ChecklistGenerator()

    # 가드 2: 존재하는 문서인지
    doc_chunks = _GENERATOR.assemble_document(doc_id)
    if not doc_chunks:
        return {
            "status": "doc_not_found",
            "doc_id": doc_id,
            "items": [],
            "message": f"문서 {doc_id}를 찾을 수 없습니다.",
        }

    # 파이프라인 실행
    result = _GENERATOR.generate_checklist(doc_id, use_mock=use_mock)

    # evidence 일치율 요약 (정성 감사는 별도)
    total_ev = sum(len(it.get("evidence", [])) for it in result["items"])
    verified_ev = sum(
        1 for it in result["items"]
        for ev in it.get("evidence", []) if ev.get("verified")
    )
    match_rate = verified_ev / total_ev if total_ev else 0.0

    return {
        "status": "ok",
        "doc_id": doc_id,
        "items": result["items"],
        "summary": {
            "item_count": len(result["items"]),
            "window_count": result["window_count"],
            "quote_match_rate": round(match_rate, 3),
            "coverage_ok": result["coverage"]["ok"],
        },
        "message": None,
    }





if __name__ == "__main__":
    gen = ChecklistGenerator()
    test_docs = ["DOC_001", "DOC_038", "DOC_068", "DOC_062", "DOC_004"]

    print("=" * 72)
    print("라우트 C 추출기 — Mock 모드 파이프라인 점검 (키 없이)")
    print("=" * 72)
    for doc_id in test_docs:
        r = gen.generate_checklist(doc_id, use_mock=True)
        cov = r["coverage"]
        cov_mark = "OK" if cov["ok"] else f"MISSING {cov['missing']}"
        print(f"\n[{doc_id}]")
        print(f"  윈도우: {r['window_count']}개 | 총 토큰: {r['total_tokens']:,}")
        print(f"  골든셋 coverage: {cov['golden_total']}개 중 누락 {len(cov['missing'])} → {cov_mark}")
        print(f"  추출(mock): {r['raw_item_count']} | quote검증 통과: {r['verified_pass']} | dedupe후: {r['deduped_count']}")
        # 토큰 비용 추정 (gpt-5-mini: 입력 $0.25/1M)
        cost = r['total_tokens'] / 1_000_000 * 0.25
        print(f"  예상 입력 비용: ${cost:.4f} (gpt-5-mini 입력단가 기준, 1회 추출)")

    print("\n" + "=" * 72)
    print("점검 완료. coverage 전부 OK면 윈도우 분할이 골든셋 정답을 안 빠뜨린 것.")
    print("=" * 72)

    # PM 인터페이스 진입점 테스트
    print("\n" + "=" * 72)
    print("generate_checklist 진입점 테스트")
    print("=" * 72)
    r = generate_checklist(doc_id="DOC_001", use_mock=True)
    print(f"  정상 호출: status={r['status']}, items={r['summary']['item_count']}, coverage_ok={r['summary']['coverage_ok']}")
    r = generate_checklist(query="자격요건 알려줘", use_mock=True)
    print(f"  doc_id 없음: status={r['status']} -> '{r['message']}'")
    r = generate_checklist(doc_id="DOC_999", use_mock=True)
    print(f"  없는 문서: status={r['status']} -> '{r['message']}'")
