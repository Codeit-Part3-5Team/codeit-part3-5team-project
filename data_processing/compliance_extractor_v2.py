# data_processing/compliance_extractor_v2.py
# 라우트 C 추출기 v2 — Pydantic 데이터 계약 기반 evidence-first 파이프라인
# 파이프라인: doc 조립 → 윈도우 → LLM추출 → verify_evidence(scope) → sentinel → dedupe → gate

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import (
    build_window_context, verify_evidence, run_sentinels, dedupe_items,
    compute_input_manifest_hash,
    LLMExtractionResponse, AuditChecklistItem, AuditEvidence, RunManifest,
)
from compliance_extractor import ChecklistGenerator, GOLDEN_COORDS, cid
from openai import LengthFinishReasonError

WINDOW_TOKEN_BUDGET = 16000
WINDOW_CHUNK_SIZE = 8
OVERLAP_CHUNKS = 1


class ComplianceExtractorV2:
    def __init__(self, max_workers=8):
        self._base = ChecklistGenerator()
        self.max_workers = max_workers

    def assemble_document(self, doc_id):
        return self._base.assemble_document(doc_id)

    def build_windows(self, chunks):
        # 청크 개수 기준 윈도우 (gpt-5-mini 속도/타임아웃 고려, 토큰 16k는 과대)
        size = WINDOW_CHUNK_SIZE
        ov = OVERLAP_CHUNKS
        if len(chunks) <= size:
            return [chunks]
        windows = []
        start = 0
        while start < len(chunks):
            end = min(start + size, len(chunks))
            windows.append(chunks[start:end])
            if end >= len(chunks):
                break
            start = end - ov  # overlap
        return windows

    def build_prompt_v2(self, window, window_id):
        blocks = []
        for c in window:
            m = c["metadata"]
            head = "[" + cid(m["doc_id"], m["chunk_index"]) + "] (section: " + str(m.get("section", "")) + ")"
            blocks.append(head + "\n" + c["page_content"])
        context = "\n\n".join(blocks)

        system = (
            "당신은 정부 입찰 RFP에서 입찰자가 준수·충족·제출해야 하는 항목을 추출하는 도구입니다.\n\n"
            "추출 대상 4종:\n"
            "- requirement: 사업 수행상 지켜야 할 요구사항\n"
            "- qualification: 입찰 참가 자격\n"
            "- submission: 제출 서류·제출 기한·제출 장소·제출 방법\n"
            "- scoring: 평가 항목·배점·평가 기준 (입찰자 행동이 아니어도 반드시 추출)\n\n"
            "규칙:\n"
            "1. 먼저 근거 quote와 chunk_id를 고르고, 그 quote의 의미만 item으로 요약하라. quote에 없는 조건·수치·기한을 item에 추가하지 마라.\n"
            "2. 각 quote는 제공된 chunk_id 중 하나 안에 그대로 존재해야 한다. 여러 chunk에 걸치면 evidence를 나눠라.\n"
            "3. 조건부 표현(~하는 경우, 해당 시, ~에 한해)은 item에서 절대 생략하지 마라.\n"
            "4. 숫자·배점·비율·날짜·시간은 item에 그대로 보존하라.\n"
            "5. 목차·사업개요·추진배경·기대효과·기관소개 같은 단순 안내는 추출하지 마라. "
            "단, 일정표 안의 제출 기한·등록 마감·질의 마감·발표 일정은 submission 또는 requirement로 추출하라.\n"
            "6. quote는 주어·조건·행동이 식별되는 완성된 절로 인용하라. 서술어 단독 인용 금지.\n\n"
            "예시:\n"
            "원문: 공동수급체를 구성하는 경우 협정서를 제출하여야 한다. → item: 공동수급체를 구성하는 경우 협정서를 제출해야 함 (조건 보존)\n"
            "원문: 제안서는 2026년 7월 1일 18:00까지 제출한다. → item: 제안서를 2026년 7월 1일 18:00까지 제출해야 함 (기한 보존)\n"
            "원문: 기술능력평가 80점, 가격평가 20점으로 한다. → item: 평가 배점은 기술능력평가 80점 가격평가 20점 (배점 추출)\n"
            "원문: 본 사업은 지역 산업 경쟁력 강화를 목적으로 한다. → 추출 안 함 (사업 배경은 대상 아님)\n"
        )
        user = "문서 조각:\n" + context + "\n\n위에서 항목을 추출하세요."
        return {"system": system, "user": user}

    def extract_window(self, window, window_id, use_mock=True, client=None):
        if use_mock:
            return self._mock_response(window, window_id), "mock"
        prompt = self.build_prompt_v2(window, window_id)
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                response_format=LLMExtractionResponse,
                max_completion_tokens=16000,
                reasoning_effort="low",
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                return None, "parse_fail"
            return parsed, "ok"
        except LengthFinishReasonError:
            return None, "length_exceeded"
        except Exception as e:
            return None, "api_error:" + type(e).__name__

    def _mock_response(self, window, window_id):
        if not window:
            return LLMExtractionResponse(items=[])
        c = window[0]
        m = c["metadata"]
        snippet = c["page_content"].strip().split("\n")[0][:40]
        return LLMExtractionResponse.model_validate({
            "items": [{
                "category": "requirement",
                "item": "(mock) " + snippet,
                "evidence": [{"chunk_id": cid(m["doc_id"], m["chunk_index"]), "quote": snippet}],
            }]
        })

    def audit_item(self, llm_item, ctx):
        audited_ev = [verify_evidence(ev.chunk_id, ev.quote, ctx) for ev in llm_item.evidence]
        item = AuditChecklistItem(
            item=llm_item.item, primary_category=llm_item.category,
            category_candidates=[], evidence=audited_ev,
        )
        return run_sentinels(item)

    def run(self, doc_id, use_mock=True, client=None):
        chunks = self.assemble_document(doc_id)
        if not chunks:
            return {"status": "doc_not_found", "doc_id": doc_id, "items": []}
        windows = self.build_windows(chunks)

        all_items = []
        windows_failed = 0

        # LLM 호출만 병렬 (느린 부분), 결과는 window_id 순서대로 수집
        def _extract(args):
            wi, w = args
            resp, status = self.extract_window(w, wi, use_mock=use_mock, client=client)
            return wi, w, resp, status

        if use_mock:
            results = [_extract((wi, w)) for wi, w in enumerate(windows)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(_extract, list(enumerate(windows))))

        # audit는 순차 (CPU 작업, 빠름) — window_id 순서 보존
        for wi, w, resp, status in sorted(results, key=lambda x: x[0]):
            if resp is None:
                windows_failed += 1
                continue
            ctx = build_window_context(wi, w, chunks)
            for llm_item in resp.items:
                all_items.append(self.audit_item(llm_item, ctx))

        deduped = dedupe_items(all_items)

        windows_completed = len(windows) - windows_failed
        verified = {"strict_verified", "relocated_verified", "cross_chunk_verified"}
        needs_review = False
        for it in deduped:
            if any(ev.match_status not in verified for ev in it.evidence):
                needs_review = True
            if it.condition_loss_risk or it.numeric_mismatch_risk or it.deadline_loss_risk:
                needs_review = True
            if not it.evidence:
                needs_review = True

        if windows_completed == 0:
            run_status = "failed"
        elif windows_failed > 0:
            run_status = "partial"
        elif needs_review:
            run_status = "complete_with_review"
        else:
            run_status = "complete_verified"

        manifest = RunManifest(
            doc_id=doc_id, status=run_status,
            windows_total=len(windows), windows_completed=windows_completed,
            windows_failed=windows_failed, window_budget=WINDOW_TOKEN_BUDGET,
            overlap_chunks=OVERLAP_CHUNKS,
            input_manifest_hash=compute_input_manifest_hash(chunks),
        )
        return {"manifest": manifest, "items": deduped,
                "raw_items": len(all_items), "deduped_items": len(deduped)}


if __name__ == "__main__":
    ex = ComplianceExtractorV2()
    for doc_id in ["DOC_001", "DOC_038", "DOC_004"]:
        r = ex.run(doc_id, use_mock=True)
        mf = r["manifest"]
        print("[" + mf.doc_id + "] status=" + mf.status)
        print("   windows " + str(mf.windows_completed) + "/" + str(mf.windows_total) +
              " (failed " + str(mf.windows_failed) + ") | dedupe " + str(r["deduped_items"]))
        print("   input_hash: " + mf.input_manifest_hash[:16] + "...")
