"""
eval_retrieval.py
Retriever Evaluation
지표: Hit Rate@5, MRR, Context Recall, Context Precision

사용법:
    python eval_retrieval.py                                      # v3 기본
    python eval_retrieval.py --golden golden_dataset_v3.json --k 5

구조:
    - evaluate_retriever()    : Hit Rate@K, MRR 계산
    - evaluate_context()      : Context Recall, Context Precision (doc_id 기반, LLM 불필요)
    - evaluate_by_category()  : 카테고리별(single_doc / multi_doc / followup) 분리 출력
    - compare()               : 이전/이후 결과 비교 출력
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "retrieval"))

import json
import argparse
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from retriever import get_retriever, load_vectorstore
from query_rewriter import rewrite_query
from query_parser import extract_metadata


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class EvalSample:
    """평가 샘플 하나"""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)
    category: str = ""        # single_doc / multi_doc / followup
    q_subtype: str = ""


# ──────────────────────────────────────────────
# 골든 데이터셋 로드
# ──────────────────────────────────────────────

def load_eval_samples_from_golden(
    json_path: str,
    use_rewrite: bool = False,
    use_system_prompt: bool = True,
    use_meta: bool = False,
    meta_agency_only: bool = False,
    meta_multi_only: bool = False,
) -> list[EvalSample]:
    """골든데이터셋 + retriever 결과로 EvalSample 리스트 생성.

    Args:
        json_path         : 골든데이터셋 JSON 경로
        use_rewrite       : True면 followup 항목에 query rewriting 적용
        use_system_prompt : query rewriting 시 시스템 프롬프트 사용 여부
        use_meta          : True면 질문에서 agency/project_name 추출 후 필터 적용
    """
    with open(json_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    vs = load_vectorstore()
    samples = []
    refusal_count = 0
    for item in golden:
        if item.get("category") == "refusal":
            refusal_count += 1
            continue

        question = item["question"]
        history = item.get("history", [])

        if use_rewrite and item.get("category") == "followup" and history:
            rewritten = rewrite_query(question, history, use_system_prompt=use_system_prompt)
            print(f"[rewrite] {question!r}\n       → {rewritten!r}")
            query = rewritten
        else:
            query = question

        # 메타데이터 필터 추출
        agency, project_name = None, None
        budget_min, budget_max = None, None
        date_field, date_min, date_max = None, None, None
        sort_by, sort_order = None, None
        category = item.get("category", "")

        meta_applicable = (
            use_meta and
            (not meta_multi_only or category == "multi_doc")
        )
        if meta_applicable:
            meta = extract_metadata(query)
            agency = meta["agency"]
            project_name = meta["project_name"] if not meta_agency_only else None
            budget_min   = meta["budget_min"]
            budget_max   = meta["budget_max"]
            date_field   = meta["date_field"]
            date_min     = meta["date_min"]
            date_max     = meta["date_max"]
            sort_by      = meta["sort_by"]
            sort_order   = meta["sort_order"]
            if agency or project_name:
                print(f"[meta] agency={agency!r}  project_name={project_name!r}")
            if budget_min is not None or budget_max is not None:
                print(f"[meta] budget min={budget_min} max={budget_max}")
            if date_field and (date_min or date_max):
                print(f"[meta] date {date_field} [{date_min}, {date_max}]")
            if sort_by:
                print(f"[meta] sort_by={sort_by} sort_order={sort_order}")

        results = get_retriever(
            query, vs,
            agency=agency,
            project_name=project_name,
            budget_min=budget_min,
            budget_max=budget_max,
            date_field=date_field,
            date_min=date_min,
            date_max=date_max,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        samples.append(EvalSample(
            question=item["question"],
            answer="",
            contexts=[doc.page_content for doc in results],
            ground_truth=item["answer"],
            relevant_doc_ids=(
                item["answer_doc_id"] if isinstance(item["answer_doc_id"], list)
                else [item["answer_doc_id"]] if isinstance(item["answer_doc_id"], str)
                else []
            ),
            retrieved_doc_ids=[doc.metadata["doc_id"] for doc in results],
            category=item.get("category", ""),
            q_subtype=item.get("q_subtype", ""),
        ))
    total = len(golden)
    print(f"[Refusal Filter] 전체 {total}건 중 {refusal_count}건 제외 → {total - refusal_count}건 평가")
    return samples


# ──────────────────────────────────────────────
# Retriever 지표 (Hit Rate@K, MRR)
# ──────────────────────────────────────────────

def hit_rate_at_k(samples: list[EvalSample], k: int = 5) -> float:
    """Hit Rate@K: 정답 문서가 상위 K개 안에 하나라도 있으면 hit"""
    hits = 0
    for s in samples:
        retrieved_k = s.retrieved_doc_ids[:k]
        if any(doc_id in retrieved_k for doc_id in s.relevant_doc_ids):
            hits += 1
    return hits / len(samples) if samples else 0.0


def mean_reciprocal_rank(samples: list[EvalSample]) -> float:
    """MRR: 첫 번째 정답 문서의 역순위 평균"""
    rr_list = []
    for s in samples:
        rr = 0.0
        for rank, doc_id in enumerate(s.retrieved_doc_ids, start=1):
            if doc_id in s.relevant_doc_ids:
                rr = 1.0 / rank
                break
        rr_list.append(rr)
    return float(np.mean(rr_list)) if rr_list else 0.0


def evaluate_retriever(samples: list[EvalSample], k: int = 5) -> dict:
    """Hit Rate@K + MRR 계산"""
    return {
        f"Hit Rate @{k}": round(hit_rate_at_k(samples, k), 4),
        "MRR (Mean Reciprocal Rank)": round(mean_reciprocal_rank(samples), 4),
    }


# ──────────────────────────────────────────────
# doc_id 기반 Context Precision, Context Recall
# ──────────────────────────────────────────────

def evaluate_context(samples: list[EvalSample], k: int = 5) -> dict:
    """
    Context Precision: 검색된 k개 중 정답 doc_id 비율
    Context Recall: 정답 doc_id가 검색 결과에 있는지 여부
    """
    precision_list = []
    recall_list = []
    for s in samples:
        retrieved_k = s.retrieved_doc_ids[:k]
        relevant = set(s.relevant_doc_ids)
        hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
        precision_list.append(hits / len(retrieved_k) if retrieved_k else 0.0)
        recall_list.append(1.0 if any(doc_id in relevant for doc_id in retrieved_k) else 0.0)

    return {
        "Context Recall": round(float(np.mean(recall_list)), 4),
        "Context Precision": round(float(np.mean(precision_list)), 4),
    }


# ──────────────────────────────────────────────
# 카테고리별 분리 출력
# ──────────────────────────────────────────────

METRIC_ORDER = [
    "Hit Rate @5",
    "MRR (Mean Reciprocal Rank)",
    "Context Recall",
    "Context Precision",
]

EVAL_CATEGORIES = ["single_doc", "multi_doc", "followup"]


def evaluate_by_category(samples: list[EvalSample], k: int = 5) -> None:
    """전체 + 카테고리별 지표 분리 출력"""
    cat_samples: dict[str, list] = {c: [] for c in EVAL_CATEGORIES}
    for s in samples:
        if s.category in cat_samples:
            cat_samples[s.category].append(s)

    def _metrics(sl):
        if not sl:
            return {m: "-" for m in METRIC_ORDER}
        r = {}
        r.update(evaluate_retriever(sl, k))
        r.update(evaluate_context(sl, k))
        return r

    all_m = _metrics(samples)
    cat_m = {c: _metrics(cat_samples[c]) for c in EVAL_CATEGORIES}
    counts = {c: len(cat_samples[c]) for c in EVAL_CATEGORIES}

    col_w = [36, 8, 12, 12, 10]
    sep = "-" * (sum(col_w) + 4)
    header = (
        f"{'항목':<{col_w[0]}} {'전체':>{col_w[1]}} "
        f"{'single_doc':>{col_w[2]}} {'multi_doc':>{col_w[3]}} {'followup':>{col_w[4]}}"
    )

    print(f"\n{sep}")
    print(header)
    print(sep)
    n_row = (
        f"{'샘플 수':<{col_w[0]}} {len(samples):>{col_w[1]}} "
        + " ".join(f"{counts[c]:>{col_w[i+2]}}" for i, c in enumerate(EVAL_CATEGORIES))
    )
    print(n_row)
    print(sep)

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    for m in METRIC_ORDER:
        row = (
            f"{m:<{col_w[0]}} {_fmt(all_m.get(m)):>{col_w[1]}} "
            + " ".join(f"{_fmt(cat_m[c].get(m)):>{col_w[i+2]}}" for i, c in enumerate(EVAL_CATEGORIES))
        )
        print(row)
    print(sep)


# ──────────────────────────────────────────────
# 이전/이후 비교 출력
# ──────────────────────────────────────────────

def compare(
    before_samples: list[EvalSample],
    after_samples: list[EvalSample],
    k: int = 5,
) -> dict:
    """이전/이후 전체 지표 계산 후 비교 테이블 출력"""
    print("▶ [이전] 평가 중...")
    before = {}
    before.update(evaluate_retriever(before_samples, k))
    before.update(evaluate_context(before_samples, k))

    print("▶ [이후] 평가 중...")
    after = {}
    after.update(evaluate_retriever(after_samples, k))
    after.update(evaluate_context(after_samples, k))

    delta = {m: round(after.get(m, 0) - before.get(m, 0), 4) for m in METRIC_ORDER}

    col_w = [36, 8, 8, 10]
    header = f"{'항목':<{col_w[0]}} {'이전':>{col_w[1]}} {'이후':>{col_w[2]}} {'변화':>{col_w[3]}}"
    sep = "-" * (sum(col_w) + 3)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for m in METRIC_ORDER:
        b = before.get(m, "-")
        a = after.get(m, "-")
        d = delta.get(m, 0)
        arrow = "▲" if d > 0 else ("▼" if d < 0 else " ")
        d_str = f"{arrow}{abs(d):.4f}" if isinstance(d, float) else "-"
        print(f"{m:<{col_w[0]}} {b:>{col_w[1]}} {a:>{col_w[2]}} {d_str:>{col_w[3]}}")
    print(sep)

    return {"before": before, "after": after, "delta": delta}


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="golden_dataset_v3.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rewrite", action="store_true",
                        help="followup 질문에 query rewriting 적용")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="query rewriting 시 시스템 프롬프트 미사용")
    parser.add_argument("--meta", action="store_true",
                        help="질문에서 agency/project_name 추출 후 메타 필터 적용")
    parser.add_argument("--meta-agency-only", action="store_true",
                        help="메타 필터 시 agency만 적용 (project_name 제외)")
    parser.add_argument("--meta-multi-only", action="store_true",
                        help="메타 필터를 multi_doc 카테고리에만 적용")
    args = parser.parse_args()

    use_sys = not args.no_system_prompt
    meta_agency_only = args.meta_agency_only
    meta_multi_only = args.meta_multi_only

    print(
        f"\n▶ 골든셋: {args.golden}"
        f"  |  rewrite: {'ON' if args.rewrite else 'OFF'}"
        f"  |  meta: {'ON' if args.meta else 'OFF'}"
        + (f" [agency only]" if meta_agency_only else "")
        + (f" [multi_doc only]" if meta_multi_only else "")
    )
    samples = load_eval_samples_from_golden(
        args.golden,
        use_rewrite=args.rewrite,
        use_system_prompt=use_sys,
        use_meta=args.meta,
        meta_agency_only=meta_agency_only,
        meta_multi_only=meta_multi_only,
    )

    flags = []
    if args.rewrite:
        flags.append(f"query rewriting{'(no sys prompt)' if not use_sys else ''}")
    if args.meta:
        meta_label = "meta filtering"
        if meta_agency_only:
            meta_label += " (agency only)"
        if meta_multi_only:
            meta_label += " (multi_doc only)"
        flags.append(meta_label)
    label = " + ".join(flags) if flags else "베이스라인"
    print(f"\n[ v3 {label} — 카테고리별 ]")
    evaluate_by_category(samples, k=args.k)
