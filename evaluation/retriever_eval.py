"""
retriever_eval.py
Retriever Evaluation
지표: Hit Rate@5, MRR, Context Recall, Context Precision

사용법:
    python evaluation/retriever_eval.py

구조:
    - evaluate_retriever()  : Hit Rate@5, MRR 계산 (retriever only)
    - evaluate_context()    : Context Recall, Context Precision 계산 (doc_id 기반, LLM 불필요)
    - compare()             : 이전/이후 결과 비교 출력
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "retrieval"))

import json
from dataclasses import dataclass, field
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from retriever import get_retriever, load_vectorstore


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class EvalSample:
    """평가 샘플 하나"""
    question: str
    answer: str                          # RAG가 생성한 답변
    contexts: list[str]                  # retriever가 가져온 청크 목록
    ground_truth: str                    # 정답 (reference answer)
    relevant_doc_ids: list[str] = field(default_factory=list)   # 정답 문서 ID
    retrieved_doc_ids: list[str] = field(default_factory=list)  # 검색된 문서 ID (순서 있음)


# ──────────────────────────────────────────────
# 골든 데이터셋 로드
# ──────────────────────────────────────────────

def load_eval_samples_from_golden(json_path: str) -> list[EvalSample]:
    """골든데이터셋 + retriever 결과로 EvalSample 리스트 생성"""
    with open(json_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    vs = load_vectorstore()
    samples = []
    for item in golden:
        if item.get("category") == "refusal":
            continue
        results = get_retriever(item["question"], vs)
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
        ))
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
    doc_id 기반 Context Precision, Context Recall 계산 (LLM 불필요)

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
# 이전/이후 비교 출력
# ──────────────────────────────────────────────

METRIC_ORDER = [
    "Hit Rate @5",
    "MRR (Mean Reciprocal Rank)",
    "Context Recall",
    "Context Precision",
]


def compare(
    before_samples: list[EvalSample],
    after_samples: list[EvalSample],
    k: int = 5,
) -> dict:
    """
    이전/이후 전체 지표 계산 후 비교 테이블 출력

    Returns:
        {"before": {...}, "after": {...}, "delta": {...}}
    """
    print("▶ [이전] 평가 중...")
    before = {}
    before.update(evaluate_retriever(before_samples, k))
    before.update(evaluate_context(before_samples, k))

    print("▶ [이후] 평가 중...")
    after = {}
    after.update(evaluate_retriever(after_samples, k))
    after.update(evaluate_context(after_samples, k))

    delta = {m: round(after.get(m, 0) - before.get(m, 0), 4) for m in METRIC_ORDER}

    # 출력
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
    samples = load_eval_samples_from_golden("evaluation/golden_dataset_v2.json")
    results = compare(samples, samples, k=5)
