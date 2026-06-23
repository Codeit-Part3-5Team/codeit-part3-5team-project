"""
Retriever Evaluation
지표: Hit Rate@5, MRR, Context Recall, Context Precision

사용법:
    pip install ragas langchain openai datasets

구조:
    - evaluate_retriever()  : Hit Rate@5, MRR 계산 (retriever only)
    - evaluate_ragas()      : Context Recall, Context Precision 계산
    - compare()             : 이전/이후 결과 비교 출력
"""

import os
import json
from dataclasses import dataclass, field
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_recall,
    context_precision,
)
from dotenv import load_dotenv

load_dotenv()


# 데이터 구조

@dataclass
class EvalSample:
    """평가 샘플 하나"""
    question: str
    answer: str                          # RAG가 생성한 답변
    contexts: list[str]                  # retriever가 가져온 청크 목록
    ground_truth: str                    # 정답 (reference answer)
    relevant_doc_ids: list[str] = field(default_factory=list)   # 정답 문서 ID
    retrieved_doc_ids: list[str] = field(default_factory=list)  # 검색된 문서 ID (순서 있음)

# 샘플 데이터 로드

def load_eval_samples(json_path: str) -> list[EvalSample]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(EvalSample(
            question=item["query"],
            answer="", #LLM 답변 없으므로 빈값
            contexts=[c["page_content"] for c in item["chunks"]],
            ground_truth="", # 정답 없으므로 빈값
            retrieved_doc_ids=[c["metadata"]["doc_id"] for c in item["chunks"]],
        ))
    return samples

# Retriever 지표 (Hit Rate@K, MRR)

def hit_rate_at_k(samples: list[EvalSample], k: int = 5) -> float:
    """
    Hit Rate@K: 정답 문서가 상위 K개 안에 하나라도 있으면 hit
    """
    hits = 0
    for s in samples:
        retrieved_k = s.retrieved_doc_ids[:k]
        if any(doc_id in retrieved_k for doc_id in s.relevant_doc_ids):
            hits += 1
    return hits / len(samples) if samples else 0.0


def mean_reciprocal_rank(samples: list[EvalSample]) -> float:
    """
    MRR: 첫 번째 정답 문서의 역순위 평균
    """
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


# RAGAS 지표

def evaluate_ragas(
    samples: list[EvalSample],
    llm=None,
    embeddings=None,
) -> dict:
    """
    RAGAS 지표 계산: Context Recall, Context Precision

    llm, embeddings 미지정 시 환경변수 OPENAI_API_KEY로 gpt-4o-mini 사용
    """
    data = {
        "question":     [s.question     for s in samples],
        "answer":       [s.answer       for s in samples],
        "contexts":     [s.contexts     for s in samples],
        "ground_truth": [s.ground_truth for s in samples],
    }
    dataset = Dataset.from_dict(data)

    metrics = [context_recall, context_precision]

    kwargs = {}
    if llm:
        kwargs["llm"] = llm
    if embeddings:
        kwargs["embeddings"] = embeddings

    result = evaluate(dataset, metrics=metrics, **kwargs)
    result_dict = result.to_pandas().mean(numeric_only=True).to_dict()

    mapping = {
        "context_recall":    "Context Recall",
        "context_precision": "Context Precision",
    }
    return {mapping.get(k, k): round(float(v), 4) for k, v in result_dict.items()}


# 이전/이후 비교 출력

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
    llm=None,
    embeddings=None,
) -> dict:
    """
    이전/이후 전체 지표 계산 후 비교 테이블 출력

    Returns:
        {"before": {...}, "after": {...}, "delta": {...}}
    """
    print("▶ [이전] 평가 중...")
    before = {}
    before.update(evaluate_retriever(before_samples, k))
    before.update(evaluate_ragas(before_samples, llm, embeddings))

    print("▶ [이후] 평가 중...")
    after = {}
    after.update(evaluate_retriever(after_samples, k))
    after.update(evaluate_ragas(after_samples, llm, embeddings))

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


# 실행 예시

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    llm = ChatOpenAI(model="gpt-5-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ── 샘플 데이터 (실제 데이터로 교체) ──
    before_samples = [
        EvalSample(
            question="파이썬에서 리스트와 튜플의 차이는?",
            answer="리스트는 변경 가능하고 튜플은 변경 불가능합니다.",
            contexts=[
                "리스트(list)는 mutable 자료형으로 요소를 추가·수정·삭제할 수 있습니다.",
                "딕셔너리는 키-값 쌍으로 구성된 자료형입니다.",
            ],
            ground_truth="리스트는 mutable(변경 가능)하며 튜플은 immutable(변경 불가능)합니다. 리스트는 [], 튜플은 ()로 생성합니다.",
            relevant_doc_ids=["doc_001"],
            retrieved_doc_ids=["doc_003", "doc_001", "doc_007"],
        ),
        EvalSample(
            question="GIL이란 무엇인가?",
            answer="GIL은 Global Interpreter Lock으로 한 번에 하나의 스레드만 파이썬 바이트코드를 실행합니다.",
            contexts=[
                "GIL(Global Interpreter Lock)은 CPython 인터프리터의 뮤텍스입니다.",
                "멀티스레딩 환경에서 GIL은 동시에 하나의 스레드만 실행되도록 제한합니다.",
            ],
            ground_truth="GIL은 CPython에서 스레드 안전성을 보장하기 위해 한 번에 하나의 스레드만 파이썬 객체에 접근하도록 제한하는 메커니즘입니다.",
            relevant_doc_ids=["doc_012"],
            retrieved_doc_ids=["doc_012", "doc_015"],
        ),
    ]

    # 이후 샘플: 동일 질문에 개선된 retriever 결과 적용
    after_samples = [
        EvalSample(
            question="파이썬에서 리스트와 튜플의 차이는?",
            answer="리스트는 mutable(변경 가능)하고 튜플은 immutable(변경 불가능)합니다. 리스트는 [], 튜플은 ()를 사용합니다.",
            contexts=[
                "리스트(list)는 mutable 자료형으로 요소를 추가·수정·삭제할 수 있습니다.",
                "튜플(tuple)은 immutable 자료형으로 한 번 생성하면 변경할 수 없습니다.",
            ],
            ground_truth="리스트는 mutable(변경 가능)하며 튜플은 immutable(변경 불가능)합니다. 리스트는 [], 튜플은 ()로 생성합니다.",
            relevant_doc_ids=["doc_001"],
            retrieved_doc_ids=["doc_001", "doc_002", "doc_003"],
        ),
        EvalSample(
            question="GIL이란 무엇인가?",
            answer="GIL(Global Interpreter Lock)은 CPython에서 스레드 안전성을 위해 한 번에 하나의 스레드만 파이썬 객체에 접근하도록 제한하는 메커니즘입니다.",
            contexts=[
                "GIL(Global Interpreter Lock)은 CPython 인터프리터의 뮤텍스입니다.",
                "GIL로 인해 멀티코어 CPU에서도 병렬 실행이 제한될 수 있습니다.",
            ],
            ground_truth="GIL은 CPython에서 스레드 안전성을 보장하기 위해 한 번에 하나의 스레드만 파이썬 객체에 접근하도록 제한하는 메커니즘입니다.",
            relevant_doc_ids=["doc_012"],
            retrieved_doc_ids=["doc_012", "doc_013"],
        ),
    ]

    results = compare(before_samples, after_samples, k=5, llm=llm, embeddings=embeddings)