"""
grade_smoke.py
grade 노드가 검색결과를 똑바로 판정하는지 육안 검증하는 smoke 스크립트.
(정밀 평가 아님 — grade 판단의 경향·오작동을 빠르게 확인하는 용도)

방법:
  v4 골든셋에서 샘플 추출 → 각 문항을 검색(retrieve) → grade_node 통과 →
  판정(sufficient/insufficient/out_of_scope)을 카테고리별 '예상'과 비교.

예상(이상적인 grade 판정):
  single_doc → sufficient   (1차 검색으로 답 충분)
  multi_doc  → 무관          (insufficient 섞여도 정상 — 재검색 여지)
  followup   → sufficient    (rewriting 후면 충분. 단 본 smoke는 원문으로 검색)
  refusal    → out_of_scope  (범위 밖 → 재검색 말고 거부로) ← 핵심 확인 대상

실행: (루트, venv-gen에서) python -m backend.evaluation.grade_smoke
"""
import json
from collections import defaultdict

from utils.config import load_config
from backend.pipeline import retrieve
from backend.graph.nodes.grade import grade_node

# 카테고리별 샘플 수 (refusal은 out_of_scope 판정이 핵심이라 전부)
_SAMPLE_PER_CAT = {"single_doc": 5, "multi_doc": 5, "followup": 5, "refusal": 13}

# 카테고리별 '이상적' grade 판정 (어긋나면 점검 대상)
_EXPECTED = {
    "single_doc": "sufficient",
    "multi_doc": None,            # insufficient/sufficient 둘 다 정상 → 판정 안 함
    "followup": "sufficient",
    "refusal": "out_of_scope",
}


def _pick_samples(golden: list[dict]) -> list[dict]:
    """카테고리별로 앞에서 N개씩 추출."""
    buckets = defaultdict(list)
    for item in golden:
        buckets[item["category"]].append(item)
    picked = []
    for cat, n in _SAMPLE_PER_CAT.items():
        picked.extend(buckets.get(cat, [])[:n])
    return picked


def main():
    config = load_config()
    path = config["evaluation"]["golden_dataset_path"]
    with open(path, encoding="utf-8") as f:
        golden = json.load(f)

    samples = _pick_samples(golden)
    print(f"[grade smoke] 샘플 {len(samples)}건 검증 시작\n")

    # 결과 집계: category → {grade값: 건수}, 그리고 어긋난 문항
    tally = defaultdict(lambda: defaultdict(int))
    mismatches = []

    for item in samples:
        qid = item["id"]
        cat = item["category"]
        question = item["question"]

        # 검색 → grade 판정 (재검색 루프 없이 grade 노드만 단독 호출)
        docs = retrieve(question, config)
        result = grade_node({"question": question, "docs": docs})
        grade = result["grade"]

        tally[cat][grade] += 1

        # 예상과 비교 (multi_doc은 예상 None이라 건너뜀)
        expected = _EXPECTED[cat]
        mark = ""
        if expected and grade != expected:
            mark = f"  ← 예상={expected}"
            mismatches.append((qid, cat, question[:40], expected, grade))

        print(f"  {qid} [{cat:10}] docs={len(docs):2} → {grade}{mark}")

    # 요약
    print("\n===== 카테고리별 판정 분포 =====")
    for cat in _SAMPLE_PER_CAT:
        dist = dict(tally[cat])
        exp = _EXPECTED[cat] or "(판정 안 함)"
        print(f"  {cat:10} (예상={exp}): {dist}")

    print(f"\n===== 어긋난 문항 ({len(mismatches)}건) =====")
    if not mismatches:
        print("  없음 — grade 판정이 예상과 일치")
    else:
        for qid, cat, q, exp, got in mismatches:
            print(f"  {qid} [{cat}] 예상={exp} / 실제={got}")
            print(f"       Q: {q}...")


if __name__ == "__main__":
    main()