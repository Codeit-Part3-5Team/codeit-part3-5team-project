# evaluation/tune_window_budget.py
# 윈도우 토큰 예산별 호출 수 vs coverage 비교 (레이턴시 최적화)
# compliance_extractor.py는 건드리지 않고 함수만 가져다 분석
# 실행: (venv) python evaluation/tune_window_budget.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_processing"))
from compliance_extractor import ChecklistGenerator

gen = ChecklistGenerator()
test_docs = ["DOC_001", "DOC_038", "DOC_068", "DOC_062", "DOC_004"]
budgets = [6000, 8000, 12000, 16000, 24000]

print("=" * 78)
print("윈도우 토큰 예산별 — 호출 수(=윈도우 수) vs 골든셋 coverage")
print("=" * 78)
print(f"{'예산':>7} | {'DOC_001':>8} {'DOC_038':>8} {'DOC_068':>8} {'DOC_062':>8} {'DOC_004':>8} | {'총호출':>6} | coverage")
print("-" * 78)

for budget in budgets:
    counts = []
    all_ok = True
    for doc_id in test_docs:
        chunks = gen.assemble_document(doc_id)
        windows = gen.build_windows(chunks, budget=budget)
        cov = gen.verify_coverage(doc_id, windows)
        counts.append(len(windows))
        if not cov["ok"]:
            all_ok = False
    total = sum(counts)
    cov_mark = "전부 OK" if all_ok else "누락발생!"
    row = " ".join(f"{c:>8}" for c in counts)
    print(f"{budget:>7,} | {row} | {total:>6} | {cov_mark}")

print("-" * 78)
print("호출 수 적을수록 빠름. coverage가 '전부 OK' 유지되는 선에서 예산 키우면 됨.")
print("주의: 예산 너무 크면 윈도우당 입력이 길어져 LLM 추출 품질 하락 가능(키 확보 후 별도 확인).")