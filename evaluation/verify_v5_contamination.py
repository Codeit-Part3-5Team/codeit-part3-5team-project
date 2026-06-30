# evaluation/verify_v5_contamination.py
# V5 hold-out 후보 8건이 개발에 쓰였는지(오염) 검증한다.
#
# burned(개발 사용) 집합 = 다음 3개 출처의 합집합:
#   1. GOLDEN_COORDS doc_id   — 추출기 개발에 좌표 박힌 문서 (compliance_extractor.py)
#   2. raw_llm_cache/*.json    — 개발 중 LLM raw 호출 캐시가 남은 문서
#   3. eval_cache/extractions/ — 평가용 추출 캐시가 생성된 문서
#
# 후보 8건 = v5_holdout_fixture_8docs.json (아인님 제공)의 doc_id
#
# 판정: burned ∩ 후보 == ∅  →  오염 없음(PASS)
#        교집합 있으면        →  해당 문서 burned이므로 V5에서 제외 필요(FAIL)
#
# ★키 소모 없음 — 파일/좌표 대조만, LLM 호출 안 함.
# ★NDA 안전 — doc_id만 다루고 원문 내용은 출력하지 않음.
#
# 실행: (venv) python evaluation/verify_v5_contamination.py [후보파일경로]
#   기본 후보파일: data/processed/v5_holdout_fixture_8docs.json
#   (Drive에서 받은 위치에 맞게 인자로 경로 전달)

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR = ROOT / "data_processing" / "compliance_extractor.py"
RAW_CACHE_DIR = ROOT / "data" / "raw_llm_cache"
EVAL_CACHE_DIR = ROOT / "data" / "eval_cache" / "extractions"


def burned_from_golden_coords() -> set:
    """compliance_extractor.py의 GOLDEN_COORDS에서 doc_id 추출."""
    text = EXTRACTOR.read_text(encoding="utf-8")
    # GOLDEN_COORDS 블록만 잘라서 "DOC_xxx" 패턴 수집
    m = re.search(r"GOLDEN_COORDS\s*=\s*\{(.*?)\}", text, re.DOTALL)
    block = m.group(1) if m else text
    return set(re.findall(r'"(DOC_\d+)"', block))


def burned_from_files(directory: Path) -> set:
    """폴더 내 파일명에서 DOC_xxx 추출 (raw_cache / eval_cache 공용)."""
    if not directory.exists():
        return set()
    docs = set()
    for f in directory.rglob("*"):
        if f.is_file():
            for did in re.findall(r"(DOC_\d+)", f.name):
                docs.add(did)
    return docs


def candidate_docs(fixture_path: Path) -> set:
    """V5 후보 fixture에서 doc_id 집합 추출. (구조 유연 대응)"""
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    docs = set()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("doc_id", "docId", "document_id") and isinstance(v, str):
                    docs.add(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)
        elif isinstance(obj, str):
            # 문자열 안에 DOC_xxx가 박혀 있을 수도
            for did in re.findall(r"DOC_\d+", obj):
                docs.add(did)

    walk(data)
    return docs


def main():
    fixture_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "data" / "processed" / "v5_holdout_fixture_8docs.json"

    print("=" * 60)
    print("V5 후보 오염 검증 (키 소모 없음, doc_id만 대조)")
    print("=" * 60)

    # 1) burned 집합 구성
    g = burned_from_golden_coords()
    r = burned_from_files(RAW_CACHE_DIR)
    e = burned_from_files(EVAL_CACHE_DIR)
    burned = g | r | e
    print(f"\n[burned 집합 = 개발 사용 문서]")
    print(f"  GOLDEN_COORDS : {sorted(g)}")
    print(f"  raw_llm_cache : {sorted(r) if r else '(없음)'}")
    print(f"  eval_cache    : {sorted(e) if e else '(없음)'}")
    print(f"  → 합집합 burned: {sorted(burned)} ({len(burned)}건)")

    # 2) 후보 집합
    if not fixture_path.exists():
        print(f"\n[중단] 후보 fixture 파일 없음: {fixture_path}")
        print("Drive에서 받은 v5_holdout_fixture_8docs.json 경로를 인자로 주세요.")
        print("  예: python evaluation/verify_v5_contamination.py data/processed/v5_holdout_fixture_8docs.json")
        sys.exit(2)

    cand = candidate_docs(fixture_path)
    print(f"\n[V5 후보 문서] ({fixture_path.name})")
    print(f"  → {sorted(cand)} ({len(cand)}건)")

    # 3) 교집합 판정
    overlap = burned & cand
    print(f"\n[오염 검증 결과]")
    if overlap:
        print(f"  ✗ FAIL — 후보 중 개발 사용 문서 발견: {sorted(overlap)}")
        print(f"  → 이 문서들은 burned이므로 V5 hold-out에서 제외해야 합니다.")
        sys.exit(1)
    else:
        print(f"  ✓ PASS — 후보 8건 모두 burned 집합과 겹치지 않음 (오염 없음)")
        print(f"  → V5 blind 조건 충족: 개발에 안 쓴 새 문서로 확인됨")

    # 4) 후보 개수 점검 (설계상 8건이어야)
    if len(cand) != 8:
        print(f"\n  [주의] 후보가 8건이 아니라 {len(cand)}건입니다. 설계(8건)와 대조 필요.")


if __name__ == "__main__":
    main()
