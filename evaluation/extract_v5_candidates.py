# evaluation/extract_v5_candidates.py
# V5 hold-out 후보 문서를 추출해 캐시에 저장한다 (채점용 캐시 데우기 + 부하 분석).
#
# 기존 eval_route_c.get_extraction()을 그대로 재사용한다 (새 추출 로직 없음).
#   - 캐시에 있으면 즉시 반환 (키 소모 0)
#   - 없으면 ComplianceExtractorV2.run(doc_id, client) 실행 후 저장 (키 소모)
#
# ★키 소모 주의 — 문서당 LLM 호출이 일어난다 (예산 관리).
#   먼저 1건만 돌려 시간/항목 수 보고, 괜찮으면 나머지 진행 권장.
#
# 실행:
#   (venv) python evaluation/extract_v5_candidates.py --doc=DOC_061        # 1건만
#   (venv) python evaluation/extract_v5_candidates.py --doc=DOC_061,DOC_003 # 여러 건
#   (venv) python evaluation/extract_v5_candidates.py --all                # 후보 8건 전부
#   (venv) python evaluation/extract_v5_candidates.py --all --force        # 캐시 무시 재추출

import sys
import time
from pathlib import Path

# .env 로드 (eval_route_c와 동일 방식)
from dotenv import load_dotenv
load_dotenv()

# eval_route_c의 get_extraction 재사용
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_route_c import get_extraction, EXTRACT_CACHE_DIR  # noqa: E402

# V5 후보 8건 (오염검증 PASS, 가벼운 순서)
V5_CANDIDATES = [
    "DOC_061",  # 가장 가벼움 (청크 150) — 1건 테스트용 권장
    "DOC_040",
    "DOC_070",
    "DOC_003",
    "DOC_048",
    "DOC_008",
    "DOC_041",
    "DOC_087",  # 가장 무거움
]


def parse_targets(args):
    if "--all" in args:
        return list(V5_CANDIDATES)
    for a in args:
        if a.startswith("--doc="):
            return [d.strip() for d in a.split("=", 1)[1].split(",") if d.strip()]
    return []


def main():
    args = sys.argv[1:]
    force = "--force" in args
    targets = parse_targets(args)

    if not targets:
        print("대상 문서를 지정하세요.")
        print("  --doc=DOC_061           (1건)")
        print("  --doc=DOC_061,DOC_003   (여러 건)")
        print("  --all                   (후보 8건 전부)")
        print("  --force                 (캐시 무시 재추출)")
        sys.exit(1)

    print("=" * 60)
    print(f"V5 후보 추출 — 대상 {len(targets)}건: {targets}")
    print(f"force(캐시무시)={force}")
    print(f"캐시 위치: {EXTRACT_CACHE_DIR}")
    print("=" * 60)

    summary = []
    for i, doc_id in enumerate(targets, 1):
        cache_file = EXTRACT_CACHE_DIR / f"{doc_id}_extracted.json"
        was_cached = cache_file.exists() and not force
        print(f"\n[{i}/{len(targets)}] {doc_id} {'(캐시 있음)' if was_cached else '(추출 실행)'}")

        t0 = time.time()
        try:
            r = get_extraction(doc_id, force=force)
        except Exception as e:
            print(f"  ✗ 실패: {type(e).__name__}: {e}")
            summary.append((doc_id, "FAIL", 0, 0.0))
            continue
        elapsed = time.time() - t0

        items = r.get("items", [])
        manifest = r.get("manifest", {})
        n_items = len(items)
        # content_type 분포 (표가 배점표인지 단순인지 보려면 primary_category 분포가 힌트)
        from collections import Counter
        cat = Counter(it.get("primary_category") for it in items)

        print(f"  항목 수: {n_items}")
        print(f"  카테고리 분포: {dict(cat)}")
        print(f"  소요 시간: {elapsed:.1f}초 {'(캐시라 0에 가까움)' if was_cached else '(실제 추출)'}")
        if manifest:
            print(f"  manifest: windows {manifest.get('windows_completed','?')}/{manifest.get('windows_total','?')}"
                  f" failed {manifest.get('windows_failed','?')}")
        summary.append((doc_id, "OK", n_items, elapsed))

    # 요약 표
    print("\n" + "=" * 60)
    print("요약 (doc_id / status / 항목수 / 시간)")
    total_items = 0
    total_time = 0.0
    for doc_id, status, n, t in summary:
        print(f"  {doc_id:9s} {status:5s} {n:5d}항목  {t:6.1f}초")
        total_items += n
        total_time += t
    print(f"  {'합계':9s}       {total_items:5d}항목  {total_time:6.1f}초")
    print("=" * 60)
    print("\n※ 키 소모량은 디스코드 !usage로 추출 전후 비교해 확인하세요.")


if __name__ == "__main__":
    main()
