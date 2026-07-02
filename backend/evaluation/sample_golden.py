"""
sample_golden.py
골든셋에서 모델 비교(바이크오프)용 대표 문항을 층화 추출한다.

목적:
    답변 생성 모델 확정(gpt vs 오픈모델) 비교를, 전체 63문항 대신 대표 표본으로
    돌려 비용을 아끼되, 시나리오 편향 없이 공평하게 뽑는다.

추출 방식:
    (category × q_subtype) 이중 층화. 각 세부유형에서 목표 비율만큼 뽑아,
    카테고리 분포뿐 아니라 유형(예산·기관·요약·거부종류 등) 다양성까지 보존한다.
    seed 고정으로 재현 가능(리포트 재현성).

실행: (루트에서) python -m backend.evaluation.sample_golden --frac 0.5 --seed 42
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_sample(items: list[dict], frac: float, seed: int) -> list[dict]:
    """
    (category, q_subtype) 버킷별로 frac 비율만큼 층화 추출한다.

    각 버킷에서 round(버킷크기 * frac)개를 뽑되, 버킷에 1개 이상 있으면
    최소 1개는 보장(유형이 통째로 빠지는 것 방지). seed 고정.

    Args:
        items: 골든셋 문항 리스트(각 dict에 category, q_subtype 포함)
        frac : 추출 비율(0~1). 예: 0.5 → 절반
        seed : 난수 시드(재현용)
    Returns:
        추출된 문항 리스트(원본 순서 유지)
    """
    rng = random.Random(seed)

    # (category, q_subtype) 기준으로 버킷 분할
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        buckets[(it.get("category"), it.get("q_subtype"))].append(it)

    picked_ids = set()
    for key, group in buckets.items():
        n = len(group)
        # 목표 개수: 비율 반올림, 최소 1개 보장(버킷이 비어있지 않으면)
        take = max(1, round(n * frac))
        take = min(take, n)
        chosen = rng.sample(group, take)
        picked_ids.update(it["id"] for it in chosen)

    # 원본 순서 유지해서 반환(id 정렬 안정성)
    return [it for it in items if it["id"] in picked_ids]


def main():
    parser = argparse.ArgumentParser(description="골든셋 층화 추출(모델 비교용)")
    parser.add_argument("--input", type=str, default="data/golden/golden_dataset_v4.json")
    parser.add_argument("--output", type=str, default="data/golden/golden_v4_sample.json")
    parser.add_argument("--frac", type=float, default=0.5, help="추출 비율(기본 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="난수 시드(재현용)")
    args = parser.parse_args()

    items = json.load(open(args.input, encoding="utf-8"))
    sample = stratified_sample(items, args.frac, args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(sample, open(args.output, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"추출 완료: {len(sample)}/{len(items)}문항 → {args.output} (frac={args.frac}, seed={args.seed})")


if __name__ == "__main__":
    main()