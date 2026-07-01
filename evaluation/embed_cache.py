# evaluation/embed_cache.py
# 라우트 C 평가 — 임베딩 캐시 (top-k 후보 선별의 비용 절감 핵심)
#
# 목적:
#   matcher LLM 판정은 비싸다(gpt-5-mini). expected 1개당 추출 항목 전체(수십~수백)를
#   판정하면 호출이 폭발한다(Q020 ~970, Q016 수천~수만). 그래서 먼저 임베딩 유사도로
#   expected당 top-k(8~10) 후보만 추려 LLM 판정 횟수를 줄인다.
#
#   임베딩(text-embedding-3-small)은 gpt-5-mini보다 수십 배 싸고, 같은 텍스트는
#   1회만 계산해 캐싱한다. 캐시 키 = norm(text) 해시 (공백제거+소문자, 캐시 적중률↑).
#
# 캐시 위치: data/embed_cache/embeddings.json  (★NDA, gitignore)
#   - 원문 의무항목 텍스트가 들어가므로 절대 git에 올리지 않는다.
#
# 사용:
#   ec = EmbedCache(client)              # client=OpenAI() 또는 None(미리 채운 캐시만 쓸 때)
#   vecs = ec.embed_many(["텍스트1", ...]) # 캐시 우선, 없는 것만 API 호출(배치)
#   sim  = cosine(vecs[0], vecs[1])

import hashlib
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
EMBED_CACHE_DIR = ROOT / "data" / "embed_cache"
EMBED_CACHE_PATH = EMBED_CACHE_DIR / "embeddings.json"

EMBED_MODEL = "text-embedding-3-small"
EMBED_MODEL_VERSION = "v1"          # 모델/차원 바뀌면 올려서 캐시 무효화
_BATCH = 128                         # OpenAI 임베딩 배치 한도 내 안전값


def _norm(text: str) -> str:
    """캐시 키 정규화 — 공백 제거 + 소문자 (Decision Pair Cache와 동일 규칙)."""
    return "".join(text.split()).lower()


def _key(text: str) -> str:
    raw = "|".join([EMBED_MODEL, EMBED_MODEL_VERSION, _norm(text)])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """코사인 유사도. 0 division 방어."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class EmbedCache:
    def __init__(self, client=None, path: Path = EMBED_CACHE_PATH):
        self.client = client
        self.path = path
        self.cache: Dict[str, List[float]] = self._load()
        self._dirty = 0
        self.api_calls = 0          # 이번 세션 실제 API 호출 텍스트 수(비용 추적)

    def _load(self) -> dict:
        if self.path.exists():
            try:
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print("  [경고] embed cache 손상 — 새로 시작")
        return {}

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)
        self._dirty = 0

    def get(self, text: str) -> Optional[List[float]]:
        return self.cache.get(_key(text))

    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        """실제 API 호출 (배치). client 없으면 에러 — 캐시에 없는 텍스트를 요구한 상황."""
        if self.client is None:
            raise RuntimeError(
                f"임베딩 캐시에 없는 텍스트 {len(texts)}건인데 client=None. "
                "client=OpenAI()를 넘기거나, 미리 캐시를 채워야 한다.")
        out: List[List[float]] = []
        for i in range(0, len(texts), _BATCH):
            chunk = texts[i:i + _BATCH]
            resp = self.client.embeddings.create(model=EMBED_MODEL, input=chunk)
            out.extend([d.embedding for d in resp.data])
            self.api_calls += len(chunk)
        return out

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트 임베딩. 캐시에 있으면 재사용, 없는 것만 배치로 API 호출.
        반환 순서 = 입력 순서. 중복 텍스트는 1회만 호출.
        """
        # 고유 텍스트만 추려 호출 (중복 제거로 비용 절감)
        need: Dict[str, None] = {}
        for t in texts:
            if self.get(t) is None:
                need[_norm(t)] = None  # norm 기준 중복 제거
        if need:
            # norm→원문 대표 하나씩 골라 호출
            rep: Dict[str, str] = {}
            for t in texts:
                n = _norm(t)
                if n in need and n not in rep:
                    rep[n] = t
            uniq_texts = list(rep.values())
            vecs = self._embed_api(uniq_texts)
            for t, v in zip(uniq_texts, vecs):
                self.cache[_key(t)] = v
                self._dirty += 1
            if self._dirty:
                self.flush()
        # 캐시에서 순서대로 반환
        result: List[List[float]] = []
        for t in texts:
            v = self.get(t)
            if v is None:
                # 이론상 도달 불가(위에서 다 채움) — 방어
                raise RuntimeError(f"임베딩 누락: {t[:40]}")
            result.append(v)
        return result


def top_k_candidates(query_vec: List[float],
                     pool_vecs: List[List[float]],
                     k: int) -> List[int]:
    """
    query(expected) 임베딩과 pool(추출 항목들) 임베딩 비교 →
    코사인 유사도 상위 k개의 pool 인덱스 반환(높은 순).
    pool이 k보다 작으면 전체 반환.
    """
    scored = [(i, cosine(query_vec, pv)) for i, pv in enumerate(pool_vecs)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:k]]


if __name__ == "__main__":
    # mock 자가검증 — API 없이 캐시/유사도/top-k 로직만 점검
    print("=== embed_cache 자가검증 (mock, API 호출 없음) ===")

    # 가짜 임베딩을 캐시에 직접 심어 로직 검증
    ec = EmbedCache(client=None, path=Path("/tmp/_mock_embed.json"))
    fake = {
        "기술평가 90점 만점": [1.0, 0.0, 0.0],
        "기술 평가 배점 90점": [0.98, 0.1, 0.0],   # 거의 같은 의미
        "회원 운영 요구사항": [0.0, 1.0, 0.0],
        "평가위원회 구성 절차": [0.0, 0.0, 1.0],
    }
    for t, v in fake.items():
        ec.cache[_key(t)] = v

    # 1) cosine 동작
    q = ec.get("기술평가 90점 만점")
    p1 = ec.get("기술 평가 배점 90점")
    p2 = ec.get("회원 운영 요구사항")
    s1, s2 = cosine(q, p1), cosine(q, p2)
    print(f"  cosine(기술평가, 기술평가배점) = {s1:.3f}  (높아야 함)")
    print(f"  cosine(기술평가, 회원운영)     = {s2:.3f}  (낮아야 함)")
    assert s1 > s2, "유사 항목이 더 높아야 한다"

    # 2) top-k 선별 — 기술평가 query에 가장 가까운 1개가 '기술 평가 배점'이어야
    pool_texts = ["회원 운영 요구사항", "기술 평가 배점 90점", "평가위원회 구성 절차"]
    pool_vecs = [ec.get(t) for t in pool_texts]
    top1 = top_k_candidates(q, pool_vecs, k=1)
    print(f"  top-1 인덱스: {top1} → '{pool_texts[top1[0]]}'  (기술 평가 배점이어야)")
    assert pool_texts[top1[0]] == "기술 평가 배점 90점"

    # 3) top-k가 pool보다 크면 전체 반환
    topk = top_k_candidates(q, pool_vecs, k=10)
    print(f"  top-10(pool=3) → {len(topk)}개 반환 (3이어야)")
    assert len(topk) == 3

    # 4) norm 중복 제거 — 공백만 다른 텍스트는 같은 키
    assert _key("기술평가 90점") == _key("기술 평가  90 점"), "norm 키 불일치"
    print("  norm 키 정규화 OK (공백 차이 무시)")

    print("\n[PASS] embed_cache 로직 자가검증 통과")
