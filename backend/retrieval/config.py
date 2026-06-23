"""
config.py
config.yaml + .env를 읽어 설정값을 한곳에서 제공.

우선순위: .env 환경변수 > config.yaml
    - 경로(FAISS_INDEX_PATH, CHUNKS_PATH)는 개인마다 다를 수 있어 .env로 덮어쓰기 허용
    - 그 외 파라미터는 config.yaml 값 사용

사용:
    from config import FAISS_INDEX_PATH, MMR_K, EMBEDDING_MODEL, ...
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# config.yaml은 이 파일과 같은 폴더 기준 — 실행 위치(cwd)와 무관하게 항상 찾음
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ── 경로 (.env 우선) ──────────────────────────────────────────────────────────
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", _cfg["paths"]["faiss_index"])
CHUNKS_PATH      = os.getenv("CHUNKS_PATH", _cfg["paths"]["chunks"])

# ── Retriever 파라미터 ────────────────────────────────────────────────────────
MMR_K           = int(_cfg["retriever"]["mmr_k"])
MMR_FETCH_K     = int(_cfg["retriever"]["mmr_fetch_k"])
MMR_LAMBDA      = float(_cfg["retriever"]["mmr_lambda"])
FUZZY_THRESHOLD = int(_cfg["retriever"]["fuzzy_threshold"])

# ── 임베딩 ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = _cfg["embedding"]["model"]

# ── 마스킹 ────────────────────────────────────────────────────────────────────
MASKING_BUSINESS_NUMBER  = bool(_cfg["masking"]["business_number"])
MASKING_CORPORATE_NUMBER = bool(_cfg["masking"]["corporate_number"])
MASKING_PHONE            = bool(_cfg["masking"]["phone"])
MASKING_EMAIL            = bool(_cfg["masking"]["email"])
MASKING_PERSON_NAME      = bool(_cfg["masking"]["person_name"])
