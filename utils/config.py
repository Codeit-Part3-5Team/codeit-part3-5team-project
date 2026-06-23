# config.yaml을 읽어 dict로 반환 (설정값 중앙 관리)
import os
import yaml

# config.yaml 경로 (backend/generation/ 아래에 위치 — 생성팀 소유 설정)
# utils/config.py 기준: 루트로 한 단계 올라가 backend/generation/config.yaml
_ROOT = os.path.dirname(os.path.dirname(__file__))   # utils의 상위 = 루트
CONFIG_PATH = os.path.join(_ROOT, "backend", "generation", "config.yaml")


# config.yaml 로드
def load_config() -> dict:
    # UTF-8 명시 (한글 깨짐/cp949 에러 방지)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)