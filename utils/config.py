# config.yaml을 읽어 dict로 반환 (설정값 중앙 관리)
import os
import yaml

# config.yaml 경로 (utils의 한 단계 위 = 루트에 위치)
# utils/config.py → 루트, 두 단계 위
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


# config.yaml 로드
def load_config() -> dict:
    # UTF-8 명시 (한글 깨짐/cp949 에러 방지)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)