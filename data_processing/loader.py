"""
RFP 데이터 로딩 모듈
- data_list.csv (메타데이터 100건 + 추출 텍스트) 로딩
- 원본 파일(hwp/pdf) 경로 매칭
"""
from pathlib import Path
import pandas as pd

# 데이터 경로 (프로젝트 루트 기준)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "중급 프로젝트" / "원본 데이터"
CSV_PATH = RAW_DIR / "data_list.csv"
FILES_DIR = RAW_DIR / "files"


def load_metadata() -> pd.DataFrame:
    """data_list.csv를 DataFrame으로 로딩 (100건)."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def summary(df: pd.DataFrame) -> None:
    """로딩된 데이터 기본 현황 출력."""
    print(f"총 행 수: {len(df)}")
    print(f"컬럼: {list(df.columns)}")
    print(f"\n파일형식 분포:\n{df['파일형식'].value_counts()}")
    print(f"\n텍스트 평균 길이: {df['텍스트'].str.len().mean():.0f}자")
    print(f"텍스트 결측: {df['텍스트'].isna().sum()}건")
    # 원본 파일 수
    if FILES_DIR.exists():
        n_files = len(list(FILES_DIR.glob('*')))
        print(f"\n원본 파일 수: {n_files}개")


if __name__ == "__main__":
    df = load_metadata()
    summary(df)