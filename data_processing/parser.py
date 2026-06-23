"""원본 100건 파싱 → 정제 → {doc_id, text} JSON 저장 (아인님 마스킹용 전달)"""
from pathlib import Path
import pandas as pd
import unicodedata
import re
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_text import extract_hwp, extract_pdf  # 검증된 파서 재사용

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "중급 프로젝트" / "원본 데이터"
CSV = RAW / "data_list.csv"
FILES = RAW / "files"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def clean_text(t: str) -> str:
    """olefile 폴백 노이즈 정제: 깨진 한자블록 + 제어문자 + 공백 정리"""
    # 1) 3글자 이상 연속 CJK 한자 블록 제거 (폴백 디코딩 노이즈)
    t = re.sub(r'[\u3400-\u9fff]{2,}', ' ', t)
    # 2) 남은 단일 한자도 제거 (RFP 본문엔 한자 거의 없음)
    t = re.sub(r'[\u3400-\u9fff]', '', t)
    # 3) 제어문자 제거 (개행/탭 제외)
    t = ''.join(c for c in t if c == '\n' or c == '\t' or ord(c) >= 32)
    # 4) 과도한 공백/줄바꿈 정리
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def main():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    file_map = {unicodedata.normalize('NFC', f.name): f
                for f in FILES.glob('*') if f.suffix in ('.hwp', '.pdf')}

    docs = []
    fail = []
    for i, row in df.iterrows():
        doc_id = f"DOC_{i+1:03d}"
        fname = unicodedata.normalize('NFC', str(row['파일명']))
        f = file_map.get(fname)
        if not f:
            fail.append((doc_id, fname, "파일없음")); continue
        raw = extract_hwp(f) if f.suffix == '.hwp' else extract_pdf(f)
        if isinstance(raw, str) and raw.startswith("__ERR__"):
            fail.append((doc_id, fname, raw[:30])); continue
        cleaned = clean_text(raw)
        docs.append({
            "doc_id": doc_id,
            "file_name": fname,
            "text": cleaned,
            "char_count": len(cleaned),
        })
        if (i+1) % 20 == 0:
            print(f"  ...{i+1}/100", flush=True)

    # JSON 한 파일로 저장
    out_path = OUT / "parsed_documents.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(docs)}건 저장 / 실패 {len(fail)}건")
    print(f"저장 위치: {out_path}")
    lens = [d['char_count'] for d in docs]
    if lens:
        print(f"정제 후 평균 {sum(lens)//len(lens):,}자 / 최소 {min(lens):,} / 최대 {max(lens):,}")
    if fail:
        print("실패:", fail)


if __name__ == "__main__":
    main()