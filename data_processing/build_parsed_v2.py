"""100건 전체 파싱 (표 포함) + 정제 + csv 금액 검증 + 저장
출력: data/processed/parsed_documents_v2.json  {doc_id, file_name, text, char_count}
"""
from pathlib import Path
import pandas as pd
import unicodedata
import re
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwp_xml_parser import parse_hwp

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw" / "중급 프로젝트" / "원본 데이터"
CSV = RAW / "data_list.csv"
FILES = RAW / "files"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def clean_text(t: str) -> str:
    """정제: 깨진 한자블록 + 제어문자 + 과도 공백 (xml이라 노이즈 적음)"""
    t = re.sub(r'[\u3400-\u9fff]{2,}', ' ', t)   # 깨진 한자블록
    t = re.sub(r'[\u3400-\u9fff]', '', t)         # 단일 한자
    t = ''.join(c for c in t if c in '\n\t' or ord(c) >= 32)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


def main():
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    # 파일명 → (사업금액, pdf여부)
    meta = {}
    for _, r in df.iterrows():
        fn = unicodedata.normalize('NFC', str(r['파일명']))
        meta[fn] = {'money': r['사업 금액'], 'fmt': str(r['파일형식']).lower()}

    file_map = {unicodedata.normalize('NFC', f.name): f
                for f in FILES.glob('*') if f.suffix in ('.hwp', '.pdf')}

    docs, fail = [], []
    money_ok, money_total = 0, 0

    for i, (_, row) in enumerate(df.iterrows()):
        doc_id = f"DOC_{i+1:03d}"
        fname = unicodedata.normalize('NFC', str(row['파일명']))
        f = file_map.get(fname)
        if not f:
            fail.append((doc_id, fname, "파일없음")); continue

        try:
            if f.suffix == '.hwp':
                try:
                    raw = parse_hwp(str(f))           # 1차: hwp5proc xml (표 포함)
                except Exception:
                    from verify_text import extract_hwp  # 2차: olefile 폴백 (어제 방식)
                    raw = extract_hwp(str(f))
                    if isinstance(raw, str) and raw.startswith("__ERR__"):
                        raise RuntimeError(raw)
            else:
                # pdf는 기존 방식 (pdfplumber)
                import pdfplumber
                with pdfplumber.open(f) as pdf:
                    raw = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception as e:
            fail.append((doc_id, fname, str(e)[:40])); continue

        cleaned = clean_text(raw)
        docs.append({
            "doc_id": doc_id,
            "file_name": fname,
            "text": cleaned,
            "char_count": len(cleaned),
        })

        # csv 금액 대조 (검증)
        m = meta.get(fname, {}).get('money')
        if pd.notna(m) and int(float(m)) > 0:
            money_total += 1
            comma = format(int(float(m)), ',')
            if comma in cleaned:
                money_ok += 1

        print(f"  [{i+1:3d}/100] {len(cleaned):>7,}자  {fname[:30]}", flush=True)

    # 저장
    out_path = OUT / "parsed_documents_v2.json"
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False, indent=1)

    # 리포트
    lens = [d['char_count'] for d in docs]
    print("\n" + "="*50)
    print(f"파싱 성공: {len(docs)}/100  (실패 {len(fail)})")
    print(f"정제 후 평균: {sum(lens)//len(lens):,}자 (v1은 26,269 — 표 포함이라 증가 예상)")
    print(f"최소 {min(lens):,} / 최대 {max(lens):,}")
    print(f"\n[csv 금액 검증]")
    print(f"  금액 있는 문서 {money_total}건 중 {money_ok}건 본문에서 일치 ({money_ok/money_total*100:.0f}%)")
    print(f"\n저장: {out_path}")
    if fail:
        print("\n[실패]")
        for d in fail: print("  ", d)


if __name__ == "__main__":
    main()