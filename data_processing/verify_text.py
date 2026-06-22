"""csv 텍스트 vs 원본 실제 텍스트 길이 비교 (100건 전수, 폴백 파서 포함)"""
from pathlib import Path
import pandas as pd
import unicodedata
import statistics

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "중급 프로젝트" / "원본 데이터"
CSV = RAW / "data_list.csv"
FILES = RAW / "files"

df = pd.read_csv(CSV, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]
file_map = {unicodedata.normalize('NFC', f.name): f
            for f in FILES.glob('*') if f.suffix in ('.hwp', '.pdf')}


def extract_pdf(path):
    import pdfplumber
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as e:
        return f"__ERR__{type(e).__name__}: {e}"


def extract_hwp(path):
    """1차: pyhwp / 2차 폴백: olefile + zlib로 BodyText 직접 디코딩"""
    import io
    # 1차: pyhwp
    try:
        from hwp5.hwp5txt import TextTransform
        from hwp5.xmlmodel import Hwp5File
        hwp = Hwp5File(str(path))
        out = io.BytesIO()
        TextTransform().transform_hwp5_to_text(hwp, out)
        hwp.close()
        text = out.getvalue().decode('utf-8', errors='ignore')
        if len(text.strip()) > 50:
            return text
    except Exception:
        pass
    # 2차 폴백: olefile
    try:
        import olefile, zlib, struct
        ole = olefile.OleFileIO(str(path))
        header = ole.openstream('FileHeader').read()
        is_compressed = bool(header[36] & 1)
        texts = []
        for entry in ole.listdir():
            if entry[0] == 'BodyText':
                data = ole.openstream(entry).read()
                if is_compressed:
                    try:
                        data = zlib.decompress(data, -15)
                    except Exception:
                        continue
                i = 0
                while i < len(data) - 4:
                    rec = struct.unpack('<I', data[i:i+4])[0]
                    tag = rec & 0x3ff
                    size = (rec >> 20) & 0xfff
                    i += 4
                    if tag == 67:  # PARA_TEXT
                        chunk = data[i:i+size]
                        try:
                            t = chunk.decode('utf-16le', errors='ignore')
                            t = ''.join(c for c in t if c == '\n' or ord(c) >= 32)
                            if t.strip():
                                texts.append(t)
                        except Exception:
                            pass
                    i += size
        ole.close()
        return "\n".join(texts) if texts else "__ERR__폴백_빈결과"
    except Exception as e:
        return f"__ERR__{type(e).__name__}: {e}"


# ========== 100건 전수 검증 ==========
print("=== csv 텍스트 vs 원본 텍스트 (100건 전수) ===\n")
results = []
err_count = 0

for i, row in df.iterrows():
    fname = unicodedata.normalize('NFC', str(row['파일명']))
    csv_len = len(str(row['텍스트']))
    f = file_map.get(fname)
    if not f:
        results.append((fname, csv_len, 0, None, "매칭실패")); continue
    full = extract_hwp(f) if f.suffix == '.hwp' else extract_pdf(f)
    if isinstance(full, str) and full.startswith("__ERR__"):
        err_count += 1
        results.append((fname, csv_len, 0, None, full[:40])); continue
    full_len = len(full)
    ratio = (csv_len / full_len * 100) if full_len else 0
    results.append((fname, csv_len, full_len, ratio, "OK"))
    if (i + 1) % 10 == 0:
        print(f"  ...{i+1}/100 처리 중", flush=True)

ok = [r for r in results if r[4] == "OK" and r[3] is not None]
ratios = [r[3] for r in ok]
print("\n" + "=" * 50)
print(f"성공 추출: {len(ok)}/100   (에러 {err_count}건)")
if ratios:
    print(f"\n[csv텍스트 / 원본텍스트 비율]")
    print(f"  평균   {statistics.mean(ratios):.1f}%")
    print(f"  중앙값 {statistics.median(ratios):.1f}%")
    print(f"  최소 {min(ratios):.1f}%  /  최대 {max(ratios):.1f}%")
    under5 = sum(1 for r in ratios if r < 5)
    print(f"  5% 미만: {under5}건")
    print(f"  원본 평균 {statistics.mean([r[2] for r in ok]):,.0f}자 / "
          f"csv 평균 {statistics.mean([r[1] for r in ok]):,.0f}자")

if err_count:
    print(f"\n[에러 파일]")
    for fn, c, fl, rt, st in results:
        if st not in ("OK",):
            print(f"  {st} | {fn[:35]}")