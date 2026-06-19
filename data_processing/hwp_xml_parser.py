"""hwp5proc xml 기반 파서 — 본문 + 표 통합 추출 (표는 본문에 녹임)
출력: {doc_id, file_name, text, char_count} (기존 구조 유지)
"""
import subprocess
import re
import xml.etree.ElementTree as ET


def _ln(tag):
    return tag.split('}')[-1] if '}' in tag else tag


def _run_xml(hwp_path):
    r = subprocess.run(['hwp5proc', 'xml', hwp_path], capture_output=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"hwp5proc 실패: {r.stderr.decode('utf-8','ignore')[:100]}")
    return r.stdout


def _cell_text(cell):
    parts = [e.text for e in cell.iter() if _ln(e.tag) == 'Text' and e.text]
    return ' '.join(' '.join(parts).split())


def _table_to_rows(table):
    rows = []
    for tr in table.iter():
        if _ln(tr.tag) != 'TableRow':
            continue
        cells = [_cell_text(c) for c in tr if _ln(c.tag) == 'TableCell']
        if any(c.strip() for c in cells):
            rows.append(cells)
    return rows


def _is_data_table(rows):
    """진짜 데이터 표인지 판정 (표지/목차/박스 제외)."""
    if len(rows) < 3:
        return False
    maxcol = max(len(r) for r in rows)
    if maxcol < 2:
        return False
    # 헤더 행에 2개 이상 셀에 내용이 있어야 데이터 표
    header = rows[0]
    filled = sum(1 for h in header if h.strip())
    return filled >= 2


def _table_to_text(table):
    rows = _table_to_rows(table)
    if not rows:
        return ""
    if not _is_data_table(rows):
        # 레이아웃 표(표지/목차/박스) → 텍스트만, [표] 마커 안 붙임
        flat = [c for r in rows for c in r if c.strip()]
        return ' '.join(flat)
    # 데이터 표 → key-value
    header = rows[0]
    out = ["[표]"]
    for r in rows[1:]:
        pairs = []
        for i, c in enumerate(r):
            if c.strip():
                key = header[i].strip() if i < len(header) and header[i].strip() else f"열{i+1}"
                pairs.append(f"{key}: {c.strip()}")
        if pairs:
            out.append(" / ".join(pairs))
    return "\n".join(out) if len(out) > 1 else ""


def parse_hwp(hwp_path):
    root = ET.fromstring(_run_xml(hwp_path))
    blocks = []

    def walk(elem):
        for child in elem:
            tag = _ln(child.tag)
            if tag == 'TableControl':
                blocks.append(_table_to_text(child))
            else:
                if tag == 'Text' and child.text and child.text.strip():
                    blocks.append(child.text.strip())
                walk(child)

    walk(root)
    return '\n'.join(b for b in blocks if b.strip())


if __name__ == "__main__":
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 else \
        r"data\raw\중급 프로젝트\원본 데이터\files\인천광역시_도시계획위원회 통합관리시스템 구축용역.hwp"
    txt = parse_hwp(f)
    print(f"추출 길이: {len(txt):,}자 / [표] {txt.count('[표]')}개")
    print("\n=== 앞 500자 ===")
    print(txt[:500])
    money = re.findall(r'[0-9][0-9,]{6,}\s*원', txt)
    print("\n금액:", set(money))