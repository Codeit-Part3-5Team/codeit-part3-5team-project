# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 베이스라인 청킹 (DE: 도혁)
#  입력 : data/processed/parsed_documents_v2.json (마스킹본이면 masked json으로 교체)
#  출력 : data/processed/chunks_v1.json  (LangChain Document 호환 dict 리스트)
#  설계 : ① 로마숫자 섹션 1차 분할  ② 표([표] 블록) content_type=table 로 분리
#         ③ 본문은 512~1024 토큰 + overlap 150 재귀 분할
#         ④ has_image_budget 플래그(csv 금액 누락 31건 추적성)
#
#  [2026-06-22 수정] 청크 중복 버그 수정 (DOC_090 505건 복제 → 0)
#    - 원인: 최종 가드의 이중 분할(split_by_tokens + _hard_slice)에서 overlap 중첩 복제
#    - 수정 ①: 가드는 overlap 없는 _simple_slice 1회만 (복제 원천 차단)
#    - 수정 ②: 최종 단계에 doc 내 동일 content 중복 제거 안전장치
#    - 수정 ③: PUA(HWP 깨진 글머리기호) 문자 정리
# =====================================================================
import os
import re
import csv
import json
import unicodedata

# --- CFG -------------------------------------------------------------
CFG = {
    "in_path":   "data/processed/masked_documents_v3.json",
    "csv_path":  "data/raw/중급 프로젝트/원본 데이터/data_list.csv",  # 금액 대조 + 메타요약
    "out_path":  "data/processed/chunks_v1.json",
    "min_tokens": 512,
    "max_tokens": 1024,
    "overlap":    150,
    "encoding":   "cl100k_base",                # text-embedding-3-small 기준
}

# --- 토큰 카운터 -----------------------------------------------------
# tiktoken 사용 가능하면 정확히, 아니면 한글 가중 근사(1자 ≈ 0.6토큰)
try:
    import tiktoken
    _enc = tiktoken.get_encoding(CFG["encoding"])
    def n_tokens(s: str) -> int:
        return len(_enc.encode(s))
    print("[info] tiktoken 정확 카운터 사용")
except Exception:
    def n_tokens(s: str) -> int:
        # 한글/한자 0.6, ascii 0.25 가중 근사
        kr = sum(1 for c in s if ord(c) > 0x3000)
        return int(kr * 0.6 + (len(s) - kr) * 0.25) + 1
    print("[warn] tiktoken 없음 → 근사 카운터 (정확도 위해 'pip install tiktoken' 권장)")

# --- PUA(Private Use Area) 정리: HWP 깨진 글머리기호 등 ----------------
# 수정③: 검색 노이즈가 되는 PUA 문자를 공백으로 치환 후 정규화
def clean_pua(text: str) -> str:
    if not text:
        return text
    # PUA 영역(글머리기호 깨짐) → 공백
    text = re.sub(r"[\uE000-\uF8FF]", " ", text)
    # 연속 공백 정리 (줄바꿈은 보존)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text

# --- 섹션 헤더 판정 --------------------------------------------------
ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
_roman_hdr = re.compile(rf"^[{ROMAN}]\s+\S.{{0,40}}$")

def is_section_header(line: str) -> bool:
    line = line.strip()
    if not _roman_hdr.match(line):
        return False
    if sum(line.count(c) for c in ROMAN) > 1:
        return False
    if re.search(r"\d{2,}\s*$", line) and len(line) > 25:
        return False
    return True

# --- 표 블록 / 본문 블록 분해 ---------------------------------------
# [2026-06-22 PM 요청] section 메타데이터에 표 헤더(컬럼명) 요약 추가
#   예: "Ⅲ. 사업비 산정 — [표: 항목/금액/일정/담당]"
#   목적: 옵션A(헤더 첫 청크만)에서 둘째+ 청크의 맥락을 section 메타로 보완
def extract_table_columns(tbl_txt: str, max_cols: int = 6) -> str:
    """표 첫 행에서 컬럼명(key)을 추출해 요약. 'key: val / key: val' 형식 기준."""
    rows = [r for r in tbl_txt.split("\n") if r.strip() and r.strip() != "[표]"]
    if not rows:
        return ""
    first = rows[0]
    cols = []
    for cell in first.split("/"):
        if ":" in cell:
            key = cell.split(":", 1)[0].strip()
            # HWP 자간 공백 정리 ('일 정' → '일정')
            key = re.sub(r"\s+", "", key)
            if key and key not in cols:
                cols.append(key)
    if not cols:
        return ""
    summary = "/".join(cols[:max_cols])
    if len(cols) > max_cols:
        summary += "…"
    return summary


def make_table_section(cur_section: str, tbl_txt: str) -> str:
    """기존 섹션명에 표 컬럼명 요약을 붙인다."""
    cols = extract_table_columns(tbl_txt)
    if cols:
        return f"{cur_section} — [표: {cols}]"
    return f"{cur_section} — [표]"


def split_blocks(text: str):
    """text 를 (kind, section, content) 블록 리스트로 분해. kind=text|table"""
    lines = text.split("\n")
    blocks, buf, cur_section = [], [], "서두"

    def flush_text():
        if buf:
            joined = "\n".join(buf).strip()
            if joined:
                blocks.append(("text", cur_section, joined))
            buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if is_section_header(line):
            flush_text()
            cur_section = line.strip()
            i += 1
            continue
        if line.strip() == "[표]":
            flush_text()
            tbl = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "[표]" or is_section_header(nxt):
                    break
                if nxt.strip() and (":" not in nxt) and ("/" not in nxt) and len(nxt.strip()) > 30:
                    break
                tbl.append(nxt)
                i += 1
            tbl_txt = "[표]\n" + "\n".join(l for l in tbl if l.strip())
            tbl_txt = tbl_txt.strip()
            # PM 요청: section에 표 헤더 요약 추가
            table_section = make_table_section(cur_section, tbl_txt)
            blocks.append(("table", table_section, tbl_txt))
            continue
        buf.append(line)
        i += 1
    flush_text()
    return blocks

# --- 토큰 기준 재귀 분할 (overlap 포함) -----------------------------
def _hard_slice(s: str, max_t: int, overlap: int):
    """줄바꿈 없는 긴 덩어리를 토큰 기준 강제 슬라이스(공백 경계 우선, overlap 포함)."""
    max_t = max(64, int(max_t * 0.9))
    words = s.split(" ")
    out, cur = [], []
    for w in words:
        cand = " ".join(cur + [w])
        if n_tokens(cand) > max_t and cur:
            out.append(" ".join(cur))
            back, acc = [], 0
            for x in reversed(cur):
                acc += n_tokens(x)
                back.insert(0, x)
                if acc >= overlap:
                    break
            cur = back + [w]
        else:
            cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out

# 수정①: 가드 전용 단순 슬라이스 — overlap 없음 (복제 원천 차단)
def _simple_slice(s: str, max_t: int):
    """가드 안전장치 전용. overlap 없이 max_t 이하로만 단순 분할.
    (overlap 중첩으로 인한 복제를 원천 차단)"""
    max_t = max(64, int(max_t * 0.9))
    words = s.split(" ")
    out, cur = [], []
    for w in words:
        if n_tokens(" ".join(cur + [w])) > max_t and cur:
            out.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out

def split_by_tokens(text: str, max_t: int, overlap: int):
    if n_tokens(text) <= max_t:
        return [text]
    units = re.split(r"\n", text)
    chunks, cur = [], []
    def emit(lines):
        chunks.append("\n".join(lines))
    for u in units:
        if n_tokens(u) > max_t:
            if cur:
                emit(cur); cur = []
            chunks.extend(_hard_slice(u, max_t, overlap))
            continue
        if cur and n_tokens("\n".join(cur + [u])) > max_t:
            emit(cur)
            back, acc = [], 0
            for line in reversed(cur):
                acc += n_tokens(line); back.insert(0, line)
                if acc >= overlap:
                    break
            cur = back + [u]
        else:
            cur.append(u)
    if cur:
        emit(cur)
    return chunks

def split_table(tbl_txt: str, max_t: int):
    """표는 행 경계로 분할. 단일 행이 max_t 초과면 그 행만 강제 슬라이스.

    [2026-06-22 옵션A 수정] 헤더는 첫 청크에만 포함, 둘째 청크부터는 본문 행만.
      - 원인: 긴 헤더(요구사항 표 등 600+토큰)를 매 청크에 반복하면
        실내용을 거의 못 담고 표가 과세분화됨 (DOC_093: 20행 → 54청크)
      - 수정: 헤더 1회만. 검색 맥락은 첫 청크가 담당, 과세분화 차단.
    """
    if n_tokens(tbl_txt) <= max_t:
        return [tbl_txt]
    rows = tbl_txt.split("\n")
    header = "\n".join(rows[:2])   # [표] + 컬럼행
    body = rows[2:]
    out, cur = [], []
    first = [True]                 # 첫 청크에만 헤더

    def cur_header():
        return header if first[0] else "[표]"   # 이후 청크는 [표] 마커만

    def flush():
        if cur:
            out.append(cur_header() + "\n" + "\n".join(cur))
            cur.clear()
            first[0] = False

    for r in body:
        h = cur_header()
        if n_tokens(h + "\n" + r) > max_t:      # 단일 행이 통째로 초과
            flush()
            h = cur_header()
            for piece in _hard_slice(r, max_t - n_tokens(h) - 1, CFG["overlap"]):
                out.append(h + "\n" + piece)
                first[0] = False
            continue
        cand = cur_header() + "\n" + "\n".join(cur + [r])
        if n_tokens(cand) > max_t and cur:
            flush()
            cur.append(r)
        else:
            cur.append(r)
    flush()
    return out

# --- has_image_budget 플래그: csv 금액 대조 ------------------------
_amt = re.compile(r"[\d,]{6,}\s*원|\d+\s*억")
def load_image_budget_doc_ids(docs):
    if not os.path.exists(CFG["csv_path"]):
        print(f"[warn] csv 없음({CFG['csv_path']}) → has_image_budget 전부 False")
        return set()
    rows = []
    with open(CFG["csv_path"], encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    name2amt = {}
    for r in rows:
        fn = r.get("파일명") or r.get("파일 명") or ""
        amt = r.get("사업금액") or r.get("사업 금액") or ""
        if fn:
            name2amt[unicodedata.normalize("NFC", fn)] = re.sub(r"[^\d]", "", amt.split(".")[0])
    flagged = set()
    for d in docs:
        fn = unicodedata.normalize("NFC", os.path.splitext(d["file_name"])[0])
        amt = ""
        for k, v in name2amt.items():
            if fn in k or k in fn:
                amt = v
                break
        if amt and len(amt) >= 7:
            text_digits = re.sub(r"[^\d]", "", d["text"])
            if amt not in text_digits:
                flagged.add(d["doc_id"])
    print(f"[info] has_image_budget 플래그: {len(flagged)}건 {sorted(flagged)}")
    return flagged

# --- 메타요약 청크: csv 구조화 메타 → 문서당 1개 검색 청크 ----------
def build_meta_chunks(docs, img_budget):
    if not os.path.exists(CFG["csv_path"]):
        print("[warn] csv 없음 → meta_summary 청크 생략")
        return []
    rows = []
    with open(CFG["csv_path"], encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    name2row = {}
    for r in rows:
        fn = r.get("파일명") or r.get("파일 명") or ""
        if fn:
            name2row[unicodedata.normalize("NFC", fn)] = r

    def g(r, *keys):
        for k in keys:
            if r.get(k):
                return r[k]
        return ""

    out = []
    for d in docs:
        fn = unicodedata.normalize("NFC", os.path.splitext(d["file_name"])[0])
        row = None
        for k, r in name2row.items():
            if fn in k or k in fn:
                row = r
                break
        if not row:
            continue
        name   = g(row, "사업명")
        amt    = g(row, "사업금액", "사업 금액").split(".")[0]
        org    = g(row, "발주기관", "발주 기관")
        opened = g(row, "공개일자", "공개 일자")
        ddl    = g(row, "입찰참여마감일", "입찰 참여 마감일")
        summ   = g(row, "사업요약", "사업 요약")
        amt_disp = f"{int(amt):,}원" if amt.isdigit() else (amt or "정보없음")

        parts = [f"[사업개요] 사업명: {name}", f"발주기관: {org}",
                 f"사업금액(예산): {amt_disp}"]
        if opened: parts.append(f"공개일자: {opened}")
        if ddl:    parts.append(f"입찰참여 마감일: {ddl}")
        if summ:   parts.append(f"사업요약:\n{summ}")
        content = clean_pua("\n".join(parts).strip())   # 수정③: PUA 정리

        out.append({
            "page_content": content,
            "metadata": {
                "doc_id": d["doc_id"],
                "file_name": d["file_name"],
                "section": "메타요약",
                "content_type": "meta_summary",
                "chunk_index": -1,
                "page": None,
                "token_count": n_tokens(content),
                "has_image_budget": d["doc_id"] in img_budget,
                "pii_masked": d.get("pii_masked"),
                "budget_amount": int(amt) if amt.isdigit() else None,
            },
        })
    print(f"[info] meta_summary 청크: {len(out)}건 (이미지금액 31건 예산 커버 포함)")
    return out


# --- 메인 -----------------------------------------------------------
def main():
    docs = json.load(open(CFG["in_path"], encoding="utf-8"))
    img_budget = load_image_budget_doc_ids(docs)

    chunks = []
    for d in docs:
        doc_id, fname = d["doc_id"], d["file_name"]
        flag = doc_id in img_budget
        pii_masked = d.get("pii_masked")
        # 수정③: 본문 PUA 정리 후 청킹
        clean_text = clean_pua(d["text"])
        ci = 0
        for kind, section, content in split_blocks(clean_text):
            pieces = (split_table(content, CFG["max_tokens"]) if kind == "table"
                      else split_by_tokens(content, CFG["max_tokens"], CFG["overlap"]))
            for p in pieces:
                p = p.strip()
                if not p:
                    continue
                chunks.append({
                    "page_content": p,
                    "metadata": {
                        "doc_id": doc_id,
                        "file_name": fname,
                        "section": section,
                        "content_type": kind,
                        "chunk_index": ci,
                        "page": None,
                        "token_count": n_tokens(p),
                        "has_image_budget": flag,
                        "pii_masked": pii_masked,
                    },
                })
                ci += 1

    # --- 작은 청크 병합 ---
    MIN_T = 60
    merged = []
    for c in chunks:
        if merged and c["metadata"]["token_count"] < MIN_T:
            p = merged[-1]
            m1, m2 = p["metadata"], c["metadata"]
            same = (m1["doc_id"] == m2["doc_id"]
                    and m1["content_type"] == m2["content_type"]
                    and m1["token_count"] + m2["token_count"] <= CFG["max_tokens"])
            if same:
                p["page_content"] += "\n" + c["page_content"]
                m1["token_count"] = n_tokens(p["page_content"])
                continue
        merged.append(c)
    chunks = merged

    # --- 최종 가드: max 초과 청크 강제 재분할 ---
    # 수정①: 이중 분할(split_by_tokens+_hard_slice) → overlap 없는 _simple_slice 1회
    #         (overlap 중첩으로 인한 복제 원천 차단)
    guarded = []
    for c in chunks:
        if c["metadata"]["token_count"] > CFG["max_tokens"]:
            for sub in _simple_slice(c["page_content"], CFG["max_tokens"]):
                nc = {"page_content": sub, "metadata": dict(c["metadata"])}
                nc["metadata"]["token_count"] = n_tokens(sub)
                guarded.append(nc)
        else:
            guarded.append(c)
    chunks = guarded

    # --- 수정②: doc 내 동일 content 중복 제거 (최종 안전장치) ---
    # 어떤 경로로든 생긴 완전 동일 청크를 제거. doc 경계로 구분(다른 문서 간
    # 공통 양식 문구는 보존), 순서 유지.
    deduped = []
    seen = set()
    removed = 0
    for c in chunks:
        key = (c["metadata"]["doc_id"], c["page_content"])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(c)
    if removed:
        print(f"[info] 중복 청크 제거: {removed}건 (doc 내 동일 content)")
    chunks = deduped

    # chunk_index 재부여(문서별)
    cnt = {}
    for c in chunks:
        did = c["metadata"]["doc_id"]
        c["metadata"]["chunk_index"] = cnt.get(did, 0)
        cnt[did] = c["metadata"]["chunk_index"] + 1

    os.makedirs(os.path.dirname(CFG["out_path"]), exist_ok=True)
    meta_chunks = build_meta_chunks(docs, img_budget)
    chunks = meta_chunks + chunks
    json.dump(chunks, open(CFG["out_path"], "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    n_tbl = sum(1 for c in chunks if c["metadata"]["content_type"] == "table")
    n_meta = sum(1 for c in chunks if c["metadata"]["content_type"] == "meta_summary")
    toks = [c["metadata"]["token_count"] for c in chunks]
    print(f"\n[done] chunks: {len(chunks)}  (meta {n_meta} / table {n_tbl} / text {len(chunks)-n_tbl-n_meta})")
    print(f"  토큰 평균 {sum(toks)//len(toks)} / 최소 {min(toks)} / 최대 {max(toks)}")
    print(f"  문서당 평균 {len(chunks)//len(docs)}청크")
    print(f"  저장: {CFG['out_path']}")

if __name__ == "__main__":
    main()
