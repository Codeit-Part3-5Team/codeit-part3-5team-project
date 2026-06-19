# -*- coding: utf-8 -*-
# =====================================================================
#  입찰메이트 RFP RAG — 베이스라인 청킹 (DE: 도혁)
#  입력 : data/processed/parsed_documents_v2.json (마스킹본이면 masked json으로 교체)
#  출력 : data/processed/chunks_v1.json  (LangChain Document 호환 dict 리스트)
#  설계 : ① 로마숫자 섹션 1차 분할  ② 표([표] 블록) content_type=table 로 분리
#         ③ 본문은 512~1024 토큰 + overlap 150 재귀 분할
#         ④ has_image_budget 플래그(csv 금액 누락 31건 추적성)
# =====================================================================
import os
import re
import csv
import json
import unicodedata

# --- CFG -------------------------------------------------------------
CFG = {
    "in_path":   "data/processed/masked_documents_v2.json",
    "csv_path":  "data/raw/중급 프로젝트/원본 데이터/data_list.csv",  # 금액 대조용
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
except Exception:
    def n_tokens(s: str) -> int:
        # 한글/한자 0.6, ascii 0.25 가중 근사
        kr = sum(1 for c in s if ord(c) > 0x3000)
        return int(kr * 0.6 + (len(s) - kr) * 0.25) + 1

# --- 섹션 헤더 판정 --------------------------------------------------
# 단독 라인이면서 "로마숫자 + 공백 + 짧은 제목" 인 경우만 본문 섹션으로 인정.
# (목차처럼 한 줄에 여러 로마숫자가 뭉친 라인은 제외)
ROMAN = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ"
_roman_hdr = re.compile(rf"^[{ROMAN}]\s+\S.{{0,40}}$")

def is_section_header(line: str) -> bool:
    line = line.strip()
    if not _roman_hdr.match(line):
        return False
    # 한 줄에 로마숫자가 2개 이상이면 목차 → 제외
    if sum(line.count(c) for c in ROMAN) > 1:
        return False
    # 페이지 번호 점선 목차 흔적 제외
    if re.search(r"\d{2,}\s*$", line) and len(line) > 25:
        return False
    return True

# --- 표 블록 / 본문 블록 분해 ---------------------------------------
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
            # 다음 [표] / 섹션헤더 / 빈줄2개 전까지 표 행으로 수집
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() == "[표]" or is_section_header(nxt):
                    break
                # 표 행은 'key: value' 패턴 위주. 패턴 깨지고 일반 문장 시작이면 종료
                if nxt.strip() and (":" not in nxt) and ("/" not in nxt) and len(nxt.strip()) > 30:
                    break
                tbl.append(nxt)
                i += 1
            tbl_txt = "[표]\n" + "\n".join(l for l in tbl if l.strip())
            blocks.append(("table", cur_section, tbl_txt.strip()))
            continue
        buf.append(line)
        i += 1
    flush_text()
    return blocks

# --- 토큰 기준 재귀 분할 (overlap 포함) -----------------------------
def _hard_slice(s: str, max_t: int, overlap: int):
    """줄바꿈 없는 긴 덩어리를 토큰 기준 강제 슬라이스(공백 경계 우선)."""
    max_t = max(64, int(max_t * 0.9))     # overlap 가산분 고려 안전 마진
    words = s.split(" ")
    out, cur = [], []
    for w in words:
        cand = " ".join(cur + [w])
        if n_tokens(cand) > max_t and cur:
            out.append(" ".join(cur))
            back, acc = [], 0
            for x in reversed(cur):
                acc += n_tokens(x); back.insert(0, x)
                if acc >= overlap:
                    break
            cur = back + [w]
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
        # 한 줄 자체가 max_t 초과면 강제 슬라이스
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
    """표는 행 경계로 분할. 단일 행이 max_t 초과면 그 행만 강제 슬라이스."""
    if n_tokens(tbl_txt) <= max_t:
        return [tbl_txt]
    rows = tbl_txt.split("\n")
    header = "\n".join(rows[:2])           # [표] + 컬럼행
    body = rows[2:]
    out, cur = [], []
    def flush():
        if cur:
            out.append(header + "\n" + "\n".join(cur))
            cur.clear()
    for r in body:
        if n_tokens(header + "\n" + r) > max_t:   # 단일 행이 통째로 초과
            flush()
            for piece in _hard_slice(r, max_t - n_tokens(header) - 1, CFG["overlap"]):
                out.append(header + "\n" + piece)
            continue
        cand = header + "\n" + "\n".join(cur + [r])
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
    """csv 사업금액이 있는데 text에서 그 금액이 안 잡히는 doc → has_image_budget=True"""
    if not os.path.exists(CFG["csv_path"]):
        print(f"[warn] csv 없음({CFG['csv_path']}) → has_image_budget 전부 False")
        return set()
    # csv 로드 (컬럼 공백 strip, 파일명 NFC)
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
        # 파일명 부분매칭 (확장자/경로 차이 흡수)
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

# --- 메인 -----------------------------------------------------------
def main():
    docs = json.load(open(CFG["in_path"], encoding="utf-8"))
    img_budget = load_image_budget_doc_ids(docs)

    chunks = []
    for d in docs:
        doc_id, fname = d["doc_id"], d["file_name"]
        flag = doc_id in img_budget
        pii_masked = d.get("pii_masked")           # 마스킹본 추가 메타(있으면 보존)
        masking_counts = d.get("masking_counts")
        ci = 0
        for kind, section, content in split_blocks(d["text"]):
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
                        "content_type": kind,            # text | table
                        "chunk_index": ci,
                        "page": None,                    # hwp → nullable
                        "token_count": n_tokens(p),
                        "has_image_budget": flag,
                        "pii_masked": pii_masked,
                    },
                })
                ci += 1

    # --- 작은 청크 병합: 같은 doc/section/content_type 내 직전 청크에 흡수 ---
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
    # chunk_index 재부여(문서별)
    cnt = {}
    for c in merged:
        did = c["metadata"]["doc_id"]
        c["metadata"]["chunk_index"] = cnt.get(did, 0)
        cnt[did] = c["metadata"]["chunk_index"] + 1
    chunks = merged

    # --- 최종 가드: max 초과 청크 강제 재분할 (근사 카운터 오차 방어) ---
    guarded = []
    for c in chunks:
        if c["metadata"]["token_count"] > CFG["max_tokens"]:
            for piece in split_by_tokens(c["page_content"], CFG["max_tokens"], CFG["overlap"]):
                for sub in _hard_slice(piece, CFG["max_tokens"], CFG["overlap"]) \
                           if n_tokens(piece) > CFG["max_tokens"] else [piece]:
                    nc = {"page_content": sub, "metadata": dict(c["metadata"])}
                    nc["metadata"]["token_count"] = n_tokens(sub)
                    guarded.append(nc)
        else:
            guarded.append(c)
    chunks = guarded
    # chunk_index 재부여(문서별)
    cnt = {}
    for c in chunks:
        did = c["metadata"]["doc_id"]
        c["metadata"]["chunk_index"] = cnt.get(did, 0)
        cnt[did] = c["metadata"]["chunk_index"] + 1

    os.makedirs(os.path.dirname(CFG["out_path"]), exist_ok=True)
    json.dump(chunks, open(CFG["out_path"], "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 요약
    n_tbl = sum(1 for c in chunks if c["metadata"]["content_type"] == "table")
    toks = [c["metadata"]["token_count"] for c in chunks]
    print(f"\n[done] chunks: {len(chunks)}  (table {n_tbl} / text {len(chunks)-n_tbl})")
    print(f"  토큰 평균 {sum(toks)//len(toks)} / 최소 {min(toks)} / 최대 {max(toks)}")
    print(f"  문서당 평균 {len(chunks)//len(docs)}청크")
    print(f"  저장: {CFG['out_path']}")

if __name__ == "__main__":
    main()
