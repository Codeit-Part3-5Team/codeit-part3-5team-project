# -*- coding: utf-8 -*-
# =====================================================================
#  골든셋 큐레이션 라벨 내용 정합성 확인 — Q015~Q020 전수
#  실행 : python evaluation/verify_curated_labels.py
#  목적 : data_authored 라벨이 가리키는 청크 내용이 질문 subtype과 맞는지
#         전수 확인 → 재매핑 대상 확정 (아인님 큐레이션 입력용)
# =====================================================================
import json

FINAL = "data/processed/chunks_v1_enriched.json"
GOLDEN = "data/processed/golden_dataset.json"

# subtype별 "내용"에 나타나야 할 핵심 키워드 (section명 아닌 본문 기준)
SUBTYPE_KW = {
    "requirement":   ["요구사항", "과업", "제안요청", "기능", "성능", "수행 방법",
                      "메뉴", "콘텐츠", "구축", "개선"],
    "qualification": ["입찰 참가 자격", "참가자격", "자격요건", "단독입찰",
                      "공동수급", "제출 서류", "제출서류", "구비서류", "서식"],
    "scoring":       ["배점", "평가항목", "기술능력 평가", "정량적 평가",
                      "정성적 평가", "심사항목", "평점", "가격 평가"],
}


def main():
    chunks = json.load(open(FINAL, encoding="utf-8"))
    golden = json.load(open(GOLDEN, encoding="utf-8"))
    body = {(c["metadata"]["doc_id"], c["metadata"]["chunk_index"]):
            (c["metadata"].get("section", ""), c["page_content"])
            for c in chunks}

    targets = ["Q015", "Q016", "Q017", "Q018", "Q019", "Q020"]
    print("=" * 70)
    print("  큐레이션 라벨 내용 정합성 (본문 키워드 기준)")
    print("=" * 70)

    for q in golden:
        if q["id"] not in targets:
            continue
        st = q.get("q_subtype", "")
        kws = SUBTYPE_KW.get(st, [])
        labs = [l for l in q.get("answer_chunk_labels", []) if l[1] != -1]
        print(f"\n[{q['id']}] {st}  (라벨 {len(labs)}개)")
        print(f"  질문: {q.get('question','')[:60]}")
        ok_cnt = 0
        for did, ci in labs:
            sec, txt = body.get((did, ci), ("(없음)", ""))
            # 본문(앞 300자)에 subtype 키워드 있는지
            hit = [k for k in kws if k in txt[:300]]
            mark = "OK " if hit else "✗  "
            if hit:
                ok_cnt += 1
            print(f"    {mark}[{ci}] {sec[:30]} | 매칭:{hit[:3]}")
        verdict = "정합" if ok_cnt == len(labs) else (
            "전부 불일치 → 재매핑" if ok_cnt == 0 else f"부분({ok_cnt}/{len(labs)})")
        print(f"  → 판정: {verdict}")

    print("\n" + "=" * 70)
    print("  ※ 본문 키워드 기준. section명이 아니라 실제 내용으로 판정.")
    print("=" * 70)


if __name__ == "__main__":
    main()
