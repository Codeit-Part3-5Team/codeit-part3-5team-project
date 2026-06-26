"""
run_eval.py — 채점 전용 (평가 환경에서 실행)

generate_answers.py가 저장한 samples JSON을 읽어 채점만 한다.
get_ai_response를 호출하지 않으므로 서비스 코드(langchain)에 의존하지 않고,
ragas가 요구하는 평가 환경(langchain 0.3 + ragas)에서 독립적으로 돌아간다.

흐름:
  [1] samples JSON 로드 (generate_answers.py 산출물)
  [2] 거부/비거부 분리 — 생성품질 지표는 비거부에만 적용
      (거부는 phase 2 llm_judge에서 ground_truth_refusal 스키마로 채점)
  [3] eval_generation(Faithfulness/AR/RAGAS) + eval_text(ROUGE/BLEU/BERTScore)
  [4] 점수·메타를 metrics_{tag}.json 으로 저장 + 콘솔 요약

실행 (프로젝트 루트, 평가 환경 venv-eval):
  python -m backend.evaluation.run_eval --samples samples_system_v2.json
  python -m backend.evaluation.run_eval --samples samples_system_v1.json

프롬프트 v1/v2 비교: 각 samples를 채점해 metrics_system_v1.json / metrics_system_v2.json 비교.
"""

import argparse
import json
import math
import re                                # judge-only 저장 파일명에서 모델명 정제용
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv          # .env 로드용
load_dotenv()                           # OPENAI_API_KEY 등 환경변수 로드 (채점 LLM 호출에 필요)

from backend.evaluation.eval_generation import evaluate_generation
from backend.evaluation.eval_text import evaluate_text
from backend.evaluation.llm_judge import evaluate_llm_judge  # 거부(refusal) 채점용

# generate_answers.py와 동일한 결과 디렉터리
RESULTS_DIR = Path("data/eval_results")


def load_samples(samples_arg: str) -> tuple[list[dict], dict]:
    """
    저장된 samples JSON을 로드한다. 파일명만 줘도 RESULTS_DIR 기준으로 찾는다.

    Args:
        samples_arg: samples 파일 경로 또는 파일명 (예: samples_system_v2.json)
    Returns:
        tuple: (samples list[dict], meta dict)
    """
    path = Path(samples_arg)
    if not path.exists():
        path = RESULTS_DIR / samples_arg   # 파일명만 준 경우 결과 폴더에서 찾기
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    # generate_answers는 {"meta":..., "samples":[...]} 구조로 저장함
    return payload["samples"], payload.get("meta", {})


def _breakdown_by_category(samples: list[dict], per_sample: list[dict]) -> dict:
    """
    비거부 샘플을 시나리오(category)별로 묶어 Faithfulness/AR 평균을 낸다.
    AR이 followup에서 유의미하다는 점이 글로벌 평균(단일답변 때문에 낮음)에 묻히지 않도록
    single_doc/multi_doc/followup로 분리 집계한다.

    Args:
        samples   : 채점한 비거부 샘플(evaluate_generation에 넘긴 것과 동일 순서)
        per_sample: evaluate_generation의 per_sample(samples와 같은 순서의 건별 점수)
    Returns:
        dict: {category: {"n":건수, "faithfulness":평균, "answer_relevancy":평균}, ...}
    """
    buckets = {}  # category → {"faith":[건별...], "ar":[건별...]}
    # 두 리스트는 같은 순서이므로 zip으로 category를 건별 점수에 다시 붙인다
    for s, row in zip(samples, per_sample):
        cat = s.get("category", "unknown")
        b = buckets.setdefault(cat, {"faith": [], "ar": []})
        b["faith"].append(row.get("faithfulness"))
        b["ar"].append(row.get("answer_relevancy"))

    def _mean(vals):
        # None/nan 제외하고 평균(유효값 없으면 None)
        nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        return round(sum(nums) / len(nums), 4) if nums else None

    return {
        cat: {
            "n": len(b["faith"]),
            "faithfulness": _mean(b["faith"]),
            "answer_relevancy": _mean(b["ar"]),
        }
        for cat, b in buckets.items()
    }


def run_refusal_only(samples_arg: str) -> None:
    """
    거부(refusal) 답변만 LLM Judge로 채점해 기존 metrics 파일에 덧붙인다.
    비거부 점수는 재채점하지 않고 기존 metrics 값을 그대로 보존한다(토큰 절약).

    Args:
        samples_arg: 채점할 samples JSON 파일명 또는 경로
    """
    # 1) samples 로드 후 거부만 추림
    samples, gen_meta = load_samples(samples_arg)
    prompt_version = gen_meta.get("prompt_version", "unknown")
    refusal = [s for s in samples if s["category"] == "refusal"]
    print(f"[refusal-only] {prompt_version}: 거부 {len(refusal)}건만 채점 (비거부 재채점 안 함)")
    if not refusal:
        print("거부 샘플이 없어 종료합니다.")
        return

    # 2) 기존 metrics 파일 로드 — 비거부 점수를 보존하기 위함(없으면 거부 결과만 단독 저장)
    metrics_path = RESULTS_DIR / f"metrics_{prompt_version}.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        print(f"경고: 기존 {metrics_path.name} 없음. 거부 결과만 단독 저장합니다.")
        metrics = {"prompt_version": prompt_version, "source_samples": samples_arg}

    # 3) 거부 채점 (LLM Judge = gpt-5-mini, refused/reason_quality/no_hallucination)
    print(f"[refusal-only] 거부 {len(refusal)}건 LLM Judge 채점 중...")
    judge_result = evaluate_llm_judge(refusal)
    metrics["refusal_judge"] = {
        "scores": judge_result["refusal"],          # 3축 평균 + count
        "per_sample": judge_result["per_sample"],   # 건별 점수·사유(케이스 분석용)
    }
    metrics["refusal_judged_at"] = datetime.now().isoformat(timespec="seconds")

    # 4) 저장 + 거부 요약 출력
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    rj = metrics["refusal_judge"]["scores"]
    print("\n========== 거부 채점 요약 ==========")
    print(f"prompt_version : {prompt_version}   (거부 {rj['count']}건)")
    print(f"refused          : {rj['refused']}   (적절히 거부했는가, 5점 만점)")
    print(f"reason_quality   : {rj['reason_quality']}   (거부 이유 명확성)")
    print(f"no_hallucination : {rj['no_hallucination']}   (거부하며 지어내지 않음)")
    print(f"저장             : {metrics_path}   (비거부 점수는 기존값 보존)")
    print("===================================")


def _judge_breakdown_by_category(per_sample: list[dict]) -> dict:
    """
    비거부(general) 건별 LLM Judge 점수를 시나리오(category)별로 묶어 평균낸다.
    per_sample에는 category가 없으므로 id 접두 규칙이 아닌, 호출부에서 붙여준
    category 필드를 사용한다(_run 내에서 주입).

    Args:
        per_sample: 건별 채점 결과(각 dict에 category 주입된 상태, is_refusal=False만 대상)
    Returns:
        dict: {category: {"n":건수, "faithfulness":평균, "relevance":평균, "accuracy":평균}}
    """
    buckets = {}  # category → {"faithfulness":[...], "relevance":[...], "accuracy":[...]}
    for row in per_sample:
        if row.get("is_refusal"):          # 거부는 축이 달라 시나리오 분해 대상에서 제외
            continue
        cat = row.get("category", "unknown")
        b = buckets.setdefault(cat, {"faithfulness": [], "relevance": [], "accuracy": []})
        for k in b:
            b[k].append(row.get(k))

    def _mean(vals):
        nums = [v for v in vals if isinstance(v, (int, float))]   # None(파싱실패) 제외
        return round(sum(nums) / len(nums), 2) if nums else None

    return {
        cat: {
            "n": len(b["faithfulness"]),
            "faithfulness": _mean(b["faithfulness"]),
            "relevance": _mean(b["relevance"]),
            "accuracy": _mean(b["accuracy"]),
        }
        for cat, b in buckets.items()
    }


def run_judge_only(samples_arg: str) -> None:
    """
    63건 전부를 LLM Judge로만 채점한다(RAGAS·eval_text 미사용). Ollama 후보 모델 비교용.
    비거부는 general 3축(faithfulness/relevance/accuracy), 거부는 refusal 3축으로
    llm_judge가 자동 분기한다. gpt 채점본을 덮지 않도록 모델별 파일로 분리 저장한다.

    Args:
        samples_arg: 채점할 samples JSON 파일명 또는 경로 (Ollama 생성 산출물)
    """
    # 1) samples 로드 — 모델명은 생성 meta(ollama_model)에서 읽어 저장 파일명에 박는다
    samples, gen_meta = load_samples(samples_arg)
    prompt_version = gen_meta.get("prompt_version", "unknown")
    model = gen_meta.get("ollama_model") or gen_meta.get("model") or "unknown"
    # 파일명에 못 쓰는 문자(콜론 등) → 하이픈. 예: "qwen3:8b" → "qwen3-8b"
    model_tag = re.sub(r"[^0-9A-Za-z._-]", "-", model)

    print(f"[judge-only] {prompt_version} / 모델={model}  전체 {len(samples)}건 LLM Judge 채점")

    # 2) 채점 — 63건 통째로 넘기면 llm_judge가 비거부/거부를 자동 분기해 양쪽 평균을 돌려줌
    #    (RAGAS·eval_text 미호출 = OpenAI 호출은 채점 건수만큼만 발생)
    judge_result = evaluate_llm_judge(samples)

    # 2-1) 시나리오별 분해를 위해 per_sample에 category를 주입(id 순서가 samples와 동일)
    for row, s in zip(judge_result["per_sample"], samples):
        row["category"] = s.get("category", "unknown")
    by_category = _judge_breakdown_by_category(judge_result["per_sample"])

    # 3) 모델별 파일로 저장 — metrics_{prompt_version}.json(gpt 채점본)을 절대 건드리지 않음
    metrics = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "eval_mode": "judge_only",                  # RAGAS 아님(LLM Judge 전용) 표식
        "prompt_version": prompt_version,
        "ollama_model": model,
        "source_samples": samples_arg,
        "generated_at": gen_meta.get("generated_at"),
        "generation_tokens": gen_meta.get("total_tokens"),
        "n_total": len(samples),
        "scores": {
            "general": judge_result["general"],     # 비거부 50건: faithfulness/relevance/accuracy
            "refusal": judge_result["refusal"],     # 거부 13건: refused/reason_quality/no_hallucination
        },
        "general_by_category": by_category,         # single/multi/followup 분해(환각 안전성 비교용)
        "per_sample": judge_result["per_sample"],   # 건별(Q번호별 점수·사유, 케이스 분석용)
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"metrics_ollama_{model_tag}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 4) 콘솔 요약
    g = judge_result["general"]
    r = judge_result["refusal"]
    print("\n========== LLM Judge 요약 (judge-only) ==========")
    print(f"모델           : {model}   (prompt={prompt_version})")
    print(f"-- 비거부 general (n={g['count']}) --")
    print(f"  faithfulness : {g['faithfulness']}")
    print(f"  relevance    : {g['relevance']}")
    print(f"  accuracy     : {g['accuracy']}")
    print("  -- 시나리오별 (Faith/Rel/Acc) --")
    for cat, sc in by_category.items():
        print(f"    {cat:<10} n={sc['n']:<2} F={sc['faithfulness']} R={sc['relevance']} A={sc['accuracy']}")
    print(f"-- 거부 refusal (n={r['count']}) --")
    print(f"  refused          : {r['refused']}")
    print(f"  reason_quality   : {r['reason_quality']}")
    print(f"  no_hallucination : {r['no_hallucination']}")
    print(f"저장           : {metrics_path}   (gpt 채점본 metrics_{prompt_version}.json 미변경)")
    print("=================================================")


def main():
    """run_eval 진입점. samples 로드 → 채점 → metrics 저장 → 요약 출력."""
    parser = argparse.ArgumentParser(description="채점 전용 (생성과 분리)")
    parser.add_argument("--samples", type=str, required=True,
                        help="채점할 samples JSON (generate_answers 산출물). 파일명 또는 경로.")
    parser.add_argument("--judge-refusal", action="store_true",
                        help="거부(refusal) 답변을 LLM Judge로 채점한다. 없으면 거부는 집계만 하고 건너뛴다.")
    parser.add_argument("--refusal-only", action="store_true",
                        help="비거부 재채점을 건너뛰고 거부만 채점해 기존 metrics에 덧붙인다(토큰 절약).")
    parser.add_argument("--judge-only", action="store_true",
                        help="63건 전부 LLM Judge로만 채점한다(RAGAS 미사용). Ollama 모델 비교용, 모델별 파일로 저장.")
    args = parser.parse_args()

    # --judge-only: Ollama 후보 비교. RAGAS 대신 LLM Judge로 전체 채점, 모델별 파일 분리 저장
    if args.judge_only:
        run_judge_only(args.samples)
        return

    # --refusal-only: 비거부는 STEP 3 값 그대로 두고, 거부만 채점해 기존 metrics에 덧붙임
    if args.refusal_only:
        run_refusal_only(args.samples)
        return

    # 1) samples 로드
    samples, gen_meta = load_samples(args.samples)
    prompt_version = gen_meta.get("prompt_version", "unknown")
    print(f"[1] samples 로드: {len(samples)}건 "
          f"(prompt_version={prompt_version}, 생성토큰={gen_meta.get('total_tokens')})")

    # 2) 거부/비거부 분리 — 생성품질 지표는 비거부에만(거부는 phase 2 llm_judge)
    non_refusal = [s for s in samples if s["category"] != "refusal"]
    refusal = [s for s in samples if s["category"] == "refusal"]
    print(f"[2] 비거부 {len(non_refusal)}건 생성품질 채점 / 거부 {len(refusal)}건은 phase 2 보류")

    # 3) 채점 (ragas judge = gpt-5-mini, config는 eval_generation 내부에서 로드)
    print("[3] 채점 중...")
    gen_scores = evaluate_generation(non_refusal)    # Faithfulness/AR/RAGAS (비거부)
    text_scores = evaluate_text(non_refusal)         # ROUGE/BLEU/BERTScore (비거부)

    # 3-1) 시나리오(category)별 Faithfulness/AR 분해 — followup AR이 글로벌 평균에 묻히지 않게
    by_category = _breakdown_by_category(non_refusal, gen_scores["per_sample"])

    # 3-2) 거부(refusal) 채점 — 플래그가 있을 때만. LLM Judge로 거부 적절성 채점
    #      (refused / reason_quality / no_hallucination 3축, gpt-5-mini가 통째로 총평)
    refusal_judge = None
    if args.judge_refusal and refusal:
        print(f"[3-2] 거부 {len(refusal)}건 LLM Judge 채점 중...")
        judge_result = evaluate_llm_judge(refusal)        # 거부 그룹 평균 + 건별 점수
        refusal_judge = {
            "scores": judge_result["refusal"],            # 3축 평균 + count
            "per_sample": judge_result["per_sample"],     # 건별 점수·사유(케이스 분석용)
        }

    # 4) 메타+점수 저장 + 콘솔 요약
    metrics = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "prompt_version": prompt_version,
        "source_samples": args.samples,
        "generated_at": gen_meta.get("generated_at"),
        "generation_tokens": gen_meta.get("total_tokens"),
        "n_total": len(samples),
        "n_generation_eval": len(non_refusal),   # 생성품질 채점 대상(비거부)
        "n_text_eval": len(non_refusal),         # 텍스트 지표 대상(비거부 = 동일)
        "n_refusal_deferred": len(refusal),      # phase 2로 미룬 거부 건수
        "scores": {
            "faithfulness": gen_scores["faithfulness"],
            "answer_relevancy": gen_scores["answer_relevancy"],
            "ragas_score": gen_scores["ragas_score"],
            "rouge1": text_scores["rouge1"],
            "rougeL": text_scores["rougeL"],
            "bleu": text_scores["bleu"],
            "bertscore_f1": text_scores["bertscore_f1"],
        },
        "scores_by_category": by_category,   # 시나리오별 Faithfulness/AR(분석·리포트용)
    }
    # 거부 채점을 한 경우에만 refusal_judge 섹션 추가(안 했으면 키 자체를 넣지 않음)
    if refusal_judge is not None:
        metrics["refusal_judge"] = refusal_judge

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / f"metrics_{prompt_version}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 콘솔 요약(1차 기록은 metrics json, 아래는 눈으로 확인용)
    print("\n========== 평가 요약 ==========")
    print(f"prompt_version : {prompt_version}   (생성품질 채점 = 비거부 {len(non_refusal)}건)")
    print(f"Faithfulness   : {metrics['scores']['faithfulness']}   (목표 >=0.75)")
    print(f"AnswerRelevance: {metrics['scores']['answer_relevancy']}   (목표 >=0.75)")
    print(f"RAGAS-Score    : {metrics['scores']['ragas_score']}   (목표 >=0.70)")
    print(f"ROUGE-1 / L    : {metrics['scores']['rouge1']} / {metrics['scores']['rougeL']}   (보조)")
    print(f"BLEU           : {metrics['scores']['bleu']}   (보조)")
    print(f"BERTScore-F1   : {metrics['scores']['bertscore_f1']}   (보조)")
    print("--- 시나리오별 (Faithfulness / AR) ---")
    for cat, sc in by_category.items():
        print(f"  {cat:<10} n={sc['n']:<2} Faith={sc['faithfulness']}  AR={sc['answer_relevancy']}")
    # 거부 채점을 한 경우에만 거부 요약 출력
    if refusal_judge is not None:
        rj = refusal_judge["scores"]
        print(f"--- 거부 채점 (LLM Judge, n={rj['count']}) ---")
        print(f"  refused         : {rj['refused']}   (적절히 거부했는가, 5점 만점)")
        print(f"  reason_quality  : {rj['reason_quality']}   (거부 이유 명확성)")
        print(f"  no_hallucination: {rj['no_hallucination']}   (거부하며 지어내지 않음)")
    print(f"저장           : {metrics_path}")
    print("================================")


if __name__ == "__main__":
    main()