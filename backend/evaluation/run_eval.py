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
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv          # .env 로드용
load_dotenv()                           # OPENAI_API_KEY 등 환경변수 로드 (채점 LLM 호출에 필요)

from backend.evaluation.eval_generation import evaluate_generation
from backend.evaluation.eval_text import evaluate_text
# phase 2(Ollama 거부 채점)에서 사용 예정:
# from backend.evaluation.llm_judge import evaluate_llm_judge

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


def main():
    """run_eval 진입점. samples 로드 → 채점 → metrics 저장 → 요약 출력."""
    parser = argparse.ArgumentParser(description="채점 전용 (생성과 분리)")
    parser.add_argument("--samples", type=str, required=True,
                        help="채점할 samples JSON (generate_answers 산출물). 파일명 또는 경로.")
    args = parser.parse_args()

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
    }
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
    print(f"저장           : {metrics_path}")
    print("================================")


if __name__ == "__main__":
    main()