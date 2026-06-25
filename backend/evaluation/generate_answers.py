"""
generate_answers.py — 답변 생성 전용 (생성 환경에서 실행)

생성과 평가를 분리한다. 이 파일은 get_ai_response로 답변만 생성해 JSON으로 저장한다.
평가(ragas 등)는 import하지 않으므로 서비스 환경(langchain 1.x)에서 그대로 돌아간다.
채점은 별도로 run_eval.py가 저장된 JSON을 읽어 수행한다(평가 환경 langchain 0.3+ragas).

흐름:
  [1] 골든셋 로드 (config의 evaluation.golden_dataset_path)
  [2] 각 문항을 get_ai_response로 통과시켜 답변 생성
      - followup은 history를 OpenAI 형식으로 변환해 전달
  [3] samples를 data/eval_results/samples_{tag}.json 으로 저장
      (NDA: contexts=RFP 원문 청크 → gitignore된 data/ 하위에 저장)

실행 (프로젝트 루트, 생성 환경 venv):
  python -m backend.evaluation.generate_answers --limit 10      # 소량 테스트
  python -m backend.evaluation.generate_answers                 # 전체
  python -m backend.evaluation.generate_answers --use-ollama --ollama-model qwen3:8b   # 트랙②용

prompt_version은 config.yaml(top-level)에서 읽는다. v1/v2 비교는 config를 바꿔가며 2번 실행.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from utils.config import load_config
from backend.pipeline import get_ai_response

# 생성 결과 저장 위치 — NDA(contexts=RFP 원문) 보호 위해 gitignore된 data/ 하위
RESULTS_DIR = Path("data/eval_results")


def convert_history(golden_history: list[dict]) -> list[dict]:
    """
    골든셋 history 형식을 generate_answer가 기대하는 OpenAI 형식으로 변환한다.

    Args:
        golden_history: [{"question": str, "answer": str}, ...] (골든셋 followup 형식)
    Returns:
        list[dict]: [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
    """
    messages = []
    for turn in golden_history or []:
        # 한 턴 = 사용자 질문 + 어시스턴트 답변 2개 메시지로 펼침
        messages.append({"role": "user", "content": turn.get("question", "")})
        messages.append({"role": "assistant", "content": turn.get("answer", "")})
    return messages


def load_golden(config: dict) -> list[dict]:
    """
    config의 골든셋 경로에서 골든셋을 로드한다.

    Args:
        config: 설정 dict (evaluation.golden_dataset_path 사용)
    Returns:
        list[dict]: 골든셋 문항 리스트
    """
    path = config["evaluation"]["golden_dataset_path"]   # 예: ./data/golden/golden_dataset_v3.json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_samples(golden_items: list[dict], config: dict,
                  use_ollama: bool = False) -> tuple[list[dict], dict]:
    """
    골든셋 각 문항을 get_ai_response로 통과시켜 채점용 samples를 만든다.

    Args:
        golden_items: 골든셋 문항 리스트
        config      : 설정 dict (prompt_version·top_k 등 주입)
        use_ollama  : True면 Ollama로 생성(트랙②), 기본 False(gpt-5-mini)
    Returns:
        tuple:
          samples (list[dict]): [{id, question, answer, contexts, ground_truth,
                                  ground_truth_refusal, category, q_subtype,
                                  elapsed_sec, tokens_used}, ...]
          gen_meta (dict)     : {total_tokens, total_elapsed_sec, count}
    """
    samples = []
    total_tokens = 0
    total_elapsed = 0.0

    for i, item in enumerate(golden_items, 1):
        # followup이면 history를 OpenAI 형식으로 변환해 전달(없으면 빈 리스트)
        history = convert_history(item.get("history"))

        # 실제 서비스 경로로 답변 생성 (config의 prompt_version이 여기서 반영됨)
        result = get_ai_response(
            query=item["question"],
            history=history,
            config=config,
            use_ollama=use_ollama,
        )

        # 채점 3종 공통 스키마로 sample 구성 (채점은 run_eval이 이 JSON을 읽어 수행)
        sample = {
            "id": item.get("id"),
            "question": item["question"],
            "answer": result["answer"],                       # 생성된 답변(채점 대상)
            "contexts": result["retrieved_chunks"],           # 검색 청크 list[str] (= RAGAS contexts)
            "ground_truth": item["answer"],                   # 정답 텍스트(문자열)
            "ground_truth_refusal": item.get("ground_truth_refusal"),  # 거부 채점용 dict(있으면)
            "category": item.get("category"),
            "q_subtype": item.get("q_subtype"),
            "elapsed_sec": result["elapsed_sec"],
            "tokens_used": result["tokens_used"],
        }
        samples.append(sample)

        total_tokens += result.get("tokens_used") or 0
        total_elapsed += result.get("elapsed_sec") or 0.0

        # 진행 상황 출력(긴 실행 중 멈춤 여부 확인용)
        print(f"  [{i}/{len(golden_items)}] {item.get('id')} ({item.get('category')}) "
              f"tokens={result['tokens_used']} {result['elapsed_sec']}s")

    gen_meta = {
        "total_tokens": total_tokens,
        "total_elapsed_sec": round(total_elapsed, 2),
        "count": len(samples),
    }
    return samples, gen_meta


def main():
    """generate_answers 진입점. 골든셋 → 답변 생성 → samples JSON 저장."""
    parser = argparse.ArgumentParser(description="답변 생성 전용 (평가와 분리)")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞에서 N건만 처리(테스트용). 미지정 시 전체.")
    parser.add_argument("--use-ollama", action="store_true",
                        help="Ollama로 답변 생성(트랙②). 기본은 gpt-5-mini.")
    parser.add_argument("--ollama-model", type=str, default=None,
                        help="Ollama 모델명(트랙②, 예: qwen3:8b). 지정 시 config.ollama_model 덮어씀.")
    args = parser.parse_args()

    config = load_config()
    # Ollama 모델을 인자로 받으면 config에 주입(모델별로 따로 생성·저장하기 위함)
    if args.ollama_model:
        config["ollama_model"] = args.ollama_model

    prompt_version = config.get("prompt_version", "system_v2")   # top-level에서 읽음(트랙③ 전환)

    # 1) 골든셋 로드 + (테스트면) 건수 제한
    golden = load_golden(config)
    if args.limit is not None:
        golden = golden[:args.limit]
    print(f"[1] 골든셋 로드: {len(golden)}건 "
          f"(prompt_version={prompt_version}, use_ollama={args.use_ollama}, "
          f"ollama_model={config.get('ollama_model') if args.use_ollama else '-'})")

    # 2) 답변 생성
    print("[2] 답변 생성 중...")
    samples, gen_meta = build_samples(golden, config, use_ollama=args.use_ollama)

    # 3) samples 저장 — 파일명 태그: 프롬프트 버전 + (ollama면 모델명)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.use_ollama:
        model_tag = (config.get("ollama_model") or "ollama").replace(":", "-").replace("/", "-")
        tag = f"{prompt_version}_ollama_{model_tag}"
    else:
        tag = prompt_version
    samples_path = RESULTS_DIR / f"samples_{tag}.json"

    # 생성 메타도 함께 저장(보고서 기록·토큰 추적용)
    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "prompt_version": prompt_version,
            "use_ollama": args.use_ollama,
            "ollama_model": config.get("ollama_model") if args.use_ollama else None,
            "limit": args.limit,
            "total_tokens": gen_meta["total_tokens"],
            "total_elapsed_sec": gen_meta["total_elapsed_sec"],
            "count": gen_meta["count"],
        },
        "samples": samples,
    }
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n생성 완료: {gen_meta['count']}건 / 누적 토큰={gen_meta['total_tokens']} / "
          f"{gen_meta['total_elapsed_sec']}s")
    print(f"저장: {samples_path}")
    print(f"→ 평가 환경(venv-eval)에서: python -m backend.evaluation.run_eval --samples {samples_path.name}")


if __name__ == "__main__":
    main()