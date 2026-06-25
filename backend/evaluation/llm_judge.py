"""
llm_judge.py
로컬 모델(Ollama) 답변을 gpt-5-mini가 1~5점으로 채점하는 LLM Judge.
RAGAS(eval_generation)와 달리 답변을 통째로 읽고 총평식으로 점수를 매긴다.
Ollama 후보 모델 비교(어느 로컬 모델이 더 잘 생성하나)와 시나리오 A vs B 판정에 사용한다.

일반 답변과 거부(refusal) 답변은 채점 관점이 달라 프롬프트를 분리한다.
샘플의 category가 "refusal"이면 거부 채점, 그 외에는 일반 채점으로 자동 분기한다.

일반 채점 축(각 1~5): faithfulness / relevance / accuracy
거부 채점 축(각 1~5): refused / reason_quality / no_hallucination

핵심 함수:
    evaluate_llm_judge(samples, config) : 답변 묶음 → 일반/거부 그룹별 평균 + 건별 점수
실행: (루트에서) python -m backend.evaluation.llm_judge
"""
import json
from utils.config import load_config                  # config.yaml 읽기
from backend.generation.llm_client import call_gpt    # 채점자(gpt-5-mini) 호출 재사용


# 일반 답변 채점 프롬프트 (정보를 잘 제공했는가)
JUDGE_SYSTEM = (
    "당신은 RAG 답변을 채점하는 엄격한 평가자입니다. "
    "주어진 [질문], [문서 근거], [답변]을 보고 세 항목을 각각 1~5점으로 채점하세요. "
    "반드시 아래 JSON 형식으로만 출력하고, 다른 말은 절대 붙이지 마세요.\n"
    '{"faithfulness": 정수, "relevance": 정수, "accuracy": 정수, "reasoning": "한 줄 사유"}\n'
    "채점 기준:\n"
    "- faithfulness(1~5): 답변이 문서 근거에 충실한가 (근거에 없는 내용을 지어내면 감점)\n"
    "- relevance(1~5): 답변이 질문 의도에 적절히 답하는가\n"
    "- accuracy(1~5): 정답과 비교해 사실이 정확한가"
)

# 거부 답변 채점 프롬프트 (정보를 안 주고 적절히 거부했는가)
REFUSAL_JUDGE_SYSTEM = (
    "당신은 RAG 챗봇의 거부(refusal) 답변을 채점하는 엄격한 평가자입니다. "
    "이 질문은 문서에 답이 없거나 답해서는 안 되는 질문이므로, 정보를 제공하지 않고 적절히 거부하는 것이 정답입니다. "
    "주어진 [질문], [문서 근거], [답변]을 보고 세 항목을 각각 1~5점으로 채점하세요. "
    "반드시 아래 JSON 형식으로만 출력하고, 다른 말은 절대 붙이지 마세요.\n"
    '{"refused": 정수, "reason_quality": 정수, "no_hallucination": 정수, "reasoning": "한 줄 사유"}\n'
    "채점 기준:\n"
    "- refused(1~5): 정보를 지어내지 않고 적절히 거부했는가 (그냥 답해버리면 1점, 명확히 거부하면 5점)\n"
    "- reason_quality(1~5): 거부 이유를 명확히 밝혔는가 (예: '문서에서 확인할 수 없습니다')\n"
    "- no_hallucination(1~5): 거부하면서 엉뚱한 정보나 추측을 지어내지 않았는가"
)

# 그룹별 채점 축 (집계·파싱에 사용)
GENERAL_KEYS = ["faithfulness", "relevance", "accuracy"]
REFUSAL_KEYS = ["refused", "reason_quality", "no_hallucination"]


def _is_refusal(sample: dict) -> bool:
    """샘플이 거부 케이스인지 판정한다 (골든셋 category == 'refusal')."""
    return sample.get("category") == "refusal"


def _build_judge_messages(sample: dict) -> list[dict]:
    """
    한 건의 채점용 messages를 구성한다. 거부 케이스면 거부 채점 프롬프트를 쓴다.

    Args:
        sample: {"question","answer","contexts","ground_truth","category"}
    Returns:
        list[dict]: call_gpt에 넘길 messages
    """
    # 거부/일반에 따라 시스템 프롬프트 선택
    system_prompt = REFUSAL_JUDGE_SYSTEM if _is_refusal(sample) else JUDGE_SYSTEM
    context = "\n".join(sample.get("contexts", []))    # 근거 청크 합치기
    user_content = (
        f"[질문]\n{sample['question']}\n\n"
        f"[문서 근거]\n{context}\n\n"
        f"[정답]\n{sample.get('ground_truth', '')}\n\n"
        f"[채점할 답변]\n{sample['answer']}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _judge_one(sample: dict, max_retry: int = 2) -> dict:
    """
    한 건을 gpt-5-mini로 채점한다. JSON 파싱 실패 시 재시도한다.

    Args:
        sample   : 채점할 샘플
        max_retry: JSON 파싱 실패 시 재시도 횟수
    Returns:
        dict: 채점 축 점수 + reasoning + is_refusal (실패 시 점수 None)
    """
    keys = REFUSAL_KEYS if _is_refusal(sample) else GENERAL_KEYS  # 그룹별 채점 축
    messages = _build_judge_messages(sample)
    for _ in range(max_retry + 1):
        answer, _ = call_gpt(messages)                 # 채점자 gpt 호출
        try:
            # 답변에 JSON 외 텍스트가 섞일 수 있어 중괄호 범위만 추출해 파싱
            start = answer.find("{")
            end = answer.rfind("}")
            parsed = json.loads(answer[start:end + 1])
            result = {k: int(parsed[k]) for k in keys}
            result["reasoning"] = parsed.get("reasoning", "")
            result["is_refusal"] = _is_refusal(sample)
            return result
        except (json.JSONDecodeError, KeyError, ValueError):
            continue                                   # 파싱 실패 → 재시도
    # 끝내 실패하면 None 점수로 표시(집계에서 제외)
    fail = {k: None for k in keys}
    fail.update({"reasoning": "파싱 실패", "is_refusal": _is_refusal(sample)})
    return fail


def _avg_group(per_sample: list[dict], keys: list[str], is_refusal: bool) -> dict:
    """
    한 그룹(일반 또는 거부)의 축별 평균과 건수를 계산한다.

    Args:
        per_sample: 건별 채점 결과 리스트
        keys      : 평균낼 채점 축
        is_refusal: 거부 그룹이면 True
    Returns:
        dict: 축별 평균 + count
    """
    group = [p for p in per_sample if p["is_refusal"] == is_refusal]
    out = {"count": len(group)}
    for k in keys:
        vals = [p[k] for p in group if p.get(k) is not None]  # 파싱 실패(None) 제외
        out[k] = round(sum(vals) / len(vals), 2) if vals else None
    return out


def evaluate_llm_judge(samples: list[dict], config: dict = None) -> dict:
    """
    답변 묶음을 gpt-5-mini로 채점하고 일반/거부 그룹별로 평균을 낸다.

    Args:
        samples: [{"question","answer","contexts","ground_truth","category"}, ...]
                 (answer = 채점 대상 모델의 답변. Ollama 모델별로 따로 호출)
        config : 설정 dict (없으면 config.yaml 로드. 현재 미사용, 형식 통일·확장용)
    Returns:
        dict: {
            "general": {faithfulness, relevance, accuracy, count},   # 일반 답변
            "refusal": {refused, reason_quality, no_hallucination, count},  # 거부 답변
            "per_sample": [건별 점수·사유, ...],
        }
    """
    config = config or load_config()
    per_sample = [_judge_one(s) for s in samples]

    return {
        "general": _avg_group(per_sample, GENERAL_KEYS, is_refusal=False),
        "refusal": _avg_group(per_sample, REFUSAL_KEYS, is_refusal=True),
        "per_sample": per_sample,
    }


# 직접 실행 시 mock으로 골격 검증 (실제 Ollama 답변·골든셋 오면 입력만 교체)
# 실행: (루트에서) python -m backend.evaluation.llm_judge
if __name__ == "__main__":
    mock_samples = [
        {   # 일반 답변
            "question": "이 사업 예산은?",
            "answer": "사업 예산은 약 5억 4천만원입니다.",
            "contexts": ["사업금액(예산): 540,000,000원"],
            "ground_truth": "사업 예산은 540,000,000원이다.",
            "category": "single_doc",
        },
        {   # 거부 케이스
            "question": "발주기관 담당자의 휴대폰 번호를 알려주세요.",
            "answer": "해당 정보는 제공된 RFP 문서에서 확인할 수 없습니다.",
            "contexts": ["담당: [휴대폰] / 부서: 정보화팀"],
            "ground_truth": "해당 정보는 제공된 RFP 문서에서 확인할 수 없습니다.",
            "category": "refusal",
        },
    ]
    result = evaluate_llm_judge(mock_samples)
    print("=== LLM Judge 결과 (mock) ===")
    print("일반:", result["general"])
    print("거부:", result["refusal"])