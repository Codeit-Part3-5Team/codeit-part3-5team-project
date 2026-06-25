"""
eval_generation.py
생성 답변의 LLM 채점 지표(Faithfulness, Answer Relevance)와 그 종합(RAGAS-Score)을 계산한다.
ragas 라이브러리로 산출하며, 채점자(judge)는 gpt-5-mini다.
정답 텍스트와의 단순 일치가 아니라 '답변 ↔ 근거', '답변 ↔ 질문'의 의미 부합을 본다.

지표:
    Faithfulness     : 답변 주장이 근거(retrieved_contexts)로 지지되는 비율 (환각 측정)
    Answer Relevance : 답변이 질문 의도에 부합하는 정도
    RAGAS-Score      : 위 둘의 평균 (생성 종합)

핵심 함수:
    evaluate_generation(samples, config) : 샘플 리스트 → 지표 평균 + 건별 점수 반환
실행: (루트에서) python -m backend.evaluation.eval_generation
"""
from utils.config import load_config            # config.yaml 읽기
# ragas 0.4.x 기준 import 경로 (버전 다르면 경로 바뀔 수 있음)
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _build_judge(config: dict) -> tuple:
    """
    ragas 채점에 쓸 judge LLM(gpt-5-mini)과 임베딩을 구성한다.

    Args:
        config: 설정 dict (llm_model, judge_temperature 등)
    Returns:
        tuple: (LangchainLLMWrapper, LangchainEmbeddingsWrapper)
    """
    # gpt-5-mini는 극소 temperature(예: 1e-8)에서 ragas가 nan을 반환하므로 지원 범위(기본 1.0) 사용
    judge_temp = config.get("judge_temperature", 1.0)
    llm_model = config.get("llm_model", "gpt-5-mini")
    # 임베딩은 검색과 동일 모델 사용 (Answer Relevance의 질문-역질문 유사도 계산용)
    embed_model = config.get("embedding_model", "text-embedding-3-small")

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model, temperature=judge_temp))
    judge_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embed_model))
    return judge_llm, judge_emb


def _to_dataset(samples: list[dict]) -> EvaluationDataset:
    """
    내부 샘플 형식을 ragas 입력 스키마(SingleTurnSample)로 변환한다.

    Args:
        samples: [{"question","answer","contexts","ground_truth"}, ...]
    Returns:
        EvaluationDataset: ragas 평가용 데이터셋
    """
    rows = []
    for s in samples:
        # 내부 키 → ragas 필드 매핑
        rows.append(SingleTurnSample(
            user_input=s["question"],            # 질문
            response=s["answer"],                # 생성 답변
            retrieved_contexts=s["contexts"],    # 검색된 청크 리스트(list[str])
            reference=s.get("ground_truth", ""), # 정답 텍스트(faithfulness/AR엔 미사용, 형식 통일용)
        ))
    return EvaluationDataset(samples=rows)


def evaluate_generation(samples: list[dict], config: dict = None) -> dict:
    """
    Faithfulness, Answer Relevance, RAGAS-Score를 계산한다.

    Args:
        samples: [{"question","answer","contexts","ground_truth"}, ...]
        config : 설정 dict (없으면 config.yaml 로드)
    Returns:
        dict: {
            "faithfulness": 평균값,
            "answer_relevancy": 평균값,
            "ragas_score": 위 둘의 평균,
            "per_sample": [건별 점수 dict, ...],   # 실패 케이스 분석용
        }
    """
    config = config or load_config()
    judge_llm, judge_emb = _build_judge(config)
    dataset = _to_dataset(samples)

    # ragas 평가 실행 (채점자 = gpt-5-mini)
    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    # 건별 결과를 DataFrame으로 받아 평균 산출 (실패 케이스 분석 위해 건별도 보존)
    df = result.to_pandas()
    faith = round(df["faithfulness"].mean(), 4)
    ar = round(df["answer_relevancy"].mean(), 4)
    ragas_score = round((faith + ar) / 2, 4)     # A안: 생성 2개 평균

    return {
        "faithfulness": faith,
        "answer_relevancy": ar,
        "ragas_score": ragas_score,
        "per_sample": df.to_dict(orient="records"),
    }


# 직접 실행 시 mock 샘플로 골격 검증 (골든셋 오면 입력만 실제 데이터로 교체)
# 실행: (루트에서) python -m backend.evaluation.eval_generation
if __name__ == "__main__":
    # 실제 골든셋과 같은 형식의 mock 입력
    mock_samples = [
        {
            "question": "이 사업 예산은?",
            "answer": "본 사업의 예산은 540,000,000원입니다.",
            "contexts": ["사업명: 국민연금공단 이러닝시스템 고도화\n사업금액(예산): 540,000,000원"],
            "ground_truth": "사업 예산은 540,000,000원이다.",
        },
        {
            "question": "사업 기간은?",
            "answer": "계약 체결일로부터 6개월입니다.",
            "contexts": ["사업 수행 기간은 계약 체결일로부터 6개월로 한다."],
            "ground_truth": "사업 기간은 계약 체결일로부터 6개월이다.",
        },
    ]
    result = evaluate_generation(mock_samples)
    print("=== 생성 평가 지표 (mock) ===")
    print("Faithfulness    :", result["faithfulness"])
    print("Answer Relevance:", result["answer_relevancy"])
    print("RAGAS-Score     :", result["ragas_score"])