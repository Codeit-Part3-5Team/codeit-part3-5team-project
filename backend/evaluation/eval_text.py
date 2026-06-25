"""
eval_text.py
생성 답변의 텍스트 품질 지표를 계산한다.
정답 답변 텍스트(ground_truth)와 생성 답변(answer)을 직접 대조하는 결정론적 지표로,
LLM 채점 지표(Faithfulness/Answer Relevance)의 교차검증용 보조 지표다.

지표:
    ROUGE-1, ROUGE-L : 정답과의 n-gram / 최장공통부분수열 겹침 (rouge-score)
    BLEU             : 생성문 기준 n-gram 정밀도 (nltk)
    BERTScore-F1     : 임베딩 기반 의미 유사도 (bert-score, 한국어)

핵심 함수:
    evaluate_text(samples) : 샘플 리스트 → 지표별 평균 dict 반환
실행: (루트에서) python -m backend.evaluation.eval_text
"""
from rouge_score import rouge_scorer            # ROUGE-1/L 계산
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction  # BLEU 계산
from bert_score import score as bert_score_fn   # BERTScore 계산

# BERTScore 한국어 평가에 쓸 언어 설정(다국어 모델 자동 선택). 필요 시 config로 분리 가능.
BERTSCORE_LANG = "ko"


def tokenize_ko(text: str) -> list[str]:
    """
    한국어 텍스트를 토큰 리스트로 변환한다.
    기본은 공백(어절) 단위. 형태소 분석기(mecab/konlpy) 도입 시 이 함수만 교체하면 된다.

    Args:
        text: 토큰화할 문자열
    Returns:
        list[str]: 토큰 리스트
    """
    # 공백 기준 분리 (형태소 분석기 도입 시 이 부분 교체예정)
    return text.split()


class _KoTokenizer:
    """
    rouge-score에 주입할 토크나이저.
    rouge-score 기본 토크나이저는 한글을 제거(숫자/영문만 남김)하므로,
    한국어를 보존하기 위해 tokenize_ko로 직접 토큰화한다.
    """
    def tokenize(self, text: str) -> list[str]:
        return tokenize_ko(text)


def calc_rouge(answer: str, ground_truth: str) -> dict:
    """
    ROUGE-1, ROUGE-L F1을 계산한다.

    Args:
        answer: 생성 답변
        ground_truth: 정답 답변 텍스트
    Returns:
        dict: {"rouge1": float, "rougeL": float}
    """
    # rouge-score 기본 토크나이저는 한글을 버리므로 커스텀 토크나이저 주입
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], tokenizer=_KoTokenizer())
    scores = scorer.score(ground_truth, answer)  # (정답, 생성) 순으로 전달
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def calc_bleu(answer: str, ground_truth: str) -> float:
    """
    BLEU(생성문 기준 n-gram 정밀도)를 계산한다.
    짧은 답변에서 점수가 0이 되는 것을 막기 위해 스무딩을 적용한다.

    Args:
        answer: 생성 답변
        ground_truth: 정답 답변 텍스트
    Returns:
        float: BLEU 점수 (0~1)
    """
    cand = tokenize_ko(answer)                   # 생성 답변 토큰
    ref = tokenize_ko(ground_truth)              # 정답 토큰 (참조는 리스트의 리스트로 전달)
    smooth = SmoothingFunction().method1         # 짧은 문장 0점 방지 스무딩
    return sentence_bleu([ref], cand, smoothing_function=smooth)


def calc_bertscore(answer: str, ground_truth: str) -> float:
    """
    BERTScore-F1(임베딩 기반 의미 유사도)을 계산한다. 표현이 달라도 의미가 같으면 높게 나온다.
    단건 디버깅용. 전체 평가는 evaluate_text가 배치로 한 번에 호출한다.

    Args:
        answer: 생성 답변
        ground_truth: 정답 답변 텍스트
    Returns:
        float: BERTScore F1 (0~1)
    """
    _, _, f1 = bert_score_fn([answer], [ground_truth], lang=BERTSCORE_LANG, verbose=False)
    return f1.mean().item()                      # 텐서 → float


def evaluate_text(samples: list[dict]) -> dict:
    """
    샘플 리스트 전체의 텍스트 품질 지표 평균을 계산한다.

    Args:
        samples: [{"answer": str, "ground_truth": str, ...}, ...]
                 (question/contexts 키가 있어도 무시, answer·ground_truth만 사용)
    Returns:
        dict: {"rouge1", "rougeL", "bleu", "bertscore_f1"} 각 평균값
    """
    rouge1_list, rougeL_list, bleu_list = [], [], []

    # ROUGE / BLEU는 건별 계산
    for s in samples:
        r = calc_rouge(s["answer"], s["ground_truth"])
        rouge1_list.append(r["rouge1"])
        rougeL_list.append(r["rougeL"])
        bleu_list.append(calc_bleu(s["answer"], s["ground_truth"]))

    # BERTScore는 모델 로드 비용이 커서 배치로 한 번에 호출 (단건 반복보다 빠름)
    answers = [s["answer"] for s in samples]
    gts = [s["ground_truth"] for s in samples]
    _, _, f1 = bert_score_fn(answers, gts, lang=BERTSCORE_LANG, verbose=False)
    bertscore_mean = f1.mean().item()

    # 평균 산출 (소수 4자리)
    n = len(samples)
    return {
        "rouge1": round(sum(rouge1_list) / n, 4),
        "rougeL": round(sum(rougeL_list) / n, 4),
        "bleu": round(sum(bleu_list) / n, 4),
        "bertscore_f1": round(bertscore_mean, 4),
    }


# 직접 실행 시 mock 샘플로 골격 검증 (골든셋 오면 입력만 실제 데이터로 교체)
# 실행: (루트에서) python -m backend.evaluation.eval_text
if __name__ == "__main__":
    # 실제 골든셋과 같은 형식의 mock 입력 (answer/ground_truth만 사용)
    mock_samples = [
        {
            "question": "이 사업 예산은?",
            "answer": "본 사업의 예산은 5억 4천만원입니다.",
            "ground_truth": "사업 예산은 540,000,000원이다.",
        },
        {
            "question": "사업 기간은?",
            "answer": "계약 체결일로부터 6개월입니다.",
            "ground_truth": "사업 기간은 계약 체결일로부터 6개월로 한다.",
        },
    ]
    result = evaluate_text(mock_samples)
    print("=== 텍스트 품질 지표 (mock) ===")
    for k, v in result.items():
        print(f"{k}: {v}")