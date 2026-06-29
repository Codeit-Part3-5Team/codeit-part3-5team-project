"""
state.py
LangGraph 오케스트레이션의 공유 상태(GraphState) 정의.

LangGraph는 노드 간에 단일 state dict를 흘려보낸다. 각 노드는 state를 받아
자기가 채울 필드만 dict로 반환하고, LangGraph가 그걸 기존 state에 병합한다.
total=False: 모든 필드가 처음부터 채워져 있지 않아도 됨(노드가 점진적으로 채움).
"""
from typing import TypedDict, Literal, Optional
from langchain_core.documents import Document  # 검색 결과 표준 객체(생성·검색 공통)


class GraphState(TypedDict, total=False):
    # --- 입력 (그래프 진입 시 주입) ---
    question: str                       # 원본 사용자 질문
    history: list[dict]                 # 대화 이력(원본 형식 그대로 보관 — 결정1: 변환은 노드 안에서만)
    selected_doc_id: Optional[str]      # 라우트 C용: UI에서 고른 문서 ID
    config: dict                        # 설정값(top_k, prompt_version 등)
    use_ollama: bool                    # 시나리오 A(Ollama)/B(gpt) 선택

    # --- question_analysis_node 산출 ---
    rewritten_question: str             # 지시어 해소된 재구성 질문(followup 대응)

    # --- routing_node 산출 ---
    route: Literal["route_a", "route_c"]  # 분기 결정(실제 분기는 조건부 엣지가 수행)

    # --- route_a / route_c 산출 ---
    docs: list[Document]                # 라우트 A: 검색된 Document 통째로 보관(결정2)
    checklist_items: list[dict]         # 라우트 C: 추출된 체크리스트 항목
    route_status: str                   # 라우트 실행 상태(ok / doc_not_found / failed 등)

    # --- answer_generation_node 산출 ---
    answer: str                         # 생성된 최종 답변
    tokens_used: int                    # 생성 토큰 수(최종 반환 dict에 사용)

    # --- self_check_node 산출 ---
    check_passed: bool                  # 검증 통과 여부
    check_flags: list[str]              # 검증에서 걸린 항목(source_missing 등)