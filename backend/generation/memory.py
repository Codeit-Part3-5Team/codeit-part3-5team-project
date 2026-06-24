# 대화 이력 관리 (v1: 단순 잘라내기)
# 이력이 길어지면 토큰이 계속 늘어나므로, 최근 N턴만 남기고 오래된 건 버린다.
# max_history는 프론트 슬라이더에서 넘어온 값을 그대로 사용 (사용자가 조절 가능).


# 최근 N턴만 남기고 잘라내기
# history: [{"role": "user"/"assistant", "content": "..."}, ...] (오래된 순)
# max_history: 남길 "턴" 수 (1턴 = user+assistant 한 쌍)
# 반환: 잘린 history 리스트
def trim_history(history: list[dict], max_history: int = 10) -> list[dict]:
    if not history:
        return []

    # 1턴 = user/assistant 2개 메시지 → 최근 max_history턴 = 뒤에서 max_history*2개
    max_messages = max_history * 2

    # 최근 것만 남김 (리스트 뒤쪽이 최신)
    return history[-max_messages:]


# 직접 실행 시 테스트
# 실행: (backend 폴더에서) python -m generation.memory
if __name__ == "__main__":
    # 5턴(10개 메시지)짜리 가짜 이력 생성
    mock_history = []
    for i in range(1, 6):
        mock_history.append({"role": "user", "content": f"질문{i}"})
        mock_history.append({"role": "assistant", "content": f"답변{i}"})

    print(f"원본: {len(mock_history)}개 메시지 (5턴)")

    # 최근 2턴만 남기기
    trimmed = trim_history(mock_history, max_history=2)
    print(f"max_history=2 → {len(trimmed)}개 메시지:")
    for msg in trimmed:
        print(f"  {msg['role']}: {msg['content']}")