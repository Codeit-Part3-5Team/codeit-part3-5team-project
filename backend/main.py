# FastAPI: 웹 서버 프레임워크
from fastapi import FastAPI
# CORS: 프론트엔드에서 백엔드 호출 허용 설정
from fastapi.middleware.cors import CORSMiddleware
# BaseModel: 요청 데이터 형식 정의할 때 사용
from pydantic import BaseModel

import time

# pipeline의 진짜 get_ai_response를 가져옴 (mock 대체)
from backend.pipeline import get_ai_response

# FastAPI 앱 인스턴스 생성 (서버의 핵심 객체)
app = FastAPI()

# CORS 미들웨어 설정
# 프론트엔드와 백엔드가 다른 포트를 사용하기 때문에 브라우저가 기본적으로 요청을 차단함
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 모든 출처 허용 (개발용. 실서비스엔 특정 도메인만 허용해야 함)
    allow_methods=["*"],
    allow_headers=["*"],
)


# 채팅 요청 데이터 형식 정의
# query: 사용자 질문 / history: 이전 대화 이력 [{role, content}, ...]
class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []
    max_history: int = 10   # 프론트 슬라이더 값 (없으면 기본 10)

# /chat 엔드포인트 (POST 방식)
# 프론트에서 query/history/max_history를 받아 get_ai_response 호출 후 dict를 JSON으로 응답
@app.post("/chat")
def chat(req: ChatRequest):
    result = get_ai_response(query=req.query, history=req.history, max_history=req.max_history)
    return result


# / 엔드포인트 (GET 방식)
# 서버 정상 동작 확인용 헬스체크
@app.get("/")
def root():
    return {"status": "ok"}

# 이 파일을 직접 실행하면 uvicorn으로 서버를 띄움
# 실행: (루트에서) python -m backend.main
if __name__ == "__main__":
    import uvicorn
    # main.py 안의 app 객체를 8000 포트로 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)