import { useState, useRef, useEffect } from "react";

// .env의 REACT_APP_API_URL을 읽음. 없으면 로컬 백엔드로 fallback (하드코딩 방지)
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function App() {
  // messages 초기값: localStorage에 저장된 대화가 있으면 복원, 없으면 빈 배열
  // (새로고침해도 대화가 유지되도록 함)
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem("chat_messages");
    return saved ? JSON.parse(saved) : [];
  });

  // messages가 바뀔 때마다 localStorage에 저장 (새로고침 대비)
  useEffect(() => {
    localStorage.setItem("chat_messages", JSON.stringify(messages));
  }, [messages]);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [maxHistory, setMaxHistory] = useState(10);
  // 사이드바 열림/닫힘 상태
  const [sidebarOpen, setSidebarOpen] = useState(true);
  // 어느 메시지의 출처가 펼쳐져 있는지 (메시지 index를 담음)
  const [openSources, setOpenSources] = useState(null);

  // 세션 ID - 탭이 열려있는 동안 유지, New Chat 시 새로 발급
  const sessionId = useRef(`session_${Date.now()}`);
  

  // AI 응답 생성 함수 - 백엔드 /chat 호출 (dict 한방 응답 받기)
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      // 이전 대화이력을 백엔드 형식 [{role, content}]으로 변환
      // 화면 messages에서 role/content만 추출 (bot → assistant로 변환)
      // 방금 추가한 이번 질문(userMessage)은 history에서 제외 (질문은 query로 따로 감)
      const history = messages.map((m) => ({
        role: m.role === "bot" ? "assistant" : "user",
        content: m.content,
      }));

      const res = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // query(이번 질문) + history(이전 대화) + max_history(슬라이더 값) 전송
        body: JSON.stringify({ query: input, history: history, max_history: maxHistory }),
      });

      // 백엔드가 dict(JSON)를 한 번에 반환 → json()으로 파싱
      const data = await res.json();

      // 봇 메시지로 추가 (answer 본문 + sources/chunks는 출처 표시용으로 함께 저장)
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: data.answer,
          elapsed: data.elapsed_sec,
          sources: data.sources,
          chunks: data.retrieved_chunks,
        },
      ]);
    } catch (e) {
      // 에러 내용 콘솔에 출력 (디버깅용)
      console.error("에러 내용:", e);
      setMessages((prev) => [
        ...prev,
        { role: "bot", content: "서버 연결 실패" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  // New Chat: 화면 초기화 + localStorage 비움 + 새 세션 ID 발급 (이전 대화와 완전히 분리)
  const startNewChat = () => {
    setMessages([]);
    localStorage.removeItem("chat_messages");   // 저장된 대화도 삭제 (새로고침해도 안 되살아나게)
    sessionId.current = `session_${Date.now()}`;
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  return (
    <div style={{ display: "flex", height: "100vh", background: "#f0f2f5", fontFamily: "sans-serif" }}>

      {/* 사이드바 */}
      {sidebarOpen && (
        <div style={{ width: "260px", background: "#0d1117", display: "flex", flexDirection: "column", padding: "16px", gap: "8px" }}>

          {/* 사이드바 토글 버튼 */}
          <button
            onClick={() => setSidebarOpen(false)}
            style={{ alignSelf: "flex-end", background: "transparent", border: "1px solid #333", color: "#fff", borderRadius: "6px", padding: "6px 10px", cursor: "pointer", marginBottom: "8px" }}
          >
            ☰
          </button>

          {/* New Chat 버튼 */}
          <button
            onClick={startNewChat}
            style={{ background: "#1a1f2e", border: "1px solid #333", color: "#fff", borderRadius: "8px", padding: "10px 14px", cursor: "pointer", textAlign: "left" }}
          >
            + New Chat
          </button>

          <div style={{ marginTop: "8px" }}>
            <div style={{ color: "#aaa", fontSize: "12px", marginBottom: "4px" }}>대화 이력 제한</div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <input
                type="range"
                min="1"
                max="20"
                value={maxHistory}
                onChange={(e) => setMaxHistory(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ color: "#fff", fontSize: "13px", minWidth: "30px" }}>{maxHistory}개</span>
            </div>
          </div>

          {/* 하단 사용자 정보 */}
          <div style={{ marginTop: "auto", background: "#1a1f2e", border: "1px solid #333", borderRadius: "8px", padding: "10px 14px", display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "#555", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "14px" }}>
              👤
            </div>
            <div>
              <div style={{ color: "#aaa", fontSize: "12px" }}>Welcome back,</div>
              <div style={{ color: "#fff", fontSize: "14px", fontWeight: "bold" }}>User</div>
            </div>
          </div>
        </div>
      )}

      {/* 메인 채팅 영역 */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>

        {/* 상단 헤더 */}
        <div style={{ display: "flex", alignItems: "center", padding: "12px 20px", background: "#fff", borderBottom: "1px solid #e0e0e0" }}>
          {/* 사이드바 닫혀있을 때 토글 버튼 */}
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              style={{ background: "transparent", border: "none", fontSize: "20px", cursor: "pointer" }}
            >
              ☰
            </button>
          )}
        </div>

        {/* 대화 내역 영역 */}
        <div style={{ flex: 1, overflowY: "auto", padding: "20px 40px", display: "flex", flexDirection: "column", gap: "16px" }}>
          {messages.map((msg, idx) => (
            <div
              key={idx}
              style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start", alignItems: "flex-end", gap: "8px" }}
            >
              {/* 봇 아바타 */}
              {msg.role === "bot" && (
                <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "#e8f5e9", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px" }}>
                  🤖
                </div>
              )}

              {/* 메시지 말풍선 */}
              <div
                style={{
                  maxWidth: "60%",
                  padding: "12px 16px",
                  borderRadius: msg.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                  background: "#fff",
                  border: msg.role === "user" ? "1px solid #ddd" : "none",
                  borderLeft: msg.role === "bot" ? "3px solid #4CAF50" : "none",
                  fontSize: "14px",
                  lineHeight: "1.5",
                  color: "#333",
                  boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
                }}
              >
                {msg.content}

                {/* 응답 시간 표시 (elapsed_sec). 0초도 표시되도록 != null 사용 */}
                {msg.elapsed != null && (
                  <div style={{ fontSize: "11px", color: "#aaa", marginTop: "6px" }}>
                    🕒 {msg.elapsed}초
                  </div>
                )}

                {/* 출처 표시 (봇 메시지 + sources 있을 때만) — 시간 블록과 형제 */}
                {msg.role === "bot" && msg.sources && msg.sources.length > 0 && (
                  <div style={{ marginTop: "8px" }}>
                    {/* 클릭하면 펼침/접힘 토글 (이미 열린 거 누르면 닫힘) */}
                    <div
                      onClick={() => setOpenSources(openSources === idx ? null : idx)}
                      style={{ fontSize: "12px", color: "#4CAF50", cursor: "pointer", userSelect: "none" }}
                    >
                      📎 출처 {msg.sources.length}개 {openSources === idx ? "▲" : "▼"}
                    </div>

                    {/* 펼쳐졌을 때만 출처 목록 + 청크 표시 */}
                    {openSources === idx && (
                      <div style={{ marginTop: "6px", padding: "8px", background: "#f6f8fa", borderRadius: "8px", fontSize: "12px", color: "#555" }}>
                        {/* 출처 목록: 문서ID / 페이지 / 점수 */}
                        {msg.sources.map((src, i) => (
                          <div key={i} style={{ marginBottom: "4px" }}>
                            • {src.doc_id} (p.{src.page}, score {src.score})
                          </div>
                        ))}

                        {/* 검색된 청크 본문 */}
                        {msg.chunks && msg.chunks.length > 0 && (
                          <div style={{ marginTop: "8px", borderTop: "1px solid #e0e0e0", paddingTop: "8px" }}>
                            <div style={{ fontWeight: "bold", marginBottom: "4px" }}>검색된 내용</div>
                            {msg.chunks.map((chunk, i) => (
                              <div key={i} style={{ marginBottom: "4px", lineHeight: "1.4" }}>{chunk}</div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* 유저 아바타 */}
              {msg.role === "user" && (
                <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "#ff8a65", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px" }}>
                  👤
                </div>
              )}
            </div>
          ))}

          {/* 응답 대기 중 표시 */}
          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "#e8f5e9", display: "flex", alignItems: "center", justifyContent: "center" }}>🤖</div>
              <div style={{ padding: "12px 16px", background: "#fff", borderRadius: "16px", fontSize: "14px", color: "#aaa" }}>입력 중...</div>
            </div>
          )}
        </div>

        {/* 입력창 영역 */}
        <div style={{ padding: "16px 40px", background: "#fff", borderTop: "1px solid #e0e0e0" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", background: "#f9f9f9", border: "1px solid #e0e0e0", borderRadius: "12px", padding: "8px 16px" }}>
            <input
              style={{ flex: 1, border: "none", background: "transparent", outline: "none", fontSize: "14px", color: "#333" }}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a new message here"
            />
            {/* 전송 버튼 */}
            <button
              onClick={sendMessage}
              disabled={loading}
              style={{ background: "transparent", border: "none", cursor: "pointer", fontSize: "18px", color: loading ? "#ccc" : "#333" }}
            >
              ▷
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;