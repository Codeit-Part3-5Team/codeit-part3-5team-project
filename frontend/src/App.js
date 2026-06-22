import { useState } from "react";
import { useChat } from "./useChat";
import { theme } from "./theme";

function App() {
  // 채팅 기능은 useChat 훅에서 가져옴 (상태·API호출·localStorage 등)
  // 화면을 바꿔도 이 훅에서 받아쓰기만 하면 기능이 그대로 연결됨
  const {
    messages,
    input, setInput,
    loading,
    maxHistory, setMaxHistory,
    openSources, setOpenSources,
    sendMessage,
    startNewChat,
    handleKeyDown,
  } = useChat();

  // 사이드바 열림/닫힘은 순수 화면 상태라 App에 둠
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // 색·폰트는 theme에서 가져옴 (바꾸려면 theme.js만 수정)
  const c = theme.colors;

  return (
    <div style={{ display: "flex", height: "100vh", background: c.bg, fontFamily: theme.font.family }}>

      {/* 사이드바 */}
      {sidebarOpen && (
        <div style={{ width: "260px", background: c.sidebar, display: "flex", flexDirection: "column", padding: "16px", gap: "8px" }}>

          {/* 사이드바 토글 버튼 */}
          <button
            onClick={() => setSidebarOpen(false)}
            style={{ alignSelf: "flex-end", background: "transparent", border: `1px solid ${c.border}`, color: "#fff", borderRadius: "6px", padding: "6px 10px", cursor: "pointer", marginBottom: "8px" }}
          >
            ☰
          </button>

          {/* New Chat 버튼 */}
          <button
            onClick={startNewChat}
            style={{ background: c.sidebarItem, border: `1px solid ${c.border}`, color: "#fff", borderRadius: "8px", padding: "10px 14px", cursor: "pointer", textAlign: "left" }}
          >
            + New Chat
          </button>

          <div style={{ marginTop: "8px" }}>
            <div style={{ color: c.textSub, fontSize: "12px", marginBottom: "4px" }}>대화 이력 제한</div>
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
          <div style={{ marginTop: "auto", background: c.sidebarItem, border: `1px solid ${c.border}`, borderRadius: "8px", padding: "10px 14px", display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: "#555", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "14px" }}>
              👤
            </div>
            <div>
              <div style={{ color: c.textSub, fontSize: "12px" }}>Welcome back,</div>
              <div style={{ color: "#fff", fontSize: "14px", fontWeight: "bold" }}>User</div>
            </div>
          </div>
        </div>
      )}

      {/* 메인 채팅 영역 */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>

        {/* 상단 헤더 */}
        <div style={{ display: "flex", alignItems: "center", padding: "12px 20px", background: c.surface, borderBottom: `1px solid ${c.borderLight}` }}>
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
                <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: c.botAvatar, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px" }}>
                  🤖
                </div>
              )}

              {/* 메시지 말풍선 */}
              <div
                style={{
                  maxWidth: "60%",
                  padding: "12px 16px",
                  borderRadius: msg.role === "user" ? "16px 16px 4px 16px" : "16px 16px 16px 4px",
                  background: c.surface,
                  border: msg.role === "user" ? `1px solid #ddd` : "none",
                  borderLeft: msg.role === "bot" ? `3px solid ${c.primary}` : "none",
                  fontSize: "14px",
                  lineHeight: "1.5",
                  color: c.textMain,
                  boxShadow: "0 1px 3px rgba(0,0,0,0.08)",
                }}
              >
                {msg.content}

                {/* 응답 시간 표시 (elapsed_sec). 0초도 표시되도록 != null 사용 */}
                {msg.elapsed != null && (
                  <div style={{ fontSize: "11px", color: c.textSub, marginTop: "6px" }}>
                    🕒 {msg.elapsed}초
                  </div>
                )}

                {/* 출처 표시 (봇 메시지 + sources 있을 때만) — 시간 블록과 형제 */}
                {msg.role === "bot" && msg.sources && msg.sources.length > 0 && (
                  <div style={{ marginTop: "8px" }}>
                    {/* 클릭하면 펼침/접힘 토글 (이미 열린 거 누르면 닫힘) */}
                    <div
                      onClick={() => setOpenSources(openSources === idx ? null : idx)}
                      style={{ fontSize: "12px", color: c.primary, cursor: "pointer", userSelect: "none" }}
                    >
                      📎 출처 {msg.sources.length}개 {openSources === idx ? "▲" : "▼"}
                    </div>

                    {/* 펼쳐졌을 때만 출처 목록 + 청크 표시 */}
                    {openSources === idx && (
                      <div style={{ marginTop: "6px", padding: "8px", background: c.surfaceAlt, borderRadius: "8px", fontSize: "12px", color: c.textSource }}>
                        {/* 출처 목록: 문서ID / 페이지 / 점수 */}
                        {msg.sources.map((src, i) => (
                          <div key={i} style={{ marginBottom: "4px" }}>
                            • {src.doc_id} (p.{src.page}, score {src.score})
                          </div>
                        ))}

                        {/* 검색된 청크 본문 */}
                        {msg.chunks && msg.chunks.length > 0 && (
                          <div style={{ marginTop: "8px", borderTop: `1px solid ${c.borderLight}`, paddingTop: "8px" }}>
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
                <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: c.userAvatar, display: "flex", alignItems: "center", justifyContent: "center", fontSize: "16px" }}>
                  👤
                </div>
              )}
            </div>
          ))}

          {/* 응답 대기 중 표시 */}
          {loading && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{ width: "32px", height: "32px", borderRadius: "50%", background: c.botAvatar, display: "flex", alignItems: "center", justifyContent: "center" }}>🤖</div>
              <div style={{ padding: "12px 16px", background: c.surface, borderRadius: "16px", fontSize: "14px", color: c.textSub }}>입력 중...</div>
            </div>
          )}
        </div>

        {/* 입력창 영역 */}
        <div style={{ padding: "16px 40px", background: c.surface, borderTop: `1px solid ${c.borderLight}` }}>
          <div style={{ display: "flex", alignItems: "center", gap: "12px", background: c.inputBg, border: `1px solid ${c.borderLight}`, borderRadius: "12px", padding: "8px 16px" }}>
            <input
              style={{ flex: 1, border: "none", background: "transparent", outline: "none", fontSize: "14px", color: c.textMain }}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a new message here"
            />
            {/* 전송 버튼 */}
            <button
              onClick={sendMessage}
              disabled={loading}
              style={{ background: "transparent", border: "none", cursor: "pointer", fontSize: "18px", color: loading ? "#ccc" : c.textMain }}
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