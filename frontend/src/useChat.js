import { useState, useRef, useEffect } from "react";

// .env의 REACT_APP_API_URL을 읽음. 없으면 로컬 백엔드로 fallback (하드코딩 방지)
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// 채팅 기능(상태 + 로직)을 모아둔 커스텀 훅
// 화면(App.js)은 이 훅에서 값/함수를 받아다 쓰기만 하면 됨
// → 화면 디자인을 바꿔도 이 훅은 그대로 재사용 가능
export function useChat() {
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

  // 화면(App.js)에서 쓸 값과 함수를 한 묶음으로 반환
  return {
    messages,
    input, setInput,
    loading,
    maxHistory, setMaxHistory,
    openSources, setOpenSources,
    sendMessage,
    startNewChat,
    handleKeyDown,
  };
}