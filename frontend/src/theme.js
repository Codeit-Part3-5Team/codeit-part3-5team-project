// 디자인 값(색·폰트·여백)을 한 곳에 모은 테마
// 색/폰트를 바꾸려면 여기 값만 수정하면 화면 전체에 반영됨
// (화면 디자인을 바꿀 때 이 파일의 값만 갈아끼우면 됨)
export const theme = {
  // 색상
  colors: {
    bg: "#f0f2f5",          // 전체 배경
    sidebar: "#0d1117",     // 사이드바 배경
    sidebarItem: "#1a1f2e", // 사이드바 내부 버튼/카드
    border: "#333",         // 사이드바 테두리
    primary: "#4CAF50",     // 포인트 색 (봇 강조·출처·화살표)
    botAvatar: "#e8f5e9",   // 봇 아바타 배경
    userAvatar: "#ff8a65",  // 유저 아바타 배경
    surface: "#fff",        // 말풍선·헤더·입력창 배경
    surfaceAlt: "#f6f8fa",  // 출처 박스 배경
    inputBg: "#f9f9f9",     // 입력창 안쪽 배경
    textMain: "#333",       // 본문 텍스트
    textSub: "#aaa",        // 보조 텍스트(시간·라벨)
    textSource: "#555",     // 출처 텍스트
    borderLight: "#e0e0e0", // 옅은 구분선
  },
  // 폰트
  font: {
    family: "sans-serif",
  },
};