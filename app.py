import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
import requests
import streamlit as st
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# --- [기본 환경 설정] ---
st.set_page_config(page_title="드론 빌더스 허브", page_icon="🛸", layout="wide")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

if not OPENAI_API_KEY:
    st.error("⚠️ .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0, max_retries=2)

# --- [사이드바: 요구 추력 계산기 & Open-Meteo 기상 관제] ---
with st.sidebar:
    st.title("🛸 드론 비행 & 설계 포털")

    # 1. 요구 추력 계산기
    st.subheader("📐 요구 추력 계산기")
    frame_type = st.selectbox("모터 개수 (프레임 타입)", ["4개 (쿼드콥터)", "6개 (헥사콥터)", "8개 (옥토콥터)"])
    motor_count = 4 if "4개" in frame_type else (6 if "6개" in frame_type else 8)

    payload_w = st.number_input("임무장비(페이로드) 무게 (kg)", min_value=0.1, max_value=50.0, value=2.0, step=0.5)
    frame_w = st.number_input("기체 무게 (배터리·프레임 포함, kg)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
    twr_target = st.slider("목표 추력비 (TWR, 권장 2.0:1)", min_value=1.5, max_value=3.0, value=2.0, step=0.1)

    total_weight = frame_w + payload_w
    req_total_thrust = total_weight * twr_target
    motor_max_thrust = req_total_thrust / motor_count
    hover_thrust = total_weight / motor_count

    st.markdown("#### 📊 계산 결과")
    c1, c2 = st.columns(2)
    c1.metric("총 이륙중량(AUW)", f"{total_weight:.1f} kg")
    c2.metric("요구 총 추력", f"{req_total_thrust:.1f} kg")
    c3, c4 = st.columns(2)
    c3.metric("모터당 최대 추력", f"{motor_max_thrust:.2f} kg", f"↑ {int(motor_max_thrust*1000)} g")
    c4.metric("호버링 추력(50%)", f"{hover_thrust:.2f} kg", f"↑ {int(hover_thrust*1000)} g")

    calc_query = f"모터 {motor_count}개 멀티콥터이고 기체 {frame_w}kg, 페이로드 {payload_w}kg(총중량 {total_weight:.1f}kg)이야. 모터당 최대추력 {motor_max_thrust:.2f}kg이 필요한데 적합한 파워콤보(모터, ESC, 프롭, 배터리) 추천과 스펙을 분석해줘"
    if st.button("💬 이 조건으로 AI 챗봇에 즉시 추천 요청"):
        st.session_state["preset_query"] = calc_query
        st.rerun()

    st.markdown("---")

    # 2. 계통 필터
    st.subheader("📁 매뉴얼 계통 필터")
    manual_cats = [
        "전체", "01_비행제어기_FC", "02_모터_구동계", "03_변속기_ESC", "04_프로펠러_공력",
        "05_전원_배터리_PDB", "06_센서_항법장치", "07_조종_데이터링크", "08_영상_FPV_짐벌",
        "09_프레임_페이로드", "10_안전_비행규정", "11_온보드컴퓨터_임베디드",
        "12_미들웨어_통신_ROS", "13_컴퓨터비전_AI_YOLO", "14_SLAM_경로계획_내비"
    ]
    selected_cat = st.selectbox("계통 선택", manual_cats)

# --- [고속 RAG & Tavily 웹 검색 함수] ---
@st.cache_resource(show_spinner=False)
def load_drone_vector_store():
    """디스크의 faiss_index를 0.05초 만에 즉시 로드"""
    p = Path("faiss_index")
    if not p.exists():
        p = Path.cwd().parent / "faiss_index"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    if p.exists() and (p / "index.faiss").exists():
        try:
            return FAISS.load_local(str(p), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            pass
    return None

def retrieve_manual_docs(query: str, cat_filter: str = "전체", k: int = 4):
    """목차 필터 및 벡터 검색 결합 (초고속 발췌)"""
    docs = []
    # 1. 목차(manual_catalog.json) 우선 매칭
    cat_p = Path("manual_catalog.json")
    if not cat_p.exists():
        cat_p = Path.cwd().parent / "manual_catalog.json"

    if cat_p.exists():
        try:
            catalog = json.loads(cat_p.read_text(encoding="utf-8"))
            q_low = query.lower()
            for item in catalog:
                if any(t in q_low for t in item.get("topics", [])) or item["category"].lower() in q_low:
                    f_path = Path("manuals") / item["relative_path"]
                    if not f_path.exists():
                        f_path = Path.cwd().parent / "manuals" / item["relative_path"]
                    if f_path.exists() and f_path.suffix.lower() == ".md":
                        docs.append({
                            "category": item["category"],
                            "source_file": item["filename"],
                            "path": str(f_path),
                            "content": f_path.read_text(encoding="utf-8")[:250],
                            "page": 1
                        })
                        if len(docs) >= 2:
                            break
        except Exception:
            pass

    # 2. FAISS 보완 검색
    if len(docs) < k:
        db = load_drone_vector_store()
        if db:
            try:
                extra = db.similarity_search(query, k=k)
                for d in extra:
                    c_name = d.metadata.get("category", "규격서")
                    if cat_filter != "전체" and c_name != cat_filter:
                        continue
                    docs.append({
                        "category": c_name,
                        "source_file": d.metadata.get("source_file", "manual.pdf"),
                        "path": d.metadata.get("file_path", ""),
                        "content": d.page_content[:250],
                        "page": d.metadata.get("page", 1)
                    })
            except Exception:
                pass
    return docs[:k]

def search_tavily(query: str, max_results: int = 3):
    """Tavily 1.5초 고속 검색"""
    if not TAVILY_API_KEY:
        return []
    clean_q = query[:60].strip()
    try:
        res = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={"api_key": TAVILY_API_KEY, "query": clean_q, "search_depth": "basic", "max_results": max_results},
            timeout=5.0
        )
        if res.status_code == 200:
            return [{"title": r.get("title", ""), "snippet": r.get("content", "")[:250], "link": r.get("url", "")} for r in res.json().get("results", [])]
    except Exception:
        pass
    return []

def route_intent(query: str):
    """질문 의도 분류 (compare, web, manual)"""
    q = query.lower()
    if any(k in q for k in ["비교", "차이", "vs", "추천", "어떤게", "골라"]):
        return "compare"
    if any(k in q for k in ["최신", "시세", "가격", "출시", "이슈", "github", "법령", "과태료"]):
        return "web"
    return "manual"

# --- [메인 화면 UI] ---
st.title("🛸 드론 빌더스 허브 (Drone Builder's Hub)")
st.caption("자율비행 드론 HW/SW 14개 계통 매뉴얼 RAG & Tavily AI 비행 안전 관제 포털")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# 이전 대화 기록 렌더링
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("files"):
            with st.expander(f"📄 참조 매뉴얼 ({len(msg['files'])}건)", expanded=False):
                for f in msg["files"]:
                    p = Path(f["path"]) if isinstance(f, dict) else Path(f)
                    name = f.get("name", p.name) if isinstance(f, dict) else p.name
                    cat = p.parent.name if p.parent.name != "manuals" else "규격서"
                    col_t, col_b = st.columns([4, 1])
                    col_t.markdown(f"• **[{cat}]** `{name}`")
                    if p.exists():
                        col_b.download_button("⬇️ 다운로드", data=p.read_bytes(), file_name=name, key=f"h_{msg.get('time', 0)}_{name}")

# 사용자 입력 처리
preset_q = st.session_state.pop("preset_query", None)
user_prompt = st.chat_input("하드웨어 배선, ROS 2 통신, YOLO 또는 비행 규정을 질문하세요...") or preset_q

if user_prompt:
    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        route = route_intent(user_prompt)

        # 1. 경로 배지 표출
        if route == "compare":
            st.info("⚖️ **검색 경로: 매뉴얼 vs 웹 비교 대조 분석**")
        elif route == "web":
            st.info("🌐 **검색 경로: Tavily AI 실시간 기술 검색**")
        else:
            st.info("📘 **검색 경로: 매뉴얼 RAG (14개 계통 기술 문서)**")

        # 2. 자료 수집
        with st.spinner("자료를 분석하고 답변을 작성 중입니다..."):
            manual_docs = retrieve_manual_docs(user_prompt, selected_cat, k=4)
            web_docs = search_tavily(user_prompt, max_results=3) if route in ["web", "compare"] else []

            # 3. 답변 생성
            if route == "compare":
                m_ctx = "\n".join([f"[{d['category']}/{d['source_file']}] {d['content']}" for d in manual_docs]) or "매뉴얼 없음"
                w_ctx = "\n".join([f"[{w['title']}] {w['snippet']}" for w in web_docs]) or "웹 검색 없음"

                fast_prompt = f"""질문: {user_prompt}
[매뉴얼 자료]
{m_ctx}
[웹 최신 자료]
{w_ctx}

위 두 자료를 종합하여 아래 3개 구분자(===)를 엄격히 지켜 한 번에 답변을 작성하세요:
===ANALYSIS===
(매뉴얼 규격과 웹 정보를 종합 대조한 핵심 엔지니어링 권고사항)
===MANUAL===
(매뉴얼에 기반한 관련 추력, 전압, 규격 핵심 수치)
===WEB===
(웹 검색에 기반한 추천 시판 부품 및 동향)"""

                res = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": fast_prompt}],
                    temperature=0.2,
                    max_tokens=850,
                    timeout=35.0
                )
                raw = res.choices[0].message.content
                p_analysis = raw.split("===ANALYSIS===")[-1].split("===MANUAL===")[0].strip() if "===ANALYSIS===" in raw else raw
                p_manual = raw.split("===MANUAL===")[-1].split("===WEB===")[0].strip() if "===MANUAL===" in raw else "매뉴얼 분석 완료"
                p_web = raw.split("===WEB===")[-1].strip() if "===WEB===" in raw else "웹 정보 분석 완료"

                tab1, tab2, tab3 = st.tabs(["⚖️ 종합 비교 대조 & 분석", "📘 매뉴얼 규격 내용", "🌐 웹 최신 정보 및 출처"])
                with tab1: st.markdown(p_analysis)
                with tab2: st.markdown(p_manual)
                with tab3:
                    st.markdown(p_web)
                    if web_docs:
                        st.markdown("##### 🔗 웹 출처")
                        for w in web_docs: st.markdown(f"- [{w['title']}]({w['link']})")

                final_answer_text = p_analysis

            else:
                # 단일 모드 (manual / web)
                if route == "web":
                    ctx = "\n".join([f"[{w['title']}] {w['snippet']}" for w in web_docs]) or "웹 자료 없음"
                    prompt = f"질문: {user_prompt}\n\n웹 검색자료:\n{ctx}\n\n위 자료를 바탕으로 엔지니어링 답변을 작성하세요."
                else:
                    ctx = "\n".join([f"[{d['category']}/{d['source_file']}] {d['content']}" for d in manual_docs]) or "매뉴얼 자료 없음"
                    prompt = f"질문: {user_prompt}\n\n매뉴얼 규격자료:\n{ctx}\n\n위 매뉴얼 자료에 근거하여 파라미터와 규격을 명확히 답변하세요."

                res = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=800,
                    timeout=30.0
                )
                final_answer_text = res.choices[0].message.content
                st.markdown(final_answer_text)

        # 4. 세로형 매뉴얼 다운로드 UI
        if manual_docs:
            st.markdown("##### 📥 인용된 원본 매뉴얼 다운로드")
            with st.expander(f"📄 매뉴얼 목록 ({len(manual_docs)}건)", expanded=True):
                for idx, doc in enumerate(manual_docs):
                    fpath = Path(doc["path"])
                    fname = doc["source_file"]
                    col_t, col_b = st.columns([4, 1])
                    col_t.markdown(f"• **[{doc['category']}]** `{fname}` (p.{doc['page']})")
                    if fpath.exists():
                        col_b.download_button("⬇️ 다운로드", data=fpath.read_bytes(), file_name=fname, key=f"dl_live_{idx}_{fname}")

        # 세션 기록 저장
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": final_answer_text,
            "files": manual_docs
        })
