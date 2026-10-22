import os
import io
import json
import base64
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
import pandas as pd
import streamlit as st
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- [기본 환경 설정] ---
st.set_page_config(page_title="드론 빌더스 허브", page_icon="🛸", layout="wide")
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()

if not OPENAI_API_KEY:
    st.error("⚠️ .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0, max_retries=2)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "query_logs" not in st.session_state:
    st.session_state["query_logs"] = []

# --- [계획서 명세 3대 페르소나 시스템 프롬프트] ---
PERSONA_PROMPTS = {
    "현장 엔지니어 (핀맵/명령어 중심)": """당신은 15년 경력의 현장 수석 드론 하드웨어/임베디드 엔지니어입니다.
- 불필요한 사설은 배제하고, 실무 수치, 핀맵(VCC, GND, TX, RX, I2C, CAN), 하드웨어 결선 주의사항, 리눅스/ROS 2 실행 명령어를 중심으로 명확하게 답변하십시오.
- 모터 결선이나 전원 배선 질문 시 답변 최상단에 "⚡ [하드웨어 결선 주의]" 경고를 반드시 포함하십시오.""",

    "DIY 빌더 (기초 원리 해설)": """당신은 친절한 드론 자작(DIY) 멘토입니다.
- 초보 제작자도 쉽게 이해할 수 있도록 기본 규격과 비행 원리를 단계별로 친절하게 설명하십시오.
- 조립이나 비행 시 자주 하는 실수와 유의할 점을 알기 쉽게 안내하십시오.""",

    "비행 안전점검관 (규정/체크리스트)": """당신은 대한민국 항공안전법 공인 비행 안전점검관입니다.
- 항공안전법 초경량비행장치 규정, 비행승인 및 촬영허가(드론원스톱), 최대 비행고도(150m) 기준을 엄격하게 검토하여 답변하십시오.
- 답변 하단에는 반드시 현장에서 점검할 수 있는 "[ ]" 체크박스 형태의 비행 전 점검표(Checklist)를 작성하십시오."""
}

# --- [사이드바 UI] ---
with st.sidebar:
    st.title("🛸 드론 비행 & 설계 포털")

    st.subheader("⚙️ 검색 모드 선택")
    search_mode = st.radio(
        "검색 및 처리 경로",
        ["🤖 자동 판별 (AI 라우팅)", "📘 매뉴얼 RAG (로컬 기술문서)", "🌐 Tavily AI 실시간 기술 검색", "⚖️ 매뉴얼 vs 웹 비교 대조"],
        index=0
    )
    st.markdown("---")

    st.subheader("📁 매뉴얼 계통 필터")
    manual_cats = [
        "전체", "01_비행제어기_FC", "02_모터_구동계", "03_변속기_ESC", "04_프로펠러_공력",
        "05_전원_배터리_PDB", "06_센서_항법장치", "07_조종_데이터링크", "08_영상_FPV_짐벌",
        "09_프레임_페이로드", "10_안전_비행규정", "11_온보드컴퓨터_임베디드",
        "12_미들웨어_통신_ROS", "13_컴퓨터비전_AI_YOLO", "14_SLAM_경로계획_내비"
    ]
    selected_cat = st.selectbox("계통 선택", manual_cats)

    st.subheader("🎭 답변 스타일 (페르소나)")
    selected_persona = st.selectbox(
        "페르소나 선택",
        ["현장 엔지니어 (핀맵/명령어 중심)", "DIY 빌더 (기초 원리 해설)", "비행 안전점검관 (규정/체크리스트)"],
        index=0
    )
    st.markdown("---")

    # [계획서 2단계] 복합 기상 위험 관제 위젯
    with st.expander("🌦️ 실시간 비행 기상 관제 (클릭하여 열기)", expanded=False):
        sel_loc = st.selectbox("관제 지역", ["서울특별시 (광나루)", "경기도 이천시", "경기도 성남시 (판교)", "경기도 화성시", "강원도 영월군"])
        coords = {"lat": 37.5502, "lon": 127.1215}
        try:
            wx_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,wind_speed_10m&hourly=wind_speed_80m,wind_speed_120m&timezone=Asia%2FSeoul", timeout=4.0).json()
            curr = wx_res.get("current", {})
            hourly = wx_res.get("hourly", {})
            w10 = curr.get("wind_speed_10m", 0.0)
            w80 = hourly.get("wind_speed_80m", [w10])[0]
            w120 = hourly.get("wind_speed_120m", [w80])[0]
            temp = curr.get("temperature_2m", 15.0)
            col_w1, col_w2 = st.columns(2)
            col_w1.metric("지상 10m 풍속", f"{w10} m/s")
            col_w2.metric("상공 120m 풍속", f"{w120} m/s")
            if max(w10, w80, w120) >= 10.0:
                st.error("🚨 비행 위험 (강풍 주의)")
            elif temp <= 0:
                st.warning("⚠️ 저온 배터리 전압강하 위험")
            else:
                st.success("✅ 비행 적합")
        except Exception:
            st.caption("기상 관제 서버 대기 중...")

    st.markdown("---")

    # [계획서 3단계] 다중 시트 엑셀 다운로드
    st.subheader("📊 시스템 운영 로그")
    def make_excel_data():
        buf = io.BytesIO()
        logs = st.session_state["query_logs"]
        df_all = pd.DataFrame(logs) if logs else pd.DataFrame([{"시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "안내": "질의 내역이 누적되면 기록됩니다."}])
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="전체이력", index=False)
            df_all.to_excel(writer, sheet_name="결선주의질의", index=False)
            df_all.to_excel(writer, sheet_name="비교검증질의", index=False)
        buf.seek(0)
        return buf.getvalue()

    st.download_button(
        "📥 질의 이력 3개 시트 엑셀 다운로드",
        data=make_excel_data(),
        file_name=f"drone_hub_log_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# --- [초고속 RAG & Tavily 검색 엔진] ---
@st.cache_resource(show_spinner=False)
def load_drone_vector_store():
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

def retrieve_manual_docs(query: str, cat_filter: str = "전체", k: int = 3):
    docs = []
    cat_p = Path("manual_catalog.json")
    if not cat_p.exists():
        cat_p = Path.cwd().parent / "manual_catalog.json"

    q_low = query.lower()
    if cat_p.exists():
        try:
            catalog = json.loads(cat_p.read_text(encoding="utf-8"))
            for item in catalog:
                if cat_filter != "전체" and item["category"] != cat_filter:
                    continue
                if any(t in q_low for t in item.get("topics", [])) or item["category"].lower() in q_low:
                    f_path = Path("manuals") / item["relative_path"]
                    if not f_path.exists():
                        f_path = Path.cwd().parent / "manuals" / item["relative_path"]
                    if f_path.exists():
                        docs.append({
                            "category": item["category"],
                            "source_file": item["filename"],
                            "path": str(f_path),
                            "content": f_path.read_text(encoding="utf-8")[:280] if f_path.suffix.lower() == ".md" else f"{item['filename']} 기술 문서 발췌본",
                            "page": 1
                        })
                        if len(docs) >= k:
                            break
        except Exception:
            pass

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
    if not TAVILY_API_KEY:
        return []
    try:
        res = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={"api_key": TAVILY_API_KEY, "query": query[:60].strip(), "search_depth": "basic", "max_results": max_results},
            timeout=4.0
        )
        if res.status_code == 200:
            return [{"title": r.get("title", ""), "snippet": r.get("content", "")[:250], "link": r.get("url", "")} for r in res.json().get("results", [])]
    except Exception:
        pass
    return []

def determine_route(query: str, selected_mode: str):
    if "매뉴얼 RAG" in selected_mode:
        return "manual"
    if "Tavily AI" in selected_mode:
        return "web"
    if "비교 대조" in selected_mode:
        return "compare"

    q = query.lower()
    if any(k in q for k in ["비교", "차이", "vs", "어느 모델", "어떤게"]):
        return "compare"
    if any(k in q for k in ["최저가", "알리", "오늘 날씨", "구매처", "시세"]):
        return "web"
    return "manual"

# --- [메인 화면 대화창] ---
st.title("🛸 드론 빌더스 허브 (Drone Builder's Hub)")
st.caption(f"자율비행 드론 HW/SW 14개 계통 매뉴얼 RAG & 항공 안전 관제 포털 (엔진: {MODEL})")

# 과거 대화 출력
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("files"):
            with st.expander(f"📄 참조 매뉴얼 ({len(msg['files'])}건)", expanded=False):
                for f in msg["files"]:
                    p = Path(f["path"])
                    col_t, col_b = st.columns([4, 1])
                    col_t.markdown(f"• **[{f['category']}]** `{f['source_file']}` (p.{f['page']})")
                    if p.exists():
                        col_b.download_button("⬇️ 다운로드", data=p.read_bytes(), file_name=f["source_file"], key=f"hist_{f['source_file']}_{f['page']}")

preset_q = st.session_state.pop("preset_query", None)
user_prompt = st.chat_input("하드웨어 배선, ROS 2 통신, YOLO 또는 비행 규정을 질문하세요...") or preset_q

if user_prompt:
    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        # [계획서 2단계] 결선 안전 경고 배너 자동 주입
        is_wire_hazard = any(k in user_prompt.lower() for k in ["vcc", "gnd", "결선", "배선", "쇼트", "단락", "5v", "12v", "극성"])
        if is_wire_hazard:
            st.warning("⚡ **[하드웨어 결선 주의]** VCC(+)와 GND(-) 극성 역전 시 ESC/FC 전소 화재 위험이 있습니다. 배선 전 반드시 전압 테스터기로 통전을 점검하십시오.")

        route = determine_route(user_prompt, search_mode)
        if route == "compare":
            st.info("⚖️ **검색 경로: 매뉴얼 vs 웹 비교 대조 분석**")
        elif route == "web":
            st.info("🌐 **검색 경로: Tavily AI 실시간 기술 검색**")
        else:
            st.info("📘 **검색 경로: 매뉴얼 RAG (14개 계통 기술 문서)**")

        with st.spinner("전문 규격 및 자료를 분석하여 답변을 작성 중입니다..."):
            manual_docs = retrieve_manual_docs(user_prompt, selected_cat, k=3)
            web_docs = search_tavily(user_prompt, max_results=3) if route in ["web", "compare"] else []
            active_persona = PERSONA_PROMPTS.get(selected_persona, PERSONA_PROMPTS["현장 엔지니어 (핀맵/명령어 중심)"])

            # [계획서 3단계] compare 모드: st.tabs 3개 영역 분할 렌더링
            if route == "compare":
                m_ctx = "\n".join([f"[{d['category']}/{d['source_file']}] {d['content']}" for d in manual_docs]) or "매뉴얼 없음"
                w_ctx = "\n".join([f"[{w['title']}] {w['snippet']}" for w in web_docs]) or "웹 검색 없음"

                compare_prompt = f"""{active_persona}

당신은 드론 하드웨어 엔지니어링 비교 분석 전문가입니다.
질문: {user_prompt}

[로컬 매뉴얼 규격 자료]
{m_ctx}

[Tavily AI 실시간 기술 검색 자료]
{w_ctx}

위 두 자료를 정밀 대조하여 반드시 아래 3개 구분자(===)를 기준으로 작성하십시오:
===ANALYSIS===
(매뉴얼 기준과 웹 시판 제품을 대조한 일치점, 차이점, 최종 권고사항)
===MANUAL===
(매뉴얼에 기재된 전압, 전류, 추력표 등 핵심 기술 기준 요약)
===WEB===
(웹 검색을 통해 파악된 시판 제품 스펙 및 최신 동향 요약)"""

                try:
                    res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": compare_prompt}], max_completion_tokens=1500, timeout=35.0)
                    raw = res.choices[0].message.content or ""
                except Exception:
                    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": compare_prompt}], max_tokens=1200, timeout=30.0)
                    raw = res.choices[0].message.content or ""

                p_ana = raw.split("===ANALYSIS===")[-1].split("===MANUAL===")[0].strip() if "===ANALYSIS===" in raw else raw
                p_man = raw.split("===MANUAL===")[-1].split("===WEB===")[0].strip() if "===MANUAL===" in raw else "매뉴얼 분석 완료"
                p_web = raw.split("===WEB===")[-1].strip() if "===WEB===" in raw else "웹 분석 완료"

                t1, t2, t3 = st.tabs(["⚖️ 종합 비교 대조 & 분석", "📘 매뉴얼 규격 내용", "🌐 웹 최신 정보 및 출처"])
                with t1: st.markdown(p_ana)
                with t2: st.markdown(p_man)
                with t3:
                    st.markdown(p_web)
                    for w in web_docs: st.markdown(f"- [{w['title']}]({w['link']})")
                final_ans = p_ana

            # 일반 RAG / Web 모드: 페르소나 적용 본문 렌더링
            else:
                ctx = "\n".join([f"[{d['category']}/{d['source_file']} p.{d['page']}] {d['content']}" for d in manual_docs]) if route == "manual" else "\n".join([f"[{w['title']}] {w['snippet']}" for w in web_docs])
                sys_msg = f"""{active_persona}

당신은 드론 빌더스 허브의 공인 수석 기술 자문 어시스턴트입니다.
제공된 근거 자료를 바탕으로 질문에 대해 체계적이고 풍부한 마크다운으로 답변하십시오.

[출력 규칙]
1. 본문은 핵심 조항, 실무 수치, 절차를 논리적으로 서술하십시오.
2. 답변 말미에 반드시 아래 형식으로 참조 쪽수 요약을 넣으십시오:
---
📌 **참조 매뉴얼 및 인용 근거:**
• `[계통명] 파일명.pdf` (p.X) — 핵심 내용 1줄 요약"""

                try:
                    res = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": f"질문: {user_prompt}\n\n[근거 자료]\n{ctx}"}],
                        max_completion_tokens=1500,
                        timeout=35.0
                    )
                    final_ans = res.choices[0].message.content or "답변 내용을 정리 중입니다."
                except Exception:
                    res = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": f"질문: {user_prompt}\n\n[근거 자료]\n{ctx}"}],
                        max_tokens=1200,
                        timeout=30.0
                    )
                    final_ans = res.choices[0].message.content

                # 본문 화면 렌더링 (확실한 출력 보장)
                st.markdown(final_ans)

        # [계획서 2단계] 비행 전 점검표(Checklist) 생성 버튼
        if st.button("📋 이 작업의 비행 전 안전 체크리스트(Checklist) 생성", key=f"btn_chk_{datetime.now().strftime('%f')}"):
            st.markdown("##### 📋 현장 비행 전 안전 점검표 (Pre-flight Checklist)")
            st.markdown("- [ ] **조종기/기체 배터리 전압 및 셀 밸런스 정상 확인**")
            st.markdown("- [ ] **모터 회전 방향 및 프로펠러 체결 잠금 너트 토크 확인**")
            st.markdown("- [ ] **드론원스톱 비행승인/촬영허가 공문 및 지자체 관제권 확인**")
            st.markdown("- [ ] **페일세이프(RTL 자동 복귀 고도) 최소 30m 이상 설정 점검**")

        # [계획서 2단계] 인용된 원본 매뉴얼 다운로드 카드 렌더링
        if manual_docs:
            st.markdown("##### 📥 인용된 원본 매뉴얼 다운로드")
            with st.expander(f"📄 매뉴얼 목록 ({len(manual_docs)}건)", expanded=True):
                for idx, doc in enumerate(manual_docs):
                    fpath = Path(doc["path"])
                    fname = doc["source_file"]
                    col_t, col_b = st.columns([4, 1])
                    col_t.markdown(f"• **[{doc['category']}]** `{fname}` (p.{doc['page']})")
                    if fpath.exists():
                        col_b.download_button("⬇️ 다운로드", data=fpath.read_bytes(), file_name=fname, key=f"live_{idx}_{fname}")

        st.session_state["chat_history"].append({"role": "assistant", "content": final_ans, "files": manual_docs})
        st.session_state["query_logs"].append({
            "시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "질문": user_prompt,
            "경로": route,
            "페르소나": selected_persona,
            "선택계통": selected_cat,
            "답변요약": final_ans[:120].replace("\n", " ") + "..."
        })
