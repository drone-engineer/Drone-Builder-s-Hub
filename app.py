import os
import json
import io
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
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()

if not OPENAI_API_KEY:
    st.error("⚠️ .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0, max_retries=2)

# 세션 상태 초기화 (대화 기록 & 엑셀 로깅 데이터)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "query_logs" not in st.session_state:
    st.session_state["query_logs"] = []

# --- [사이드바: 검색모드 / 접이식 계산기 2종 / 기상관제 / 계통필터 / 3개 시트 엑셀] ---
with st.sidebar:
    st.title("🛸 드론 비행 & 설계 포털")

    # 1. 검색 모드 선택 (라디오 4종)
    st.subheader("⚙️ 검색 모드 선택")
    search_mode = st.radio(
        "검색 및 처리 경로",
        ["🤖 자동 판별 (AI 라우팅)", "📘 매뉴얼 RAG (로컬 기술문서)", "🌐 Tavily AI 실시간 기술 검색", "⚖️ 매뉴얼 vs 웹 비교 대조"],
        index=0
    )
    st.markdown("---")

    # 2. 요구 추력 계산기 (접이식 Expander로 개선)
    with st.expander("📐 요구 추력 계산기 (클릭하여 열기)", expanded=False):
        frame_type = st.selectbox("모터 개수 (프레임 타입)", ["4개 (쿼드콥터)", "6개 (헥사콥터)", "8개 (옥토콥터)"], key="sb_frame")
        motor_count = 4 if "4개" in frame_type else (6 if "6개" in frame_type else 8)

        payload_w = st.number_input("임무장비(페이로드) 무게 (kg)", min_value=0.1, max_value=50.0, value=2.0, step=0.5, key="sb_pay")
        frame_w = st.number_input("기체 무게 (배터리·프레임 포함, kg)", min_value=0.5, max_value=100.0, value=5.0, step=0.5, key="sb_frame_w")
        twr_target = st.slider("목표 추력비 (TWR, 권장 2.0:1)", min_value=1.5, max_value=3.0, value=2.0, step=0.1, key="sb_twr")

        total_weight = frame_w + payload_w
        req_total_thrust = total_weight * twr_target
        motor_max_thrust = req_total_thrust / motor_count
        hover_thrust = total_weight / motor_count

        st.markdown("##### 📊 계산 결과")
        c1, c2 = st.columns(2)
        c1.metric("총 이륙중량(AUW)", f"{total_weight:.1f} kg")
        c2.metric("요구 총 추력", f"{req_total_thrust:.1f} kg")
        c3, c4 = st.columns(2)
        c3.metric("모터당 최대 추력", f"{motor_max_thrust:.2f} kg", f"↑ {int(motor_max_thrust*1000)} g")
        c4.metric("호버링 추력(50%)", f"{hover_thrust:.2f} kg", f"↑ {int(hover_thrust*1000)} g")

        calc_query = f"모터 {motor_count}개 멀티콥터이고 기체 {frame_w}kg, 페이로드 {payload_w}kg(총중량 {total_weight:.1f}kg)이야. 모터당 최대추력 {motor_max_thrust:.2f}kg이 필요한데 적합한 파워콤보(모터, ESC, 프롭, 배터리) 추천과 스펙을 분석해줘"
        if st.button("💬 이 조건으로 AI 챗봇에 즉시 추천 요청", key="btn_calc_query"):
            st.session_state["preset_query"] = calc_query
            st.rerun()

    # 3. 비행시간 분석 & 모터 역설계 시뮬레이터 (접이식)
    with st.expander("⏱️ 비행시간 분석 & 모터 역설계 시뮬레이터", expanded=False):
        calc_tab1, calc_tab2 = st.tabs(["🎯 목표시간 모터 역추천", "📊 순방향 체공시간 계산"])

        with calc_tab1:
            st.caption("원하는 비행시간(예: 60분)을 설정하면, 필수 최소 효율(g/W)을 역산하고 적합 모터를 추천합니다.")
            rev_mc = st.selectbox("모터 개수", [4, 6, 8], index=0, format_func=lambda x: f"{x}개 ({'쿼드' if x==4 else '헥사' if x==6 else '옥토'}콥터)", key="rev_mc")
            rw1, rw2 = st.columns(2)
            rev_dry = rw1.number_input("기체 공허중량 (kg)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, key="rev_dry")
            rev_pay = rw2.number_input("임무장비 무게 (kg)", min_value=0.0, max_value=50.0, value=3.0, step=0.5, key="rev_pay")
            rb1, rb2 = st.columns(2)
            rev_bs = rb1.selectbox("배터리 전압(S)", [6, 12, 14], index=1, format_func=lambda x: f"{x}S ({round(x*3.7, 1)}V)", key="rev_bs")
            rev_bmah = rb2.number_input("배터리 용량 (mAh)", min_value=5000, max_value=120000, value=30000, step=2000, key="rev_bmah")
            rev_tmin = st.number_input("🎯 목표 비행시간 (분)", min_value=10, max_value=120, value=60, step=5, key="rev_tmin")

            est_batt_wt = round((rev_bmah / 1000.0) * (rev_bs / 12.0) * 0.205, 2)
            rev_auw = round(rev_dry + rev_pay + est_batt_wt, 2)
            v_nom_rev = rev_bs * 3.7
            total_wh_rev = (rev_bmah * v_nom_rev) / 1000.0
            usable_wh_rev = total_wh_rev * 0.80
            max_allowed_w = usable_wh_rev / (rev_tmin / 60.0)
            motor_allowed_w = round(max_allowed_w / rev_mc, 1)
            rev_thrust_per_motor = round((rev_auw * 1000.0) / rev_mc, 1)
            required_eff = round(rev_thrust_per_motor / motor_allowed_w, 2) if motor_allowed_w > 0 else 0

            st.markdown("##### 📐 역산된 모터 스펙 요구조건")
            r1, r2 = st.columns(2)
            r1.metric("예상 총 이륙중량", f"{rev_auw} kg")
            r2.metric("축당 호버링 요구추력", f"{int(rev_thrust_per_motor)} g")
            r3, r4 = st.columns(2)
            r3.metric("모터당 허용 전력", f"{motor_allowed_w} W 이하")
            r4.metric("필수 최소 효율", f"{required_eff} g/W 이상")

            if required_eff > 11.5:
                st.warning("⚠️ 요구 효율이 11.5 g/W를 초과합니다. 대형 프롭 및 고효율 VTOL 모터가 필요합니다.")
            elif required_eff >= 8.5:
                st.info("💡 산업용 대형 파워시스템(24~28인치 프롭 / 저KV)으로 실현 가능한 범위입니다.")
            else:
                st.success("🟢 일반 상용 파워시스템으로 충분히 달성 가능한 안정적 수치입니다.")

            rev_prompt = f"기체 공허중량 {rev_dry}kg, 임무장비 {rev_pay}kg, 배터리 {rev_bs}S {rev_bmah}mAh(추정무게 {est_batt_wt}kg)로 총 이륙중량 약 {rev_auw}kg인 {rev_mc}축 기체야. 목표 비행시간 {rev_tmin}분을 달성하려면 축당 호버링 추력 {int(rev_thrust_per_motor)}g에서 모터 1개당 소모전력이 {motor_allowed_w}W 이하, 최소 효율이 {required_eff}g/W 이상이어야 해. 로컬 매뉴얼 제원표와 최신 스펙을 대조해서 이를 충족할 수 있는 구체적인 모터 모델명, 프로펠러 인치, ESC 조합을 역설계 추천해줘."
            if st.button("🚀 이 역설계 조건으로 최적 모터 검색 및 분석", key="btn_rev_search"):
                st.session_state["preset_query"] = rev_prompt
                st.rerun()

        with calc_tab2:
            st.caption("기체 중량(AUW)과 배터리 스펙으로 예상 호버링 시간을 계산합니다.")
            fwd_auw = st.number_input("기체 총 이륙중량 (AUW, kg)", min_value=0.5, max_value=200.0, value=7.0, step=0.5, key="fwd_auw")
            fb1, fb2 = st.columns(2)
            fwd_s = fb1.selectbox("배터리 셀 수 (S)", [4, 6, 12, 14], index=2, format_func=lambda x: f"{x}S ({round(x*3.7, 1)}V)", key="fwd_s")
            fwd_mah = fb2.number_input("배터리 용량 (mAh)", min_value=1000, max_value=100000, value=22000, step=1000, key="fwd_mah")
            fe1, fe2 = st.columns(2)
            fwd_eff = fe1.slider("호버링 추진 효율 (g/W)", min_value=4.0, max_value=12.0, value=8.0, step=0.1, key="fwd_eff")
            fwd_dod = fe2.slider("배터리 방전 심도 (DOD %)", min_value=50, max_value=95, value=80, step=5, key="fwd_dod")

            v_nom_f = fwd_s * 3.7
            total_wh_f = (fwd_mah * v_nom_f) / 1000.0
            usable_wh_f = total_wh_f * (fwd_dod / 100.0)
            hover_power_f = (fwd_auw * 1000.0) / fwd_eff
            hover_curr_f = hover_power_f / v_nom_f
            flight_time_f = (usable_wh_f / hover_power_f) * 60.0
            ft_sec = int((flight_time_f - int(flight_time_f)) * 60)
            req_c = round(hover_curr_f / (fwd_mah / 1000.0), 2)

            st.markdown("##### 📊 체공 시간 분석 결과")
            fc1, fc2 = st.columns(2)
            fc1.metric("예상 체공시간", f"{int(flight_time_f)}분 {ft_sec}초")
            fc2.metric("호버링 소모 전력", f"{round(hover_power_f, 1)} W")
            fc3, fc4 = st.columns(2)
            fc3.metric("호버링 소모 전류", f"{round(hover_curr_f, 1)} A")
            fc4.metric("요구 방전율", f"{req_c} C")

            fwd_prompt = f"AUW {fwd_auw}kg 기체이고 {fwd_s}S {fwd_mah}mAh LiPo 배터리를 사용할 때 호버링 시간이 약 {int(flight_time_f)}분으로 계산되었어. 호버링 소모전류 {round(hover_curr_f, 1)}A를 고려할 때 적합한 배터리 C-rate와 전원분배장치(PDB/커넥터) 규격을 권고해줘."
            if st.button("💬 이 조건으로 배터리·PDB 규격 분석 요청", key="btn_fwd_query"):
                st.session_state["preset_query"] = fwd_prompt
                st.rerun()

    # 4. Open-Meteo 실시간 상공 비행 기상 관제 위젯 (접이식)
    with st.expander("🌦️ 실시간 비행 기상 관제 (클릭하여 열기)", expanded=False):
        REGION_PRESETS = {
            "서울특별시 (광나루 비행장)": {"lat": 37.5502, "lon": 127.1215},
            "경기도 이천시 (테스트 필드)": {"lat": 37.2723, "lon": 127.4350},
            "경기도 성남시 (판교 IT밸리)": {"lat": 37.3947, "lon": 127.1112},
            "경기도 화성시 (비봉 드론비행장)": {"lat": 37.2285, "lon": 126.8920},
            "강원도 영월군 (드론전용공역)": {"lat": 37.1837, "lon": 128.4619}
        }
        sel_region = st.selectbox("비행 관제 지역", list(REGION_PRESETS.keys()), index=0)
        coords = REGION_PRESETS[sel_region]

        try:
            wx_url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current=temperature_2m,wind_speed_10m,wind_gusts_10m&hourly=wind_speed_80m,wind_speed_120m&timezone=Asia%2FSeoul"
            wx_res = requests.get(wx_url, timeout=4.0).json()
            curr = wx_res.get("current", {})
            hourly = wx_res.get("hourly", {})
            temp = curr.get("temperature_2m", 0.0)
            w10 = curr.get("wind_speed_10m", 0.0)
            gust = curr.get("wind_gusts_10m", 0.0)
            w80 = hourly.get("wind_speed_80m", [w10])[0]
            w120 = hourly.get("wind_speed_120m", [w80])[0]

            wc1, wc2 = st.columns(2)
            wc1.metric("지상 기온", f"{temp} °C")
            wc1.metric("지상 10m 풍속", f"{w10} m/s")
            wc2.metric("상공 80m 순항풍", f"{w80} m/s")
            wc2.metric("120m(법정한계)", f"{w120} m/s")

            max_w = max(w10, w80, w120, gust)
            if max_w >= 10.0:
                st.error(f"🚨 **비행 위험 (강풍 {max_w}m/s)**: 실외 비행 즉시 중단 권고")
            elif max_w >= 6.5:
                st.warning(f"⚠️ **비행 주의 (풍속 {max_w}m/s)**: 자세제어 주의")
            else:
                st.success(f"✅ **비행 적합 (최대 {max_w}m/s)**: 안정적 비행 가능")
        except Exception:
            st.caption("기상 관제 서버 응답 대기 중...")

    st.markdown("---")

    # 5. 매뉴얼 계통 필터
    st.subheader("📁 매뉴얼 계통 필터")
    manual_cats = [
        "전체", "01_비행제어기_FC", "02_모터_구동계", "03_변속기_ESC", "04_프로펠러_공력",
        "05_전원_배터리_PDB", "06_센서_항법장치", "07_조종_데이터링크", "08_영상_FPV_짐벌",
        "09_프레임_페이로드", "10_안전_비행규정", "11_온보드컴퓨터_임베디드",
        "12_미들웨어_통신_ROS", "13_컴퓨터비전_AI_YOLO", "14_SLAM_경로계획_내비"
    ]
    selected_cat = st.selectbox("계통 선택", manual_cats)

    st.markdown("---")

    # 6. 질문과 답변 3개 시트 엑셀 다운로드 (완벽 복원)
    st.subheader("📊 시스템 운영 로그")
    log_count = len(st.session_state["query_logs"])
    st.caption(f"누적 질의 기록: **{log_count}건**")

    # 엑셀 데이터 생성 함수 (pd.ExcelWriter 기반 3개 시트 분할)
    def generate_excel_bytes():
        buffer = io.BytesIO()
        logs = st.session_state["query_logs"]
        if not logs:
            df_all = pd.DataFrame([{"시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "질문": "초기 상태", "경로": "안내", "답변": "질의 이력이 누적되면 이곳에 기록됩니다."}])
            df_warn = df_all.copy()
            df_comp = df_all.copy()
        else:
            df_all = pd.DataFrame(logs)
            # 결선 주의 질의 필터링
            df_warn = df_all[df_all["질문"].str.contains("결선|배선|vcc|gnd|쇼트|전압|화재", case=False, na=False)]
            if df_warn.empty:
                df_warn = pd.DataFrame([{"안내": "결선 및 전원 관련 주의 질의 없음"}])
            # 비교/검증 질의 필터링
            df_comp = df_all[df_all["경로"].str.contains("compare|비교", case=False, na=False)]
            if df_comp.empty:
                df_comp = pd.DataFrame([{"안내": "비교 대조 질의 없음"}])

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="전체이력", index=False)
            df_warn.to_excel(writer, sheet_name="결선주의질의", index=False)
            df_comp.to_excel(writer, sheet_name="비교검증질의", index=False)
        buffer.seek(0)
        return buffer.getvalue()

    st.download_button(
        label="📥 질의 이력 3개 시트 엑셀 다운로드",
        data=generate_excel_bytes(),
        file_name=f"drone_hub_logs_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# --- [초고속 2~3초 RAG & Tavily 웹 검색 엔진] ---
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

def retrieve_manual_docs(query: str, cat_filter: str = "전체", k: int = 4):
    docs = []
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

def determine_route(query: str, selected_mode: str):
    if "매뉴얼 RAG" in selected_mode:
        return "manual"
    if "Tavily AI" in selected_mode:
        return "web"
    if "비교 대조" in selected_mode:
        return "compare"

    q = query.lower()
    if any(k in q for k in ["비교", "차이", "vs", "추천", "어떤게", "골라"]):
        return "compare"
    if any(k in q for k in ["최신", "시세", "가격", "출시", "이슈", "github", "법령", "과태료"]):
        return "web"
    return "manual"

# --- [메인 화면 UI] ---
st.title("🛸 드론 빌더스 허브 (Drone Builder's Hub)")
st.caption(f"자율비행 드론 HW/SW 14개 계통 매뉴얼 RAG & Tavily AI 비행 안전 관제 포털 (엔진: {MODEL})")

# 이전 대화 렌더링
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
                        col_b.download_button("⬇️ 다운로드", data=p.read_bytes(), file_name=name, key=f"hist_{name}_{f.get('page', 1)}")

# 사용자 입력 처리
preset_q = st.session_state.pop("preset_query", None)
user_prompt = st.chat_input("하드웨어 배선, ROS 2 통신, YOLO 또는 비행 규정을 질문하세요...") or preset_q

if user_prompt:
    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        route = determine_route(user_prompt, search_mode)

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

            # 3. gpt-6-luna 답변 생성
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
                if route == "web":
                    ctx = "\n".join([f"[{w['title']}] {w['snippet']}" for w in web_docs]) or "웹 자료 없음"
                    prompt = f"질문: {user_prompt}\n\n웹 검색자료:\n{ctx}\n\n위 자료를 바탕으로 엔지니어링 답변을 작성하세요."
                else:
                    ctx = "\n".join([f"[{d['category']}/{d['source_file']}] {d['content']}" for d in manual_docs]) or "매뉴얼 자료 없음"
                    prompt = f"질문: {user_prompt}\n\n매뉴얼 규격자료:\n{ctx}\n\n위 매뉴얼 자료에 근거하여 파라미터와 규격을 명확히 답변하세요."

                res = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
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
                        col_b.download_button("⬇️ 다운로드", data=fpath.read_bytes(), file_name=fname, key=f"live_{idx}_{fname}")

        # 5. 세션 기록 및 엑셀 로깅 데이터 실시간 적재
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": final_answer_text,
            "files": manual_docs
        })
        st.session_state["query_logs"].append({
            "시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "질문": user_prompt,
            "경로": route,
            "선택계통": selected_cat,
            "답변요약": final_answer_text[:120].replace("\n", " ") + "..."
        })
