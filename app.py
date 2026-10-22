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
MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()

if not OPENAI_API_KEY:
    st.error("⚠️ .env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# 타임아웃 45초 확보
client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0, max_retries=2)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "query_logs" not in st.session_state:
    st.session_state["query_logs"] = []

# --- [모터 보간 연산 함수] ---
def interpolate_motor_benchmark(motor_name: str, target_thrust_g: float):
    bpath = Path("motor_benchmarks.json")
    if not bpath.exists():
        bpath = Path.cwd().parent / "motor_benchmarks.json"
    if not bpath.exists():
        return None
    try:
        data = json.loads(bpath.read_text(encoding="utf-8"))
        m_info = data.get(motor_name)
        if not m_info:
            return None
        curve = sorted(m_info["curve"], key=lambda x: x["thrust_g"])
        if target_thrust_g <= curve[0]["thrust_g"]:
            return {**curve[0], "status": "호버링 여유 (저부하)"}
        if target_thrust_g >= curve[-1]["thrust_g"]:
            return {**curve[-1], "status": "⚠️ 모터 최대 한계 초과 위험"}
        for i in range(len(curve) - 1):
            p1, p2 = curve[i], curve[i+1]
            if p1["thrust_g"] <= target_thrust_g <= p2["thrust_g"]:
                ratio = (target_thrust_g - p1["thrust_g"]) / (p2["thrust_g"] - p1["thrust_g"])
                return {
                    "throttle": round(p1["throttle"] + ratio * (p2["throttle"] - p1["throttle"]), 1),
                    "current_a": round(p1["current_a"] + ratio * (p2["current_a"] - p1["current_a"]), 2),
                    "power_w": round(p1["power_w"] + ratio * (p2["power_w"] - p1["power_w"]), 1),
                    "eff_gw": round(p1["eff_gw"] + ratio * (p2["eff_gw"] - p1["eff_gw"]), 2),
                    "status": "✅ 안정적 호버링 (권장 40~55%)"
                }
    except Exception:
        pass
    return None

# --- [사이드바 UI] ---
with st.sidebar:
    st.title("🛸 드론 비행 & 설계 포털")

    # [1] 검색 모드 선택
    st.subheader("⚙️ 검색 모드 선택")
    search_mode = st.radio(
        "검색 및 처리 경로",
        ["🤖 자동 판별 (AI 라우팅)", "📘 매뉴얼 RAG (로컬 기술문서)", "🌐 Tavily AI 실시간 기술 검색", "⚖️ 매뉴얼 vs 웹 비교 대조"],
        index=0
    )
    st.markdown("---")

    # [2] 요구 추력 계산기 (접이식)
    with st.expander("📐 요구 추력 계산기 (클릭하여 열기)", expanded=False):
        frame_type = st.selectbox("모터 개수", ["4개 (쿼드콥터)", "6개 (헥사콥터)", "8개 (옥토콥터)"], key="sb_frame")
        motor_count = 4 if "4개" in frame_type else (6 if "6개" in frame_type else 8)
        payload_w = st.number_input("임무장비(페이로드) 무게 (kg)", min_value=0.1, max_value=50.0, value=2.0, step=0.5, key="sb_pay")
        frame_w = st.number_input("기체 무게 (배터리 포함, kg)", min_value=0.5, max_value=100.0, value=5.0, step=0.5, key="sb_frame_w")
        twr_target = st.slider("목표 추력비 (TWR)", 1.5, 3.0, 2.0, 0.1, key="sb_twr")

        total_weight = frame_w + payload_w
        req_total_thrust = total_weight * twr_target
        motor_max_thrust = req_total_thrust / motor_count
        hover_thrust = total_weight / motor_count

        c1, c2 = st.columns(2)
        c1.metric("총 이륙중량", f"{total_weight:.1f} kg")
        c2.metric("요구 총 추력", f"{req_total_thrust:.1f} kg")
        c3, c4 = st.columns(2)
        c3.metric("모터당 최대 추력", f"{motor_max_thrust:.2f} kg")
        c4.metric("호버링 추력(50%)", f"{hover_thrust:.2f} kg")

        calc_q = f"모터 {motor_count}개 기체이고 AUW {total_weight:.1f}kg이야. 모터당 최대추력 {motor_max_thrust:.2f}kg이 필요한데 적합한 파워콤보(모터, ESC, 프롭, 배터리) 추천과 스펙을 분석해줘"
        if st.button("💬 이 조건으로 AI 추천 요청", key="btn_calc_query"):
            st.session_state["preset_query"] = calc_q
            st.rerun()

    # [3] 비행시간 분석 & 모터 역설계 시뮬레이터 (3개 탭 완전 복원)
    with st.expander("⏱️ 비행시간 분석 & 모터 역설계 시뮬레이터", expanded=False):
        tab1, tab2, tab3 = st.tabs(["🎯 목표시간 모터 역추천", "📊 순방향 체공시간", "🔬 실측 벤치마크 시뮬레이터"])

        with tab1:
            st.caption("목표 비행시간(예: 60분) 설정 시 요구 최소 효율(g/W)을 역산합니다.")
            r_mc = st.selectbox("모터 개수", [4, 6, 8], index=0, key="r_mc")
            col_rw1, col_rw2 = st.columns(2)
            r_dry = col_rw1.number_input("공허중량 (kg)", 0.5, 50.0, 5.0, 0.5, key="r_dry")
            r_pay = col_rw2.number_input("장비무게 (kg)", 0.0, 50.0, 3.0, 0.5, key="r_pay")
            col_rb1, col_rb2 = st.columns(2)
            r_bs = col_rb1.selectbox("배터리 전압", [6, 12, 14], index=1, format_func=lambda x: f"{x}S ({round(x*3.7,1)}V)", key="r_bs")
            r_mah = col_rb2.number_input("배터리 용량 (mAh)", 5000, 120000, 30000, 2000, key="r_mah")
            r_tmin = st.number_input("🎯 목표 비행시간 (분)", 10, 120, 60, 5, key="r_tmin")

            est_b_wt = round((r_mah / 1000.0) * (r_bs / 12.0) * 0.205, 2)
            r_auw = round(r_dry + r_pay + est_b_wt, 2)
            v_nom_r = r_bs * 3.7
            usable_wh_r = ((r_mah * v_nom_r) / 1000.0) * 0.80
            motor_allowed_w = round((usable_wh_r / (r_tmin / 60.0)) / r_mc, 1)
            thrust_per_m = round((r_auw * 1000.0) / r_mc, 1)
            req_eff_gw = round(thrust_per_m / motor_allowed_w, 2) if motor_allowed_w > 0 else 0

            rc1, rc2 = st.columns(2)
            rc1.metric("총 이륙중량", f"{r_auw} kg")
            rc2.metric("축당 호버링 추력", f"{int(thrust_per_m)} g")
            rc3, rc4 = st.columns(2)
            rc3.metric("모터당 허용 전력", f"{motor_allowed_w} W 이하")
            rc4.metric("필수 최소 효율", f"{req_eff_gw} g/W 이상")

            rev_p = f"AUW {r_auw}kg {r_mc}축 기체에서 목표 비행시간 {r_tmin}분을 달성하려면 축당 호버링 추력 {int(thrust_per_m)}g에서 모터 소모전력이 {motor_allowed_w}W 이하, 최소효율이 {req_eff_gw}g/W 이상이어야 해. 이를 충족하는 모터, 프롭, ESC 조합을 역설계 추천해줘."
            if st.button("🚀 이 역설계 조건으로 최적 모터 검색", key="btn_rev_search"):
                st.session_state["preset_query"] = rev_p
                st.rerun()

        with tab2:
            st.caption("기체 중량과 배터리 스펙으로 예상 호버링 시간을 계산합니다.")
            f_auw = st.number_input("총 이륙중량 (AUW, kg)", 0.5, 100.0, float(total_weight), 0.5, key="f_auw")
            f_s = st.selectbox("배터리 셀 수 (S)", [4, 6, 12, 14], index=2, key="f_s")
            f_mah = st.number_input("배터리 용량 (mAh)", 1000, 100000, 22000, 1000, key="f_mah")
            f_eff = st.slider("호버링 추진 효율 (g/W)", 4.0, 12.0, 8.0, 0.1, key="f_eff")
            f_dod = st.slider("가용 방전심도 (DOD %)", 50, 95, 80, 5, key="f_dod")

            v_nom_f = f_s * 3.7
            usable_wh_f = ((f_mah * v_nom_f) / 1000.0) * (f_dod / 100.0)
            hover_p_f = (f_auw * 1000.0) / f_eff
            flight_t_f = (usable_wh_f / hover_p_f) * 60.0
            st.metric("예상 체공시간", f"{int(flight_t_f)}분 {int((flight_t_f - int(flight_t_f))*60)}초")

        with tab3:
            st.caption("실측 시험 데이터시트(JSON) 기반 오차 없는 체공시간 보간 연산")
            bpath_chk = Path("motor_benchmarks.json")
            if not bpath_chk.exists():
                bpath_chk = Path.cwd().parent / "motor_benchmarks.json"
            motor_opts = list(json.loads(bpath_chk.read_text(encoding="utf-8")).keys()) if bpath_chk.exists() else []
            sel_m = st.selectbox("실측 시험 모터", motor_opts, index=0, key="sel_m")
            col_b1, col_b2 = st.columns(2)
            b_mc_val = col_b1.selectbox("축 수", [4, 6, 8], index=0, key="b_mc_val")
            b_auw_val = col_b2.number_input("총 이륙중량 (kg)", 1.0, 80.0, 14.0, 0.5, key="b_auw_val")
            req_thrust_g = (b_auw_val * 1000.0) / b_mc_val
            inter_res = interpolate_motor_benchmark(sel_m, req_thrust_g)
            if inter_res:
                bc1, bc2 = st.columns(2)
                bc1.metric("호버링 스로틀", f"{inter_res['throttle']} %", inter_res['status'])
                bc2.metric("실측 모터 효율", f"{inter_res['eff_gw']} g/W")

    # [4] 실시간 비행 기상 관제
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
            col_w1, col_w2 = st.columns(2)
            col_w1.metric("지상 10m 풍속", f"{w10} m/s")
            col_w2.metric("상공 120m 풍속", f"{w120} m/s")
            if max(w10, w80, w120) >= 10.0:
                st.error("🚨 비행 위험 (강풍)")
            else:
                st.success("✅ 비행 적합")
        except Exception:
            st.caption("기상 관제 서버 대기 중...")

    st.markdown("---")

    # [5] 매뉴얼 계통 필터
    st.subheader("📁 매뉴얼 계통 필터")
    manual_cats = [
        "전체", "01_비행제어기_FC", "02_모터_구동계", "03_변속기_ESC", "04_프로펠러_공력",
        "05_전원_배터리_PDB", "06_센서_항법장치", "07_조종_데이터링크", "08_영상_FPV_짐벌",
        "09_프레임_페이로드", "10_안전_비행규정", "11_온보드컴퓨터_임베디드",
        "12_미들웨어_통신_ROS", "13_컴퓨터비전_AI_YOLO", "14_SLAM_경로계획_내비"
    ]
    selected_cat = st.selectbox("계통 선택", manual_cats)

    st.markdown("---")

    # [6] 3개 시트 엑셀 다운로드
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

# 라우터: 14개 계통 매뉴얼 주제는 무조건 'manual' 우선 (외부 웹 지연 원천 방어)
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

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("files"):
            with st.expander(f"📄 참조 매뉴얼 ({len(msg['files'])}건)", expanded=False):
                for f in msg["files"]:
                    p = Path(f["path"])
                    col_t, col_b = st.columns([4, 1])
                    col_t.markdown(f"• **[{f['category']}]** `{f['source_file']}`")
                    if p.exists():
                        col_b.download_button("⬇️ 다운로드", data=p.read_bytes(), file_name=f["source_file"], key=f"h_{f['source_file']}_{f['page']}")

preset_q = st.session_state.pop("preset_query", None)
user_prompt = st.chat_input("하드웨어 배선, ROS 2 통신, YOLO 또는 비행 규정을 질문하세요...") or preset_q

if user_prompt:
    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        route = determine_route(user_prompt, search_mode)

        if route == "compare":
            st.info("⚖️ **검색 경로: 매뉴얼 vs 웹 비교 대조 분석**")
        elif route == "web":
            st.info("🌐 **검색 경로: Tavily AI 실시간 기술 검색**")
        else:
            st.info("📘 **검색 경로: 매뉴얼 RAG (14개 계통 기술 문서)**")

        with st.spinner("자료를 분석하고 답변을 작성 중입니다..."):
            manual_docs = retrieve_manual_docs(user_prompt, selected_cat, k=3)
            web_docs = search_tavily(user_prompt, max_results=3) if route in ["web", "compare"] else []

            if route == "compare":
                m_ctx = "\n".join([f"[{d['category']}/{d['source_file']}] {d['content']}" for d in manual_docs]) or "매뉴얼 없음"
                w_ctx = "\n".join([f"[{w['title']}] {w['snippet']}" for w in web_docs]) or "웹 검색 없음"

                fast_p = f"""질문: {user_prompt}
[매뉴얼 자료]
{m_ctx}
[웹 자료]
{w_ctx}

위 자료를 종합하여 아래 3개 구분자로만 작성하세요:
===ANALYSIS===
(종합 비교 대조 권고사항)
===MANUAL===
(매뉴얼 기준 규격 수치)
===WEB===
(웹 최신 동향)"""
                res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": fast_p}], max_completion_tokens=850, timeout=35.0)
                raw = res.choices[0].message.content
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
            else:
                ctx = "\n".join([f"[{d['category']}/{d['source_file']}] {d['content']}" for d in manual_docs]) if route == "manual" else "\n".join([f"[{w['title']}] {w['snippet']}" for w in web_docs])
                res = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "너는 드론 전문 엔지니어링 어시스턴트야. 제공된 규격과 법규 자료에 근거하여 명확하고 신뢰성 있게 한국어로 답변해."},
                        {"role": "user", "content": f"질문: {user_prompt}\n\n근거 자료:\n{ctx}"}
                    ],
                    max_completion_tokens=800,
                    timeout=30.0
                )
                final_ans = res.choices[0].message.content
                st.markdown(final_ans)

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
            "선택계통": selected_cat,
            "답변요약": final_ans[:120].replace("\n", " ") + "..."
        })
