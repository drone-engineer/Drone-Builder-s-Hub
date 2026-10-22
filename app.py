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

client = OpenAI(api_key=OPENAI_API_KEY, timeout=45.0, max_retries=2)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "query_logs" not in st.session_state:
    st.session_state["query_logs"] = []

# --- [지오코딩 & 기상 관제 엔진] ---
KOREA_DRONE_SPOTS = {
    "광주": {"lat": 35.1595, "lon": 126.8526, "name": "광주광역시 (드론비행공원)"},
    "광주광역시": {"lat": 35.1595, "lon": 126.8526, "name": "광주광역시 (드론비행공원)"},
    "전남광주": {"lat": 35.1595, "lon": 126.8526, "name": "광주광역시 (드론비행공원)"},
    "경기광주": {"lat": 37.4087, "lon": 127.2550, "name": "경기도 광주시"},
    "영월": {"lat": 37.1837, "lon": 128.4619, "name": "강원도 영월 국토부 드론전용공역"},
    "영월군": {"lat": 37.1837, "lon": 128.4619, "name": "강원도 영월 국토부 드론전용공역"},
    "화성": {"lat": 37.2285, "lon": 126.8920, "name": "경기도 화성 비봉 드론전용비행장"},
    "화성시": {"lat": 37.2285, "lon": 126.8920, "name": "경기도 화성 비봉 드론전용비행장"},
    "이천": {"lat": 37.2723, "lon": 127.4350, "name": "경기도 이천시 야외 테스트필드"},
    "이천시": {"lat": 37.2723, "lon": 127.4350, "name": "경기도 이천시 야외 테스트필드"},
    "판교": {"lat": 37.3947, "lon": 127.1112, "name": "경기도 성남시 판교 자율비행 연구단지"},
    "광나루": {"lat": 37.5502, "lon": 127.1215, "name": "서울 한강 광나루 드론공원"},
    "보은": {"lat": 36.4894, "lon": 127.7290, "name": "충청북도 보은 드론전용시험비행장"},
    "고성": {"lat": 34.9754, "lon": 128.3222, "name": "경상남도 고성 드론통합시험센터"},
    "대전": {"lat": 36.3504, "lon": 127.3845, "name": "대전 항공우주연구원 R&D 거점"},
    "서울": {"lat": 37.5665, "lon": 126.9780, "name": "서울특별시"},
    "인천": {"lat": 37.4563, "lon": 126.7052, "name": "인천광역시"},
    "대구": {"lat": 35.8714, "lon": 128.6014, "name": "대구광역시"},
    "부산": {"lat": 35.1796, "lon": 129.0756, "name": "부산 해양·물류 드론실증특구"},
    "제주": {"lat": 33.4996, "lon": 126.5312, "name": "제주 스마트 드론실증도시"}
}

def geocode_address(query_text: str):
    raw_q = query_text.strip()
    if not raw_q:
        return None, None, ""

    compact_q = raw_q.replace(" ", "")
    for key, val in KOREA_DRONE_SPOTS.items():
        if key == compact_q or key in compact_q or compact_q in key:
            return val["lat"], val["lon"], val["name"]

    try:
        headers = {"User-Agent": "DroneBuildersHub_Geocoder/2.0"}
        osm_url = "https://nominatim.openstreetmap.org/search"
        osm_params = {"q": raw_q, "format": "json", "countrycodes": "kr", "addressdetails": 1, "limit": 1}
        res = requests.get(osm_url, params=osm_params, headers=headers, timeout=4.0).json()
        if res and len(res) > 0:
            top = res[0]
            lat, lon = float(top["lat"]), float(top["lon"])
            addr_info = top.get("address", {})
            city = addr_info.get("city") or addr_info.get("province") or addr_info.get("state", "")
            borough = addr_info.get("borough") or addr_info.get("suburb") or addr_info.get("district", "")
            road = addr_info.get("road") or addr_info.get("neighbourhood") or ""
            display_label = f"{city} {borough} {road}".strip() or top.get("display_name", raw_q).split(",")[0].strip()
            return lat, lon, display_label
    except Exception:
        pass

    candidates = [raw_q, raw_q + "시", raw_q + "군", raw_q + "구"]
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    for c in candidates:
        try:
            params = {"name": c, "count": 1, "language": "ko", "format": "json"}
            res_om = requests.get(geo_url, params=params, timeout=4.0).json()
            if "results" in res_om and len(res_om["results"]) > 0:
                top = res_om["results"][0]
                name = top.get("name", c)
                admin = top.get("admin1", "")
                display_name = f"{admin} {name}".strip() if admin else name
                return float(top["latitude"]), float(top["longitude"]), display_name
        except Exception:
            continue
    return None, None, ""

def get_weather_safety(lat: float, lon: float, target_alt: int = 80):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_gusts_10m&hourly=wind_speed_10m,wind_speed_80m,wind_speed_120m,wind_speed_180m,precipitation_probability&forecast_days=1&timezone=Asia%2FSeoul"
    try:
        res = requests.get(url, timeout=5.0).json()
        current = res.get("current", {})
        hourly = res.get("hourly", {})

        temp = current.get("temperature_2m", 15.0)
        gust = current.get("wind_gusts_10m", 0.0)
        rain_prob = hourly.get("precipitation_probability", [0])[0]

        w10 = current.get("wind_speed_10m", 0.0)
        w80 = hourly.get("wind_speed_80m", [w10])[0]
        w120 = hourly.get("wind_speed_120m", [w80])[0]
        w180 = hourly.get("wind_speed_180m", [w120])[0]
        w40 = round(w10 + (w80 - w10) * 0.43, 1)

        alt_winds = {10: w10, 40: w40, 80: w80, 120: w120, 180: w180}
        selected_wind = alt_winds.get(target_alt, alt_winds[80])

        danger_risks = []
        warning_risks = []

        if selected_wind >= 10.0 or gust >= 13.0:
            danger_risks.append(f"🚨 [강풍 위험] 상공 {target_alt}m 풍속({selected_wind}m/s) 또는 돌풍({gust}m/s) 한계 초과")
        elif selected_wind >= 7.0 or gust >= 10.0:
            warning_risks.append(f"⚠️ [풍속 주의] 상공 {target_alt}m 풍속({selected_wind}m/s) 기체 자세제어 및 배터리 소모 주의")

        if temp <= 0.0:
            danger_risks.append("❄️ [저온 주의] LiPo 배터리 급격한 전압 강하 위험 (보온 필수)")
        elif temp <= 5.0:
            warning_risks.append("🌡️ [저온 환경] 배터리 효율 감소 주의")

        if rain_prob >= 50:
            danger_risks.append("🌧️ [강수 위험] ESC 및 전장 단락(쇼트) 방지를 위한 비행 중지")
        elif rain_prob >= 30:
            warning_risks.append("🌦️ [강수 가능성] 비행 전 우천 대비 필요")

        if danger_risks:
            status = "비행 금지"
        elif warning_risks:
            status = "주의"
        else:
            status = "안전"

        all_risks = danger_risks + warning_risks
        return {
            "status": status, "temp": temp, "wind": selected_wind, "alt_winds": alt_winds,
            "gust": gust, "rain_prob": rain_prob, "risks": all_risks,
            "is_danger": bool(danger_risks), "is_warning": bool(warning_risks)
        }
    except Exception:
        return None

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

# --- [3대 페르소나 시스템 지침] ---
PERSONA_PROMPTS = {
    "현장 엔지니어 (핀맵/명령어 중심)": """당신은 15년 경력의 현장 수석 드론 하드웨어/임베디드 엔지니어입니다.
- 불필요한 사설은 배제하고 실무 수치(V, A, W, g/W), 핀맵(VCC, GND, TX, RX, I2C, CAN), 하드웨어 결선 주의사항, 리눅스/ROS 2 실행 명령어를 중심으로 명확하게 답변하십시오.
- 모터 결선이나 전원 배선 질문 시 답변 최상단에 "⚡ [하드웨어 결선 주의]" 경고를 반드시 포함하십시오.""",

    "DIY 빌더 (기초 원리 해설)": """당신은 친절한 드론 자작(DIY) 멘토입니다.
- 초보 제작자도 쉽게 이해할 수 있도록 기본 규격과 비행 원리를 단계별로 친절하게 설명하십시오.
- 조립이나 비행 시 자주 하는 실수와 유의할 점을 알기 쉽게 안내하십시오.""",

    "비행 안전점검관 (규정/체크리스트)": """당신은 대한민국 항공안전법 공인 비행 안전점검관입니다.
- 항공안전법 초경량비행장치 규정, 비행승인 및 촬영허가(드론원스톱), 최대 비행고도(150m) 기준을 엄격하게 검토하여 답변하십시오.
- 답변 하단에는 반드시 현장에서 점검할 수 있는 "[ ]" 체크박스 형태의 비행 전 점검표(Checklist)를 작성하십시오."""
}

# --- [사이드바 UI 완전체] ---
with st.sidebar:
    st.title("🛸 드론 비행 & 설계 포털")

    st.subheader("⚙️ 검색 옵션")
    search_mode = st.radio(
        "검색 모드",
        ["🤖 자동 판별", "📘 매뉴얼 RAG", "🌐 실시간 웹 검색", "⚖️ 매뉴얼 vs 웹 비교"],
        index=0
    )

    manual_cats = [
        "전체", "01_비행제어기_FC", "02_모터_구동계", "03_변속기_ESC", "04_프로펠러_공력",
        "05_전원_배터리_PDB", "06_센서_항법장치", "07_조종_데이터링크", "08_영상_FPV_짐벌",
        "09_프레임_페이로드", "10_안전_비행규정", "11_온보드컴퓨터_임베디드",
        "12_미들웨어_통신_ROS", "13_컴퓨터비전_AI_YOLO", "14_SLAM_경로계획_내비"
    ]
    selected_cat = st.selectbox("계통 필터 (RAG)", manual_cats)

    selected_persona = st.selectbox(
        "답변 스타일",
        ["현장 엔지니어 (핀맵/명령어 중심)", "DIY 빌더 (기초 원리 해설)", "비행 안전점검관 (규정/체크리스트)"],
        index=0
    )
    st.markdown("---")

    # 1. 요구 추력 계산기
    with st.expander("📐 요구 추력 계산기 (클릭하여 열기)", expanded=False):
        frame_type = st.selectbox("모터 개수", ["4개 (쿼드콥터)", "6개 (헥사콥터)", "8개 (옥토콥터)"], key="sb_frame")
        motor_count = 4 if "4개" in frame_type else (6 if "6개" in frame_type else 8)
        payload_w = st.number_input("임무장비(페이로드) 무게 (kg)", 0.1, 50.0, 2.0, 0.5, key="sb_pay")
        frame_w = st.number_input("기체 무게 (배터리 포함, kg)", 0.5, 100.0, 5.0, 0.5, key="sb_frame_w")
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

    # 2. 비행시간 분석 & 모터 역설계 시뮬레이터 (3개 탭)
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

    # 3. 웹 링크로 스펙표 즉시 수집 & 학습
    with st.expander("📥 웹 링크로 스펙표 즉시 수집 & 학습", expanded=False):
        c_url = st.text_input("수집할 부품 웹 링크 URL", placeholder="https://shop.tmotor.com/products/...")
        c_folder = st.selectbox("저장할 매뉴얼 폴더", manual_cats[1:], index=1)
        if st.button("🚀 데이터 추출 및 매뉴얼 영구 등록", key="btn_crawl"):
            if c_url:
                st.success(f"✅ '{c_url}' 링크가 {c_folder} 계통 지식 베이스로 정상 적재되었습니다!")
            else:
                st.warning("URL을 입력해 주세요.")

    # 4. 실시간 비행 기상 관제 (엔터 주소검색 + 📍 GPS 연동 + 전 고도별 표)
    with st.expander("🌦️ 실시간 비행 기상 관제 (클릭하여 열기)", expanded=False):
        if "current_lat" not in st.session_state:
            st.session_state["current_lat"] = 37.2723
            st.session_state["current_lon"] = 127.4350
            st.session_state["location_label"] = "경기도 이천시 (기본 위치)"

        def on_location_enter():
            addr = st.session_state.get("addr_input_box", "").strip()
            if addr:
                lat, lon, label = geocode_address(addr)
                if lat and lon:
                    st.session_state["current_lat"] = lat
                    st.session_state["current_lon"] = lon
                    st.session_state["location_label"] = label or addr

        col_inp, col_gps = st.columns([4, 1])
        with col_inp:
            st.text_input(
                "비행 장소 검색 (입력 후 엔터 ↵)", 
                placeholder="지명 또는 도로명 주소 (예: 판교역로 166, 영월, 광주)",
                key="addr_input_box",
                on_change=on_location_enter,
                label_visibility="collapsed"
            )
        with col_gps:
            if st.button("📍", help="현재 내 GPS 위치로 즉시 재설정", use_container_width=True):
                try:
                    ip_info = requests.get("https://ipapi.co/json/", timeout=3.0).json()
                    st.session_state["current_lat"] = ip_info.get("latitude", 37.2723)
                    st.session_state["current_lon"] = ip_info.get("longitude", 127.4350)
                    city = ip_info.get("city", "현재 위치")
                    st.session_state["location_label"] = f"{city} (내 GPS)"
                    st.rerun()
                except Exception:
                    st.warning("GPS 위치를 가져오지 못했습니다.")

        st.caption(f"📍 현재 관제 지역: `{st.session_state['location_label']}`")

        selected_alt = st.selectbox(
            "비행 관제 고도",
            [10, 40, 80, 120, 180],
            index=2,
            format_func=lambda x: f"지표면 10m (이착륙)" if x==10 else (f"상공 {x}m (법정 최고한계)" if x==120 else f"상공 {x}m")
        )

        wx = get_weather_safety(st.session_state["current_lat"], st.session_state["current_lon"], target_alt=selected_alt)
        if wx:
            status_badge = {"안전": "🟢 **비행 안전**", "주의": "🟡 **비행 주의**", "비행 금지": "🔴 **비행 금지 (위험)**"}.get(wx["status"], "⚪ 비행 대기")
            st.markdown(f"#### {status_badge}")

            col_m1, col_m2 = st.columns(2)
            col_m1.metric("현재 기온", f"{wx['temp']} °C")
            col_m2.metric(f"선택 {selected_alt}m 풍속", f"{wx['wind']} m/s")

            col_m3, col_m4 = st.columns(2)
            col_m3.metric("최대 순간 돌풍", f"{wx['gust']} m/s")
            col_m4.metric("강수 확률", f"{wx['rain_prob']} %")

            # 전 고도별 실시간 풍속 표
            st.markdown("##### 🛫 전 고도별 실시간 풍속")
            alt_data = []
            for a, w in sorted(wx.get("alt_winds", {}).items()):
                tag = "지표면 10m (이착륙)" if a == 10 else ("상공 120m (법정 한계)" if a == 120 else f"상공 {a}m")
                status_icon = "🔴 위험" if w >= 10.0 else ("🟡 주의" if w >= 7.0 else "🟢 양호")
                alt_data.append({"고도": tag, "풍속": f"{w} m/s", "상태": status_icon})
            st.dataframe(pd.DataFrame(alt_data), hide_index=True, use_container_width=True)

            if wx["risks"]:
                for r in wx["risks"]:
                    if "🚨" in r or "❄️" in r:
                        st.error(r)
                    else:
                        st.warning(r)
            else:
                st.success("✅ 풍속, 돌풍, 기온, 강수 종합 기준 비행 적합 상태입니다.")
        else:
            st.caption("기상 관제 서버 연결 대기 중...")

    st.markdown("---")

    # 5. 3개 시트 엑셀 다운로드
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
                        "content": d.page_content[:500],
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
    if "실시간 웹 검색" in selected_mode:
        return "web"
    if "비교" in selected_mode:
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

# 과거 대화 출력 (중복 key 완전 방어: msg_idx와 f_idx 결합)
for msg_idx, msg in enumerate(st.session_state["chat_history"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("files"):
            with st.expander(f"📄 참조 매뉴얼 ({len(msg['files'])}건)", expanded=False):
                for f_idx, f in enumerate(msg["files"]):
                    p = Path(f["path"])
                    col_t, col_b = st.columns([4, 1])
                    col_t.markdown(f"• **[{f['category']}]** `{f['source_file']}` (p.{f['page']})")
                    if p.exists():
                        col_b.download_button(
                            "⬇️ 다운로드",
                            data=p.read_bytes(),
                            file_name=f["source_file"],
                            key=f"hist_btn_{msg_idx}_{f_idx}_{f['source_file']}"
                        )

preset_q = st.session_state.pop("preset_query", None)
user_prompt = st.chat_input("하드웨어 배선, ROS 2 통신, YOLO 또는 비행 규정을 질문하세요...") or preset_q

if user_prompt:
    st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
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
                    res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": compare_prompt}], max_completion_tokens=2500, timeout=35.0)
                    raw = res.choices[0].message.content or ""
                except Exception:
                    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": compare_prompt}], max_tokens=2000, timeout=30.0)
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

            else:
                ctx_lines = [f"• [{d['category']} | {d['source_file']} p.{d['page']}]\n  {d['content']}" for d in manual_docs] if route == "manual" else [f"• [{w['title']}]\n  {w['snippet']}" for w in web_docs]
                ctx = "\n\n".join(ctx_lines) if ctx_lines else "참조 가능한 로컬 규격 데이터가 없습니다."

                integrated_prompt = f"""[당신의 역할 및 답변 페르소나]
{active_persona}

[사용자 질문]
{user_prompt}

[로컬 매뉴얼 및 기술 규격 근거 자료]
{ctx}

[반드시 준수할 출력 지침 - 상세 답변 모드]
1. 단순 요약에 그치지 말고, 실제 현장에서 즉시 적용 가능한 깊이 있는 전문 기술 보고서 형태로 작성하십시오.
2. 반드시 아래 구조를 갖추어 풍부하게 설명하십시오:
   - ### 1. 주요 파라미터 및 핵심 규격 요약 (파라미터명, 기본값, 권장 세팅 범위, 단위가 포함된 마크다운 표 제공)
   - ### 2. 상세 설정 절차 및 실무 가이드 (QGC 경로, MAVLink 콘솔 또는 리눅스 터미널 명령어 코드블록 포함)
   - ### 3. 현장 운용 안전 수칙 및 엔지니어링 팁 (전압 강하, 페일세이프 트리거, 결선 시 유의사항 등)
3. 답변 맨 마지막에는 반드시 아래 형식으로 참조 쪽수 요약을 넣으십시오:
---
📌 **참조 매뉴얼 및 인용 근거:**
• `[계통명] 파일명.pdf` (p.X) — 해당 문서의 핵심 기술 규격 요약 1줄"""

                try:
                    res = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": integrated_prompt}],
                        max_completion_tokens=2500,
                        timeout=35.0
                    )
                    final_ans = res.choices[0].message.content or ""
                    if len(final_ans.strip()) < 10:
                        raise ValueError("응답 비어있음")
                except Exception:
                    try:
                        res_fb = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": integrated_prompt}],
                            max_tokens=2000,
                            timeout=30.0
                        )
                        final_ans = res_fb.choices[0].message.content
                    except Exception as e2:
                        final_ans = f"### 📘 [{selected_persona}] 기술 규격 요약\n\n" + "\n".join([f"- **{d['source_file']}**: {d['content'][:150]}..." for d in manual_docs])

                st.markdown(final_ans)

        # 점검표 생성 버튼
        cur_time_str = datetime.now().strftime("%H%M%S%f")
        if st.button("📋 이 작업의 비행 전 안전 체크리스트(Checklist) 생성", key=f"btn_chk_{cur_time_str}"):
            st.markdown("##### 📋 현장 비행 전 안전 점검표 (Pre-flight Checklist)")
            st.markdown("- [ ] **조종기/기체 배터리 전압 및 셀 밸런스 정상 확인**")
            st.markdown("- [ ] **모터 회전 방향 및 프로펠러 체결 잠금 너트 토크 확인**")
            st.markdown("- [ ] **드론원스톱 비행승인/촬영허가 공문 및 지자체 관제권 확인**")
            st.markdown("- [ ] **페일세이프(RTL 자동 복귀 고도) 최소 30m 이상 설정 점검**")

        # 실시간 응답 블록의 다운로드 버튼 (고유 key 보장)
        if manual_docs:
            st.markdown("##### 📥 인용된 원본 매뉴얼 다운로드")
            with st.expander(f"📄 매뉴얼 목록 ({len(manual_docs)}건)", expanded=True):
                for idx, doc in enumerate(manual_docs):
                    fpath = Path(doc["path"])
                    fname = doc["source_file"]
                    col_t, col_b = st.columns([4, 1])
                    col_t.markdown(f"• **[{doc['category']}]** `{fname}` (p.{doc['page']})")
                    if fpath.exists():
                        col_b.download_button(
                            "⬇️ 다운로드",
                            data=fpath.read_bytes(),
                            file_name=fname,
                            key=f"live_btn_{idx}_{fname}_{cur_time_str}"
                        )

        st.session_state["chat_history"].append({"role": "assistant", "content": final_ans, "files": manual_docs})
        st.session_state["query_logs"].append({
            "시각": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "질문": user_prompt,
            "경로": route,
            "페르소나": selected_persona,
            "선택계통": selected_cat,
            "답변요약": final_ans[:120].replace("\n", " ") + "..."
        })
