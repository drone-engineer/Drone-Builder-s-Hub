import os
import io
import json
import base64
import re
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TypedDict
from bs4 import BeautifulSoup
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from langgraph.graph import StateGraph, START, END

# ==============================================================================
# 1. API 키 및 환경 변수 로드
# ==============================================================================
load_dotenv()

def read_secret(key_name, default=""):
    if key_name in st.secrets:
        return str(st.secrets[key_name]).strip()
    return os.getenv(key_name, default).strip()

OPENAI_API_KEY = read_secret("OPENAI_API_KEY")
TAVILY_API_KEY = read_secret("TAVILY_API_KEY") or read_secret("TAVILY_KEY")
MODEL = read_secret("OPENAI_MODEL", "gpt-6-luna")

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("운영팀에서_"):
    st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .streamlit/secrets.toml을 확인해 주세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

# ==============================================================================
# 2. 국내 대표 드론 비행 거점 사전 및 지오코딩 엔진
# ==============================================================================
KOREA_DRONE_SPOTS = {
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
    "부산": {"lat": 35.1796, "lon": 129.0756, "name": "부산 해양·물류 드론실증특구"},
    "제주": {"lat": 33.4996, "lon": 126.5312, "name": "제주 스마트 드론실증도시"}
}

def geocode_address(query_text: str):
    raw_q = query_text.strip()
    if not raw_q:
        return None, None, ""

    compact_q = raw_q.replace(" ", "")

    # 1. 공인 드론 공역 및 주요 거점 우선 매칭
    for key, val in KOREA_DRONE_SPOTS.items():
        if key == compact_q or key in compact_q or compact_q in key:
            return val["lat"], val["lon"], val["name"]

    # 2. 국내 상세 도로명/지번 주소 정밀 검색 (OpenStreetMap Nominatim 한국 엔진)
    try:
        headers = {"User-Agent": "DroneBuildersHub_Geocoder/2.0 (contact: admin@dronehub.local)"}
        osm_url = "https://nominatim.openstreetmap.org/search"
        osm_params = {
            "q": raw_q,
            "format": "json",
            "countrycodes": "kr",
            "addressdetails": 1,
            "limit": 1
        }
        res = requests.get(osm_url, params=osm_params, headers=headers, timeout=5).json()
        if res and len(res) > 0:
            top = res[0]
            lat = float(top["lat"])
            lon = float(top["lon"])

            # 주소 정보 한글 정제
            addr_info = top.get("address", {})
            city = addr_info.get("city") or addr_info.get("province") or addr_info.get("state", "")
            borough = addr_info.get("borough") or addr_info.get("suburb") or addr_info.get("district", "")
            road = addr_info.get("road") or addr_info.get("neighbourhood") or ""

            if road:
                display_label = f"{city} {borough} {road}".strip()
            elif borough:
                display_label = f"{city} {borough}".strip()
            else:
                display_label = top.get("display_name", raw_q).split(",")[0].strip()

            return lat, lon, display_label
    except Exception:
        pass

    # 3. Open-Meteo 폴백 검색 (도시/행정구역명 보정)
    candidates = [raw_q, raw_q + "시", raw_q + "군", raw_q + "구"]
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    for c in candidates:
        try:
            params = {"name": c, "count": 1, "language": "ko", "format": "json"}
            res_om = requests.get(geo_url, params=params, timeout=4).json()
            if "results" in res_om and len(res_om["results"]) > 0:
                top = res_om["results"][0]
                name = top.get("name", c)
                admin = top.get("admin1", "")
                display_name = f"{admin} {name}".strip() if admin else name
                return float(top["latitude"]), float(top["longitude"]), display_name
        except Exception:
            continue

    return None, None, ""

def get_weather_safety(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_gusts_10m&hourly=wind_speed_80m,wind_speed_120m,precipitation_probability&forecast_days=1"
    try:
        res = requests.get(url, timeout=5).json()
        current = res.get("current", {})
        hourly = res.get("hourly", {})

        temp = current.get("temperature_2m", 20.0)
        w_10m = current.get("wind_speed_10m", 0.0)
        gust = current.get("wind_gusts_10m", 0.0)
        w_80m = hourly.get("wind_speed_80m", [w_10m])[0]
        w_120m = hourly.get("wind_speed_120m", [w_10m])[0]
        rain_prob = hourly.get("precipitation_probability", [0])[0]

        risks = []
        if w_80m >= 10.0 or gust >= 13.0:
            risks.append("🚨 [강풍 위험] 상공 80m 풍속/돌풍 한계 초과 (기체 자세제어 상실 우려)")
        if temp <= 0.0:
            risks.append("❄️ [저온 주의] LiPo 배터리 급격한 전압 강하 위험")
        if rain_prob >= 40:
            risks.append("🌧️ [강수 위험] ESC 및 전장 단락(쇼트) 방지를 위한 비행 중지 권고")

        status = "안전" if not risks else ("주의" if len(risks) == 1 else "비행 금지")
        return {
            "status": status, "temp": temp, "w_10m": w_10m, "w_80m": w_80m,
            "w_120m": w_120m, "gust": gust, "rain_prob": rain_prob, "risks": risks
        }
    except Exception:
        return None

# ==============================================================================
# 3. 드론 14개 계통 PDF + Markdown 통합 RAG 인덱서
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_drone_vector_store():
    """디스크에 구워둔 FAISS 인덱스를 0.05초 만에 즉시 로드 (질문 지연 원천 제거)"""
    idx_path = Path("faiss_index")
    if not idx_path.exists():
        idx_path = Path.cwd().parent / "faiss_index"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    if idx_path.exists() and (idx_path / "index.faiss").exists():
        try:
            return FAISS.load_local(str(idx_path), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"인덱스 로드 오류: {e}")
    return None


def route_relevant_manuals_by_catalog(query: str, max_docs: int = 2):
    """
    1단계: 마스터 목차(manual_catalog.json)를 스캔하여 질문과 관련성이 가장 높은 문서 1~2개만 초고속 선별
    """
    catalog_path = Path("manual_catalog.json")
    if not catalog_path.exists():
        catalog_path = Path.cwd().parent / "manual_catalog.json"

    if not catalog_path.exists():
        return []

    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    q_lower = query.lower()
    scored_docs = []

    for item in catalog:
        score = 0
        # 1. 파일명 및 카테고리 매칭 점수
        if any(term in item["filename"].lower() for term in q_lower.split()):
            score += 3
        if any(term in item["category"].lower() for term in q_lower.split()):
            score += 2
        # 2. 목차 키워드 매칭 점수
        for topic in item.get("topics", []):
            if topic in q_lower:
                score += 2

        if score > 0:
            scored_docs.append((score, item))

    # 점수 높은 순으로 상위 max_docs개 선정
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    selected = [doc["relative_path"] for score, doc in scored_docs[:max_docs]]
    return selected


def retrieve_drone_manuals(query: str, category_filter: str = "전체", k: int = 4):
    """
    2단계: 목차에서 선별된 문서가 있다면 해당 문서 위주로 국소 정밀 검색 (1초 내 완료)
    """
    manuals_path = Path("manuals")
    if not manuals_path.exists():
        manuals_path = Path.cwd().parent / "manuals"

    # 1단계 목차 라우팅 실행
    target_rel_files = route_relevant_manuals_by_catalog(query, max_docs=2)

    results = []

    # 선별된 파일이 있으면 해당 파일들의 본문만 직접 빠르게 읽어 청크 구성
    if target_rel_files:
        for rel_path in target_rel_files:
            target_f = manuals_path / rel_path
            if target_f.exists() and target_f.suffix.lower() == ".md":
                try:
                    content = target_f.read_text(encoding="utf-8")
                    # 질문 관련 핵심 문단 위주로 300자 발췌
                    results.append({
                        "category": target_f.parent.name,
                        "source_file": target_f.name,
                        "file_path": str(target_f),
                        "page": 1,
                        "content": content[:350]
                    })
                except Exception:
                    pass

    # 결과가 부족하면 기존 벡터스토어에서 k=3개 보완
    if len(results) < 2:
        db = get_drone_vector_store()
        if db is not None:
            try:
                extra_docs = db.similarity_search(query, k=3)
                for d in extra_docs:
                    results.append({
                        "category": d.metadata.get("category", "일반"),
                        "source_file": d.metadata.get("source_file", "문서"),
                        "file_path": d.metadata.get("file_path", ""),
                        "page": d.metadata.get("page", 1),
                        "content": d.page_content[:300]
                    })
            except Exception:
                pass

    return results[:k]


def extract_search_keyword(raw_text: str) -> str:
    text = raw_text.lower()

    thrust_match = re.search(r'최대\s*추력\s*([0-9.]+)\s*kg', text) or re.search(r'추력\s*([0-9.]+)\s*kg', text)
    thrust_val = f"{thrust_match.group(1)}kg" if thrust_match else ""

    payload_match = re.search(r'페이로드\s*([0-9.]+)\s*kg', text)
    payload_val = f"{payload_match.group(1)}kg" if payload_match else ""

    keywords = ["drone", "multicopter"]
    if "모터" in text or "motor" in text:
        keywords.append("brushless motor")
    if "파워콤보" in text or "추진" in text or "propulsion" in text:
        keywords.append("propulsion combo")
    if "esc" in text or "변속기" in text:
        keywords.append("ESC")
    if "프롭" in text or "프로펠러" in text:
        keywords.append("propeller")

    if thrust_val:
        keywords.append(f"{thrust_val} thrust")
    elif payload_val:
        keywords.append(f"{payload_val} payload")
    else:
        clean_words = [w for w in re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text).split() if len(w) > 1][:4]
        keywords.extend(clean_words)

    return " ".join(keywords)

def search_drone_tech(query: str, max_results: int = 3):
    """Tavily 초고속 AI 검색 (basic 모드 + 6초 타임아웃 방어)"""
    if not TAVILY_API_KEY:
        return []

    q_lower = query.lower()
    headers = {"Content-Type": "application/json"}
    url = "https://api.tavily.com/search"

    # 검색 쿼리 간소화 (지나치게 긴 쿼리 축약)
    clean_q = query.strip()
    if len(clean_q) > 60:
        clean_q = clean_q[:60]

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": clean_q,
        "search_depth": "basic",  # 고속 1.5초 모드
        "max_results": max_results
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=6)
        if res.status_code == 200:
            results = res.json().get("results", [])
            clean_results = []
            for r in results:
                clean_results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "link": r.get("url", "")
                })
            return clean_results
    except Exception as e:
        print(f"Tavily 타임아웃/오류 방어: {e}")
    return []

def format_section_to_markdown(data):
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        return "\n".join([f"• {format_section_to_markdown(item)}" for item in data])
    if isinstance(data, dict):
        lines = []
        for key, val in data.items():
            lines.append(f"#### 📌 {key}")
            if isinstance(val, list):
                for item in val:
                    lines.append(f"- {item}")
            elif isinstance(val, dict):
                lines.append(format_section_to_markdown(val))
            else:
                lines.append(f"{val}\n")
            lines.append("")
        return "\n".join(lines).strip()
    return str(data)

# ==============================================================================
# 5. 3대 페르소나별 상세 행동 지침 및 서식 정의
# ==============================================================================
PERSONA_PROMPTS = {
    "현장 엔지니어 (핀맵/명령어 중심)": """당신은 15년 차 베테랑 드론 현장 엔지니어입니다.
[답변 스타일]
- 군더더기 서론/결론을 일절 생략하고 기술 스펙, 물리 수치, 핀맵, 결선 기준을 최우선으로 직설 서술하세요.
- 결선 정보(VCC, GND, 5V, 12V, CAN, UART, PWM 신호선)와 CLI 설정 명령어를 반드시 코드블록이나 표로 명확히 제시하세요.
- 전류(A), 최대 추력(kg/g), 배터리 셀 수(S) 등 한계치를 수치로 명시하세요.""",

    "DIY 빌더 (기초 원리 해설)": """당신은 드론 제작 커뮤니티의 친절한 수석 빌더 멘토입니다.
[답변 스타일]
- 초보 제작자도 원리를 쉽게 이해할 수 있도록 어려운 용어는 쉬운 비유와 함께 단계별로 풀어서 설명하세요.
- 왜 이 모터와 프롭, 변속기를 조합해야 하는지(KV값과 프롭 크기의 상관관계, 발열 원인 등) 기초 원리를 친절하게 해설하세요.
- 입문자가 조립 시 흔히 저지르는 실수(역회전 프롭 장착, 냉땜, 전원 쇼트) 방지 팁을 덧붙이세요.""",

    "비행 안전점검관 (규정/체크리스트)": """당신은 국토교통부 표준 및 항공안전기술원 기준의 비행 안전점검관입니다.
[답변 스타일]
- 안전성과 법적 규정 준수를 최우선으로 삼아 보수적이고 엄격한 어조로 평가하세요.
- 총 이륙중량(AUW) 대비 안전 여유율(호버링 추력비 2:1 이상), 배터리 방전 위험, ESC 허용 전류 마진을 엄격히 검토하세요.
- 답변 말미에 현장에서 바로 대조할 수 있는 '비행 전 필수 안전 점검 체크리스트(- [ ] )'를 반드시 3~4개 포함하세요."""
}

class DroneQAState(TypedDict, total=False):
    question: str
    selected_mode: str
    selected_category: str
    persona: str
    route: str
    manual_context: list
    web_results: list
    answer: str
    compare_sections: dict
    referenced_files: list
    error: str

def route_node(state: DroneQAState) -> DroneQAState:
    mode = state.get("selected_mode", "🤖 자동 판별")
    if mode == "📘 매뉴얼 RAG":
        return {"route": "manual"}
    elif mode == "🌐 실시간 웹 검색":
        return {"route": "web"}
    elif mode == "⚖️ 매뉴얼 vs 웹 비교":
        return {"route": "compare"}

    q = state.get("question", "").lower()
    compare_kw = ["비교", "차이", "vs", "상충", "다른점", "추천", "어떤게", "골라줘", "선정", "조합"]
    web_kw = ["최신", "시세", "가격", "출시", "버그", "깃허브", "github", "이슈", "과태료", "법령", "구매처", "날씨", "풍속"]
    manual_kw = ["모터", "추력", "esc", "변속기", "fc", "배선", "결선", "핀맵", "설정", "파라미터", "체크리스트", "ros", "yolo", "캘리브레이션"]

    if any(k in q for k in compare_kw):
        chosen = "compare"
    elif any(k in q for k in web_kw):
        chosen = "web"
    elif any(k in q for k in manual_kw):
        chosen = "manual"
    else:
        chosen = "manual"

    return {"route": chosen}

def manual_node(state: DroneQAState) -> DroneQAState:
    chunks = retrieve_drone_manuals(state.get("question", ""), category_filter=state.get("selected_category", "전체"))
    return {"manual_context": chunks}

def web_node(state: DroneQAState) -> DroneQAState:
    try:
        items = search_drone_tech(state.get("question", ""))
        return {"web_results": items, "error": ""}
    except Exception as e:
        return {"web_results": [], "error": str(e)}

def compare_node(state: DroneQAState) -> DroneQAState:
    chunks = retrieve_drone_manuals(state.get("question", ""), category_filter=state.get("selected_category", "전체"))
    try:
        items = search_drone_tech(state.get("question", ""))
        return {"manual_context": chunks, "web_results": items, "error": ""}
    except Exception as e:
        return {"manual_context": chunks, "web_results": [], "error": str(e)}

def answer_node(state: DroneQAState) -> DroneQAState:
    q = state.get("question", "")
    persona_choice = state.get("persona", "현장 엔지니어 (핀맵/명령어 중심)")
    persona_rule = PERSONA_PROMPTS.get(persona_choice, PERSONA_PROMPTS["현장 엔지니어 (핀맵/명령어 중심)"])
    route = state.get("route", "manual")
    manual_docs = state.get("manual_context", [])
    web_docs = state.get("web_results", [])

    referenced_files = []
    seen_f = set()
    for d in manual_docs:
        fp = d.get("file_path", "")
        if fp and fp not in seen_f and os.path.exists(fp):
            ext = Path(fp).suffix.lower()
            mime_type = "text/markdown" if ext == ".md" else "application/pdf"
            referenced_files.append({"name": d["source_file"], "path": fp, "mime": mime_type, "type": "md" if ext == ".md" else "pdf"})
            seen_f.add(fp)

    if route == "compare":
        m_lines = [f"[{d['category']} | {d['source_file']} p.{d['page']}] {d['content'][:400]}" for d in manual_docs]
        m_txt = "\n".join(m_lines) if m_lines else "로컬 매뉴얼 자료 없음"

        w_lines = [f"[{i}] {w['title']}: {w['snippet']}" for i, w in enumerate(web_docs, start=1)]
        w_txt = "\n".join(w_lines) if w_lines else "웹 검색 자료 없음"

        compare_prompt = f"""{persona_rule}

현재 작업: 사용자 질문에 대해 [로컬 매뉴얼 규격]과 [Tavily AI 실시간 기술 검색]을 대조 분석합니다.
질문: {q}

[로컬 매뉴얼 규격 자료]
{m_txt}

[Tavily AI 실시간 기술 검색 자료]
{w_txt}

반드시 위 페르소나 스타일과 출력 지침을 준수하여, 아래 3개 구분자(===)를 엄격히 지켜 답변을 상세히 작성하세요:

===ANALYSIS===
(매뉴얼 기준과 웹 시판 제품을 대조한 일치점, 차이점, 최종 파워콤보 선정 및 현장 권고사항을 페르소나 관점에서 체계적으로 서술)

===MANUAL===
(매뉴얼에 기재된 전압, 최대 전류, 추력표, 프로펠러 규격 등 핵심 기술 기준 요약)

===WEB===
(웹 검색을 통해 파악된 시판 파워시스템 모델명, 스펙 특징 및 최신 동향 요약, 문장 끝에 [1], [2] 출처 번호 명시)"""

        try:
            res_fast = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": compare_prompt}]
            )
            raw_text = res_fast.choices[0].message.content.strip()

            if "===ANALYSIS===" in raw_text:
                part_analysis = raw_text.split("===ANALYSIS===")[-1].split("===MANUAL===")[0].strip()
            else:
                part_analysis = raw_text

            if "===MANUAL===" in raw_text:
                part_manual = raw_text.split("===MANUAL===")[-1].split("===WEB===")[0].strip()
            else:
                part_manual = "매뉴얼 규격 분석 요약이 완료되었습니다."

            if "===WEB===" in raw_text:
                part_web = raw_text.split("===WEB===")[-1].strip()
            else:
                part_web = "웹 검색 스펙 분석이 완료되었습니다."

        except Exception as e:
            part_analysis = f"비교 분석 생성 중 오류 발생: {e}"
            part_manual = "매뉴얼 데이터를 불러오는 중 오류가 발생했습니다."
            part_web = "웹 검색 데이터를 불러오는 중 오류가 발생했습니다."

        if locals().get('web_docs'):
            links = [f"**[{i}] {w['title']}**  \n{w['link']}" for i, w in enumerate(web_docs, start=1)]
            part_web += "\n\n---\n**🌐 Tavily AI 실시간 기술 출처:**\n\n" + "\n\n".join(links)

        return {
            "route": "compare",
            "answer": part_analysis,
            "compare_sections": {
                "analysis": part_analysis,
                "manual": part_manual,
                "web": part_web
            },
            "referenced_files": referenced_files
        }

    context = []
    if locals().get('manual_docs'):
        m_lines = [f"[{d['category']} | {d['source_file']} p.{d['page']}] {d['content'][:400]}" for d in manual_docs]
        context.append("[로컬 기술 매뉴얼]\n" + "\n".join(m_lines))
    if locals().get('web_docs'):
        w_lines = [f"[{i}] {w['title']}: {w['snippet']}" for i, w in enumerate(web_docs, start=1)]
        context.append("[Tavily AI 기술 검색 결과]\n" + "\n".join(w_lines))

    system_rules = f"""당신은 드론 빌더스 허브(Drone Builder's Hub)의 전문 어시스턴트입니다.
{persona_rule}

[공통 필수 안전 지침]
1. 하드웨어 결선, 전원(VCC, GND, 5V, 12V), 프로펠러 장착 시 쇼트 및 비행 안전 수칙을 명시하세요.
2. 페이로드(유상하중)와 총 이륙중량(AUW), 호버링 추력비(최소 2:1 권장)를 고려하여 모터 1개당 요구 추력을 정확히 계산해 설명하세요.
3. 웹 검색 결과를 사용할 경우 문장 끝에 [1], [2] 번호를 명시하세요.
4. 원문이 외국어라도 한국어 표준 항공/전기전자 기술 용어로 정제하여 답변하세요.
"""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_rules},
                {"role": "user", "content": f"질문: {q}\n\n" + "\n\n".join(context)}
            ]
        )
        base_ans = res.choices[0].message.content.strip()

        footer = []
        if locals().get('manual_docs'):
            summary_prompt = """아래 매뉴얼 발췌본들을 바탕으로, 각 파일과 페이지에 적힌 핵심 사양을 '1줄의 한국어 요약문'으로 번역/정제하세요.
[발췌본]
""" + "\n".join([f"- `{d['category']}/{d['source_file']}` (p.{d['page']}): {d['content'][:150].replace(chr(10), ' ')}" for d in manual_docs]) + """
[출력 양식] 각 항목별로 아래 형식만 줄바꿈하여 출력하세요:
• `경로/파일명.pdf` (p.X) — [한국어로 번역 정제된 1줄 요약]
"""
            trans_res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": summary_prompt}])
            doc_page_map = {}
            if 'manual_docs' in locals() and manual_docs:
                for d in manual_docs:
                    k = (f"{d['category']}/{d['source_file']}", d.get('page', 1))
                    if k not in doc_page_map:
                        doc_page_map[k] = d['content'].replace('\n', ' ').replace('~', '-').strip()[:70]
            summary_items = [f"- `{doc}` (p.{pg}) — {snip.replace('~', '-')}..." for (doc, pg), snip in doc_page_map.items()]
            footer.append("📌 **참조 매뉴얼 및 페이지별 내용 요약:**\n\n" + "\n\n".join(summary_items))

        if locals().get('web_docs'):
            links = [f"**[{i}] {w['title']}**  \n{w['link']}" for i, w in enumerate(web_docs, start=1)]
            footer.append("**🌐 Tavily AI 실시간 기술 출처:**\n\n" + "\n\n".join(links))

        final_out = base_ans
        if footer:
            final_out += "\n\n---\n" + "\n\n".join(footer)
        return {"route": route, "answer": final_out, "referenced_files": referenced_files, "compare_sections": {}}
    except Exception as e:
        return {"route": route, "answer": f"답변 생성 오류: {e}", "referenced_files": [], "compare_sections": {}}

builder = StateGraph(DroneQAState)
builder.add_node("route_node", route_node)
builder.add_node("manual_node", manual_node)
builder.add_node("web_node", web_node)
builder.add_node("compare_node", compare_node)
builder.add_node("answer_node", answer_node)

builder.add_edge(START, "route_node")
builder.add_conditional_edges("route_node", lambda s: s.get("route", "manual"), {
    "manual": "manual_node", "web": "web_node", "compare": "compare_node"
})
builder.add_edge("manual_node", "answer_node")
builder.add_edge("web_node", "answer_node")
builder.add_edge("compare_node", "answer_node")
builder.add_edge("answer_node", END)
drone_agent = builder.compile()

# ==============================================================================
# 6. 실시간 질의 이력 로거 및 상시 엑셀 빌더
# ==============================================================================
LOG_FILE = Path("history_log.csv")

def append_query_log(entry: dict):
    df_new = pd.DataFrame([entry])
    if not LOG_FILE.exists():
        df_new.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")
    else:
        df_new.to_csv(LOG_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")

def generate_multi_sheet_excel():
    if LOG_FILE.exists():
        try:
            df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame(columns=["일시", "질문", "처리경로", "계통필터", "페르소나", "결선주의", "소요시간_초", "답변요약"])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="전체이력", index=False)
        df_danger = df[df["결선주의"] == "YES"] if "결선주의" in df.columns else pd.DataFrame(columns=df.columns)
        df_danger.to_excel(writer, sheet_name="결선주의질의", index=False)
        df_compare = df[df["처리경로"] == "compare"] if "처리경로" in df.columns else pd.DataFrame(columns=df.columns)
        df_compare.to_excel(writer, sheet_name="비교검증질의", index=False)
    output.seek(0)
    return output.getvalue()

# ==============================================================================
# 7. Streamlit 화면 렌더링
# ==============================================================================
st.set_page_config(page_title="드론 빌더스 허브", page_icon="🛸", layout="centered")

st.markdown("""
<style>
.block-container { max-width: 900px !important; padding-top: 1.2rem !important; margin: 0 auto !important; }
h1 { white-space: nowrap !important; }
div[data-testid="stSidebarContent"] { padding-top: 1rem !important; }
.route-badge {
    display: inline-block;
    padding: 0.25rem 0.65rem;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 6px;
    margin-bottom: 0.75rem;
}
.badge-manual { background-color: #E3F2FD; color: #0D47A1; border: 1px solid #BBDEFB; }
.badge-web { background-color: #E8F5E9; color: #1B5E20; border: 1px solid #C8E6C9; }
.badge-compare { background-color: #FFF3E0; color: #E65100; border: 1px solid #FFE0B2; }
</style>
""", unsafe_allow_html=True)

st.title("🛸 드론 빌더스 허브 (Drone Builder's Hub)")
st.caption("자율비행 드론 HW/SW 14개 계통 매뉴얼 RAG & Tavily AI 비행 안전 관제 포털")

with st.sidebar:
    st.subheader("⚙️ 검색 옵션")
    selected_mode = st.radio("검색 모드", ["🤖 자동 판별", "📘 매뉴얼 RAG", "🌐 실시간 웹 검색", "⚖️ 매뉴얼 vs 웹 비교"])
    selected_category = st.selectbox("계통 필터 (RAG)", ["전체", "01_비행제어기_FC", "02_모터_구동계", "03_변속기_ESC", "05_전원_배터리_PDB", "11_온보드컴퓨터_임베디드", "12_미들웨어_통신_ROS", "13_컴퓨터비전_AI_YOLO"])
    persona = st.selectbox("답변 스타일", ["현장 엔지니어 (핀맵/명령어 중심)", "DIY 빌더 (기초 원리 해설)", "비행 안전점검관 (규정/체크리스트)"])

    st.markdown("---")
    # ==============================================================================
    # 요구 추력 계산기
    # ==============================================================================
    with st.expander("📐 요구 추력 계산기 (클릭하여 열기)", expanded=False):
        st.caption("기체 중량과 임무장비에 필요한 모터당 적정 추력과 파워콤보 체급을 자동 산출합니다.")

        calc_motor_count = st.selectbox("모터 개수 (프레임 타입)", [4, 6, 8, 12], index=0, format_func=lambda x: f"{x}개 ({'쿼드' if x==4 else '헥사' if x==6 else '옥토' if x==8 else '도데카'}콥터)")
        calc_payload = st.number_input("임무장비(페이로드) 무게 (kg)", min_value=0.0, max_value=150.0, value=20.0, step=0.5)
        calc_frame_weight = st.number_input("기체 무게 (배터리·프레임 포함, kg)", min_value=0.5, max_value=150.0, value=8.0, step=0.5)
        calc_twr = st.slider("목표 추력비 (TWR, 권장 2.0:1)", min_value=1.5, max_value=3.5, value=2.0, step=0.1)

        auw = round(calc_payload + calc_frame_weight, 2)
        total_thrust_req = round(auw * calc_twr, 2)
        thrust_per_motor_max = round(total_thrust_req / calc_motor_count, 2)
        thrust_per_motor_hov = round(auw / calc_motor_count, 2)

        st.markdown("##### 📊 계산 결과")
        col_c1, col_c2 = st.columns(2)
        col_c1.metric("총 이륙중량 (AUW)", f"{auw} kg")
        col_c2.metric("요구 총 추력", f"{total_thrust_req} kg")

        col_c3, col_c4 = st.columns(2)
        col_c3.metric("모터당 최대 추력", f"{thrust_per_motor_max} kg", delta=f"{int(thrust_per_motor_max*1000)} g")
        col_c4.metric("호버링 추력(50%)", f"{thrust_per_motor_hov} kg", delta=f"{int(thrust_per_motor_hov*1000)} g")

        if thrust_per_motor_max < 1.5:
            rec_class = "💡 소형 기체 (5~7인치 / 4S~6S / 2207~2806급)"
        elif thrust_per_motor_max < 4.0:
            rec_class = "💡 중소형 기체 (13~18인치 / 6S LiPo / 4014~5010급)"
        elif thrust_per_motor_max < 9.0:
            rec_class = "💡 중대형 산업용 (21-24인치 / 12S LiPo / Hobbywing X6급, 6215-8120급)"
        elif thrust_per_motor_max < 16.0:
            rec_class = "💡 대형 헤비리프팅 (28~34인치 / 12S~14S / Hobbywing X8, T-Motor P80급)"
        else:
            rec_class = "💡 초대형 산업/특수기 (36~42인치 / 14S~18S / Hobbywing X9/X11급 이상)"

        st.info(rec_class)

        query_prompt = f"모터 {calc_motor_count}개짜리 멀티콥터이고 기체무게 {calc_frame_weight}kg, 페이로드 {calc_payload}kg(AUW {auw}kg)이야. 모터 1개당 최대 추력 {thrust_per_motor_max}kg(호버링 {thrust_per_motor_hov}kg)이 필요한데 적합한 파워콤보(모터, ESC, 프롭, 배터리)를 추천하고 스펙을 분석해줘"
        if st.button("💬 이 조건으로 AI 챗봇에 즉시 추천 요청", use_container_width=True):
            st.session_state["calc_query_send"] = query_prompt
            st.rerun()

    # ==============================================================================
    # ⏱️ 체공 시간 분석 & 역설계 모터 추천 시뮬레이터
    # ==============================================================================
    with st.expander("⏱️ 비행시간 분석 & 모터 역설계 시뮬레이터", expanded=False):
        calc_tab1, calc_tab2 = st.tabs(["🎯 목표시간 모터 역추천", "📊 순방향 체공시간 계산"])

        # --------------------------------------------------------------------------
        # 탭 1: 목표 비행시간 기준 모터/추진계 역설계
        # --------------------------------------------------------------------------
        with calc_tab1:
            st.caption("원하는 비행시간(예: 60분)을 설정하면, 이를 달성하기 위한 요구 효율(g/W)과 적합 모터 규격을 역산합니다.")

            rev_motor_cnt = st.selectbox("모터 개수", [4, 6, 8], index=0, format_func=lambda x: f"{x}개 ({'쿼드' if x==4 else '헥사' if x==6 else '옥토'}콥터)", key="rev_mc")
            col_rw1, col_rw2 = st.columns(2)
            with col_rw1:
                rev_dry_wt = st.number_input("기체 공허중량 (kg)", min_value=0.5, max_value=50.0, value=5.0, step=0.5, key="rev_dry")
            with col_rw2:
                rev_payload = st.number_input("임무장비 무게 (kg)", min_value=0.0, max_value=50.0, value=3.0, step=0.5, key="rev_pay")

            col_rb1, col_rb2 = st.columns(2)
            with col_rb1:
                rev_batt_s = st.selectbox("배터리 전압(S)", [6, 12, 14], index=1, format_func=lambda x: f"{x}S ({round(x*3.7, 1)}V)", key="rev_bs")
            with col_rb2:
                rev_batt_mah = st.number_input("배터리 용량 (mAh)", min_value=5000, max_value=120000, value=30000, step=2000, key="rev_bmah")

            rev_target_min = st.number_input("🎯 목표 비행시간 (분)", min_value=10, max_value=120, value=60, step=5, key="rev_tmin")

            # 리포 배터리 대략적 중량 산출 (12S 기준 약 0.19~0.22 kg / 1000mAh)
            est_batt_wt = round((rev_batt_mah / 1000.0) * (rev_batt_s / 12.0) * 0.205, 2)
            rev_auw = round(rev_dry_wt + rev_payload + est_batt_wt, 2)

            # 에너지 및 요구 전력 역산
            v_nom_rev = rev_batt_s * 3.7
            total_wh_rev = (rev_batt_mah * v_nom_rev) / 1000.0
            usable_wh_rev = total_wh_rev * 0.80 # 80% 안전 방전 기준

            # 목표 시간 달성을 위한 총 허용 소비전력 (W)
            max_allowed_power_w = usable_wh_rev / (rev_target_min / 60.0)
            motor_allowed_power_w = round(max_allowed_power_w / rev_motor_cnt, 1)

            # 축당 요구 호버링 추력 및 최소 요구 효율
            rev_thrust_per_motor_g = round((rev_auw * 1000.0) / rev_motor_cnt, 1)
            required_eff_gw = round(rev_thrust_per_motor_g / motor_allowed_power_w, 2) if motor_allowed_power_w > 0 else 0

            st.markdown("##### 📐 역산된 모터 스펙 요구조건")
            r_c1, r_c2 = st.columns(2)
            r_c1.metric("예상 총 이륙중량", f"{rev_auw} kg", help=f"기체+장비 {rev_dry_wt+rev_payload}kg + 추정 배터리 {est_batt_wt}kg")
            r_c2.metric("축당 호버링 요구추력", f"{int(rev_thrust_per_motor_g)} g")

            r_c3, r_c4 = st.columns(2)
            r_c3.metric("모터당 허용 전력", f"{motor_allowed_power_w} W 이하")
            r_c4.metric("필수 최소 효율", f"{required_eff_gw} g/W 이상")

            if required_eff_gw > 11.5:
                st.warning("⚠️ 요구 효율이 11.5 g/W를 초과합니다. 초대형 프로펠러(30인치 이상) 및 초고효율 VTOL 모터 또는 배터리 용량 증설이 필수적입니다.")
            elif required_eff_gw >= 8.5:
                st.info("💡 고급 산업용 파워시스템(24~28인치 프롭 / 저KV)으로 실현 가능한 범위입니다.")
            else:
                st.success("🟢 일반 상용 파워시스템으로 충분히 달성 가능한 안정적 수치입니다.")

            rev_prompt = f"기체 공허중량 {rev_dry_wt}kg, 임무장비 {rev_payload}kg, 배터리 {rev_batt_s}S {rev_batt_mah}mAh(추정무게 {est_batt_wt}kg)로 총 이륙중량 약 {rev_auw}kg인 {rev_motor_cnt}축 기체야. 목표 비행시간 {rev_target_min}분을 달성하려면 축당 호버링 추력 {int(rev_thrust_per_motor_g)}g에서 모터 1개당 소모전력이 {motor_allowed_power_w}W 이하, 최소 효율이 {required_eff_gw}g/W 이상이어야 해. 로컬 매뉴얼 제원표와 최신 스펙을 대조해서 이를 충족할 수 있는 구체적인 모터 모델명, 프로펠러 인치, ESC 조합을 역설계 추천해줘."
            if st.button("🚀 이 역설계 조건으로 최적 모터 검색 및 분석", use_container_width=True, key="btn_rev_search"):
                st.session_state["calc_query_send"] = rev_prompt
                st.rerun()

        # --------------------------------------------------------------------------
        # 탭 2: 기존 순방향 체공시간 계산
        # --------------------------------------------------------------------------
        with calc_tab2:
            st.caption("기체 중량(AUW)과 배터리 스펙으로 예상 호버링 시간을 계산합니다.")
            calc_flight_auw = st.number_input("기체 총 이륙중량 (AUW, kg)", min_value=0.5, max_value=200.0, value=float(auw) if 'auw' in locals() else 13.0, step=0.5, key="ft_auw_tab2")

            c_b1, c_b2 = st.columns(2)
            with c_b1:
                batt_s_f = st.selectbox("배터리 셀 수 (S)", [4, 6, 12, 14], index=2, format_func=lambda x: f"{x}S ({round(x*3.7, 1)}V)", key="ft_s_tab2")
            with c_b2:
                batt_mah_f = st.number_input("배터리 용량 (mAh)", min_value=1000, max_value=100000, value=22000, step=1000, key="ft_mah_tab2")

            col_eff1, col_eff2 = st.columns(2)
            with col_eff1:
                hover_eff_f = st.slider("호버링 추진 효율 (g/W)", min_value=4.0, max_value=12.0, value=8.0, step=0.1, key="ft_eff_tab2")
            with col_eff2:
                dod_limit_f = st.slider("배터리 방전 심도 (DOD %)", min_value=50, max_value=95, value=80, step=5, key="ft_dod_tab2")

            v_nom_f = batt_s_f * 3.7
            total_wh_f = (batt_mah_f * v_nom_f) / 1000.0
            usable_wh_f = total_wh_f * (dod_limit_f / 100.0)

            hover_power_f = (calc_flight_auw * 1000.0) / hover_eff_f
            hover_curr_f = hover_power_f / v_nom_f
            flight_time_f = (usable_wh_f / hover_power_f) * 60.0
            ft_sec_f = int((flight_time_f - int(flight_time_f)) * 60)
            req_c_rate_f = round(hover_curr_f / (batt_mah_f / 1000.0), 2)

            st.markdown("##### 📊 체공 시간 분석 결과")
            fc1, fc2 = st.columns(2)
            fc1.metric("예상 체공시간", f"{int(flight_time_f)}분 {ft_sec_f}초")
            fc2.metric("호버링 소모 전력", f"{round(hover_power_f, 1)} W")

            fc3, fc4 = st.columns(2)
            fc3.metric("호버링 소모 전류", f"{round(hover_curr_f, 1)} A")
            fc4.metric("요구 방전율", f"{req_c_rate_f} C")

            ft_fwd_prompt = f"AUW {calc_flight_auw}kg 기체이고 {batt_s_f}S {batt_mah_f}mAh LiPo 배터리를 사용할 때 호버링 시간이 약 {int(flight_time_f)}분으로 계산되었어. 호버링 소모전류 {round(hover_curr_f, 1)}A를 고려할 때 적합한 배터리 C-rate와 전원분배장치(PDB/커넥터) 규격을 권고해줘."
            if st.button("💬 이 조건으로 AI 분석 요청", use_container_width=True, key="btn_fwd_query"):
                st.session_state["calc_query_send"] = ft_fwd_prompt
                st.rerun()


    st.markdown("---")
    # ==============================================================================
    # 웹 링크 기반 스펙/추력표 자동 추출기 (단일 + 컬렉션 일괄)
    # ==============================================================================
    with st.expander("📥 웹 링크로 스펙표 즉시 수집 & 학습", expanded=False):
        if "crawl_success_msg" in st.session_state:
            st.success(st.session_state["crawl_success_msg"])

        st.caption("제조사 단일 제품 또는 컬렉션(목록) URL을 넣으면 사양과 추력표를 추출해 매뉴얼로 영구 저장합니다.")
        crawl_url = st.text_input("부품 웹페이지 또는 컬렉션 링크 (URL)", placeholder="https://shop.tmotor.com/collections/...")
        crawl_cat = st.selectbox("저장할 계통 폴더", ["02_모터_구동계", "03_변속기_ESC", "01_비행제어기_FC", "05_전원_배터리_PDB"], index=0)
        batch_crawl = st.checkbox("🔗 목록/컬렉션 URL인 경우 하위 제품 전체 일괄 수집", value=True)

        if st.button("🚀 데이터 추출 및 매뉴얼 영구 등록", use_container_width=True):
            raw_url = crawl_url.strip()
            if not raw_url:
                st.warning("링크(URL)를 입력해주세요.")
            else:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                save_dir = Path("manuals") / crawl_cat
                save_dir.mkdir(parents=True, exist_ok=True)

                target_urls = [raw_url]

                # 목록 URL 감지 시 하위 /products/ 링크 일괄 수집
                if batch_crawl and ("/collections/" in raw_url or "/category/" in raw_url or "/products" not in raw_url):
                    with st.spinner("목록 페이지에서 하위 제품 링크 추출 중..."):
                        try:
                            list_res = requests.get(raw_url, headers=headers, timeout=12)
                            if list_res.status_code == 200:
                                list_soup = BeautifulSoup(list_res.text, "html.parser")
                                found = set()
                                for a in list_soup.find_all("a", href=True):
                                    href = a["href"]
                                    if "/products/" in href:
                                        clean_u = href.split("?")[0]
                                        if not clean_u.startswith("http"):
                                            parsed_uri = urlparse(raw_url)
                                            clean_u = f"{parsed_uri.scheme}://{parsed_uri.netloc}{clean_u}"
                                        found.add(clean_u)
                                if found:
                                    target_urls = list(found)
                                    st.info(f"🔎 총 {len(target_urls)}개의 하위 제품 링크를 발견했습니다. 순차 수집을 시작합니다.")
                        except Exception as e:
                            st.warning(f"목록 파싱 중 오류(단일 URL로 진행): {e}")

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                saved_count = 0

                for idx, u in enumerate(target_urls, 1):
                    status_text.text(f"[{idx}/{len(target_urls)}] 데이터시트 수집 중: {u[:45]}...")
                    try:
                        p_res = requests.get(u, headers=headers, timeout=12)
                        if p_res.status_code == 200:
                            p_soup = BeautifulSoup(p_res.text, "html.parser")
                            for s in p_soup(["script", "style", "nav", "footer", "header"]):
                                s.decompose()
                            raw_body_text = p_soup.get_text(separator="\n", strip=True)[:14000]

                            extract_prompt = f"""아래 드론 부품 웹페이지 텍스트에서 [기본 사양]과 [추력/전류/전압 테스트 데이터 표]를 누락 없이 정확한 마크다운 문서로 변환하세요.
수치와 단위(V, A, W, g, kg, rpm, gf/W)를 원문 그대로 보존해야 합니다.

[웹페이지 원문]
{raw_body_text}

[출력 양식]
# [제조사 및 제품 모델명] 사양 데이터시트
- 원문 출처: {u}
## 1. 기본 사양
(사양 목록)
## 2. 벤치 테스트 추력-전류 시험표
| 스로틀(%) | 전압(V) | 전류(A) | 추력(g) | 추력(kg) | 소비전력(W) | 효율(gf/W) |
| ... |
"""
                            gpt_res = client.chat.completions.create(
                                model=MODEL,
                                messages=[{"role": "user", "content": extract_prompt}]
                            )
                            parsed_markdown = gpt_res.choices[0].message.content.strip()

                            first_line = parsed_markdown.splitlines()[0] if parsed_markdown.splitlines() else f"Product_{idx}"
                            clean_fname = re.sub(r'[^a-zA-Z0-9가-힣_-]', '_', first_line).strip('_')[:35] or f"Spec_{idx}"

                            save_path = save_dir / f"{clean_fname}.md"
                            save_path.write_text(parsed_markdown, encoding="utf-8")
                            saved_count += 1
                    except Exception:
                        pass

                    progress_bar.progress(float(idx) / float(len(target_urls)))

                if saved_count > 0:
                    st.cache_resource.clear()
                    st.session_state["crawl_success_msg"] = f"✅ 총 {saved_count}개 부품 규격표 등록 완료! 매뉴얼 DB에 영구 반영되었습니다."
                    st.balloons()
                    st.rerun()
                else:
                    st.error("스펙 데이터를 추출하지 못했습니다. 링크를 확인해주세요.")

    st.markdown("---")
    st.subheader("📊 시스템 운영 로그")
    excel_bytes = generate_multi_sheet_excel()
    st.download_button(
        label="📥 질의 이력 3개 시트 엑셀 다운로드",
        data=excel_bytes,
        file_name=f"drone_hub_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
    if LOG_FILE.exists():
        st.caption("✅ 질문 내역이 실시간으로 누적 기록 중입니다.")
    else:
        st.caption("💡 질문을 입력하면 이력이 자동으로 누적됩니다.")

    st.markdown("---")
    with st.expander("🌦️ 실시간 비행 기상 관제 (클릭)", expanded=False):
        if "current_lat" not in st.session_state:
            st.session_state["current_lat"] = 37.2723
            st.session_state["current_lon"] = 127.4350
            st.session_state["location_label"] = "이천시 (현재 위치)"

        search_addr = st.text_input("🔍 비행 장소 검색", placeholder="영월, 화성, 광나루, 이천, 판교, 보은")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("위치 검색", use_container_width=True):
                if search_addr.strip():
                    lat, lon, label = geocode_address(search_addr)
                    if lat and lon:
                        st.session_state["current_lat"] = lat
                        st.session_state["current_lon"] = lon
                        st.session_state["location_label"] = label or search_addr
        with c2:
            if st.button("📍 내 GPS", use_container_width=True):
                try:
                    ip_info = requests.get("https://ipapi.co/json/", timeout=3).json()
                    st.session_state["current_lat"] = ip_info.get("latitude", 37.2723)
                    st.session_state["current_lon"] = ip_info.get("longitude", 127.4350)
                    city = ip_info.get("city", "현재 위치")
                    st.session_state["location_label"] = f"{city} (내 위치)"
                except Exception:
                    pass

        st.caption(f"📍 현재 지역: `{st.session_state['location_label']}`")
        wx = get_weather_safety(lat=st.session_state["current_lat"], lon=st.session_state["current_lon"])
        if wx:
            status_color = {"안전": "🟢", "주의": "🟡", "비행 금지": "🔴"}.get(wx["status"], "⚪")
            st.markdown(f"**{status_color} 비행 안전 등급: {wx['status']}**")
            m1, m2 = st.columns(2)
            m1.metric("기온", f"{wx['temp']} °C")
            m2.metric("상공 80m", f"{wx['w_80m']} m/s")
            m1.metric("돌풍", f"{wx['gust']} m/s")
            m2.metric("강수확률", f"{wx['rain_prob']}%")
            if wx["risks"]:
                for r in wx["risks"]:
                    st.warning(r)

# 대화 세션
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def render_route_badge(route_str: str):
    if route_str == "manual":
        st.markdown('<span class="route-badge badge-manual">📘 검색 경로: 매뉴얼 RAG (14개 계통 기술 문서)</span>', unsafe_allow_html=True)
    elif route_str == "web":
        st.markdown('<span class="route-badge badge-web">🌐 검색 경로: Tavily AI 실시간 기술 검색</span>', unsafe_allow_html=True)
    elif route_str == "compare":
        st.markdown('<span class="route-badge badge-compare">⚖️ 검색 경로: 매뉴얼 vs 웹 비교 대조 분석</span>', unsafe_allow_html=True)

for idx, c in enumerate(st.session_state["chat_history"]):
    with st.chat_message(c["role"]):
        if c["role"] == "assistant":
            render_route_badge(c.get("route", "manual"))

        if c.get("is_compare") and c.get("sections"):
            tab1, tab2, tab3 = st.tabs(["⚖️ 종합 비교 대조 & 분석", "📘 매뉴얼 규격 내용", "🌐 웹 최신 정보 및 출처"])
            with tab1:
                st.markdown(format_section_to_markdown(c["sections"].get("analysis", "")))
            with tab2:
                st.markdown(format_section_to_markdown(c["sections"].get("manual", "")))
            with tab3:
                st.markdown(format_section_to_markdown(c["sections"].get("web", "")))
        else:
            st.markdown(c["content"])

    # --- [세로형 매뉴얼 다운로드 목록 (전체 파일명 표출)] ---
    ref_list = locals().get('ref_files', [])
    if ref_list:
        st.markdown('##### 📥 인용된 원본 매뉴얼 다운로드')
        with st.expander(f'📄 전체 매뉴얼 목록 열기 ({len(ref_list)}건)', expanded=True):
            for f_idx, f_item in enumerate(ref_list):
                fpath = Path(f_item['path']) if isinstance(f_item, dict) else Path(f_item)
                fname = f_item.get('name', fpath.name) if isinstance(f_item, dict) else fpath.name
                c_cat = fpath.parent.name if fpath.parent.name != 'manuals' else '규격서'
                col_txt, col_btn = st.columns([4, 1])
                with col_txt:
                    st.markdown(f'• **[{c_cat}]** `{fname}`')
                with col_btn:
                    if fpath.exists():
                        st.download_button(
                            label='⬇️ 다운로드',
                            data=fpath.read_bytes(),
                            file_name=fname,
                            key=f'live_dl_{f_idx}_{fname}'
                        )
user_input = st.chat_input("하드웨어 배선, ROS 2 통신, YOLO 또는 비행 규정을 질문하세요...")

if "calc_query_send" in st.session_state and st.session_state["calc_query_send"]:
    user_input = st.session_state.pop("calc_query_send")

if user_input:
    clean_q = user_input.strip()
    if not clean_q:
        st.warning("질문을 입력해 주세요.")
    else:
        danger_kw = ["vcc", "gnd", "쇼트", "단락", "5v", "12v", "전원", "배선", "역전압"]
        is_danger = any(dk in clean_q.lower() for dk in danger_kw)
        if is_danger:
            st.warning("⚡ [하드웨어 결선 주의] VCC/GND 극성 및 전압 레벨을 멀티미터로 사전에 반드시 실측하십시오.")

        st.session_state["chat_history"].append({"role": "user", "content": clean_q, "files": [], "is_compare": False, "route": ""})
        with st.chat_message("user"):
            st.markdown(clean_q)

        with st.chat_message("assistant"):
            with st.spinner("Tavily AI 기술 검색 및 응답 생성 중..."):
                t_start = datetime.now()
                res = drone_agent.invoke({
                    "question": clean_q,
                    "selected_mode": selected_mode,
                    "selected_category": selected_category,
                    "persona": persona
                })
                t_elapsed = round((datetime.now() - t_start).total_seconds(), 2)

                ans = res.get("answer", "답변 생성 실패")
                ref_files = res.get("referenced_files", [])
                cmp_sections = res.get("compare_sections", {})
                route_used = res.get("route", "manual")
                is_cmp = bool(route_used == "compare" and cmp_sections)

                safe_summary = (ans if isinstance(ans, str) else str(ans))[:120].replace("\n", " ")

                append_query_log({
                    "일시": datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S"),
                    "질문": clean_q,
                    "처리경로": route_used,
                    "계통필터": selected_category,
                    "페르소나": persona,
                    "결선주의": "YES" if is_danger else "NO",
                    "소요시간_초": t_elapsed,
                    "답변요약": safe_summary
                })

                render_route_badge(route_used)

                if is_cmp:
                    tab1, tab2, tab3 = st.tabs(["⚖️ 종합 비교 대조 & 분석", "📘 매뉴얼 규격 내용", "🌐 웹 최신 정보 및 출처"])
                    with tab1:
                        st.markdown(format_section_to_markdown(cmp_sections.get("analysis", "")))
                    with tab2:
                        st.markdown(format_section_to_markdown(cmp_sections.get("manual", "")))
                    with tab3:
                        st.markdown(format_section_to_markdown(cmp_sections.get("web", "")))
                else:
                    st.markdown(ans)

                if locals().get('ref_files') or locals().get('web_docs'):
                    st.markdown("##### 📚 인용된 공식 근거 및 출처 자료")

                    # 1. 로컬 매뉴얼 (PDF 및 MD)
                    if locals().get('ref_files'):
                        st.caption("📄 **로컬 지식 베이스 규격서 (PDF / Markdown):**")
                        cols = st.columns(min(len(ref_files), 4))
                        for col_idx, f_info in enumerate(ref_files):
                            with cols[col_idx % 4]:
                                try:
                                    with open(f_info["path"], "rb") as f_data:
                                        icon = "📝" if f_info.get("type") == "md" else "📄"
                                        st.download_button(
                                            label=f"{icon} {f_info['name'][:22]}",
                                            data=f_data.read(),
                                            file_name=f_info["name"],
                                            mime=f_info.get("mime", "application/octet-stream"),
                                            key=f"live_down_{col_idx}_{datetime.now().timestamp()}"
                                        )
                                except Exception:
                                    pass

                    # 2. 웹 검색 공식 출처 링크 카드
                    if locals().get('web_docs'):
                        st.caption("🌐 **실시간 웹 기술 데이터시트 링크:**")
                        w_cols = st.columns(min(len(web_docs), 2))
                        for w_idx, w_item in enumerate(web_docs[:4]):
                            with w_cols[w_idx % 2]:
                                st.markdown(f"- 🔗 [{w_item['title'][:32]}...]({w_item['link']})")

                st.markdown("---")
                if st.button("📋 이 작업의 비행 전 안전 체크리스트(Checklist) 생성", key=f"chk_btn_{len(st.session_state['chat_history'])}"):
                    with st.spinner("점검 체크리스트 생성 중..."):
                        chk_prompt = f"질문: {clean_q}\n답변: {ans}\n위 하드웨어/소프트웨어 설정 후 비행장에 나가기 전에 반드시 수행해야 할 3~5가지 핵심 안전 점검 사항을 마크다운 체크박스('- [ ] ') 형식으로 요약 작성해줘."
                        chk_res = client.chat.completions.create(
                            model=MODEL,
                            messages=[{"role": "user", "content": chk_prompt}]
                        )
                        st.markdown("### ✈️ 비행 전 현장 필수 점검표 (Pre-flight Checklist)")
                        st.markdown(chk_res.choices[0].message.content)

                st.session_state["chat_history"].append({
                    "role": "assistant", 
                    "content": ans,
                    "files": ref_files,
                    "is_compare": is_cmp,
                    "sections": cmp_sections,
                    "route": route_used
                })

                st.rerun()