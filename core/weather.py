import requests

def get_flight_weather(lat: float, lon: float):
    """Open-Meteo API로부터 지상/80m/120m 풍속 및 기온 수집 및 종합 판정"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,wind_gusts_10m&hourly=wind_speed_80m,wind_speed_120m&timezone=Asia%2FSeoul"
    try:
        res = requests.get(url, timeout=4.0).json()
        curr = res.get("current", {})
        hourly = res.get("hourly", {})
        temp = curr.get("temperature_2m", 0.0)
        w10 = curr.get("wind_speed_10m", 0.0)
        gust = curr.get("wind_gusts_10m", 0.0)
        w80 = hourly.get("wind_speed_80m", [w10])[0]
        w120 = hourly.get("wind_speed_120m", [w80])[0]
        max_w = max(w10, w80, w120, gust)

        if max_w >= 10.0:
            status, level = f"🚨 비행 위험 (강풍 {max_w}m/s)", "error"
        elif max_w >= 6.5:
            status, level = f"⚠️ 비행 주의 (풍속 {max_w}m/s)", "warning"
        else:
            status, level = f"✅ 비행 적합 (최대 {max_w}m/s)", "success"

        return {
            "temp": temp, "w10": w10, "gust": gust,
            "w80": w80, "w120": w120, "max_w": max_w,
            "status": status, "level": level
        }
    except Exception:
        return None