import json
from pathlib import Path

def interpolate_motor_benchmark(motor_name: str, target_thrust_g: float):
    """실측 제원표(JSON)에서 목표 추력에 해당하는 스로틀, 전력(W), 전류(A) 선형 보간"""
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
                t_throttle = round(p1["throttle"] + ratio * (p2["throttle"] - p1["throttle"]), 1)
                t_current = round(p1["current_a"] + ratio * (p2["current_a"] - p1["current_a"]), 2)
                t_power = round(p1["power_w"] + ratio * (p2["power_w"] - p1["power_w"]), 1)
                t_eff = round(p1["eff_gw"] + ratio * (p2["eff_gw"] - p1["eff_gw"]), 2)
                status = "✅ 안정적 호버링 (권장 40-55%)" if 40 <= t_throttle <= 58 else ("⚠️ 고부하 호버링" if t_throttle > 58 else "저부하 운용")
                return {"throttle": t_throttle, "current_a": t_current, "power_w": t_power, "eff_gw": t_eff, "status": status}
    except Exception:
        pass
    return None

def calc_reverse_specs(dry_wt: float, payload_wt: float, batt_s: int, batt_mah: int, target_min: int, motor_cnt: int):
    """목표 비행시간(역설계) 기준 요구 파워 및 최소 효율 역산"""
    est_batt_wt = round((batt_mah / 1000.0) * (batt_s / 12.0) * 0.205, 2)
    auw = round(dry_wt + payload_wt + est_batt_wt, 2)
    v_nom = batt_s * 3.7
    usable_wh = ((batt_mah * v_nom) / 1000.0) * 0.80
    max_allowed_w = usable_wh / (target_min / 60.0)
    motor_allowed_w = round(max_allowed_w / motor_cnt, 1)
    thrust_per_motor = round((auw * 1000.0) / motor_cnt, 1)
    req_eff = round(thrust_per_motor / motor_allowed_w, 2) if motor_allowed_w > 0 else 0
    return {
        "auw": auw, "est_batt_wt": est_batt_wt,
        "thrust_per_motor": thrust_per_motor,
        "motor_allowed_w": motor_allowed_w,
        "req_eff": req_eff
    }