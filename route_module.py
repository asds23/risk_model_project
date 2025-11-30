#!/usr/bin/env python3
# stage3_route_module.py
"""
Этап 3: построение мультимодального маршрута и расчёт риска на каждом модуле.
Зависимость: data_fetching_and_risk_mapping.get_point_risk
Помещай файл рядом с data_fetching_and_risk_mapping.py
"""

import math
import json
import csv
from datetime import datetime
from typing import List, Tuple, Optional

# импортируем функцию получения риска по точке (MET.NO → расширенные признаки)
try:
    from data_fetching_and_risk_mapping import get_point_risk
except Exception as e:
    # Если импорт не удался (например, файл в другой папке), сообщим понятную ошибку
    raise ImportError("Нужно поместить data_fetching_and_risk_mapping.py рядом с этим файлом. Ошибка: " + str(e))


# ---------------------------
# 1) Утилиты геометрии
# ---------------------------
def haversine(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    """Расстояние в км между двумя точками (lat, lon)."""
    lat1, lon1 = a
    lat2, lon2 = b
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    aa = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(aa))


def interpolate_points(a: Tuple[float,float], b: Tuple[float,float], n_segments: int) -> List[Tuple[float,float]]:
    """Разбить отрезок a->b на n_segments (вернёт n_segments+1 точку)."""
    lat1, lon1 = a
    lat2, lon2 = b
    pts = []
    for i in range(n_segments + 1):
        t = i / n_segments
        lat = lat1 + (lat2 - lat1) * t
        lon = lon1 + (lon2 - lon1) * t
        pts.append((round(lat,6), round(lon,6)))
    return pts


# ---------------------------
# 2) Уязвимости модулей (базовые мультипликаторы)
# ---------------------------
MODULE_VULN = {
    "truck": {"label": "Авто", "mult": 1.0},
    "rail":  {"label": "Ж/д",  "mult": 0.8},
    "ship":  {"label": "Мор/реч", "mult": 1.4},
    "air":   {"label": "Авиа", "mult": 1.2}
}

# дополнительные веса для специфических опасностей (добавляются сверху)
MODULE_SPECIFIC_WEIGHTS = {
    "truck": {"fog": 0.4, "precipitation": 0.3, "wind": 0.2, "ice_temp": 0.4},
    "rail":  {"fog": 0.3, "precipitation": 0.25, "wind": 0.1, "ice_temp": 0.3},
    "ship":  {"wave_height": 0.6, "wind": 0.25, "precipitation": 0.15},
    "air":   {"wind_gust": 0.5, "visibility": 0.3, "thunder": 0.3}
}


# ---------------------------
# 3) Функции расчёта модульных рисков
# ---------------------------
def compute_module_specific_extra(risk_details: dict, module: str) -> float:
    """
    По расширенным деталям риска (из data_fetching_and_risk_mapping.map_weather_to_risk)
    вычисляем дополнительный модульный риск 0-10, основанный на специфических параметрах.
    risk_details — словарь детальных нормализованных рисков (0-10).
    """
    w = MODULE_SPECIFIC_WEIGHTS.get(module, {})
    extra = 0.0

    # Truck/Rail: fog, precipitation, ice_temp (приближённо через dew_point/temperature)
    if module in ("truck", "rail"):
        fog = risk_details.get("fog_risk", 0)
        precip = max(risk_details.get("precipitation_1h_risk", 0),
                     risk_details.get("precipitation_6h_risk", 0))
        temp = risk_details.get("temperature_risk", 0)
        dew_diff = risk_details.get("dew_point_risk", 0)
        # примерная вероятность льда — если temp около 0 и dew близко к temp -> риск гололеда
        ice_temp = 0
        if temp >= 3 and temp <= 12 and dew_diff < 2:
            ice_temp = 6  # условный риск гололеда
        extra = (fog * w.get("fog",0) + precip * w.get("precipitation",0) + ice_temp * w.get("ice_temp",0)) 
    elif module == "ship":
        wave = risk_details.get("wave_height_risk", 0)
        wind = risk_details.get("wind_speed_risk", 0)
        precip = max(risk_details.get("precipitation_6h_risk", 0), risk_details.get("precipitation_12h_risk", 0))
        extra = (wave * w.get("wave_height",0) + wind * w.get("wind",0) + precip * w.get("precipitation",0))
    elif module == "air":
        gust = risk_details.get("wind_gust_risk", 0)
        # visibility отсутствует в details — приближённо используем cloud/ humidity to infer low visibility
        vis_like = max(risk_details.get("fog_risk",0), (10 - risk_details.get("cloud_risk",0)))
        thunder = risk_details.get("prob_thunder_risk", 0)
        extra = (gust * w.get("wind_gust",0) + vis_like * w.get("visibility",0) + thunder * w.get("thunder",0))
    else:
        extra = 0.0

    # Ограничение 0-10
    return round(min(10, extra), 2)


def compute_module_risk(base_total_risk: float, risk_details: dict, module: str) -> float:
    """
    Итоговый риск для модуля:
      module_risk = clamp(base_total_risk * mult + extra_specific, 0, 10)
    """
    mult = MODULE_VULN.get(module, {}).get("mult", 1.0)
    extra = compute_module_specific_extra(risk_details, module)
    raw = base_total_risk * mult + extra * 0.15  # extra добавляется с небольшим весом
    return round(min(10, max(0, raw)), 2)


# ---------------------------
# 4) Основная функция построения мультимодального маршрута
# ---------------------------
def build_multimodal_route(
    start_coords: Tuple[float,float],
    end_coords: Tuple[float,float],
    modules: Optional[List[str]] = None,
    segments_per_leg: int = 5
):
    """
    Построить маршрут с учётом последовательности модулей.
      start_coords, end_coords: (lat,lon)
      modules: список модулей, по умолчанию ["truck"] (весь путь автоперевозка)
      segments_per_leg: сколько интерполяций на каждую «лего-часть» между сменой модуля.
    Логика разбивки:
      - если modules = ["truck","rail","ship","truck"] => разбиваем маршрут на 3 legs (между сменами)
      - leg i будет иметь segments_per_leg сегментов (точек = segments_per_leg + 1)
    Возвращает словарь с детальной информацией.
    """
    if modules is None or len(modules) == 0:
        modules = ["truck"]

    # Разбиваем маршрут пропорционально на number_of_legs
    num_legs = max(1, len(modules))
    # Простейшее разделение — равные промежутки по геодистанции
    lat1, lon1 = start_coords
    lat2, lon2 = end_coords

    # Формируем промежуточные контрольные точки для смен модулей:
    control_points = interpolate_points(start_coords, end_coords, n_segments=num_legs)
    # control_points имеет num_legs+1 точку — между ними legs
    route = {
        "start": start_coords,
        "end": end_coords,
        "modules": modules,
        "legs": []
    }

    for i in range(len(modules)):
        leg_start = control_points[i]
        leg_end = control_points[i+1]
        module = modules[i]
        pts = interpolate_points(leg_start, leg_end, n_segments=segments_per_leg)
        leg = {
            "module": module,
            "start": leg_start,
            "end": leg_end,
            "segments": []
        }
        for idx, pt in enumerate(pts):
            # Получаем расширенные погодные данные и общий риск в точке
            pr = get_point_risk(pt[0], pt[1])  # использует MET.NO + расширенная нормализация
            base_total = pr["risk"]["total_risk"]
            details = pr["risk"]["details"]

            module_risk = compute_module_risk(base_total, details, module)

            seg = {
                "segment_index": idx,
                "coords": pt,
                "point_risk": base_total,
                "risk_details": details,
                "module_risk": module_risk,
                "weather_raw": pr["weather_raw"]
            }
            leg["segments"].append(seg)
        route["legs"].append(leg)

    # Сводные метрики
    # например: максимум риска по маршруту для каждого модуля
    summary = {}
    for leg in route["legs"]:
        module = leg["module"]
        max_module = max(s["module_risk"] for s in leg["segments"])
        avg_module = round(sum(s["module_risk"] for s in leg["segments"]) / len(leg["segments"]), 2)
        summary[module] = {"max_risk": max_module, "avg_risk": avg_module}
    route["summary"] = summary
    route["generated_at"] = datetime.utcnow().isoformat()

    return route


# ---------------------------
# 5) Экспорт результатов
# ---------------------------
def save_route_json(route: dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(route, f, ensure_ascii=False, indent=2)


def save_route_csv(route: dict, filename: str):
    # Сохраняем плоскую таблицу: leg_index, module, segment_index, lat, lon, point_risk, module_risk
    rows = []
    for leg_idx, leg in enumerate(route["legs"]):
        for seg in leg["segments"]:
            rows.append({
                "leg_index": leg_idx,
                "module": leg["module"],
                "segment_index": seg["segment_index"],
                "lat": seg["coords"][0],
                "lon": seg["coords"][1],
                "point_risk": seg["point_risk"],
                "module_risk": seg["module_risk"]
            })
    keys = ["leg_index","module","segment_index","lat","lon","point_risk","module_risk"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------
# 6) Демонстрация (пример использования)
# ---------------------------
def demo():
    # Пример: Москва -> Владивосток, цепочка модулей: truck -> rail -> ship -> truck
    # Координаты (lat, lon)
    start = (55.7558, 37.6176)      # Москва
    end = (43.1155, 131.8855)      # Владивосток
    modules = ["truck", "rail", "ship", "truck"]
    print("Старт построения маршрута:", start, "->", end, "модули:", modules)
    route = build_multimodal_route(start, end, modules=modules, segments_per_leg=3)
    print("Summary:", route["summary"])
if __name__ == "__main__":
    demo()
