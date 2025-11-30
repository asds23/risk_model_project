#!/usr/bin/env python3
# ml_model_core.py
"""
Ядро модели множественной регрессии для маршрутов с рисками.
Использует данные MET.NO и построение мультимодальных маршрутов.
"""

import random
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import os

# Импортируем локальные модули
from route_module import (
    build_multimodal_route,
    haversine,
    MODULE_VULN
)

# -----------------------------
# 1) Геокодирование адресов
# -----------------------------
CITY_COORDS = {
    "Москва": (55.7558, 37.6176),
    "Санкт-Петербург": (59.9311, 30.3609),
    "Новосибирск": (55.0084, 82.9357),
    "Екатеринбург": (56.8389, 60.6057)
}

def geocode_address(address: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "RiskModel/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data and len(data) > 0:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except:
        pass
    for city, coords in CITY_COORDS.items():
        if city.lower() in address.lower():
            return coords
    return round(random.uniform(55.0, 56.5), 6), round(random.uniform(36.0, 38.5), 6)

# -----------------------------
# 2) Генерация признаков маршрута
# -----------------------------
def aggregate_route_features(route: dict) -> dict:
    segments = []
    module_counts = {m: 0 for m in MODULE_VULN.keys()}

    for leg in route["legs"]:
        module_counts[leg["module"]] += 1
        for seg in leg["segments"]:
            segments.append(seg)

    if not segments:
        return {}

    feats = {"segments_total": len(segments), "legs_total": len(route["legs"])}

    temps, winds = [], []
    for seg in segments:
        weather = seg.get("weather_raw", {}).get("properties", {}).get("timeseries", [])
        if weather:
            details = weather[0]["data"]["instant"]["details"]
            temps.append(details.get("air_temperature", 0.0))
            winds.append(details.get("wind_speed", 0.0))

    feats["temp_mean"] = float(np.mean(temps)) if temps else 0.0
    feats["temp_std"] = float(np.std(temps)) if temps else 0.0
    feats["wind_mean"] = float(np.mean(winds)) if winds else 0.0
    feats["wind_std"] = float(np.std(winds)) if winds else 0.0

    for m in MODULE_VULN.keys():
        feats[f"module_count_{m}"] = module_counts[m]

    start = segments[0]["coords"]
    end = segments[-1]["coords"]
    feats["distance_km"] = float(haversine(start, end))
    return feats

def aggregate_route_features_test(route: dict) -> dict:
    segments = []
    module_counts = {m: 0 for m in MODULE_VULN.keys()}

    for leg in route["legs"]:
        module_counts[leg["module"]] += 1
        for seg in leg["segments"]:
            segments.append(seg)

    if not segments:
        return {}

    feats = {"segments_total": len(segments), "legs_total": len(route["legs"])}

    temps, winds = [], []
    for seg in segments:
        weather = seg.get("weather_raw", {}).get("properties", {}).get("timeseries", [])
        if weather:
            details = weather[0]["data"]["instant"]["details"]
            temps.append(details.get("air_temperature", 0.0))
            winds.append(details.get("wind_speed", 0.0))

    feats["temp_mean"] = float(np.mean(temps)) if temps else 0.0
    feats["temp_std"] = float(np.std(temps)) if temps else 0.0
    feats["wind_mean"] = float(np.mean(winds)) if winds else 0.0
    feats["wind_std"] = float(np.std(winds)) if winds else 0.0

    for m in MODULE_VULN.keys():
        feats[f"module_count_{m}"] = module_counts[m]

    start = segments[0]["coords"]
    end = segments[-1]["coords"]
    feats["distance_km"] = float(haversine(start, end))
    return feats

# -----------------------------
# 3) Класс модели с обучением и предсказанием
# -----------------------------
class RiskMultiRegressor:
    def __init__(self):
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=80, max_depth=12, random_state=42, n_jobs=-1
            )
        )
        self.feature_columns = None
        self.target_names = [f"avg_module_risk_{m}" for m in MODULE_VULN.keys()]

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        self.feature_columns = list(X_numeric.columns)
        self.model.fit(X_numeric, Y[self.target_names].values)

    def predict(self, route: dict) -> dict:
        if self.model is None:
            raise ValueError("Модель не обучена. Используйте метод fit() или load().")
        X = self._prepare_features(route)
        y_pred = self.model.predict(X)
        return {name: float(y_pred[0, i]) for i, name in enumerate(self.target_names)}

    def _prepare_features(self, route: dict) -> pd.DataFrame:
        feats = aggregate_route_features(route)
        row = {c: float(feats.get(c, 0.0)) for c in self.feature_columns}
        return pd.DataFrame([row])

    def save(self, filename="risk_model.pkl"):
        path = os.path.join(os.getcwd(), filename)
        joblib.dump({
            "model": self.model,
            "feature_columns": self.feature_columns,
            "target_names": self.target_names
        }, path)

    def load(self, filename="risk_model.pkl"):
        path = os.path.join(os.getcwd(), filename)
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.target_names = data["target_names"]

# -----------------------------
# 4) Утилита: построение маршрута по адресам
# -----------------------------
def route_from_addresses(addr_from: str, addr_to: str, modules=None, segments_per_leg=5):
    start_coords = geocode_address(addr_from)
    end_coords = geocode_address(addr_to)
    return build_multimodal_route(start_coords, end_coords, modules=modules, segments_per_leg=segments_per_leg)
