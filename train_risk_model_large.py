"""
Этап 2: обучение модели RiskMultiRegressor на множественных данных
Вход: адреса "От" и "До"
Выход: обучленная модель с прогнозом module_risk и геоданными
"""

import os
import pandas as pd
from ml_model import RiskMultiRegressor, route_from_addresses, aggregate_route_features

# -----------------------------
# 1) Большой набор обучающих маршрутов
# -----------------------------
training_routes = [
    ("улица 1 Мая, 8, Балашиха, Московская область", "ул. Малая Ордынка, 31, Москва", ["truck", "rail"]),
    ("ул. Ленина, 10, Новосибирск", "пр. Литейный, 20, Санкт-Петербург", ["truck", "air"]),
    ("ул. Мира, 15, Екатеринбург", "ул. Тверская, 7, Москва", ["truck", "rail", "ship"]),
    ("ул. Советская, 25, Казань", "ул. Красная, 10, Самара", ["truck", "rail"]),
    ("ул. Карла Маркса, 5, Омск", "ул. Ленина, 100, Красноярск", ["truck", "air"]),
    ("ул. Пушкина, 12, Воронеж", "ул. Советская, 55, Ростов-на-Дону", ["truck", "rail"]),
    ("ул. Октябрьская, 30, Тюмень", "ул. Ленина, 40, Екатеринбург", ["truck", "rail"]),
    ("ул. Гагарина, 50, Нижний Новгород", "ул. Большая Покровская, 2, Нижний Новгород", ["truck", "air"]),
    ("ул. Лермонтова, 7, Уфа", "ул. Карла Маркса, 22, Пермь", ["truck", "rail", "ship"]),
    ("ул. Молодежная, 1, Челябинск", "ул. Красная, 33, Екатеринбург", ["truck", "rail"]),
    ("ул. Московская, 10, Ярославль", "ул. Советская, 5, Кострома", ["truck", "rail"]),
    ("ул. Советская, 12, Тула", "ул. Ленина, 3, Калуга", ["truck", "rail"]),
    ("ул. Ленина, 7, Смоленск", "ул. Октябрьская, 20, Брянск", ["truck", "rail"]),
    ("ул. Пушкина, 15, Орел", "ул. Советская, 25, Курск", ["truck", "rail"]),
    ("ул. Красная, 9, Краснодар", "ул. Советская, 40, Ростов-на-Дону", ["truck", "rail", "ship"]),
]

# -----------------------------
# 2) Генерация данных для обучения
# -----------------------------
feature_rows = []
target_rows = []

for start, end, modules in training_routes:
    print(f"Генерируем маршрут: {start[:30]} -> {end[:30]} с модулями {modules}")
    route = route_from_addresses(start, end, modules=modules, segments_per_leg=3)

    # Формируем признаки маршрута
    features = aggregate_route_features(route)
    feature_rows.append(features)

    # Формируем целевые значения — риски модулей
    targets = {}
    for leg in route["legs"]:
        mod = leg["module"]
        risks = [seg["module_risk"] for seg in leg["segments"]]
        targets[f"avg_module_risk_{mod}"] = float(pd.Series(risks).mean())
    target_rows.append(targets)

# -----------------------------
# 3) Подготовка DataFrame для обучения
# -----------------------------
X_df = pd.DataFrame(feature_rows).fillna(0)
Y_df = pd.DataFrame(target_rows).fillna(0)

print("Признаки для обучения:")
print(X_df.head())
print("Целевые значения:")
print(Y_df.head())

# -----------------------------
# 4) Обучение модели
# -----------------------------
model = RiskMultiRegressor()
model.feature_columns = list(X_df.columns)  # сохраняем имена признаков
print("Начинаем обучение модели...")
model.fit(X_df, Y_df)

# Сохраняем модель в текущую директорию
current_dir = os.getcwd()
model_filename = os.path.join(current_dir, "risk_model_trained_large.pkl")
model.save(model_filename)

print(f"Модель обучена и сохранена: {model_filename}")

# -----------------------------
# 5) Дополнительно: вывод значимых геоданных с адресами
# -----------------------------
for i, route in enumerate(training_routes):
    start, end, modules = route
    print(f"\nМаршрут {i+1}: {start[:30]} -> {end[:30]} с модулями {modules}")
