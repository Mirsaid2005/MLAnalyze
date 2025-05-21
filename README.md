# MLAnalyze
# Анализ стоимости футболистов FIFA с помощью Random Forest

## Цель проекта
Построение модели машинного обучения для предсказания рыночной стоимости футболистов на основе их характеристик.

## Основные шаги

1. **Загрузка данных**
   - Использован датасет `fifa_players.csv`
   - Выбраны 12 ключевых признаков и целевая переменная `value_euro`

2. **Предобработка данных**
   - Удаление строк с пропущенными значениями
   - Автоматическое кодирование категориальных переменных (One-Hot Encoding)

3. **Разделение данных**
   - 80% данных - обучающая выборка
   - 20% данных - тестовая выборка
   - Фиксированный random_state для воспроизводимости

4. **Обучение модели**
   - Random Forest Regressor (100 деревьев)
   - Параметры по умолчанию

5. **Оценка результатов**
   - Метрики качества: RMSE, MAE, R²
   - Анализ важности признаков

## Ключевые результаты

- **Точность модели**: R² = 0.969 (модель объясняет 96.9% дисперсии)
- **Средняя ошибка**: ±823,443 евро (MAE)
- **Главные факторы стоимости**:
  1. Общий рейтинг (82.7% влияния)
  2. Потенциал игрока (9.3%)
  3. Возраст (2.7%)

## Полный код

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Загрузка данных
data = pd.read_csv("fifa_players.csv")

# Выбор признаков
features = [
    "age", "height_cm", "weight_kgs", "overall_rating", "potential",
    "wage_euro", "crossing", "finishing", "dribbling", "stamina", "strength", "vision"
]
target = "value_euro"

# Предобработка
data_clean = data[features + [target]].dropna()
X = pd.get_dummies(data_clean[features]) 
y = data_clean[target]

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание и оценка
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Важность признаков
importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
print(importance.sort_values("Importance", ascending=False))
