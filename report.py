import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Ensure TensorFlow uses only CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.config.set_visible_devices([], "GPU")

# Загрузка и предварительная обработка данных ---
print("--- 1. Загрузка и предварительная обработка данных ---")

# Загрузка наборов данных с обработкой ошибок
try:
    red_wine = pd.read_csv("winequality-red.csv", sep=";")
    white_wine = pd.read_csv("winequality-white.csv", sep=";")
except FileNotFoundError as e:
    print(
        f"Ошибка: Не удалось найти файл данных. Убедитесь, что файлы CSV находятся в той же директории, что и скрипт. {e}"
    )
    exit()  # Завершить выполнение скрипта, если файлы не найдены

# Добавление столбца 'type'
red_wine["type"] = "red"
white_wine["type"] = "white"

# Объединение наборов данных
wine_data = pd.concat([red_wine, white_wine], ignore_index=True)

print("Объединенный набор данных:")
print(wine_data.head())
print(wine_data.info())
print(wine_data.describe())

# Исследовательский анализ данных (EDA) ---
print("\n--- 2. Исследовательский анализ данных (EDA) ---")

# Проверка на пропуски
print("\nПропущенные значения:\n", wine_data.isnull().sum())

# Визуализация распределений числовых признаков (раскомментировать для Jupyter)
# plt.figure(figsize=(15, 10))
# wine_data.hist(bins=20, figsize=(15, 10))
# plt.suptitle("Распределение признаков")
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

# Визуализация корреляционной матрицы (раскомментировать для Jupyter)
# plt.figure(figsize=(12, 10))
# sns.heatmap(wine_data.drop("type", axis=1).corr(), annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Корреляционная матрица признаков вина")
# plt.show()

# Ящиковые диаграммы для выявления выбросов (раскомментировать для Jupyter)
# plt.figure(figsize=(15, 10))
# for i, column in enumerate(wine_data.drop(["quality", "type"], axis=1).columns):
#     plt.subplot(3, 4, i + 1)
#     sns.boxplot(y=wine_data[column])
#     plt.title(column)
# plt.tight_layout()
# plt.show()

print(
    "Исследовательский анализ данных завершен. Визуализации могут быть сгенерированы при необходимости."
)

# Подготовка данных для моделирования ---
print("\n--- 3. Подготовка данных для моделирования ---")

# Отделение признаков (X) от целевой переменной (y)
# Включаем 'type' в признаки и применяем One-Hot Encoding
X = wine_data.drop("quality", axis=1)
y = wine_data["quality"]

# One-Hot Encoding для столбца 'type'
X = pd.get_dummies(
    X, columns=["type"], drop_first=True
)  # drop_first=True избегает дамми-ловушки

# Нормализация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Данные подготовлены. Размеры выборок:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Разработка и обучение нейронной сети
print("\n--- 4. Разработка и обучение нейронной сети ---")

# Определение модели нейронной сети
model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dense(32, activation="relu"),
        Dense(1, activation="linear"),  # Выходной слой для регрессии
    ]
)

# Компиляция модели
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Обучение модели (verbose=1 для отображения прогресса)
history = model.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1
)

print("Обучение модели завершено.")

# Оценка модели и визуализация результатов
print("\n--- 5. Оценка модели и визуализация результатов ---")

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Оценка метрик
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"R^2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Визуализация предсказанных vs. фактических значений (scatterplot)
plt.figure(figsize=(10, 7))
sns.scatterplot(x=y_test, y=y_pred.flatten(), alpha=0.6)

# Добавление линии идеального предсказания
min_quality = wine_data["quality"].min()
max_quality = wine_data["quality"].max()
plt.plot(
    [min_quality, max_quality],
    [min_quality, max_quality],
    color="red",
    linestyle="--",
    lw=2,
    label="Идеальное предсказание",
)

plt.xlabel("Фактическое качество")
plt.ylabel("Предсказанное качество")
plt.title("Фактическое vs. Предсказанное качество вина")
plt.grid(True)
plt.legend()
plt.savefig("actual_vs_predicted_scatterplot.png")
plt.show()
plt.close()

print(
    "Визуализация фактических vs. предсказанных значений сохранена как actual_vs_predicted_scatterplot.png"
)

# Оригинальная визуализация stripplot (закомментирована, но оставлена для справки)
# plt.figure(figsize=(10, 7))
# sns.stripplot(x=y_test, y=y_pred.flatten(), alpha=0.6, jitter=0.2)
# plt.plot(
#     [min_quality, max_quality],
#     [min_quality, max_quality],
#     color="red",
#     linestyle="--",
#     lw=2,
# )
# plt.xlabel("Фактическое качество (дискретное)")
# plt.ylabel("Предсказанное качество")
# plt.title("Фактическое vs. Предсказанное качество вина")
# plt.grid(True)
# plt.xticks(sorted(y_test.unique()))
# plt.savefig("actual_vs_predicted_stripplot.png")
# plt.show()
# plt.close()
# print(
#     "Визуализация фактических vs. предсказанных значений сохранена как actual_vs_predicted_stripplot.png"
# )


print(
    "\nВесь код объединен в один файл. Для использования в Jupyter Notebook просто скопируйте содержимое этого файла в ячейку."
)
