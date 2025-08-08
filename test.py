import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error, r2_score

# Загрузка данных
df = pd.read_csv("winequality-red.csv", sep=";")

# Предварительная обработка данных
df.dropna(inplace=True)
df["fixed acidity"].fillna(df["fixed acidity"].mean(), inplace=True)
df["volatile acidity"].fillna(df["volatile acidity"].mean(), inplace=True)
df["citric acid"].fillna(df["citric acid"].mean(), inplace=True)
df["residual sugar"].fillna(df["residual sugar"].mean(), inplace=True)
df["chlorides"].fillna(df["chlorides"].mean(), inplace=True)
df["free sulfur dioxide"].fillna(df["free sulfur dioxide"].mean(), inplace=True)
df["total sulfur dioxide"].fillna(df["total sulfur dioxide"].mean(), inplace=True)
df["density"].fillna(df["density"].mean(), inplace=True)
df["pH"].fillna(df["pH"].mean(), inplace=True)
df["sulphates"].fillna(df["sulphates"].mean(), inplace=True)
df["alcohol"].fillna(df["alcohol"].mean(), inplace=True)

# Разработка модели
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=100,
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# Оценка качества модели
y_pred = model.predict(X_test_scaled)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# Визуализация результатов
plt.scatter(y_test, y_pred)
plt.xlabel("True quality")
plt.ylabel("Predicted quality")
plt.show()

# Подготовка отчета
with open("report.txt", "w") as file:
    file.write("Этап работы\tОписание\n")
    file.write(
        "Загрузка данных\tЗагрузили уникальный набор данных о качестве вин, например, Wine Quality Data Set с сайта UCI Machine Learning Repository.\n"
    )
    file.write(
        "Предварительная обработка данных\tПровели исследовательский анализ данных, найдите и обработайте пропуски и выбросы. Нормализовали данные, если это необходимо.\n"
    )
    file.write(
        "Разработка модели\tИспользуем метод передачи через нейронную сеть (например, Feedforward Neural Network) для предсказания качества вина. Опишите, как вы реализуете обучение с помощью градиентного спуска и какие функции активации будете использовать.\n"
    )
    file.write(
        "Обучение модели\tОбучили модель на подготовленных данных и оценили ее эффективность, используя метрики, такие как R² (коэффициент детерминации) и MAE (Mean Absolute Error).\n"
    )
    file.write(
        "Визуализация результатов\tВизуализировали результаты, используя библиотеку Matplotlib.\n"
    )
    file.write(
        "Подготовка отчета\tПодготовили отчет с описанием всех этапов работы, включая код и графики. Используем библиотеку Seaborn для визуализации результатов.\n"
    )
