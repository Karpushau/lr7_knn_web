import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Загружаем данные и обучаем модель KNN (кэшируем, чтобы не обучать каждый раз)
@st.cache_resource
def train_model():
    # Загружаем стандартный датасет Ирисов
    iris = load_iris()
    
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    
    # Создаем и обучаем модель KNN (K=3)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    return knn, iris.target_names

# Инициализируем модель
model, target_names = train_model()

# Интерфейс приложения
st.title("🌸 Классификация Ирисов (Метод KNN)")
st.write("Введите параметры цветка, чтобы алгоритм определил его сорт:")

# Создаем ползунки для ввода 4 признаков цветка
sepal_length = st.slider("Длина чашелистика (см)", 4.0, 8.0, 5.8)
sepal_width = st.slider("Ширина чашелистика (см)", 2.0, 4.5, 3.0)
petal_length = st.slider("Длина лепестка (см)", 1.0, 7.0, 3.7)
petal_width = st.slider("Ширина лепестка (см)", 0.1, 2.5, 1.2)

# Кнопка для запуска предсказания
if st.button("Определить сорт"):
    # Собираем введенные значения в массив
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    # Делаем предсказание
    prediction = model.predict(features)[0]
    predicted_class = target_names[prediction]
    
    # Выводим результат на экран
    st.success(f"Модель KNN предсказывает сорт: **{predicted_class}**")
