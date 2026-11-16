import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Настройка страницы
st.set_page_config(page_title="Подсчет растений", layout="centered")
st.title("🌱 Подсчет комнатных растений")
st.write("Загрузите фотографию комнаты, и нейросеть посчитает горшки с растениями.")


# Загружаем обученную модель (убедись, что папка train-6 совпадает с твоей!)
@st.cache_resource
def load_model():
    # ВНИМАНИЕ: проверь, что путь к best.pt правильный!
    return YOLO("runs/detect/train-6/weights/best.pt")


model = load_model()

# Кнопка для загрузки файла
uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Показываем исходное фото
    image = Image.open(uploaded_file)
    st.image(image, caption="Исходное изображение", use_container_width=True)

    if st.button("Посчитать растения"):
        with st.spinner("Нейросеть анализирует..."):
            # Конвертируем картинку для нейросети
            img_array = np.array(image)

            # Запускаем модель
            results = model(img_array)

            # Получаем результат (картинку с рамками)
            res_plotted = results[0].plot()

            # Считаем количество найденных объектов
            count = len(results[0].boxes)

            st.success(f"### Найдено растений: {count}")
            st.image(res_plotted, caption="Результат работы нейросети", use_container_width=True)
