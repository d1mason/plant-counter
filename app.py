import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import json
import os
import pandas as pd
from datetime import datetime
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

            history_file = "history.json"
            new_record = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "count": int(count)
            }
            if os.path.exists(history_file):
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(new_record)
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=4)
# --- Блок истории и скачивания отчета ---
st.markdown("---")
st.subheader("📊 История подсчетов")

if os.path.exists("history.json"):
    with open("history.json", "r", encoding="utf-8") as f:
        history_data = json.load(f)

    if len(history_data) > 0:
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Скачать отчет в Excel (CSV)",
            data=csv,
            file_name=f"plant_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("История пока пуста.")
else:
    st.info("История пока пуста.")
