from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Шампунь: прогноз качества", layout="wide")

st.title("Система прогнозирования качества шампуня")
st.markdown("Это MVP-интерфейс для обзора данных, прогноза варок и модели Champion/Challenger.")

st.header("Обзор данных")
col1, col2, col3 = st.columns(3)
col1.metric("Варок в базе", "-")
col2.metric("Размеченных варок", "-")
col3.metric("Ошибок данных", "-")

st.header("Прогноз варки")
batch_id = st.number_input("ID варки", min_value=1, step=1)
checkpoint = st.number_input("Этап до", min_value=1, step=1)
if st.button("Сделать прогноз"):
    st.warning("Прогнозный функционал пока не реализован полностью.")

st.header("Модели")
st.write("Champion и Challenger появятся после обучения моделей.")
