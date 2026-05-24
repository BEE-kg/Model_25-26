import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

MU0 = 4 * math.pi * 1e-7

st.set_page_config(page_title="Моделирование катушки", layout="wide")
st.title("Моделирование оптимизации катушки")

with st.sidebar:
    wire_length = st.number_input("Длина провода L, м", min_value=0.1, max_value=1000.0, value=50.0, step=1.0)
    wire_diameter_mm = st.number_input("Диаметр провода d, мм", min_value=0.01, max_value=20.0, value=1.0, step=0.1)
    frame_diameter_cm = st.number_input("Диаметр каркаса D, см", min_value=0.1, max_value=100.0, value=5.0, step=0.5)
    current = st.number_input("Ток I, А", min_value=0.01, max_value=100.0, value=2.0, step=0.1)
    mu_r = st.number_input("Относительная магнитная проницаемость", min_value=1.0, max_value=5000.0, value=1.0, step=1.0)
    range_factor = st.slider("Диапазон длин катушки", 2, 12, 6, 1)
    points = st.slider("Количество точек моделирования", 300, 8000, 2000, 100)

wire_diameter = wire_diameter_mm / 1000
frame_diameter = frame_diameter_cm / 100
frame_radius = frame_diameter / 2
turn_circumference = math.pi * frame_diameter
section_area = math.pi * frame_diameter ** 2 / 4

dense_turn_length = math.sqrt(turn_circumference ** 2 + wire_diameter ** 2)
dense_turns = int(wire_length // dense_turn_length)

if dense_turns < 1:
    st.error("Провода недостаточно даже для одного витка.")
    st.stop()

dense_length = dense_turns * wire_diameter
max_model_length = min(max(dense_length * range_factor, wire_diameter * 40), wire_length * 0.95)
coil_lengths = np.linspace(wire_diameter, max_model_length, points)
wire_limit = np.sqrt(np.maximum(wire_length ** 2 - coil_lengths ** 2, 0)) / turn_circumference
packing_limit = coil_lengths / wire_diameter
turns_array = np.floor(np.minimum(wire_limit, packing_limit)).astype(int)
valid = turns_array >= 1
coil_lengths = coil_lengths[valid]
turns_array = turns_array[valid]

if len(coil_lengths) == 0:
    st.error("Для выбранных параметров невозможно построить катушку.")
    st.stop()

pitch = coil_lengths / turns_array
used_wire = np.sqrt((turns_array * turn_circumference) ** 2 + coil_lengths ** 2)
remaining_wire = wire_length - used_wire
turn_density = turns_array / coil_lengths
magnetic_field = MU0 * mu_r * turns_array * current / np.sqrt(coil_lengths ** 2 + frame_diameter ** 2)
inductance = MU0 * mu_r * turns_array ** 2 * section_area / coil_lengths
energy = inductance * current ** 2 / 2
optimal_index = int(np.argmax(magnetic_field))
optimal_length = float(coil_lengths[optimal_index])
optimal_turns = int(turns_array[optimal_index])
optimal_pitch = float(pitch[optimal_index])
optimal_used_wire = float(used_wire[optimal_index])
optimal_remaining_wire = float(remaining_wire[optimal_index])
max_field = float(magnetic_field[optimal_index])
optimal_inductance = float(inductance[optimal_index])
optimal_energy = float(energy[optimal_index])

with st.sidebar:
    selected_length = st.slider("Проверяемая длина катушки l, м", float(coil_lengths[0]), float(coil_lengths[-1]), optimal_length, float((coil_lengths[-1] - coil_lengths[0]) / 1000))

selected_index = int(np.argmin(np.abs(coil_lengths - selected_length)))
selected_length = float(coil_lengths[selected_index])
selected_turns = int(turns_array[selected_index])
selected_pitch = float(pitch[selected_index])
selected_used_wire = float(used_wire[selected_index])
selected_remaining_wire = float(remaining_wire[selected_index])
selected_field = float(magnetic_field[selected_index])
selected_inductance = float(inductance[selected_index])
selected_energy = float(energy[selected_index])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Оптимальное число витков", f"{optimal_turns}")
k2.metric("Оптимальная длина", f"{optimal_length:.4f} м")
k3.metric("Максимальная индукция", f"{max_field * 1000:.4f} мТл")
k4.metric("Индуктивность", f"{optimal_inductance * 1000:.4f} мГн")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Шаг намотки", f"{optimal_pitch * 1000:.3f} мм")
k6.metric("Использовано провода", f"{optimal_used_wire:.3f} м")
k7.metric("Остаток провода", f"{optimal_remaining_wire:.3f} м")
k8.metric("Энергия поля", f"{optimal_energy * 1000:.6f} мДж")

data = pd.DataFrame({
    "l_m": coil_lengths,
    "N": turns_array,
    "pitch_mm": pitch * 1000,
    "turn_density_1_per_m": turn_density,
    "used_wire_m": used_wire,
    "remaining_wire_m": remaining_wire,
    "B_T": magnetic_field,
    "B_mT": magnetic_field * 1000,
    "L_H": inductance,
    "L_mH": inductance * 1000,
    "W_J": energy
})

tab1, tab2, tab3, tab4 = st.tabs(["B=f(l)", "Дополнительные зависимости", "Визуализация намотки", "Таблица и экспорт"])

with tab1:
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=data["l_m"], y=data["B_mT"], mode="lines", name="B(l)"))
    fig_b.add_trace(go.Scatter(x=[optimal_length], y=[max_field * 1000], mode="markers", name="Максимум", marker=dict(size=12)))
    fig_b.add_trace(go.Scatter(x=[selected_length], y=[selected_field * 1000], mode="markers", name="Проверяемая длина", marker=dict(size=10, symbol="diamond")))
    fig_b.update_layout(xaxis_title="Длина катушки l, м", yaxis_title="Магнитная индукция B, мТл", height=540)
    st.plotly_chart(fig_b, width="stretch")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Проверяемая длина", f"{selected_length:.4f} м")
    a2.metric("Число витков", f"{selected_turns}")
    a3.metric("Индукция", f"{selected_field * 1000:.4f} мТл")
    a4.metric("Индуктивность", f"{selected_inductance * 1000:.4f} мГн")
    st.success("Главный график задания построен: магнитная индукция B показана как функция длины катушки l, а максимум отмечен отдельной точкой.")

with tab2:
    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(x=data["l_m"], y=data["N"], mode="lines", name="N(l)"))
    fig_n.add_trace(go.Scatter(x=[optimal_length], y=[optimal_turns], mode="markers", name="Оптимум", marker=dict(size=12)))
    fig_n.update_layout(xaxis_title="Длина катушки l, м", yaxis_title="Число витков N", height=430)
    st.plotly_chart(fig_n, width="stretch")

    fig_l = go.Figure()
    fig_l.add_trace(go.Scatter(x=data["l_m"], y=data["L_mH"], mode="lines", name="Индуктивность"))
    fig_l.add_trace(go.Scatter(x=[optimal_length], y=[optimal_inductance * 1000], mode="markers", name="Оптимум", marker=dict(size=12)))
    fig_l.update_layout(xaxis_title="Длина катушки l, м", yaxis_title="Индуктивность, мГн", height=430)
    st.plotly_chart(fig_l, width="stretch")

    fig_pitch = go.Figure()
    fig_pitch.add_trace(go.Scatter(x=data["l_m"], y=data["pitch_mm"], mode="lines", name="Шаг намотки"))
    fig_pitch.add_hline(y=wire_diameter_mm, line_dash="dash", annotation_text="диаметр провода")
    fig_pitch.update_layout(xaxis_title="Длина катушки l, м", yaxis_title="Шаг между витками, мм", height=430)
    st.plotly_chart(fig_pitch, width="stretch")

with tab3:
    display_turns = max(1, min(selected_turns, 140))
    t = np.linspace(0, 2 * math.pi * display_turns, max(300, display_turns * 35))
    x = selected_length * t / (2 * math.pi * display_turns)
    y = frame_radius * np.cos(t)
    z = frame_radius * np.sin(t)

    fig_3d = go.Figure()
    fig_3d.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name="Витки"))
    fig_3d.update_layout(height=560, scene=dict(xaxis_title="l, м", yaxis_title="y, м", zaxis_title="z, м", aspectmode="data"))
    st.plotly_chart(fig_3d, width="stretch")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Равномерный шаг", f"{selected_pitch * 1000:.3f} мм")
    b2.metric("Использовано провода", f"{selected_used_wire:.3f} м")
    b3.metric("Остаток провода", f"{selected_remaining_wire:.3f} м")
    b4.metric("Плотность витков", f"{selected_turns / selected_length:.2f} 1/м")
    st.info("На рисунке показана равномерная намотка. Для каждого значения l программа заново подбирает целое число витков и одинаковый шаг вдоль всей длины катушки.")

with tab4:
    st.dataframe(data, width="stretch")
    csv = data.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Скачать таблицу CSV", csv, "coil_modeling_results.csv", "text/csv")
    report = f"""Оптимальное число витков: {optimal_turns}
Диаметр каркаса: {frame_diameter:.6f} м
Диаметр провода: {wire_diameter:.6f} м
Длина провода: {wire_length:.6f} м
Ток: {current:.6f} А
Относительная магнитная проницаемость: {mu_r:.6f}
Оптимальная длина катушки: {optimal_length:.6f} м
Шаг намотки: {optimal_pitch:.9f} м
Максимальная магнитная индукция: {max_field:.9f} Тл
Максимальная магнитная индукция: {max_field * 1000:.6f} мТл
Индуктивность при оптимальной длине: {optimal_inductance:.9f} Гн
Индуктивность при оптимальной длине: {optimal_inductance * 1000:.6f} мГн
Энергия магнитного поля: {optimal_energy:.9f} Дж
Использовано провода: {optimal_used_wire:.6f} м
Остаток провода: {optimal_remaining_wire:.6f} м
"""
    st.download_button("Скачать результаты TXT", report.encode("utf-8-sig"), "coil_modeling_summary.txt", "text/plain")
