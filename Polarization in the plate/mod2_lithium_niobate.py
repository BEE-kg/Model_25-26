import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Моделирование поляризации в пластинке LiNbO₃",
    page_icon="🧊",
    layout="wide"
)

def refractive_indices_linbo3(lambda_nm):
    lam = np.asarray(lambda_nm, dtype=float) / 1000.0
    lam2 = lam ** 2
    ne2 = (
        1
        + 2.9804 / (1 - 0.02047 / lam2)
        + 0.5981 / (1 - 0.0666 / lam2)
        + 8.9543 / (1 - 416.08 / lam2)
    )
    no2 = (
        1
        + 2.6734 / (1 - 0.01764 / lam2)
        + 1.2290 / (1 - 0.05914 / lam2)
        + 12.614 / (1 - 474.60 / lam2)
    )
    return np.sqrt(no2), np.sqrt(ne2)

def phase_retardation(delta_n, thickness_um, wavelength_nm):
    return 2 * np.pi * thickness_um * delta_n / (wavelength_nm / 1000.0)

def wrap_phase(delta):
    return np.mod(delta, 2 * np.pi)

def electric_field_curve(ax, ay, phase, t_points, time_angle):
    x = ax * np.cos(t_points)
    y = ay * np.cos(t_points + phase)
    tc = np.deg2rad(time_angle)
    xc = ax * np.cos(tc)
    yc = ay * np.cos(tc + phase)
    return x, y, xc, yc

def stokes_parameters(ex, ey):
    s0 = abs(ex) ** 2 + abs(ey) ** 2
    s1 = abs(ex) ** 2 - abs(ey) ** 2
    s2 = 2 * np.real(ex * np.conj(ey))
    s3 = -2 * np.imag(ex * np.conj(ey))
    psi = 0.5 * np.rad2deg(np.arctan2(s2, s1))
    value = np.clip(s3 / s0, -1, 1)
    chi = 0.5 * np.rad2deg(np.arcsin(value))
    return s0, s1, s2, s3, psi, chi

def polarization_type(ax, ay, phase_mod):
    coupling = abs(2 * ax * ay * np.sin(phase_mod))
    equal_amplitudes = abs(ax - ay)
    quarter = min(abs(phase_mod - np.pi / 2), abs(phase_mod - 3 * np.pi / 2))
    linear_condition = coupling < 0.03
    circular_condition = equal_amplitudes < 0.03 and quarter < 0.05
    if circular_condition:
        return "близкая к круговой"
    if linear_condition:
        return "линейная"
    return "эллиптическая"

def analyzer_intensity(ex, ey, angles_deg):
    theta = np.deg2rad(angles_deg)
    projection = ex * np.cos(theta) + ey * np.sin(theta)
    return np.abs(projection) ** 2

def ellipse_figure(x, y, xc, yc, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="траектория конца вектора E",
            line=dict(width=4)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, xc],
            y=[0, yc],
            mode="lines+markers",
            name="текущий вектор E",
            line=dict(width=5),
            marker=dict(size=[1, 10])
        )
    )
    fig.add_hline(y=0, line_width=1, opacity=0.35)
    fig.add_vline(x=0, line_width=1, opacity=0.35)
    fig.update_layout(
        title=title,
        xaxis_title="компонента вдоль обыкновенной оси",
        yaxis_title="компонента вдоль необыкновенной оси",
        height=520,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=30, r=30, t=70, b=70)
    )
    fig.update_xaxes(range=[-1.1, 1.1], constrain="domain", zeroline=True)
    fig.update_yaxes(range=[-1.1, 1.1], scaleanchor="x", scaleratio=1, zeroline=True)
    return fig

def polarization_3d_figure(ax, ay, phase, title):
    z = np.linspace(0, 2, 900)
    arg = 2 * np.pi * z
    x = ax * np.cos(arg)
    y = ay * np.cos(arg + phase)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            name="пространственная развёртка E",
            line=dict(width=7)
        )
    )
    for pos in np.linspace(0, 2, 9):
        a = 2 * np.pi * pos
        ex_now = ax * np.cos(a)
        ey_now = ay * np.cos(a + phase)
        fig.add_trace(
            go.Scatter3d(
                x=[0, ex_now],
                y=[0, ey_now],
                z=[pos, pos],
                mode="lines",
                line=dict(width=4),
                showlegend=False
            )
        )
    fig.update_layout(
        title=title,
        height=620,
        scene=dict(
            xaxis_title="Eₒ",
            yaxis_title="Eₑ",
            zaxis_title="координата распространения, λ",
            xaxis=dict(range=[-1.1, 1.1]),
            yaxis=dict(range=[-1.1, 1.1]),
            zaxis=dict(range=[0, 2])
        ),
        margin=dict(l=0, r=0, t=70, b=0)
    )
    return fig

def line_marker(fig, x_value, y_values, name):
    for value in y_values:
        fig.add_trace(
            go.Scatter(
                x=[x_value],
                y=[value],
                mode="markers",
                marker=dict(size=10),
                name=name,
                showlegend=False
            )
        )
    return fig

st.title("Моделирование изменения поляризации света в пластинке LiNbO₃")

with st.sidebar:
    st.header("Параметры модели")
    wavelength_nm = st.slider("Длина волны λ, нм", 450.0, 650.0, 532.0, 1.0)
    thickness_unit = st.selectbox("Единица толщины пластинки", ["мкм", "мм"])
    if thickness_unit == "мкм":
        thickness_value = st.number_input("Толщина пластинки d, мкм", 0.1, 5000.0, 100.0, 1.0)
        thickness_um = thickness_value
    else:
        thickness_value = st.number_input("Толщина пластинки d, мм", 0.001, 5.0, 0.1, 0.001, format="%.3f")
        thickness_um = thickness_value * 1000.0

    input_angle_deg = st.slider("Угол входной линейной поляризации α, град", 0.0, 90.0, 45.0, 1.0)
    time_angle_deg = st.slider("Фаза времени для отображения вектора E, град", 0.0, 360.0, 35.0, 1.0)
    analyzer_angle_deg = st.slider("Угол анализатора, град", 0.0, 180.0, 90.0, 1.0)
    samples = st.slider("Точек на кривой поляризации", 200, 1200, 600, 50)

    st.divider()
    index_mode = st.radio(
        "Показатели преломления",
        ["формулы LiNbO₃ из задания", "ручное значение Δn"],
        index=0
    )

    if index_mode == "ручное значение Δn":
        no_manual = st.number_input("nₒ", 1.0, 4.0, 2.30, 0.001, format="%.4f")
        delta_n_manual = st.number_input("Δn = nₑ - nₒ", -1.0, 1.0, -0.09, 0.001, format="%.4f")

if index_mode == "формулы LiNbO₃ из задания":
    no_value, ne_value = refractive_indices_linbo3(wavelength_nm)
else:
    no_value = no_manual
    ne_value = no_manual + delta_n_manual

delta_n = ne_value - no_value
phase = phase_retardation(delta_n, thickness_um, wavelength_nm)
phase_mod = wrap_phase(phase)
phase_waves = phase / (2 * np.pi)

alpha = np.deg2rad(input_angle_deg)
ao = float(np.cos(alpha))
ae = float(np.sin(alpha))

t = np.linspace(0, 2 * np.pi, samples)
xin, yin, xcin, ycin = electric_field_curve(ao, ae, 0.0, t, time_angle_deg)
xout, yout, xcout, ycout = electric_field_curve(ao, ae, phase_mod, t, time_angle_deg)

ex = ao + 0j
ey = ae * np.exp(1j * phase_mod)
s0, s1, s2, s3, psi_deg, chi_deg = stokes_parameters(ex, ey)
pol_type = polarization_type(ao, ae, phase_mod)
axis_ratio = abs(np.tan(np.deg2rad(chi_deg)))
axis_ratio = 0.0 if axis_ratio < 1e-12 else axis_ratio

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("nₒ", f"{no_value:.5f}")
c2.metric("nₑ", f"{ne_value:.5f}")
c3.metric("Δn", f"{delta_n:.5f}")
c4.metric("Δφ mod 2π", f"{phase_mod:.3f} рад")
c5.metric("Выход", pol_type)

with st.expander("Используемая физическая модель"):
    st.markdown(
        """
Модель основана на разложении входного линейно поляризованного света на две взаимно ортогональные компоненты: вдоль обыкновенной и необыкновенной осей кристалла.  
В кристалле LiNbO₃ эти компоненты распространяются с разными показателями преломления, поэтому между ними появляется фазовая задержка.
"""
    )
    st.latex(r"\Delta n = n_e - n_o")
    st.latex(r"\Delta \varphi = \frac{2\pi d}{\lambda}(n_e-n_o)")
    st.latex(r"E_o(t)=A_o\cos(\omega t), \qquad E_e(t)=A_e\cos(\omega t+\Delta \varphi)")
    st.markdown(
        """
Если фазовая задержка равна 0 или π, поляризация остаётся линейной.  
Если амплитуды компонент равны, а фазовая задержка близка к π/2 или 3π/2, поляризация становится близкой к круговой.  
В остальных случаях на выходе получается эллиптическая поляризация.
"""
    )

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Вход и выход",
        "3D-визуализация",
        "Спектральные зависимости",
        "Фазовая задержка и анализатор",
        "Таблица и экспорт"
    ]
)

with tab1:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            ellipse_figure(xin, yin, xcin, ycin, "Входная линейная поляризация"),
            width="stretch"
        )
    with right:
        st.plotly_chart(
            ellipse_figure(xout, yout, xcout, ycout, "Выходная поляризация после пластинки"),
            width="stretch"
        )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Фазовая задержка", f"{phase:.3f} рад")
    r2.metric("Фазовая задержка", f"{phase_waves:.3f} λ")
    r3.metric("Угол эллипса ψ", f"{psi_deg:.2f}°")
    r4.metric("Эллиптичность χ", f"{chi_deg:.2f}°")

    st.markdown(
        f"""
При выбранных параметрах входной свет является линейно поляризованным под углом **{input_angle_deg:.1f}°** к обыкновенной оси.  
После прохождения через пластинку между ортогональными компонентами появляется фазовый сдвиг, поэтому состояние на выходе: **{pol_type}**.
"""
    )

with tab2:
    st.markdown(
        """
3D-график показывает пространственную развёртку электрического поля вдоль направления распространения. 
Для линейной поляризации конец вектора лежит в одной плоскости, а при появлении фазовой задержки между компонентами возникает эллиптическая или круговая форма.
"""
    )
    left3d, right3d = st.columns(2)
    with left3d:
        st.plotly_chart(
            polarization_3d_figure(ao, ae, 0.0, "3D-развёртка входной линейной поляризации"),
            width="stretch"
        )
    with right3d:
        st.plotly_chart(
            polarization_3d_figure(ao, ae, phase_mod, "3D-развёртка выходной поляризации"),
            width="stretch"
        )

with tab3:
    wavelengths = np.linspace(450, 650, 501)
    no_arr, ne_arr = refractive_indices_linbo3(wavelengths)
    dn_arr = ne_arr - no_arr

    fig_n = go.Figure()
    fig_n.add_trace(go.Scatter(x=wavelengths, y=no_arr, mode="lines", name="nₒ(λ)", line=dict(width=4)))
    fig_n.add_trace(go.Scatter(x=wavelengths, y=ne_arr, mode="lines", name="nₑ(λ)", line=dict(width=4)))
    fig_n.add_vline(x=wavelength_nm, line_width=2, line_dash="dash")
    fig_n.add_trace(go.Scatter(x=[wavelength_nm], y=[no_value], mode="markers", marker=dict(size=11), name="текущее nₒ"))
    fig_n.add_trace(go.Scatter(x=[wavelength_nm], y=[ne_value], mode="markers", marker=dict(size=11), name="текущее nₑ"))
    fig_n.update_layout(
        title="Дисперсия показателей преломления LiNbO₃",
        xaxis_title="длина волны λ, нм",
        yaxis_title="показатель преломления",
        height=520,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=30, r=30, t=70, b=70)
    )
    st.plotly_chart(fig_n, width="stretch")

    fig_dn = go.Figure()
    fig_dn.add_trace(go.Scatter(x=wavelengths, y=dn_arr, mode="lines", name="Δn(λ)=nₑ-nₒ", line=dict(width=4)))
    fig_dn.add_vline(x=wavelength_nm, line_width=2, line_dash="dash")
    fig_dn.add_trace(go.Scatter(x=[wavelength_nm], y=[delta_n], mode="markers", marker=dict(size=11), name="текущее Δn"))
    fig_dn.update_layout(
        title="Двулучепреломление Δn от длины волны",
        xaxis_title="длина волны λ, нм",
        yaxis_title="Δn",
        height=460,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=30, r=30, t=70, b=70)
    )
    st.plotly_chart(fig_dn, width="stretch")

with tab4:
    phase_arr = phase_retardation(dn_arr, thickness_um, wavelengths)
    phase_mod_arr = wrap_phase(phase_arr)
    phase_waves_arr = phase_arr / (2 * np.pi)

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=wavelengths, y=phase_waves_arr, mode="lines", name="Δφ / 2π", line=dict(width=4)))
    fig_phase.add_vline(x=wavelength_nm, line_width=2, line_dash="dash")
    fig_phase.add_trace(go.Scatter(x=[wavelength_nm], y=[phase_waves], mode="markers", marker=dict(size=11), name="текущее значение"))
    fig_phase.update_layout(
        title="Фазовая задержка в долях периода",
        xaxis_title="длина волны λ, нм",
        yaxis_title="Δφ / 2π",
        height=460,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=30, r=30, t=70, b=70)
    )
    st.plotly_chart(fig_phase, width="stretch")

    fig_phase_mod = go.Figure()
    fig_phase_mod.add_trace(go.Scatter(x=wavelengths, y=phase_mod_arr, mode="lines", name="Δφ mod 2π", line=dict(width=4)))
    fig_phase_mod.add_vline(x=wavelength_nm, line_width=2, line_dash="dash")
    fig_phase_mod.add_trace(go.Scatter(x=[wavelength_nm], y=[phase_mod], mode="markers", marker=dict(size=11), name="текущее значение"))
    fig_phase_mod.update_layout(
        title="Фазовая задержка, приведённая к интервалу 0...2π",
        xaxis_title="длина волны λ, нм",
        yaxis_title="Δφ mod 2π, рад",
        height=460,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=30, r=30, t=70, b=70)
    )
    st.plotly_chart(fig_phase_mod, width="stretch")

    angles = np.linspace(0, 180, 721)
    intensities = analyzer_intensity(ex, ey, angles)
    current_intensity = analyzer_intensity(ex, ey, np.array([analyzer_angle_deg]))[0]

    fig_an = go.Figure()
    fig_an.add_trace(go.Scatter(x=angles, y=intensities, mode="lines", name="I(θ)", line=dict(width=4)))
    fig_an.add_vline(x=analyzer_angle_deg, line_width=2, line_dash="dash")
    fig_an.add_trace(go.Scatter(x=[analyzer_angle_deg], y=[current_intensity], mode="markers", marker=dict(size=11), name="текущее значение"))
    fig_an.update_layout(
        title="Интенсивность после линейного анализатора",
        xaxis_title="угол анализатора θ, град",
        yaxis_title="нормированная интенсивность",
        height=460,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=30, r=30, t=70, b=70)
    )
    st.plotly_chart(fig_an, width="stretch")

with tab5:
    table = pd.DataFrame(
        {
            "λ, нм": wavelengths,
            "nₒ": no_arr,
            "nₑ": ne_arr,
            "Δn": dn_arr,
            "Δφ, рад": phase_arr,
            "Δφ mod 2π, рад": phase_mod_arr,
            "Δφ / 2π": phase_waves_arr
        }
    )

    current_table = pd.DataFrame(
        {
            "Параметр": [
                "длина волны λ, нм",
                "толщина пластинки d, мкм",
                "угол входной поляризации α, град",
                "nₒ",
                "nₑ",
                "Δn",
                "Δφ, рад",
                "Δφ mod 2π, рад",
                "Δφ / 2π",
                "тип выходной поляризации",
                "угол ориентации эллипса ψ, град",
                "эллиптичность χ, град",
                "отношение малой полуоси к большой"
            ],
            "Значение": [
                f"{wavelength_nm:.3f}",
                f"{thickness_um:.3f}",
                f"{input_angle_deg:.3f}",
                f"{no_value:.6f}",
                f"{ne_value:.6f}",
                f"{delta_n:.6f}",
                f"{phase:.6f}",
                f"{phase_mod:.6f}",
                f"{phase_waves:.6f}",
                pol_type,
                f"{psi_deg:.6f}",
                f"{chi_deg:.6f}",
                f"{axis_ratio:.6f}"
            ]
        }
    )

    st.subheader("Текущие расчётные значения")
    st.dataframe(current_table, width="stretch", hide_index=True)

    st.subheader("Таблица по диапазону длин волн 450–650 нм")
    st.dataframe(table, width="stretch", hide_index=True)

    csv_table = table.to_csv(index=False).encode("utf-8-sig")
    csv_current = current_table.to_csv(index=False).encode("utf-8-sig")

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Скачать таблицу по спектру CSV",
            data=csv_table,
            file_name="linbo3_spectral_table.csv",
            mime="text/csv"
        )
    with col_b:
        st.download_button(
            "Скачать текущие параметры CSV",
            data=csv_current,
            file_name="linbo3_current_state.csv",
            mime="text/csv"
        )
