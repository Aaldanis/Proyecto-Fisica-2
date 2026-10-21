# =============================================================================
# ASISTENTE VIRTUAL DE ESTADÍSTICA APLICADA
# Universidad Mariano Gálvez de Guatemala
# Código completamente comentado línea por línea
# =============================================================================

# -----------------------------------------------------------------------------
# IMPORTACIONES
# -----------------------------------------------------------------------------
import streamlit as st                  # Crea la interfaz web
import numpy as np                      # Cálculos matemáticos (media, desviación, etc.)
import pandas as pd                     # Tablas de datos (frecuencias)
import matplotlib.pyplot as plt         # Gráficas (histogramas, dispersión)
from scipy import stats                 # Estadística avanzada (moda, normal, correlación)
import re                               # Busca letras o caracteres inválidos
import os
from dotenv import load_dotenv
from collections import Counter
import google.generativeai as genai
load_dotenv()

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(                     # Configura el título y diseño de la página
    page_title="Asistente de Estadística UMG",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------------------------
# DATOS QUE DEBES EDITAR
# -----------------------------------------------------------------------------
INTEGRANTES = "Edgar Aldanis Chávez Estrada y Erick Adolfo de Leon Rivera"   # Nombres de los integrantes
SECCION = "Ing. M.A Reyes Baltazar Mazá Guerra"                                       # Sección del curso
ASIGNATURA = "Fisica 2"                         # Nombre de la asignatura
UNIVERSIDAD = "Universidad Mariano Gálvez de Guatemala"     # Universidad
ADICIONAL = "Mas opciones y asistente IA en la barra lateral izquierda"

# URL del vídeo de fondo (déjalo vacío si no quieres vídeo)
VIDEO_FONDO = "https://github.com/Aaldanis/Wallpapers/raw/refs/heads/main/celestial-veil.1920x1080.mp4"                                            # Ejemplo: "https://...mp4"

# -----------------------------------------------------------------------------
# ESTILOS CSS (diseño visual)
# -----------------------------------------------------------------------------
with open("styles.css", "r", encoding="utf-8") as f:
    CSS = f"<style>{f.read()}</style>"
st.html(CSS)                            # Aplica los estilos a la página

# Vídeo de fondo (solo si se puso una URL)
if VIDEO_FONDO:
    st.html(f'<video autoplay muted loop playsinline id="video-fondo"><source src="{VIDEO_FONDO}" type="video/mp4"></video>')

# -----------------------------------------------------------------------------
# FUNCIONES
# -----------------------------------------------------------------------------

from backend import parsear_datos, calcular


def mostrar_procedimiento(res, unidad=""):
    """Muestra el procedimiento completo con fórmulas y valores sustituidos."""
    u = f" {unidad}" if unidad else ""                          # Unidad (si existe)
    datos = res["datos"]
    n = res["n"]
    media = res["media"]

    st.markdown("#### Procedimiento completo")
    st.write(f"**Datos:** {', '.join(map(str, datos))}{u}")
    st.write(f"**n =** {n}")

    # Media
    suma = np.sum(datos)
    st.markdown("**1. Media aritmética**")
    st.latex(r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}")
    suma_str = " + ".join([str(x) for x in datos])
    st.latex(rf"\bar{{x}} = \frac{{{suma_str}}}{{{n}}} = \frac{{{suma:.4f}}}{{{n}}} = {media:.4f}")

    # Mediana
    st.markdown("**2. Mediana**")
    st.latex(r"\text{Mediana} = \begin{cases} x_{\frac{n+1}{2}} & \text{si } n \text{ es impar} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{si } n \text{ es par} \end{cases}")
    ordenados = np.sort(datos)
    st.write(f"Datos ordenados: {', '.join(map(str, ordenados))}")
    if n % 2 == 1:
        pos = (n + 1) // 2
        st.markdown(f"n = {n} (impar)")
        st.markdown(f"$$\\text{{Posición}} = \\frac{{n+1}}{{2}} = \\frac{{{n}+1}}{{2}} = {pos}$$")
        st.markdown(f"Mediana = dato en posición {pos} = **{res['mediana']:.4f}{u}**")
    else:
        pos1 = n // 2
        pos2 = pos1 + 1
        st.markdown(f"n = {n} (par)")
        st.markdown(f"$$\\text{{Mediana}} = \\frac{{x_{{{pos1}}} + x_{{{pos2}}}}}{{2}} = \\frac{{{ordenados[pos1-1]:.4f} + {ordenados[pos2-1]:.4f}}}{{2}} = {res['mediana']:.4f}$$")

    # Moda
    st.markdown("**3. Moda**")
    st.latex(r"\text{Moda} = \text{valor con mayor frecuencia}")
    if isinstance(res["moda"], str):
        st.write(f"Moda: **{res['moda']}**")
    else:
        frec = Counter(datos)[res["moda"]]
        st.markdown(f"El valor {res['moda']} aparece {frec} veces.")
        st.markdown(f"$$\\text{{Moda}} = {res['moda']}$$")

    # Dispersión
    st.markdown("**4. Medidas de dispersión**")
    st.markdown("- **Rango**")
    st.latex(r"R = x_{\max} - x_{\min}")
    st.latex(rf"R = {np.max(datos):.4f} - {np.min(datos):.4f} = {res['rango']:.4f}")

    desviaciones = datos - media
    cuadrados = desviaciones ** 2
    suma_cuadrados = np.sum(cuadrados)

    st.markdown("- **Desviación estándar muestral (s)**")
    st.latex(r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}")
    if n <= 10:
        terminos = " + ".join([f"({x:.2f}-{media:.2f})^2" for x in datos])
        st.latex(rf"\sum (x_i-\bar{{x}})^2 = {terminos} = {suma_cuadrados:.4f}")
    st.latex(rf"s = \sqrt{{\frac{{{suma_cuadrados:.4f}}}{{{n-1}}}}} = {res['std_m']:.4f}")

    st.markdown("- **Desviación estándar poblacional (σ)**")
    st.latex(r"\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{n}}")
    st.latex(rf"\sigma = \sqrt{{\frac{{{suma_cuadrados:.4f}}}{{{n}}}}} = {res['std_p']:.4f}")

    st.markdown("- **Varianza muestral**")
    st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1}")
    st.latex(rf"s^2 = \frac{{{suma_cuadrados:.4f}}}{{{n-1}}} = {res['var_m']:.4f}")

    st.success(f"**Resultado final:** Media = {media:.4f}{u} | s = {res['std_m']:.4f}{u}")
    st.info(f"La desviación estándar muestral (s) usa n-1. La poblacional (σ) usa n.")


# -----------------------------------------------------------------------------
# ENCABEZADO DE LA PÁGINA
# -----------------------------------------------------------------------------
st.html(f"""
<div class="glass-card">
    <div style="color:#67e8f9; font-weight:600;">{UNIVERSIDAD}</div>
    <div style="color:#67e8f9; font-weight:600;">{ADICIONAL}</div>
    <h1 class="hero-title">Asistente de Estadística Aplicada</h1>
    <p style="color:#e2e8f0;"><b>Integrantes:</b> {INTEGRANTES} · <b>Profesor:</b> {SECCION}</p>
</div>
""")

# -----------------------------------------------------------------------------
# MENÚ LATERAL
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Opciones")
    seccion = st.radio("", ["Módulos principales", "Ejercicios de ejemplo", "Practicar"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Asistente IA")
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            api_key = ""
    if api_key:
        st.success("API KEY encontrada")
    else:
        st.warning("API Key no configurada (.env o Streamlit Secrets)")
    pregunta_ia = st.text_area("Pregunta sobre Estadística", placeholder="Ej: ¿Cuándo se usa la desviación estándar muestral?", height=80, key="pregunta_ia")
    if st.button("Preguntar a Gemini"):
        if not api_key:
            st.error("API Key no configurada. Agrégala en .env o en Streamlit Secrets.")
        elif not pregunta_ia.strip():
            st.warning("Escribe una pregunta")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-2.0-flash-lite")
                prompt = f"""Eres un asistente de Estadística Aplicada. 
Solo puedes responder preguntas relacionadas con:
- Estadística descriptiva (media, mediana, moda, rango, dispersión)
- Datos experimentales y unidades
- Tablas de frecuencia y gráficas
- Probabilidad y distribución normal
- Correlación y regresión lineal
- Diferencia entre desviación muestral y poblacional

Si la pregunta NO está relacionada con estos temas, responde EXACTAMENTE:
"Este asistente solo trabaja y responde al tema de Estadística Aplicada."

Pregunta del usuario: {pregunta_ia}"""
                response = model.generate_content(prompt)
                st.markdown("**Respuesta:**")
                st.info(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    st.markdown("---")
    st.caption(f"{INTEGRANTES}\n{SECCION}")

# -----------------------------------------------------------------------------
# MÓDULOS PRINCIPALES
# -----------------------------------------------------------------------------
if seccion == "Módulos principales":
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Descriptiva", "2. Datos Experimentales", "3. Tablas y Gráficas",
        "4. Probabilidad", "5. Correlación"
    ])

    # ----- Módulo 1: Estadística Descriptiva -----
    with tab1:
        st.html('<div class="glass-card">')
        st.subheader("Estadística Descriptiva")
        unidad = st.text_input("Unidad (opcional)", placeholder="A, m, kg...")
        datos = st.text_area("Datos", placeholder="2.1, 2.3, 2.2, 2.4, 2.0", height=100)
        if st.button("Calcular descriptiva", key="btn1"):
            arr, err = parsear_datos(datos)
            if err:
                st.error(err)
            else:
                r = calcular(arr)
                mostrar_procedimiento(r, unidad)
                u = f" {unidad}" if unidad else ""
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Media", f"{r['media']:.4f}{u}")
                c2.metric("Mediana", f"{r['mediana']:.4f}{u}")
                c3.metric("s", f"{r['std_m']:.4f}{u}")
                c4.metric("Rango", f"{r['rango']:.4f}{u}")
        st.html('</div>')

    # ----- Módulo 2: Datos Experimentales -----
    with tab2:
        st.html('<div class="glass-card">')
        st.subheader("Datos Experimentales")
        st.info("Convierte primero las mediciones a la misma unidad.")
        col1, col2 = st.columns([3, 1])
        with col1:
            datos = st.text_area("Mediciones", placeholder="1.20, 1.22, 1.19, 1.21", height=100)
        with col2:
            unidad = st.text_input("Unidad", placeholder="m")
        if st.button("Procesar mediciones", key="btn2"):
            arr, err = parsear_datos(datos)
            if err:
                st.error(err)
            else:
                r = calcular(arr)
                u = unidad.strip() if unidad else ""
                st.markdown("### Paso 1: Datos")
                st.write(f"{', '.join(map(str, arr))} {u}  |  n = {r['n']}")
                st.markdown("### Paso 2: Media")
                st.markdown(f"$$\\bar{{x}} = {r['media']:.4f}\\ {u}$$")
                st.markdown("### Paso 3: Dispersión (ambas)")
                st.markdown(f"$$s = {r['std_m']:.4f}\\ {u}$$")
                st.markdown(f"$$\\sigma = {r['std_p']:.4f}\\ {u}$$")
                st.markdown("### Paso 4: Resultado experimental")
                st.markdown(f"$$({r['media']:.4f} \\pm {r['std_m']:.4f})\\ {u}$$")
                st.success(f"Resultado (con s): ({r['media']:.4f} ± {r['std_m']:.4f}) {u}")
                st.info(f"σ = {r['std_p']:.4f} {u}")
                st.markdown("---")
                st.markdown("### Procedimiento completo")
                mostrar_procedimiento(r, u)
        st.html('</div>')

    # ----- Módulo 3: Tablas y Gráficas -----
    with tab3:
        st.html('<div class="glass-card">')
        st.subheader("Tablas de Frecuencia y Gráficas")
        datos = st.text_area("Datos", height=90, placeholder="3.1, 3.6, 3.8, 3.9, 4.2, ...")
        tipo = st.radio("Gráfica", ["Histograma", "Diagrama de dispersión"], horizontal=True)
        datos_y = st.text_area("Datos Y", height=70) if tipo == "Diagrama de dispersión" else ""
        if st.button("Generar", key="btn3"):
            arr, err = parsear_datos(datos)
            if err:
                st.error(err)
            else:
                n = len(arr)
                # Regla de Sturges para número de clases
                num_clases = int(np.ceil(1 + 3.322 * np.log10(n)))
                num_clases = max(3, min(num_clases, 12))
                st.info(f"Número de clases (Regla de Sturges): **{num_clases}**")

                minimo, maximo = arr.min(), arr.max()
                bins = np.linspace(minimo, maximo + 0.0001, num_clases + 1)
                frecuencias, bordes = np.histogram(arr, bins=bins)

                filas = []
                frec_acum = 0
                for i in range(num_clases):
                    lim_inf, lim_sup = bordes[i], bordes[i + 1]
                    marca = (lim_inf + lim_sup) / 2
                    f_abs = int(frecuencias[i])
                    frec_acum += f_abs
                    f_rel = f_abs / n
                    intervalo = f"[{lim_inf:.2f} – {lim_sup:.2f})" if i < num_clases - 1 else f"[{lim_inf:.2f} – {lim_sup:.2f}]"
                    filas.append({
                        "Clase (Intervalo)": intervalo,
                        "Marca de clase (xᵢ)": round(marca, 3),
                        "Frecuencia absoluta (fᵢ)": f_abs,
                        "Frecuencia relativa (hᵢ)": f"{f_rel:.2f} ({f_rel*100:.0f}%)",
                        "Frecuencia acumulada (Fᵢ)": frec_acum
                    })
                filas.append({
                    "Clase (Intervalo)": "Total",
                    "Marca de clase (xᵢ)": "",
                    "Frecuencia absoluta (fᵢ)": n,
                    "Frecuencia relativa (hᵢ)": "1.00 (100%)",
                    "Frecuencia acumulada (Fᵢ)": ""
                })
                st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

                fig, ax = plt.subplots()
                if tipo == "Histograma":
                    ax.hist(arr, bins=bins, color="#38bdf8", edgecolor="white")
                    ax.set_title("Histograma")
                else:
                    y, e = parsear_datos(datos_y)
                    if e or len(y) != len(arr):
                        st.error("Serie Y inválida")
                    else:
                        ax.scatter(arr, y, color="#c084fc")
                        ax.set_title("Diagrama de dispersión")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
        st.html('</div>')

    # ----- Módulo 4: Probabilidad -----
    with tab4:
        st.html('<div class="glass-card">')
        st.subheader("Distribución Normal")
        modo = st.radio("¿Cómo quieres ingresar los parámetros?",
                        ["Ingresar parámetros teóricos (μ y σ)", "Ingresar datos en bruto (lista)"],
                        horizontal=True)
        mu = 0.0
        sigma = 1.0
        if modo == "Ingresar parámetros teóricos (μ y σ)":
            c1, c2 = st.columns(2)
            mu = c1.number_input("Media μ", value=0.0)
            sigma = c2.number_input("Desviación σ", value=1.0, min_value=0.01)
        else:
            datos_brutos = st.text_area("Datos en bruto", placeholder="2.1, 2.3, 2.2, 2.4, 2.0", height=100)
            if datos_brutos.strip():
                arr, err = parsear_datos(datos_brutos)
                if err:
                    st.error(err)
                else:
                    mu = float(np.mean(arr))
                    sigma = float(np.std(arr, ddof=1))
                    st.success(f"Parámetros calculados: μ = {mu:.4f} | σ (s) = {sigma:.4f}")

        tipo = st.selectbox("Calcular", [
            "P(X ≤ x) — Probabilidad acumulada a la izquierda",
            "P(X ≥ x) — Probabilidad acumulada a la derecha",
            "P(a ≤ X ≤ b) — Probabilidad entre dos valores"
        ])
        if "P(a ≤ X ≤ b)" in tipo:
            a = st.number_input("a", value=-1.0)
            b = st.number_input("b", value=1.0)
        else:
            x = st.number_input("x", value=0.0)

        if st.button("Calcular probabilidad", key="btn4"):
            if sigma <= 0:
                st.error("La desviación estándar debe ser mayor que 0.")
            else:
                dist = stats.norm(mu, sigma)
                st.latex(r"Z = \frac{X - \mu}{\sigma}")
                if "P(X ≤ x)" in tipo:
                    z = (x - mu) / sigma
                    st.latex(rf"Z = \frac{{{x} - {mu:.4f}}}{{{sigma:.4f}}} = {z:.4f}")
                    st.success(f"P(X ≤ {x}) = {dist.cdf(x):.6f}")
                elif "P(X ≥ x)" in tipo:
                    st.success(f"P(X ≥ {x}) = {1 - dist.cdf(x):.6f}")
                else:
                    st.success(f"P({a} ≤ X ≤ {b}) = {dist.cdf(b) - dist.cdf(a):.6f}")
        st.html('</div>')

    # ----- Módulo 5: Correlación -----
    with tab5:
        st.html('<div class="glass-card">')
        st.subheader("Correlación y Regresión Lineal")
        c1, c2 = st.columns(2)
        dx = c1.text_area("X", placeholder="1,2,3,4,5")
        dy = c2.text_area("Y", placeholder="2,4,5,4,6")
        if st.button("Calcular", key="btn5"):
            x, ex = parsear_datos(dx)
            y, ey = parsear_datos(dy)
            if ex or ey:
                st.error(ex or ey)
            elif len(x) != len(y):
                st.error("Deben tener la misma cantidad")
            else:
                media_x = np.mean(x)
                media_y = np.mean(y)
                dx_arr = x - media_x
                dy_arr = y - media_y
                suma_dxdy = np.sum(dx_arr * dy_arr)
                suma_dx2 = np.sum(dx_arr ** 2)
                suma_dy2 = np.sum(dy_arr ** 2)
                r, _ = stats.pearsonr(x, y)
                pendiente, intercepto, *_ = stats.linregress(x, y)

                st.markdown("### Paso 1: Medias")
                st.markdown(f"$$\\bar{{x}} = {media_x:.4f}$$")
                st.markdown(f"$$\\bar{{y}} = {media_y:.4f}$$")

                st.markdown("### Paso 2: Coeficiente de correlación (r)")
                st.latex(r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}")
                st.markdown(f"$$\\sum (x_i - \\bar{{x}})(y_i - \\bar{{y}}) = {suma_dxdy:.4f}$$")
                st.markdown(f"$$\\sum (x_i - \\bar{{x}})^2 = {suma_dx2:.4f}$$")
                st.markdown(f"$$\\sum (y_i - \\bar{{y}})^2 = {suma_dy2:.4f}$$")
                st.markdown(f"$$r = \\frac{{{suma_dxdy:.4f}}}{{\\sqrt{{{suma_dx2:.4f} \\times {suma_dy2:.4f}}}}} = {r:.4f}$$")

                st.markdown("### Paso 3: Recta de regresión")
                st.latex(r"\hat{y} = a + bx")
                st.markdown(f"$$b = {pendiente:.4f}$$")
                st.markdown(f"$$a = {intercepto:.4f}$$")
                st.markdown(f"$$\\hat{{y}} = {intercepto:.4f} + {pendiente:.4f}x$$")

                fuerza = "fuerte" if abs(r) > 0.7 else "moderada" if abs(r) > 0.4 else "débil"
                direccion = "positiva" if r > 0 else "negativa"
                st.info(f"Interpretación: correlación {fuerza} y {direccion}.")
                st.metric("r", f"{r:.4f}")
                st.metric("r²", f"{r**2:.4f}")

                fig, ax = plt.subplots()
                ax.scatter(x, y, color="#38bdf8")
                ax.plot(x, intercepto + pendiente * x, color="#c084fc", label="Recta de regresión")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
        st.html('</div>')

# -----------------------------------------------------------------------------
# EJERCICIOS DE EJEMPLO
# -----------------------------------------------------------------------------
elif seccion == "Ejercicios de ejemplo":
    st.html('<div class="glass-card">')
    st.subheader("Ejercicios de ejemplo")
    ejemplos = {
        "Corriente (ejemplo del proyecto)": "2.1, 2.3, 2.2, 2.4, 2.0",
        "Notas": "85, 90, 78, 92, 88, 76, 95, 82",
        "Alturas (cm)": "165, 170, 168, 172, 160, 175, 169",
        "Tiempos (s)": "0.25, 0.28, 0.22, 0.30, 0.26, 0.24",
        "Temperaturas": "22.5, 23.1, 21.8, 22.9, 23.4, 22.0",
        "Pesos (g)": "50.2, 49.8, 50.5, 49.9, 50.1, 50.3",
        "Velocidades": "45, 52, 48, 55, 50, 47, 53",
        "Puntajes": "70, 85, 60, 90, 75, 80, 65, 88",
        "pH": "7.1, 7.0, 7.2, 6.9, 7.1, 7.3",
        "Energía (kWh)": "120, 135, 110, 145, 125, 130, 140",
        "Presiones": "1.01, 1.02, 0.99, 1.03, 1.00, 1.01",
        "Distancias (m)": "10.5, 11.2, 10.8, 11.0, 10.9, 11.1"
    }
    ej = st.selectbox("Elige", list(ejemplos.keys()))
    st.code(ejemplos[ej])
    if st.button("Resolver ejemplo"):
        arr, err = parsear_datos(ejemplos[ej])
        if not err:
            mostrar_procedimiento(calcular(arr))
    st.html('</div>')

# -----------------------------------------------------------------------------
# PRÁCTICA
# -----------------------------------------------------------------------------
elif seccion == "Practicar":
    st.html('<div class="glass-card">')
    st.subheader("Práctica con retroalimentación")
    if "prac" not in st.session_state:
        st.session_state.prac = None
    if st.button("Generar ejercicio nuevo"):
        n = np.random.randint(6, 11)
        base = np.random.uniform(15, 90)
        st.session_state.prac = np.round(np.random.normal(base, base * 0.12, n), 2)
        st.rerun()
    if st.session_state.prac is not None:
        datos = st.session_state.prac
        st.code(", ".join(map(str, datos)))
        c1, c2 = st.columns(2)
        mu_u = c1.number_input("Tu media", format="%.4f", step=0.01)
        s_u = c2.number_input("Tu desviación estándar muestral", format="%.4f", step=0.01)
        if st.button("Verificar"):
            mu_ok, s_ok = np.mean(datos), np.std(datos, ddof=1)
            if abs(mu_u - mu_ok) < 0.05 and abs(s_u - s_ok) < 0.05:
                st.success("¡Correcto!")
            else:
                st.warning("Revisa:")
                if abs(mu_u - mu_ok) >= 0.05:
                    st.error(f"Media correcta: {mu_ok:.4f}")
                if abs(s_u - s_ok) >= 0.05:
                    st.error(f"s correcta: {s_ok:.4f}")
            st.markdown("**Procedimiento correcto**")
            mostrar_procedimiento(calcular(datos))
    else:
        st.info("Genera un ejercicio para empezar.")
    st.html('</div>')

st.markdown("---")
st.caption(f"{INTEGRANTES} · {ASIGNATURA} · {SECCION}")
