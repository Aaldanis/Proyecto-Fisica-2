# =============================================================================
# ASISTENTE VIRTUAL DE ESTADÍSTICA APLICADA
# Universidad Mariano Gálvez de Guatemala
# =============================================================================
# Este archivo es la interfaz principal de la aplicación hecha con Streamlit.
# Controla menús, pestañas, entradas de datos, botones y la forma de mostrar resultados.

# -----------------------------------------------------------------------------
# IMPORTACIONES
# -----------------------------------------------------------------------------
import streamlit as st                  # Librería principal para crear la interfaz web
import numpy as np                      # Cálculos matemáticos (media, desviación, arrays)
import pandas as pd                     # Manejo de tablas de datos (frecuencias)
import matplotlib.pyplot as plt         # Creación de gráficas (histogramas, dispersión)
from scipy import stats                 # Estadística avanzada (moda, normal, correlación)
import re                               # Expresiones regulares para detectar letras o caracteres inválidos
import os                               # Acceso a variables de entorno del sistema
from dotenv import load_dotenv          # Carga variables desde un archivo .env
from collections import Counter         # Contador de frecuencias (útil para calcular la moda)
import google.generativeai as genai     # Librería para usar el modelo de IA Gemini de Google

# Carga las variables de entorno del archivo .env (si existe)
load_dotenv()

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
# Configura el título de la pestaña del navegador, el ícono y el diseño ancho
st.set_page_config(
    page_title="Asistente de Estadística UMG",   # Título que aparece en la pestaña
    page_icon="📊",                              # Ícono de la pestaña
    layout="wide"                                # La página ocupa todo el ancho disponible
)

# -----------------------------------------------------------------------------
# DATOS QUE DEBES EDITAR
# -----------------------------------------------------------------------------
INTEGRANTES = "Edgar Aldanis Chávez Estrada y Erick Adolfo de Leon Rivera"   # Nombres de los integrantes
SECCION = "Ing. M.A Reyes Baltazar Mazá Guerra"                              # Nombre del profesor
ASIGNATURA = "Fisica 2"                                                      # Nombre de la asignatura
UNIVERSIDAD = "Universidad Mariano Gálvez de Guatemala"                      # Nombre de la universidad
ADICIONAL = "Mas opciones y asistente IA en la barra lateral izquierda"      # Mensaje adicional superior

# URL del vídeo de fondo (déjalo vacío "" si no quieres vídeo)
VIDEO_FONDO = "https://github.com/Aaldanis/Wallpapers/raw/refs/heads/main/celestial-veil.1920x1080.mp4"

# -----------------------------------------------------------------------------
# ESTILOS CSS (diseño visual)
# -----------------------------------------------------------------------------
# Abre el archivo styles.css y lee todo su contenido
with open("styles.css", "r", encoding="utf-8") as f:
    CSS = f"<style>{f.read()}</style>"   # Encierra el CSS dentro de una etiqueta <style>

# Aplica los estilos CSS a la página
st.html(CSS)

# Vídeo de fondo (solo se muestra si se definió una URL)
if VIDEO_FONDO:
    # Inserta un elemento HTML de vídeo que se reproduce automáticamente, en bucle y sin sonido
    st.html(f'<video autoplay muted loop playsinline id="video-fondo"><source src="{VIDEO_FONDO}" type="video/mp4"></video>')

# -----------------------------------------------------------------------------
# FUNCIONES
# -----------------------------------------------------------------------------
# Importa las funciones de cálculo desde el archivo backend.py
from backend import parsear_datos, calcular


def mostrar_procedimiento(res, unidad=""):
    """Muestra el procedimiento completo con fórmulas y valores sustituidos."""
    # Si hay unidad, le agrega un espacio al inicio para que se vea bien
    u = f" {unidad}" if unidad else ""
    datos = res["datos"]      # Extrae los datos originales
    n = res["n"]              # Cantidad de datos
    media = res["media"]      # Media aritmética ya calculada

    st.markdown("#### Procedimiento completo")
    st.write(f"**Datos:** {', '.join(map(str, datos))}{u}")
    st.write(f"**n =** {n}")

    # ==================== 1. MEDIA ARITMÉTICA ====================
    suma = np.sum(datos)                     # Suma de todos los datos
    st.markdown("**1. Media aritmética**")
    st.latex(r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}")   # Fórmula general
    suma_str = " + ".join([str(x) for x in datos])        # Cadena con la suma escrita
    st.latex(rf"\bar{{x}} = \frac{{{suma_str}}}{{{n}}} = \frac{{{suma:.4f}}}{{{n}}} = {media:.4f}")

    # ==================== 2. MEDIANA ====================
    st.markdown("**2. Mediana**")
    st.latex(r"\text{Mediana} = \begin{cases} x_{\frac{n+1}{2}} & \text{si } n \text{ es impar} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{si } n \text{ es par} \end{cases}")
    ordenados = np.sort(datos)               # Ordena los datos de menor a mayor
    st.write(f"Datos ordenados: {', '.join(map(str, ordenados))}")
    if n % 2 == 1:                           # Si n es impar
        pos = (n + 1) // 2
        st.markdown(f"n = {n} (impar)")
        st.markdown(f"$$\\text{{Posición}} = \\frac{{n+1}}{{2}} = \\frac{{{n}+1}}{{2}} = {pos}$$")
        st.markdown(f"Mediana = dato en posición {pos} = **{res['mediana']:.4f}{u}**")
    else:                                    # Si n es par
        pos1 = n // 2
        pos2 = pos1 + 1
        st.markdown(f"n = {n} (par)")
        st.markdown(f"$$\\text{{Mediana}} = \\frac{{x_{{{pos1}}} + x_{{{pos2}}}}}{{2}} = \\frac{{{ordenados[pos1-1]:.4f} + {ordenados[pos2-1]:.4f}}}{{2}} = {res['mediana']:.4f}$$")

    # ==================== 3. MODA ====================
    st.markdown("**3. Moda**")
    st.latex(r"\text{Moda} = \text{valor con mayor frecuencia}")
    if isinstance(res["moda"], str):         # Si no hay moda única
        st.write(f"Moda: **{res['moda']}**")
    else:
        frec = Counter(datos)[res["moda"]]   # Cuenta cuántas veces aparece la moda
        st.markdown(f"El valor {res['moda']} aparece {frec} veces.")
        st.markdown(f"$$\\text{{Moda}} = {res['moda']}$$")

    # ==================== 4. MEDIDAS DE DISPERSIÓN ====================
    st.markdown("**4. Medidas de dispersión**")
    st.markdown("- **Rango**")
    st.latex(r"R = x_{\max} - x_{\min}")
    st.latex(rf"R = {np.max(datos):.4f} - {np.min(datos):.4f} = {res['rango']:.4f}")

    desviaciones = datos - media             # Desviación de cada dato respecto a la media
    cuadrados = desviaciones ** 2            # Eleva al cuadrado cada desviación
    suma_cuadrados = np.sum(cuadrados)       # Suma de los cuadrados

    st.markdown("- **Desviación estándar muestral (s)**")
    st.latex(r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}")
    if n <= 10:                              # Si hay pocos datos, muestra el detalle
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
    # Radio buttons para elegir la sección principal
    seccion = st.radio("", ["Módulos principales", "Ejercicios de ejemplo", "Practicar"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Asistente IA")
    # Intenta obtener la API Key de Gemini
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            api_key = ""
    if api_key:
        st.success("Conexión con IA exitosa")
    else:
        st.warning("API Key no configurada (.env o Streamlit Secrets)")
    # Caja de texto para preguntas a la IA
    pregunta_ia = st.text_area("Pregunta sobre Estadística", placeholder="Ej: ¿Cuándo se usa la desviación estándar muestral?", height=80, key="pregunta_ia")
    if st.button("Preguntar al asistente"):
        if not api_key:
            st.error("API Key no configurada. Agrégala en .env o en Streamlit Secrets.")
        elif not pregunta_ia.strip():
            st.warning("Escribe una pregunta")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("models/gemini-3.5-flash-lite")
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
    # Crea las 5 pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Descriptiva", "2. Datos Experimentales", "3. Tablas y Gráficas",
        "4. Probabilidad", "5. Correlación"
    ])

    # ----- Módulo 1: Estadística Descriptiva -----
    with tab1:
        st.subheader("Estadística Descriptiva")
        st.caption("Calcula media, mediana, moda, rango y medidas de dispersión (s y σ) con procedimiento completo.")
        unidad = st.text_input("Unidad (opcional)", placeholder="A, m, kg...")
        datos = st.text_area("Datos", placeholder="2.1, 2.3, 2.2, 2.4, 2.0", height=100)
        if st.button("Calcular descriptiva", key="btn1"):
            arr, err = parsear_datos(datos)     # Convierte el texto a números y valida
            if err:
                st.error(err)
            else:
                r = calcular(arr)               # Calcula todas las medidas
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
        st.subheader("Datos Experimentales")
        st.caption("Procesa series de mediciones, maneja unidades y muestra el resultado como x̄ ± s.")
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
        st.subheader("Tablas de Frecuencia y Gráficas")
        st.caption("Genera tablas de frecuencia por intervalos (Regla de Sturges) e histogramas o diagramas de dispersión.")
        datos = st.text_area("Datos", height=90, placeholder="3.1, 3.6, 3.8, 3.9, 4.2, ...")
        tipo = st.radio("Gráfica", ["Histograma", "Diagrama de dispersión"], horizontal=True)
        datos_y = st.text_area("Datos Y", height=70) if tipo == "Diagrama de dispersión" else ""
        if st.button("Generar", key="btn3"):
            arr, err = parsear_datos(datos)
            if err:
                st.error(err)
            else:
                n = len(arr)
                if n > 5000:
                    st.warning("Hay muchos datos. La gráfica puede tardar un poco.")
                # Regla de Sturges para calcular el número de clases
                num_clases = int(np.ceil(1 + 3.322 * np.log10(n)))
                num_clases = max(3, num_clases)
                st.info(f"Número de clases (Regla de Sturges): **{num_clases}**")

                minimo, maximo = float(arr.min()), float(arr.max())
                rango = maximo - minimo
                if rango == 0:
                    rango = 1.0
                amplitud = rango / num_clases
                bordes = [minimo + i * amplitud for i in range(num_clases + 1)]
                bordes[-1] = maximo + amplitud * 0.001

                frecuencias, _ = np.histogram(arr, bins=bordes)

                filas = []
                frec_acum = 0
                for i in range(num_clases):
                    lim_inf = bordes[i]
                    lim_sup = bordes[i + 1]
                    marca = (lim_inf + lim_sup) / 2
                    f_abs = int(frecuencias[i])
                    frec_acum += f_abs
                    f_rel = f_abs / n if n > 0 else 0
                    if i < num_clases - 1:
                        intervalo = f"[{lim_inf:.2f} – {lim_sup:.2f})"
                    else:
                        intervalo = f"[{lim_inf:.2f} – {lim_sup:.2f}]"
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
                    try:
                        ax.hist(arr, bins=bordes, color="#38bdf8", edgecolor="white")
                        ax.set_title("Histograma")
                        ax.set_xlabel("Valores")
                        ax.set_ylabel("Frecuencia")
                    except Exception as e:
                        st.error(f"No se pudo generar el histograma: {e}")
                else:
                    y, e = parsear_datos(datos_y)
                    if e or len(y) != len(arr):
                        st.error("Serie Y inválida o no tiene la misma cantidad de datos que X.")
                    else:
                        try:
                            ax.scatter(arr, y, color="#c084fc")
                            ax.set_title("Diagrama de dispersión")
                            ax.set_xlabel("X")
                            ax.set_ylabel("Y")
                        except Exception as ex:
                            st.error(f"Error al graficar: {ex}")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
        st.html('</div>')

    # ----- Módulo 4: Probabilidad -----
    with tab4:
        st.subheader("Distribución Normal")
        st.caption("Calcula probabilidades de la distribución normal usando parámetros teóricos o datos en bruto.")
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
                    st.success(f"P(X ≤ {x}) = {dist.cdf(x)*100:.4f}%")
                elif "P(X ≥ x)" in tipo:
                    z = (x - mu) / sigma
                    st.latex(rf"Z = \frac{{{x} - {mu:.4f}}}
