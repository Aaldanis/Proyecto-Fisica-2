# =============================================================================
# ASISTENTE VIRTUAL DE ESTADÍSTICA APLICADA
# Universidad Mariano Gálvez de Guatemala
# =============================================================================
# Este archivo es la interfaz principal de la aplicación (hecha con Streamlit).
# Controla todo lo que ve el usuario: menús, pestañas, entradas de datos,
# botones y la forma en que se muestran los resultados.
# -----------------------------------------------------------------------------
# IMPORTACIONES
# -----------------------------------------------------------------------------
import streamlit as st                  # Librería que crea la interfaz web interactiva
import numpy as np                      # Cálculos matemáticos (media, desviación, arrays, etc.)
import pandas as pd                     # Manejo de tablas de datos (frecuencias, dataframes)
import matplotlib.pyplot as plt         # Creación de gráficas (histogramas, diagramas de dispersión)
from scipy import stats                 # Estadística avanzada (moda, normal, correlación de Pearson)
import re                               # Expresiones regulares (buscar letras o caracteres inválidos)
import os                               # Acceso a variables de entorno del sistema
from dotenv import load_dotenv          # Carga variables desde un archivo .env
from collections import Counter         # Contador de frecuencias (útil para la moda)
import google.generativeai as genai     # Librería para usar el modelo de IA de Google (Gemini)

# Cargamos las variables de entorno del archivo .env (si existe)
load_dotenv()

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
# Configuramos el título de la pestaña del navegador, el ícono y el diseño ancho
st.set_page_config(
    page_title="Asistente de Estadística UMG",   # Título que aparece en la pestaña del navegador
    page_icon="📊",                              # Ícono de la pestaña
    layout="wide"                                # Diseño de la página: ocupa todo el ancho disponible
)

# -----------------------------------------------------------------------------
# DATOS QUE DEBES EDITAR (información del proyecto)
# -----------------------------------------------------------------------------
INTEGRANTES = "Edgar Aldanis Chávez Estrada y Erick Adolfo de Leon Rivera"   # Nombres de los integrantes del equipo
SECCION = "Ing. M.A Reyes Baltazar Mazá Guerra"                              # Nombre del profesor
ASIGNATURA = "Fisica 2"                                                      # Nombre de la asignatura
UNIVERSIDAD = "Universidad Mariano Gálvez de Guatemala"                      # Nombre de la universidad
ADICIONAL = "Mas opciones y asistente IA en la barra lateral izquierda"      # Mensaje adicional que se muestra arriba

# URL del vídeo de fondo (déjalo vacío "" si no quieres vídeo de fondo)
VIDEO_FONDO = "https://github.com/Aaldanis/Wallpapers/raw/refs/heads/main/celestial-veil.1920x1080.h264"

# -----------------------------------------------------------------------------
# ESTILOS CSS (diseño visual de la página)
# -----------------------------------------------------------------------------
# Abrimos el archivo styles.css y leemos todo su contenido
with open("styles.css", "r", encoding="utf-8") as f:
    CSS = f"<style>{f.read()}</style>"   # Encerramos el CSS dentro de una etiqueta <style>

# Aplicamos los estilos CSS a la página de Streamlit
st.html(CSS)

# Vídeo de fondo (solo se muestra si se puso una URL válida)
if VIDEO_FONDO:
    # Insertamos un elemento HTML de vídeo que se reproduce automáticamente, en bucle y sin sonido
    st.html(f'<video autoplay muted loop playsinline id="video-fondo"><source src="{VIDEO_FONDO}" type="video/mp4"></video>')

# -----------------------------------------------------------------------------
# FUNCIONES (importadas desde backend.py)
# -----------------------------------------------------------------------------
# Importamos las funciones de cálculo que están en el archivo backend.py
from backend import parsear_datos, calcular


# -----------------------------------------------------------------------------
# FUNCIÓN: mostrar_procedimiento
# Muestra paso a paso cómo se calculan media, mediana, moda y desviaciones,
# con las fórmulas y los valores sustituidos.
# -----------------------------------------------------------------------------
def mostrar_procedimiento(res, unidad=""):
    """
    Muestra el procedimiento completo con fórmulas y valores sustituidos.
    
    Parámetros:
        res (dict): Diccionario con los resultados de la función calcular()
        unidad (str): Unidad de medida opcional (ej. "A", "m", "kg")
    """
    # Si hay unidad, le agregamos un espacio al inicio para que se vea bien
    u = f" {unidad}" if unidad else ""
    
    # Extraemos los datos y valores principales del diccionario de resultados
    datos = res["datos"]
    n = res["n"]
    media = res["media"]

    # Título de la sección de procedimiento
    st.markdown("#### Procedimiento completo")
    
    # Mostramos los datos originales y la cantidad de datos
    st.write(f"**Datos:** {', '.join(map(str, datos))}{u}")
    st.write(f"**n =** {n}")

    # ==================== 1. MEDIA ARITMÉTICA ====================
    suma = np.sum(datos)                     # Calculamos la suma de todos los datos
    st.markdown("**1. Media aritmética**")
    st.latex(r"\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}")   # Mostramos la fórmula general
    
    # Creamos una cadena con la suma escrita (ej. "2.1 + 2.3 + 2.2")
    suma_str = " + ".join([str(x) for x in datos])
    # Mostramos la fórmula con los valores sustituidos
    st.latex(rf"\bar{{x}} = \frac{{{suma_str}}}{{{n}}} = \frac{{{suma:.4f}}}{{{n}}} = {media:.4f}")

    # ==================== 2. MEDIANA ====================
    st.markdown("**2. Mediana**")
    # Fórmula de la mediana según si n es par o impar
    st.latex(r"\text{Mediana} = \begin{cases} x_{\frac{n+1}{2}} & \text{si } n \text{ es impar} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{si } n \text{ es par} \end{cases}")
    
    # Ordenamos los datos de menor a mayor
    ordenados = np.sort(datos)
    st.write(f"Datos ordenados: {', '.join(map(str, ordenados))}")
    
    # Calculamos según si la cantidad de datos es impar o par
    if n % 2 == 1:   # Si n es impar
        pos = (n + 1) // 2
        st.markdown(f"n = {n} (impar)")
        st.markdown(f"$$\\text{{Posición}} = \\frac{{n+1}}{{2}} = \\frac{{{n}+1}}{{2}} = {pos}$$")
        st.markdown(f"Mediana = dato en posición {pos} = **{res['mediana']:.4f}{u}**")
    else:            # Si n es par
        pos1 = n // 2
        pos2 = pos1 + 1
        st.markdown(f"n = {n} (par)")
        st.markdown(f"$$\\text{{Mediana}} = \\frac{{x_{{{pos1}}} + x_{{{pos2}}}}}{{2}} = \\frac{{{ordenados[pos1-1]:.4f} + {ordenados[pos2-1]:.4f}}}{{2}} = {res['mediana']:.4f}$$")

    # ==================== 3. MODA ====================
    st.markdown("**3. Moda**")
    st.latex(r"\text{Moda} = \text{valor con mayor frecuencia}")
    
    # Si la moda es un texto (no hay moda única), lo mostramos directamente
    if isinstance(res["moda"], str):
        st.write(f"Moda: **{res['moda']}**")
    else:
        # Contamos cuántas veces aparece el valor de la moda
        frec = Counter(datos)[res["moda"]]
        st.markdown(f"El valor {res['moda']} aparece {frec} veces.")
        st.markdown(f"$$\\text{{Moda}} = {res['moda']}$$")

    # ==================== 4. MEDIDAS DE DISPERSIÓN ====================
    st.markdown("**4. Medidas de dispersión**")
    
    # --- Rango ---
    st.markdown("- **Rango**")
    st.latex(r"R = x_{\max} - x_{\min}")
    st.latex(rf"R = {np.max(datos):.4f} - {np.min(datos):.4f} = {res['rango']:.4f}")

    # Calculamos las desviaciones de cada dato respecto a la media
    desviaciones = datos - media
    cuadrados = desviaciones ** 2          # Elevamos al cuadrado cada desviación
    suma_cuadrados = np.sum(cuadrados)     # Sumamos todos los cuadrados

    # --- Desviación estándar muestral (s) ---
    st.markdown("- **Desviación estándar muestral (s)**")
    st.latex(r"s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n-1}}")
    
    # Si hay pocos datos (≤ 10), mostramos el detalle de la suma de cuadrados
    if n <= 10:
        terminos = " + ".join([f"({x:.2f}-{media:.2f})^2" for x in datos])
        st.latex(rf"\sum (x_i-\bar{{x}})^2 = {terminos} = {suma_cuadrados:.4f}")
    
    st.latex(rf"s = \sqrt{{\frac{{{suma_cuadrados:.4f}}}{{{n-1}}}}} = {res['std_m']:.4f}")

    # --- Desviación estándar poblacional (σ) ---
    st.markdown("- **Desviación estándar poblacional (σ)**")
    st.latex(r"\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{n}}")
    st.latex(rf"\sigma = \sqrt{{\frac{{{suma_cuadrados:.4f}}}{{{n}}}}} = {res['std_p']:.4f}")

    # --- Varianza muestral ---
    st.markdown("- **Varianza muestral**")
    st.latex(r"s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1}")
    st.latex(rf"s^2 = \frac{{{suma_cuadrados:.4f}}}{{{n-1}}} = {res['var_m']:.4f}")

    # Mensaje final con el resultado principal
    st.success(f"**Resultado final:** Media = {media:.4f}{u} | s = {res['std_m']:.4f}{u}")
    st.info(f"La desviación estándar muestral (s) usa n-1. La poblacional (σ) usa n.")


# -----------------------------------------------------------------------------
# ENCABEZADO DE LA PÁGINA (lo que se ve arriba del todo)
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
# MENÚ LATERAL (barra izquierda)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Opciones")
    
    # Radio buttons para elegir la sección principal de la app
    seccion = st.radio(
        "", 
        ["Módulos principales", "Ejercicios de ejemplo", "Practicar"], 
        label_visibility="collapsed"   # Ocultamos la etiqueta del radio
    )
    
    st.markdown("---")   # Línea separadora
    st.markdown("### Asistente IA")
    
    # Intentamos obtener la API Key de Gemini (primero del archivo .env, luego de los secrets de Streamlit)
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            api_key = ""
    
    # Mostramos el estado de la conexión con la IA
    if api_key:
        st.success("Conexión con IA exitosa")
    else:
        st.warning("API Key no configurada (.env o Streamlit Secrets)")
    
    # Caja de texto para que el usuario haga preguntas a la IA
    pregunta_ia = st.text_area(
        "Pregunta sobre Estadística", 
        placeholder="Ej: ¿Cuándo se usa la desviación estándar muestral?", 
        height=80, 
        key="pregunta_ia"
    )
    
    # Botón para enviar la pregunta a la IA
    if st.button("Preguntar al asistente"):
        if not api_key:
            st.error("API Key no configurada. Agrégala en .env o en Streamlit Secrets.")
        elif not pregunta_ia.strip():
            st.warning("Escribe una pregunta")
        else:
            try:
                # Configuramos la API de Google Gemini
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("models/gemini-3.5-flash-lite")
                
                # Prompt que limita al modelo a responder solo sobre estadística aplicada
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
                
                # Generamos la respuesta y la mostramos
                response = model.generate_content(prompt)
                st.markdown("**Respuesta:**")
                st.info(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.caption(f"{INTEGRANTES}\n{SECCION}")

# -----------------------------------------------------------------------------
# MÓDULOS PRINCIPALES (las 5 pestañas de cálculo)
# -----------------------------------------------------------------------------
if seccion == "Módulos principales":
    # Creamos 5 pestañas
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Descriptiva", "2. Datos Experimentales", "3. Tablas y Gráficas",
        "4. Probabilidad", "5. Correlación"
    ])

    # ==================== MÓDULO 1: ESTADÍSTICA DESCRIPTIVA ====================
    with tab1:
        st.subheader("Estadística Descriptiva")
        st.caption("Calcula media, mediana, moda, rango y medidas de dispersión (s y σ) con procedimiento completo.")
        
        # Campo opcional para la unidad de medida
        unidad = st.text_input("Unidad (opcional)", placeholder="A, m, kg...")
        
        # Caja de texto para ingresar los datos
        datos = st.text_area("Datos", placeholder="2.1, 2.3, 2.2, 2.4, 2.0", height=100)
        
        # Botón para calcular
        if st.button("Calcular descriptiva", key="btn1"):
            # Convertimos el texto a números y validamos
            arr, err = parsear_datos(datos)
            if err:
                st.error(err)          # Si hay error, lo mostramos
            else:
                # Calculamos todas las medidas
                r = calcular(arr)
                # Mostramos el procedimiento completo
                mostrar_procedimiento(r, unidad)
                
                # Preparamos la unidad para mostrarla
                u = f" {unidad}" if unidad else ""
                
                # Mostramos 4 métricas principales en columnas
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Media", f"{r['media']:.4f}{u}")
                c2.metric("Mediana", f"{r['mediana']:.4f}{u}")
                c3.metric("s", f"{r['std_m']:.4f}{u}")
                c4.metric("Rango", f"{r['rango']:.4f}{u}")
        
        st.html('</div>')   # Cierre de la tarjeta visual (si se usaba)

    # ==================== MÓDULO 2: DATOS EXPERIMENTALES ====================
    with tab2:
        st.subheader("Datos Experimentales")
        st.caption("Procesa series de mediciones, maneja unidades y muestra el resultado como x̄ ± s.")
        st.info("Convierte primero las mediciones a la misma unidad.")
        
        # Dos columnas: una para los datos y otra para la unidad
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
                
                # Mostramos los pasos de forma clara
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

    # ==================== MÓDULO 3: TABLAS Y GRÁFICAS ====================
    with tab3:
        st.subheader("Tablas de Frecuencia y Gráficas")
        st.caption("Genera tablas de frecuencia por intervalos (Regla de Sturges) e histogramas o diagramas de dispersión.")
        
        datos = st.text_area("Datos", height=90, placeholder="3.1, 3.6, 3.8, 3.9, 4.2, ...")
        
        # El usuario elige el tipo de gráfica
        tipo = st.radio("Gráfica", ["Histograma", "Diagrama de dispersión"], horizontal=True)
        
        # Si elige diagrama de dispersión, pedimos también los datos de Y
        datos_y = st.text_area("Datos Y", height=70) if tipo == "Diagrama de dispersión" else ""
        
        if st.button("Generar", key="btn3"):
            arr, err = parsear_datos(datos)
            if err:
                st.error(err)
            else:
                n = len(arr)
                if n > 5000:
                    st.warning("Hay muchos datos. La gráfica puede tardar un poco.")
                
                # Regla de Sturges para calcular el número de clases (intervalos)
                num_clases = int(np.ceil(1 + 3.322 * np.log10(n)))
                num_clases = max(3, num_clases)   # Mínimo 3 clases
                st.info(f"Número de clases (Regla de Sturges): **{num_clases}**")
                
                minimo, maximo = float(arr.min()), float(arr.max())
                rango = maximo - minimo
                if rango == 0:
                    rango = 1.0   # Evitamos división por cero
                
                # Calculamos la amplitud de cada intervalo
                amplitud = rango / num_clases
                
                # Creamos los bordes de los intervalos
                bordes = [minimo + i * amplitud for i in range(num_clases + 1)]
                bordes[-1] = maximo + amplitud * 0.001   # Aseguramos que el máximo entre
                
                # Calculamos las frecuencias con NumPy
                frecuencias, _ = np.histogram(arr, bins=bordes)
                
                # Construimos la tabla de frecuencias fila por fila
                filas = []
                frec_acum = 0
                for i in range(num_clases):
                    lim_inf = bordes[i]
                    lim_sup = bordes[i + 1]
                    marca = (lim_inf + lim_sup) / 2          # Marca de clase (punto medio)
                    f_abs = int(frecuencias[i])              # Frecuencia absoluta
                    frec_acum += f_abs                       # Frecuencia acumulada
                    f_rel = f_abs / n if n > 0 else 0        # Frecuencia relativa
                    
                    # Formato del intervalo (el último se cierra con ])
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
                
                # Agregamos la fila de totales
                filas.append({
                    "Clase (Intervalo)": "Total",
                    "Marca de clase (xᵢ)": "",
                    "Frecuencia absoluta (fᵢ)": n,
                    "Frecuencia relativa (hᵢ)": "1.00 (100%)",
                    "Frecuencia acumulada (Fᵢ)": ""
                })
                
                # Mostramos la tabla
                st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
                
                # Creamos la gráfica
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
                    # Diagrama de dispersión
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
                st.pyplot(fig)      # Mostramos la gráfica en Streamlit
                plt.close()         # Cerramos la figura para liberar memoria
        
        st.html('</div>')

    # ==================== MÓDULO 4: PROBABILIDAD (DISTRIBUCIÓN NORMAL) ====================
    with tab4:
        st.subheader("Distribución Normal")
        st.caption("Calcula probabilidades de la distribución normal usando parámetros teóricos o datos en bruto.")
        
        # El usuario elige cómo quiere ingresar los parámetros
        modo = st.radio(
            "¿Cómo quieres ingresar los parámetros?",
            ["Ingresar parámetros teóricos (μ y σ)", "Ingresar datos en bruto (lista)"],
            horizontal=True
        )
        
        mu = 0.0
        sigma = 1.0
        
        if modo == "Ingresar parámetros teóricos (μ y σ)":
            c1, c2 = st.columns(2)
            mu = c1.number_input("Media μ", value=0.0)
            sigma = c2.number_input("Desviación σ", value=1.0, min_value=0.01)
        else:
            # Calculamos μ y σ a partir de los datos ingresados
            datos_brutos = st.text_area("Datos en bruto", placeholder="2.1, 2.3, 2.2, 2.4, 2.0", height=100)
            if datos_brutos.strip():
                arr, err = parsear_datos(datos_brutos)
                if err:
                    st.error(err)
                else:
                    mu = float(np.mean(arr))
                    sigma = float(np.std(arr, ddof=1))
                    st.success(f"Parámetros calculados: μ = {mu:.4f} | σ (s) = {sigma:.4f}")
        
        # Tipo de probabilidad que se quiere calcular
        tipo = st.selectbox("Calcular", [
            "P(X ≤ x) — Probabilidad acumulada a la izquierda",
            "P(X ≥ x) — Probabilidad acumulada a la derecha",
            "P(a ≤ X ≤ b) — Probabilidad entre dos valores"
        ])
        
        # Pedimos los valores según el tipo de probabilidad
        if "P(a ≤ X ≤ b)" in tipo:
            a = st.number_input("a", value=-1.0)
            b = st.number_input("b", value=1.0)
        else:
            x = st.number_input("x", value=0.0)
        
        if st.button("Calcular probabilidad", key="btn4"):
            if sigma <= 0:
                st.error("La desviación estándar debe ser mayor que 0.")
            else:
                # Creamos la distribución normal con los parámetros dados
                dist = stats.norm(mu, sigma)
                st.latex(r"Z = \frac{X - \mu}{\sigma}")
                
                if "P(X ≤ x)" in tipo:
                    z = (x - mu) / sigma
                    st.latex(rf"Z = \frac{{{x} - {mu:.4f}}}{{{sigma:.4f}}} = {z:.4f}")
                    st.success(f"P(X ≤ {x}) = {dist.cdf(x)*100:.4f}%")
                
                elif "P(X ≥ x)" in tipo:
                    z = (x - mu) / sigma
                    st.latex(rf"Z = \frac{{{x} - {mu:.4f}}}{{{sigma:.4f}}} = {z:.4f}")
                    st.success(f"P(X ≥ {x}) = {(1 - dist.cdf(x))*100:.2f}%")
                
                else:
                    # Probabilidad entre a y b
                    za = (a - mu) / sigma
                    zb = (b - mu) / sigma
                    st.latex(r"Z_1 = \frac{a - \mu}{\sigma}, \quad Z_2 = \frac{b - \mu}{\sigma}")
                    st.latex(rf"Z_1 = \frac{{{a} - {mu:.4f}}}{{{sigma:.4f}}} = {za:.4f}, \quad Z_2 = \frac{{{b} - {mu:.4f}}}{{{sigma:.4f}}} = {zb:.4f}")
                    
                    prob = dist.cdf(b) - dist.cdf(a)
                    st.success(f"P({a} ≤ X ≤ {b}) = {prob * 100:.2f}%")
        
        st.html('</div>')

    # ==================== MÓDULO 5: CORRELACIÓN Y REGRESIÓN ====================
    with tab5:
        st.subheader("Correlación y Regresión Lineal")
        st.caption("Calcula el coeficiente de correlación (r) y la recta de regresión lineal con procedimiento paso a paso.")
        
        c1, c2 = st.columns(2)
        dx = c1.text_area("X", placeholder="1,2,3,4,5", key="tx_x")
        dy = c2.text_area("Y", placeholder="2,4,5,4,6", key="tx_y")
        
        if st.button("Calcular", key="btn5"):
            x, ex = parsear_datos(dx)
            y, ey = parsear_datos(dy)
            
            if ex or ey:
                st.error(ex or ey)
            elif len(x) < 2 or len(y) < 2:
                st.error("Debes ingresar al menos 2 pares de datos.")
            elif len(x) != len(y):
                st.error("Las variables X e Y deben tener la misma cantidad de elementos.")
            else:
                # Calculamos las medias
                media_x = np.mean(x)
                media_y = np.mean(y)
                
                # Desviaciones respecto a la media
                dx_arr = x - media_x
                dy_arr = y - media_y
                
                # Sumas necesarias para la correlación y regresión
                suma_dxdy = np.sum(dx_arr * dy_arr)
                suma_dx2 = np.sum(dx_arr ** 2)
                suma_dy2 = np.sum(dy_arr ** 2)
                
                # Coeficiente de correlación de Pearson
                r, _ = stats.pearsonr(x, y)
                
                # Pendiente (b) e intercepto (a) de la recta
                b = suma_dxdy / suma_dx2 if suma_dx2 != 0 else 0
                a = media_y - (b * media_x)
                r2 = r ** 2   # Coeficiente de determinación
                
                # --- Paso 1: Medias ---
                st.markdown("### Paso 1: Medias")
                st.latex(rf"\bar{{x}} = {media_x:.4f}, \quad \bar{{y}} = {media_y:.4f}")
                
                # --- Paso 2: Coeficiente de Correlación ---
                st.markdown("### Paso 2: Coeficiente de correlación (r)")
                st.latex(r"r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}")
                
                st.latex(rf"\sum (x_i - \bar{{x}})(y_i - \bar{{y}}) = {suma_dxdy:.4f}")
                st.latex(rf"\sum (x_i - \bar{{x}})^2 = {suma_dx2:.4f}, \quad \sum (y_i - \bar{{y}})^2 = {suma_dy2:.4f}")
                st.latex(rf"r = \frac{{{suma_dxdy:.4f}}}{{\sqrt{{{suma_dx2:.4f} \times {suma_dy2:.4f}}}}} = {r:.4f}")
                
                # --- Paso 3: Recta de Regresión ---
                st.markdown("### Paso 3: Recta de regresión")
                st.latex(r"\hat{y} = a + bx")
                
                st.latex(r"b = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}")
                st.latex(rf"b = \frac{{{suma_dxdy:.4f}}}{{{suma_dx2:.4f}}} = {b:.4f}")
                
                st.latex(r"a = \bar{y} - b\bar{x}")
                st.latex(rf"a = {media_y:.4f} - ({b:.4f} \cdot {media_x:.4f}) = {a:.4f}")
                
                # Ecuación final con signo correcto
                signo = "+" if b >= 0 else "-"
                st.latex(rf"\hat{{y}} = {a:.4f} {signo} {abs(b):.4f}x")
                
                # Interpretación de la correlación
                st.markdown("---")
                fuerza = "fuerte" if abs(r) > 0.7 else "moderada" if abs(r) > 0.4 else "débil"
                direccion = "positiva" if r > 0 else "negativa"
                
                st.info(f"**Interpretación:** Existe una correlación **{fuerza} y {direccion}** entre las variables.")
                
                # Métricas principales
                m1, m2, m3 = st.columns(3)
                m1.metric("Coef. Correlación (r)", f"{r:.4f}")
                m2.metric("Coef. Determinación (r²)", f"{r2:.4f}")
                m3.metric("Variabilidad explicada", f"{r2 * 100:.2f}%")
                
                # Gráfica con estilo oscuro
                fig, ax = plt.subplots(figsize=(7, 4))
                fig.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#161b22')
                
                # Puntos observados
                ax.scatter(x, y, color="#38bdf8", s=60, label="Datos observados", zorder=3)
                
                # Recta de regresión
                x_line = np.linspace(min(x), max(x), 100)
                y_line = a + b * x_line
                ax.plot(x_line, y_line, color="#f87171", linewidth=2, label="Recta de regresión")
                
                # Estilo de los ejes
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#30363d')
                
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.legend(facecolor='#21262d', edgecolor='none', labelcolor='white')
                ax.grid(color='#30363d', linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                plt.close()

# -----------------------------------------------------------------------------
# EJERCICIOS DE EJEMPLO
# -----------------------------------------------------------------------------
elif seccion == "Ejercicios de ejemplo":
    st.subheader("Ejercicios de ejemplo")
    
    # Diccionario con ejemplos predefinidos
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
    
    # Menú desplegable para elegir un ejemplo
    ej = st.selectbox("Elige", list(ejemplos.keys()))
    st.code(ejemplos[ej])   # Mostramos los datos del ejemplo
    
    if st.button("Resolver ejemplo"):
        arr, err = parsear_datos(ejemplos[ej])
        if not err:
            # Calculamos y mostramos el procedimiento completo
            mostrar_procedimiento(calcular(arr))
    
    st.html('</div>')

# -----------------------------------------------------------------------------
# PRÁCTICA (ejercicios aleatorios con retroalimentación)
# -----------------------------------------------------------------------------
elif seccion == "Practicar":
    st.subheader("Práctica con retroalimentación")
    
    # Inicializamos la variable de sesión para guardar el ejercicio actual
    if "prac" not in st.session_state:
        st.session_state.prac = None
    
    if st.button("Generar ejercicio nuevo"):
        # Generamos entre 6 y 10 datos aleatorios con distribución normal
        n = np.random.randint(6, 11)
        base = np.random.uniform(15, 90)
        st.session_state.prac = np.round(np.random.normal(base, base * 0.12, n), 2)
        st.rerun()   # Recargamos la página para mostrar el nuevo ejercicio
    
    if st.session_state.prac is not None:
        datos = st.session_state.prac
        st.code(", ".join(map(str, datos)))   # Mostramos los datos generados
        
        c1, c2 = st.columns(2)
        mu_u = c1.number_input("Tu media", format="%.4f", step=0.01)
        s_u = c2.number_input("Tu desviación estándar muestral", format="%.4f", step=0.01)
        
        if st.button("Verificar"):
            # Calculamos los valores correctos
            mu_ok, s_ok = np.mean(datos), np.std(datos, ddof=1)
            
            # Comparamos con una tolerancia de 0.05
            if abs(mu_u - mu_ok) < 0.05 and abs(s_u - s_ok) < 0.05:
                st.success("¡Correcto!")
            else:
                st.warning("Revisa:")
                if abs(mu_u - mu_ok) >= 0.05:
                    st.error(f"Media correcta: {mu_ok:.4f}")
                if abs(s_u - s_ok) >= 0.05:
                    st.error(f"s correcta: {s_ok:.4f}")
            
            # Mostramos el procedimiento correcto
            st.markdown("**Procedimiento correcto**")
            mostrar_procedimiento(calcular(datos))
    else:
        st.info("Genera un ejercicio para empezar.")
    
    st.html('</div>')

# -----------------------------------------------------------------------------
# PIE DE PÁGINA
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(f"{INTEGRANTES} · {ASIGNATURA} · {SECCION}")
