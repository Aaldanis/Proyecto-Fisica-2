"""
BACKEND - FUNCIONES DE CÁLCULO ESTADÍSTICO
Este archivo contiene las funciones que hacen los cálculos matemáticos
del asistente de estadística. Separa la lógica de cálculo de la interfaz
(app.py) para que el código sea más ordenado y fácil de mantener.
"""
# -----------------------------------------------------------------------------
# IMPORTACIONES (librerías necesarias)
# -----------------------------------------------------------------------------
import numpy as np                  # Librería para cálculos numéricos rápidos (media, desviación, arrays, etc.)
from scipy import stats             # Librería de estadística científica (moda, distribución normal, correlación, etc.)
from collections import Counter     # Contador de frecuencias (para encontrar la moda fácilmente)
import re                           # Módulo de expresiones regulares (para buscar letras o caracteres inválidos en el texto)
# -----------------------------------------------------------------------------
# FUNCIÓN: parsear_datos
# Convierte el texto que escribe el usuario en una lista de números (array de NumPy).
# También valida que no haya letras ni texto inválido.
# -----------------------------------------------------------------------------
def parsear_datos(texto):
    """
    Convierte texto en números. Rechaza letras.
    
    Parámetros:
        texto (str): Cadena de texto con números separados por comas, espacios, puntos y coma o saltos de línea.
    
    Retorna:
        tuple: (array de NumPy con los números, None) si todo está bien
               (None, mensaje de error) si hay algún problema
    """
    # Si el texto está vacío o solo tiene espacios, devolvemos un error
    if not texto or not texto.strip():
        return None, "Escribe algunos números."
    
    # Buscamos si hay alguna letra (a-z o A-Z) en el texto
    # Si encontramos una letra, rechazamos la entrada
    if re.search(r"[a-zA-Z]", texto):
        return None, "Error: solo se permiten números. Detecté una letra o texto inválido."
    
    # Normalizamos los separadores: cambiamos punto y coma, saltos de línea y tabuladores por comas
    texto = texto.replace(";", ",").replace("\n", ",").replace("\t", ",")
    
    # Dividimos el texto por comas o espacios (uno o más)
    partes = re.split(r"[,\s]+", texto.strip())
    
    # Lista vacía donde iremos guardando los números válidos
    nums = []
    
    # Recorremos cada parte (cada posible número)
    for p in partes:
        p = p.strip()               # Quitamos espacios al inicio y final
        if not p:                   # Si la parte está vacía, la saltamos
            continue
        try:
            # Intentamos convertir la parte a número decimal (float)
            nums.append(float(p))
        except ValueError:
            # Si no se puede convertir, devolvemos un error específico
            return None, f"Error: '{p}' no es un número válido."
    
    # Si después de todo tenemos menos de 2 números, no podemos calcular estadísticas
    if len(nums) < 2:
        return None, "Se necesitan al menos 2 números."
    
    # Convertimos la lista a un array de NumPy y devolvemos éxito (sin error)
    return np.array(nums), None
# -----------------------------------------------------------------------------
# FUNCIÓN: calcular
# Calcula las principales medidas estadísticas de un conjunto de datos.
# -----------------------------------------------------------------------------
def calcular(datos):
    """
    Calcula media, mediana, moda, rango y desviaciones.
    
    Parámetros:
        datos (np.array): Array de NumPy con los números ya validados.
    
    Retorna:
        dict: Diccionario con todos los resultados calculados.
    """
    # Calculamos la moda usando SciPy
    # keepdims=True mantiene la forma del resultado para poder acceder fácilmente
    moda_res = stats.mode(datos, keepdims=True)
    
    # Si la moda aparece más de una vez, la tomamos; si no, decimos que no hay moda única
    moda = moda_res.mode[0] if moda_res.count[0] > 1 else "Sin moda única"
    
    # Devolvemos un diccionario con todos los resultados
    return {
        "n": len(datos),                                    # Cantidad de datos (tamaño de la muestra)
        "media": np.mean(datos),                            # Media aritmética
        "mediana": np.median(datos),                        # Mediana (valor central)
        "moda": moda,                                       # Moda (valor más frecuente) o mensaje
        "rango": np.max(datos) - np.min(datos),             # Rango (máximo - mínimo)
        "var_m": np.var(datos, ddof=1),                     # Varianza muestral (usa n-1)
        "std_m": np.std(datos, ddof=1),                     # Desviación estándar muestral (s)
        "std_p": np.std(datos, ddof=0),                     # Desviación estándar poblacional (σ)
        "datos": datos                                      # Guardamos los datos originales para usarlos después
    }
