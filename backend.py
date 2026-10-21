"""
Backend - Funciones de cálculo estadístico
"""
import numpy as np
from scipy import stats
from collections import Counter
import re


def parsear_datos(texto):
    """Convierte texto en números. Rechaza letras."""
    if not texto or not texto.strip():
        return None, "Escribe algunos números."
    if re.search(r"[a-zA-Z]", texto):
        return None, "Error: solo se permiten números. Detecté una letra o texto inválido."
    texto = texto.replace(";", ",").replace("\n", ",").replace("\t", ",")
    partes = re.split(r"[,\s]+", texto.strip())
    nums = []
    for p in partes:
        p = p.strip()
        if not p:
            continue
        try:
            nums.append(float(p))
        except ValueError:
            return None, f"Error: '{p}' no es un número válido."
    if len(nums) < 2:
        return None, "Se necesitan al menos 2 números."
    return np.array(nums), None


def calcular(datos):
    """Calcula media, mediana, moda, rango y desviaciones."""
    moda_res = stats.mode(datos, keepdims=True)
    moda = moda_res.mode[0] if moda_res.count[0] > 1 else "Sin moda única"
    return {
        "n": len(datos),
        "media": np.mean(datos),
        "mediana": np.median(datos),
        "moda": moda,
        "rango": np.max(datos) - np.min(datos),
        "var_m": np.var(datos, ddof=1),
        "std_m": np.std(datos, ddof=1),
        "std_p": np.std(datos, ddof=0),
        "datos": datos
    }
