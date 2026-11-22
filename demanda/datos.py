"""Genera ventas semanales SINTÉTICAS de una cadena ficticia de supermercados ("Mercado Istmo").

3 tiendas × 4 categorías, enero 2023 – diciembre 2025, con:
- tendencia por tienda, estacionalidad anual por categoría,
- semanas de feriados panameños (Carnaval, Semana Santa, Fiestas Patrias, Navidad),
- promociones aleatorias que suben la demanda,
- ruido multiplicativo.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 5
RUTA = Path(__file__).resolve().parent.parent / "datos" / "ventas_semanales.csv"

TIENDAS = {"Colón Centro": (1.0, 0.04), "Panamá Norte": (1.6, 0.07), "David": (0.8, 0.02)}  # escala, crecimiento anual
CATEGORIAS = {
    # base semanal (unidades), amplitud estacional, semana pico, sensibilidad a feriados
    "Bebidas": (5200, 0.22, 14, 0.35),
    "Abarrotes": (8800, 0.06, 50, 0.20),
    "Lácteos": (4100, 0.05, 30, 0.10),
    "Licores": (1300, 0.30, 50, 0.90),
}

# Semanas ISO con feriados o eventos de alto consumo en Panamá (aprox. por año).
FERIADOS = {
    2023: {7: "Carnaval", 14: "Semana Santa", 44: "Fiestas Patrias", 45: "Fiestas Patrias", 47: "Fiestas Patrias", 51: "Navidad", 52: "Fin de año"},
    2024: {6: "Carnaval", 13: "Semana Santa", 44: "Fiestas Patrias", 45: "Fiestas Patrias", 47: "Fiestas Patrias", 51: "Navidad", 52: "Fin de año"},
    2025: {9: "Carnaval", 16: "Semana Santa", 44: "Fiestas Patrias", 45: "Fiestas Patrias", 47: "Fiestas Patrias", 51: "Navidad", 52: "Fin de año"},
    2026: {7: "Carnaval", 14: "Semana Santa", 44: "Fiestas Patrias", 45: "Fiestas Patrias", 47: "Fiestas Patrias", 51: "Navidad", 52: "Fin de año"},
}


def es_feriado(fechas: pd.Series) -> pd.Series:
    iso = fechas.dt.isocalendar()
    return pd.Series([w in FERIADOS.get(y, {}) for y, w in zip(iso.year, iso.week)], index=fechas.index).astype(int)


def generar(semilla: int = SEMILLA) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    semanas = pd.date_range("2023-01-02", "2025-12-29", freq="W-MON")
    filas = []
    for tienda, (escala, crec) in TIENDAS.items():
        for cat, (base, amp, pico, sens) in CATEGORIAS.items():
            promo = rng.random(len(semanas)) < 0.12
            for i, f in enumerate(semanas):
                semana = f.isocalendar().week
                anios = i / 52
                estacional = 1 + amp * np.cos(2 * np.pi * (semana - pico) / 52)
                feriado = semana in FERIADOS.get(f.isocalendar().year, {})
                media = base * escala * (1 + crec) ** anios * estacional
                media *= (1 + sens) if feriado else 1
                media *= 1.18 if promo[i] else 1
                filas.append({"semana": f.date(), "tienda": tienda, "categoria": cat,
                              "promocion": int(promo[i]), "feriado": int(feriado),
                              "unidades": int(rng.poisson(media * rng.lognormal(0, 0.06)))})
    return pd.DataFrame(filas)


if __name__ == "__main__":
    RUTA.parent.mkdir(exist_ok=True)
    df = generar()
    df.to_csv(RUTA, index=False)
    print(f"{len(df):,} filas · {df.semana.nunique()} semanas · {df.groupby(['tienda', 'categoria']).ngroups} series → {RUTA.name}")
