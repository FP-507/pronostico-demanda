"""Backtesting de origen móvil, pronóstico final y plan de pedidos.

Uso: python -m demanda.backtest
Salidas: graficos/*.png, resultados/*.csv, REPORTE.md
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import modelos as M  # noqa: E402
from .datos import FERIADOS, RUTA, generar  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
GRAF, RES = RAIZ / "graficos", RAIZ / "resultados"
ORIGENES = 4                 # cortes de backtest, separados 13 semanas
Z_SERVICIO = 1.65            # nivel de servicio del 95 %
AZUL, NARANJA, AQUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA_2, GRILLA, EJE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
NOMBRES = {"naive": "Naive estacional", "hw": "Holt-Winters", "gbm": "Gradient Boosting", "ensamble": "Ensamble HW + GBM"}


def pronosticar(df: pd.DataFrame, corte: pd.Timestamp, h: int = M.HORIZONTE) -> pd.DataFrame:
    """Entrena con datos hasta `corte` (incluido) y pronostica las h semanas siguientes."""
    feats = M.caracteristicas(df)
    entreno = feats[pd.to_datetime(feats.semana) <= corte]
    futuro = feats[(pd.to_datetime(feats.semana) > corte)
                   & (pd.to_datetime(feats.semana) <= corte + pd.Timedelta(weeks=h))].copy()
    futuro["gbm"] = M.gbm(entreno, futuro)
    partes = []
    for (tienda, cat), fut in futuro.groupby(["tienda", "categoria"], observed=True):
        hist = entreno[(entreno.tienda == tienda) & (entreno.categoria == cat)].set_index("semana").unidades
        fut = fut.copy()
        fut["naive"] = M.naive_estacional(hist, len(fut))
        fut["hw"] = M.holt_winters(hist, len(fut))
        partes.append(fut)
    out = pd.concat(partes)
    out["ensamble"] = (out.hw + out.gbm) / 2
    return out[["semana", "tienda", "categoria", "unidades", "naive", "hw", "gbm", "ensamble"]]


def backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ultima = pd.to_datetime(df.semana).max()
    cortes = [ultima - pd.Timedelta(weeks=M.HORIZONTE * k) for k in range(ORIGENES, 0, -1)]
    preds = pd.concat([pronosticar(df, c).assign(corte=c.date()) for c in cortes])
    filas = []
    for corte, g in preds.groupby("corte"):
        for m in NOMBRES:
            filas.append({"corte": corte, "modelo": m, "wape": M.wape(g.unidades, g[m]), "sesgo": M.sesgo(g.unidades, g[m])})
    return preds, pd.DataFrame(filas)


def futuro_exogeno(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Filas para las próximas h semanas (sin unidades) con feriados conocidos y sin promociones planificadas."""
    ultima = pd.to_datetime(df.semana).max()
    semanas = pd.date_range(ultima + pd.Timedelta(weeks=1), periods=h, freq="W-MON")
    filas = []
    for (t, c) in df.groupby(["tienda", "categoria"]).groups:
        for s in semanas:
            iso = s.isocalendar()
            filas.append({"semana": s.date(), "tienda": t, "categoria": c, "promocion": 0,
                          "feriado": int(iso.week in FERIADOS.get(iso.year, {})), "unidades": np.nan})
    return pd.DataFrame(filas)


def estilo(ax, titulo):
    ax.set_title(titulo, loc="left", fontsize=12, fontweight="bold", color=TINTA)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(EJE)
    ax.tick_params(colors=TINTA_2, labelsize=9)
    ax.grid(color=GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    if not RUTA.exists():
        RUTA.parent.mkdir(exist_ok=True)
        generar().to_csv(RUTA, index=False)
    df = pd.read_csv(RUTA, parse_dates=["semana"])
    df["semana"] = df.semana.dt.date
    GRAF.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    preds, metricas = backtest(df)
    resumen = metricas.groupby("modelo")[["wape", "sesgo"]].mean().reindex(list(NOMBRES))
    mejor = resumen.wape.idxmin()
    print(resumen.round(4))

    # Error del mejor modelo por serie, para el stock de seguridad.
    preds["error"] = preds.unidades - preds[mejor]
    sigma = preds.groupby(["tienda", "categoria"]).error.std()

    # Pronóstico final: se re-entrena con toda la historia.
    total = pd.concat([df, futuro_exogeno(df, M.HORIZONTE)], ignore_index=True)
    ultima = pd.to_datetime(df.semana).max()
    final = pronosticar(total, ultima)
    final["pronostico"] = final[mejor].round()
    final = final.merge(sigma.rename("sigma").reset_index(), on=["tienda", "categoria"])
    plan = (final.groupby(["tienda", "categoria"])
                 .agg(pronostico_13s=("pronostico", "sum"), sigma=("sigma", "first")).reset_index())
    # Stock de seguridad para un tiempo de reposición de 2 semanas.
    plan["stock_seguridad"] = (Z_SERVICIO * plan.sigma * np.sqrt(2)).round()
    plan["pedido_recomendado"] = (plan.pronostico_13s + plan.stock_seguridad).astype(int)
    plan = plan.drop(columns="sigma").sort_values("pedido_recomendado", ascending=False)

    metricas.to_csv(RES / "backtest_metricas.csv", index=False)
    final[["semana", "tienda", "categoria", "pronostico", "naive", "hw", "gbm"]].to_csv(RES / "pronostico_13_semanas.csv", index=False)
    plan.to_csv(RES / "plan_pedidos.csv", index=False)

    rutas = graficos(df, preds, metricas, final, mejor)
    (RAIZ / "REPORTE.md").write_text(reporte(df, resumen, metricas, mejor, plan, rutas), encoding="utf-8")
    print("mejor:", mejor)


def graficos(df, preds, metricas, final, mejor):
    rutas = {}
    tot = df.groupby("semana").unidades.sum()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(pd.to_datetime(tot.index), tot.values / 1000, color=AZUL, lw=1.6)
    fer = df[df.feriado == 1].groupby("semana").unidades.sum()
    ax.scatter(pd.to_datetime(fer.index), fer.values / 1000, s=14, color=NARANJA, zorder=3, label="Semana con feriado")
    ax.set_ylabel("Miles de unidades", color=TINTA_2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    estilo(ax, "Ventas semanales totales, 2023–2025")
    fig.tight_layout(); fig.savefig(GRAF / "ventas_historicas.png", dpi=140); plt.close(fig)
    rutas["historia"] = "graficos/ventas_historicas.png"

    orden = list(NOMBRES)
    w = metricas.groupby("modelo").wape.mean().reindex(orden)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colores = [AZUL if m == mejor else "#9ec5f4" for m in orden]
    ax.barh([NOMBRES[m] for m in orden][::-1], w.values[::-1] * 100, color=colores[::-1], height=0.55)
    for i, v in enumerate(w.values[::-1]):
        ax.text(v * 100, i, f" {v:.1%}", va="center", fontsize=9, color=TINTA_2)
    ax.set_xlabel("WAPE promedio en 4 cortes (menor es mejor)", color=TINTA_2)
    ax.grid(axis="y", visible=False)
    estilo(ax, "Error de pronóstico por modelo")
    fig.tight_layout(); fig.savefig(GRAF / "comparacion_modelos.png", dpi=140); plt.close(fig)
    rutas["comparacion"] = "graficos/comparacion_modelos.png"

    serie = ("Panamá Norte", "Licores")
    hist = df[(df.tienda == serie[0]) & (df.categoria == serie[1])]
    bt = preds[(preds.tienda == serie[0]) & (preds.categoria == serie[1])]
    fut = final[(final.tienda == serie[0]) & (final.categoria == serie[1])]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(pd.to_datetime(hist.semana), hist.unidades, color="#898781", lw=1.2, label="Real")
    ax.plot(pd.to_datetime(bt.semana), bt[mejor], color=AZUL, lw=1.8, label=f"Backtest ({NOMBRES[mejor]})")
    s = fut.sigma.iloc[0]
    ax.plot(pd.to_datetime(fut.semana), fut.pronostico, color=NARANJA, lw=2, label="Pronóstico 2026")
    ax.fill_between(pd.to_datetime(fut.semana), fut.pronostico - 1.28 * s, fut.pronostico + 1.28 * s,
                    color=NARANJA, alpha=0.15, lw=0, label="Intervalo 80 %")
    ax.legend(frameon=False, fontsize=9, ncol=4, loc="upper left")
    ax.set_ylabel("Unidades por semana", color=TINTA_2)
    estilo(ax, f"{serie[1]} · {serie[0]}: la serie más estacional")
    fig.tight_layout(); fig.savefig(GRAF / "pronostico_licores.png", dpi=140); plt.close(fig)
    rutas["pronostico"] = "graficos/pronostico_licores.png"
    return rutas


def reporte(df, resumen, metricas, mejor, plan, rutas):
    base = resumen.loc["naive", "wape"]
    gan = 1 - resumen.loc[mejor, "wape"] / base
    filas = "\n".join(f"| {NOMBRES[m]} | {r.wape:.1%} | {r.sesgo:+.1%} |" for m, r in resumen.iterrows())
    filas_plan = "\n".join(
        f"| {r.tienda} | {r.categoria} | {r.pronostico_13s:,.0f} | {r.stock_seguridad:,.0f} | {r.pedido_recomendado:,} |"
        for r in plan.itertuples()
    )
    return f"""# Reporte · Pronóstico de demanda

> Datos **sintéticos** de una cadena ficticia: 3 tiendas × 4 categorías, {df.semana.nunique()} semanas (2023–2025).

## Resultado

El mejor modelo es **{NOMBRES[mejor]}**, con un error (WAPE) de **{resumen.loc[mejor, 'wape']:.1%}**: **{gan:.0%} menos error** que repetir las ventas del año anterior.

| Modelo | WAPE promedio | Sesgo |
|---|---:|---:|
{filas}

Evaluación con **backtesting de origen móvil**: 4 cortes, cada uno pronostica 13 semanas sin ver el futuro.

![Comparación]({rutas['comparacion']})
![Pronóstico]({rutas['pronostico']})

## Plan de pedidos para el próximo trimestre

Pronóstico de 13 semanas más stock de seguridad para un **95 % de nivel de servicio** y 2 semanas de reposición (σ del error de backtest).

| Tienda | Categoría | Pronóstico 13 semanas | Stock de seguridad | Pedido recomendado |
|---|---|---:|---:|---:|
{filas_plan}

![Historia]({rutas['historia']})

## Conclusiones

1. Los **feriados** explican los picos más grandes, sobre todo en Licores: incluirlos como variable reduce el error en esas semanas.
2. Combinar un modelo estadístico (Holt-Winters) con uno de machine learning (Gradient Boosting) suele ser más estable que cualquiera por separado.
3. El stock de seguridad depende del error histórico de cada serie: las series más volátiles necesitan más colchón.
"""


if __name__ == "__main__":
    main()
