import numpy as np
import pandas as pd
import pytest

from demanda import modelos as M
from demanda.backtest import futuro_exogeno, pronosticar
from demanda.datos import generar


@pytest.fixture(scope="module")
def ventas():
    df = generar(semilla=1)
    df["semana"] = pd.to_datetime(df.semana).dt.date
    return df


def test_metricas():
    assert M.wape([100, 100], [90, 110]) == pytest.approx(0.1)
    assert M.sesgo([100, 100], [110, 110]) == pytest.approx(0.1)


def test_naive_estacional_repite_anio_anterior():
    s = pd.Series(np.arange(104))
    assert list(M.naive_estacional(s, 3)) == [52, 53, 54]


def test_rezagos_no_filtran_el_futuro(ventas):
    f = M.caracteristicas(ventas)
    serie = f[(f.tienda == "David") & (f.categoria == "Lácteos")].reset_index(drop=True)
    # El rezago más corto es el horizonte: nunca usa ventas de las últimas 13 semanas.
    assert min(M.LAGS) >= M.HORIZONTE
    assert serie.loc[20, "lag_13"] == serie.loc[7, "unidades"]


def test_pronostico_tiene_horizonte_y_modelos(ventas):
    corte = pd.Timestamp(max(ventas.semana)) - pd.Timedelta(weeks=13)
    p = pronosticar(ventas, corte)
    assert len(p) == 12 * M.HORIZONTE
    assert {"naive", "hw", "gbm", "ensamble"} <= set(p.columns)
    assert (p[["hw", "gbm"]] > 0).all().all()
    assert M.wape(p.unidades, p.ensamble) < M.wape(p.unidades, p.naive) * 1.2


def test_futuro_exogeno_marca_feriados(ventas):
    fut = futuro_exogeno(ventas, 13)
    assert len(fut) == 12 * 13
    assert fut.unidades.isna().all()
    assert fut.feriado.sum() > 0           # Carnaval cae en el primer trimestre
