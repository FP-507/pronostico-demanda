"""Modelos de pronóstico: línea base estacional, Holt-Winters y gradient boosting global."""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

HORIZONTE = 13  # semanas (un trimestre)
LAGS = [13, 14, 15, 26, 52]


def wape(real, pred) -> float:
    """Error porcentual absoluto ponderado: sum|e| / sum(real). Robusto a valores pequeños."""
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float(np.abs(real - pred).sum() / np.abs(real).sum())


def sesgo(real, pred) -> float:
    real, pred = np.asarray(real, float), np.asarray(pred, float)
    return float((pred - real).sum() / real.sum())


def naive_estacional(historia: pd.Series, h: int = HORIZONTE) -> np.ndarray:
    """Repite la misma semana del año anterior."""
    return historia.iloc[-52:-52 + h].to_numpy() if h < 52 else historia.iloc[-52:].to_numpy()[:h]


def holt_winters(historia: pd.Series, h: int = HORIZONTE) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(historia.to_numpy(float), trend="add", damped_trend=True,
                                      seasonal="mul", seasonal_periods=52, initialization_method="estimated")
        return modelo.fit(optimized=True).forecast(h)


def caracteristicas(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de calendario y rezagos de al menos HORIZONTE semanas (pronóstico directo sin recursión)."""
    df = df.sort_values(["tienda", "categoria", "semana"]).copy()
    semana_anio = pd.to_datetime(df.semana).dt.isocalendar().week.astype(int)
    df["sem_sin"] = np.sin(2 * np.pi * semana_anio / 52)
    df["sem_cos"] = np.cos(2 * np.pi * semana_anio / 52)
    df["t"] = (pd.to_datetime(df.semana) - pd.Timestamp("2023-01-02")).dt.days / 7
    g = df.groupby(["tienda", "categoria"]).unidades
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    df["media_13_25"] = g.transform(lambda s: s.shift(13).rolling(13).mean())
    for col in ("tienda", "categoria"):
        df[col] = df[col].astype("category")
    return df


COLUMNAS_GBM = ["tienda", "categoria", "promocion", "feriado", "sem_sin", "sem_cos", "t",
                *[f"lag_{l}" for l in LAGS], "media_13_25"]


def gbm(entreno: pd.DataFrame, futuro: pd.DataFrame) -> np.ndarray:
    """Un solo modelo para todas las series; aprende el log de las unidades."""
    X = entreno.dropna(subset=COLUMNAS_GBM)
    modelo = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, max_leaf_nodes=20,
                                           l2_regularization=0.5, categorical_features="from_dtype",
                                           random_state=0)
    modelo.fit(X[COLUMNAS_GBM], np.log1p(X.unidades))
    return np.expm1(modelo.predict(futuro[COLUMNAS_GBM]))
