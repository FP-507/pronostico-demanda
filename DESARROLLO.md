# Notas de desarrollo · Pronóstico de demanda

Guía para retomar el proyecto: cómo está armado, por qué y cómo cambiarlo.

## Puesta en marcha

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
python -m demanda.datos        # datos/ventas_semanales.csv (12 series × 157 semanas, semilla 5)
python -m demanda.backtest     # backtest, pronóstico final, plan de pedidos, gráficos y REPORTE.md
pytest -q
```

`python -m demanda.backtest` genera los datos si no existen. Holt-Winters se ajusta 12 series × 5 veces (4 cortes y el final), así que tarda alrededor de un minuto.

## Módulos

| Archivo | Responsabilidad | Funciones clave |
|---|---|---|
| `demanda/datos.py` | Datos sintéticos y calendario de feriados (`FERIADOS`, por año y semana ISO) | `generar`, `es_feriado` |
| `demanda/modelos.py` | Métricas, modelos y variables. Sin E/S | `wape`, `sesgo`, `naive_estacional`, `holt_winters`, `caracteristicas`, `gbm` |
| `demanda/backtest.py` | Orquestación: backtest, pronóstico final, plan de pedidos, gráficos, reporte | `pronosticar`, `backtest`, `futuro_exogeno`, `main` |

Salidas: `resultados/backtest_metricas.csv`, `resultados/pronostico_13_semanas.csv`, `resultados/plan_pedidos.csv`, `graficos/*.png` y `REPORTE.md`.

## Flujo de `backtest.main`

1. **Backtest:** 4 cortes separados por 13 semanas, contando hacia atrás desde la última. En cada corte `pronosticar()` entrena con el pasado y predice las 13 semanas siguientes con los 4 modelos.
2. **Selección:** el modelo con menor WAPE promedio.
3. **Incertidumbre:** desviación estándar del error del mejor modelo, por serie, durante el backtest.
4. **Pronóstico final:** se agregan 13 semanas futuras con `futuro_exogeno()` (feriados conocidos, sin promociones) y se re-entrena con toda la historia.
5. **Plan de pedidos:** `pronóstico de 13 semanas + Z × σ × √(semanas de reposición)`, con Z = 1,65 (95 % de servicio) y 2 semanas de reposición.

## Decisiones

- **Rezagos de 13 semanas o más** (`LAGS = [13, 14, 15, 26, 52]`). El modelo puede pronosticar las 13 semanas del horizonte en forma directa, sin usar sus propias predicciones como entrada (sin recursión ni acumulación de error) y sin fuga de datos. `test_rezagos_no_filtran_el_futuro` lo verifica.
- **Un modelo global de Gradient Boosting** para las 12 series. Aprende patrones compartidos (feriados, promociones) con más datos que un modelo por serie. Tienda y categoría entran como variables categóricas nativas (`categorical_features="from_dtype"`).
- **Objetivo en logaritmo** (`log1p` / `expm1`): estabiliza la varianza entre series de escalas muy distintas (Licores frente a Abarrotes).
- **Semana del año en seno y coseno,** para que la semana 52 quede "cerca" de la semana 1.
- **Holt-Winters amortiguado y multiplicativo:** la estacionalidad crece con el nivel y la tendencia no se extrapola sin límite.
- **Ensamble simple** (promedio HW + GBM): suele ser más estable que cada modelo solo y no necesita ajustar pesos.
- **WAPE en vez de MAPE:** MAPE explota con semanas de poca venta. El sesgo indica si se pronostica de más o de menos (sesgo negativo: faltante de inventario).
- **Feriados fijos por semana ISO** en un diccionario, para poder editarlos a mano cada año.

## Tests

`tests/test_demanda.py`: métricas con valores calculados a mano, la línea base estacional, que los rezagos no filtren el futuro, que el pronóstico tenga 12 × 13 filas y todos los modelos (y que el ensamble no sea mucho peor que la línea base), y que el futuro exógeno marque feriados. La CI los corre en cada push.

## Cómo extenderlo

- **Nuevo modelo:** escribir la función en `modelos.py`, agregar su columna en `pronosticar()` y su nombre en `NOMBRES` de `backtest.py`. El backtest, la selección y los gráficos lo incluyen solos.
- **Nueva variable para el GBM:** calcularla en `caracteristicas()` y sumarla a `COLUMNAS_GBM`. Si depende de ventas, el rezago tiene que ser de `HORIZONTE` o más.
- **Feriados de otro año:** agregar una entrada a `FERIADOS` en `datos.py` con las semanas ISO.
- **Promociones planificadas:** hoy `futuro_exogeno()` pone `promocion = 0`; recibir un calendario de promociones mejoraría el pronóstico final.
- **Otro nivel de servicio o tiempo de reposición:** `Z_SERVICIO` y el `np.sqrt(2)` en `main()`.

## Pendientes

- Intervalos de predicción más formales (cuantiles con `loss="quantile"` en el GBM) en vez de suponer error normal.
- Pasar las semanas de reposición a una constante en lugar del `2` dentro de `np.sqrt(2)`.
- Las cifras del README están escritas a mano a partir de `resultados/backtest_metricas.csv`.
