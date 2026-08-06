# -*- coding: utf-8 -*-
"""
KitagawaHealth v2.1.0 — Ejemplo de uso con datos sintéticos
===========================================================
Este archivo se ejecuta tal cual, sin datos externos:

    python example_usage.py

Genera un conjunto de datos ficticio de brechas por sexo en cinco causas y
tres años, y recorre las funciones principales del módulo: descomposición,
verificación formal, comparación entre años, inferencia por bootstrap y
exportación.
"""
import numpy as np
import pandas as pd

from kitagawa_health import KitagawaDecomposer, analyze_health_gap

SEMILLA = 2026

# ── 1. Datos sintéticos ───────────────────────────────────────────────────
# Cinco causas de muerte, tres años, tasas por 100.000 y población expuesta.
rng = np.random.default_rng(SEMILLA)
causas = ['Cardiovascular', 'Neoplasias', 'Respiratorias',
          'Externas', 'Metabólicas']

filas = []
for anio in (2015, 2020, 2025):
    for causa in causas:
        filas.append({
            'Año': anio,
            'Causa': causa,
            'Tasa hombres': rng.uniform(20, 180),
            'Tasa mujeres': rng.uniform(15, 150),
            'Población hombres': int(rng.integers(40_000, 200_000)),
            'Población mujeres': int(rng.integers(40_000, 200_000)),
        })
df = pd.DataFrame(filas)
print(f"Datos de ejemplo: {df.shape[0]} filas, {df.Causa.nunique()} causas, "
      f"{df.Año.nunique()} años\n")

# ── 2. Descomposición ─────────────────────────────────────────────────────
kd = KitagawaDecomposer(
    data           = df,
    stratum_col    = 'Causa',
    rate_A_col     = 'Tasa hombres',
    rate_B_col     = 'Tasa mujeres',
    count_A_col    = 'Población hombres',
    count_B_col    = 'Población mujeres',
    year_col       = 'Año',
    group_A_label  = 'Hombres',
    group_B_label  = 'Mujeres',
    rate_weighting = 'auto',   # pondera las tasas por la población expuesta
    random_state   = SEMILLA,
).run()

print("── Resumen anual ──")
print(kd.annual_summary().to_string(index=False), "\n")

print("── Estratos que más aportan a la brecha (2025) ──")
print(kd.summary_table(top_n=3).to_string(index=False), "\n")

# ── 3. Verificación formal de las identidades ─────────────────────────────
kd.check_identity()
print()

# ── 4. Cambio de la brecha entre dos años ─────────────────────────────────
# Con la v2.1.0 la descomposición es exacta: efecto_tasa + efecto_estructura
# reproduce delta_total y el resultado no depende de ningún año base.
cambio = kd.compare_years(2015, 2025)
print("── Cambio de la brecha 2015 → 2025 ──")
print(f"  delta_total       = {cambio.delta_total.sum():9.4f}")
print(f"  efecto_tasa       = {cambio.efecto_tasa.sum():9.4f}")
print(f"  efecto_estructura = {cambio.efecto_estructura.sum():9.4f}")
print(f"  residuo           = {abs(cambio.residuo.sum()):9.2e}  (debe ser ~0)\n")

# ── 5. Inferencia por bootstrap ───────────────────────────────────────────
# Las tasas son por 100.000, de modo que el modelo adecuado es 'poisson'.
# Para prevalencias en porcentaje se usa 'binomial' con rate_scale=100.
bs = kd.bootstrap(
    n_boot           = 500,
    rate_uncertainty = 'poisson',
    rate_scale       = 100_000,
    conf_level       = 0.95,
    random_state     = SEMILLA,
)
print("── Componentes con intervalo de confianza al 95 % ──")
print(bs[bs.Año == 2025].to_string(index=False), "\n")

# ── 6. Gráficos y exportación ─────────────────────────────────────────────
# Descomente para generar las figuras:
# kd.plot_decomposition()
# kd.plot_strata_ranked(top_n=5)
# kd.plot_temporal_evolution()
# kd.plot_compare_years(2015, 2025)

kd.export_results('resultados_ejemplo.xlsx')
print("Resultados exportados a resultados_ejemplo.xlsx "
      "(incluye la hoja Inferencia_Bootstrap)\n")

# ── 7. La misma tubería en una sola llamada ───────────────────────────────
kd2 = analyze_health_gap(
    data          = df,
    stratum_col   = 'Causa',
    rate_A_col    = 'Tasa hombres',
    rate_B_col    = 'Tasa mujeres',
    count_A_col   = 'Población hombres',
    count_B_col   = 'Población mujeres',
    year_col      = 'Año',
    group_A_label = 'Hombres',
    group_B_label = 'Mujeres',
    plot          = False,
)
print("analyze_health_gap() reproduce el mismo resultado:",
      np.allclose(kd.results_.C_total, kd2.results_.C_total))
