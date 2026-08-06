# -*- coding: utf-8 -*-
"""
Validación de KitagawaHealth v2.1.0 con datos reales
====================================================
Fuente: BASE_ENDES_2009_2025_completa.xlsx, hoja DATOS
        (INEI, Indicadores de Resultados de los Programas Presupuestales)

ADVERTENCIA SOBRE LOS PESOS
---------------------------
n_muestra es el TAMAÑO MUESTRAL departamental de la ENDES, no la población
infantil. La ENDES sobremuestrea los departamentos pequeños y usa factores
de expansión, de modo que el agregado ponderado por n_muestra NO reproduce
la cifra nacional del INEI (2015: 46,5 % vs 43,5 % publicado en anemia).
Por eso el "efecto de estructura" que se estima aquí mide RECOMPOSICIÓN
MUESTRAL, un artefacto del diseño de la encuesta, no un desplazamiento
demográfico. Sirve como diagnóstico de la comparabilidad de los promedios
departamentales entre años; no debe leerse como hallazgo epidemiológico.
Para una lectura demográfica hay que reemplazar n_muestra por la población
de 6 a 35 meses (anemia) o menor de 5 años (DCI) por departamento.
"""
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, '/home/claude')
from kitagawa_health import KitagawaDecomposer

RUTA = '/mnt/user-data/uploads/BASE_ENDES_2009_2025_completa.xlsx'
SALIDA = '/mnt/user-data/outputs/kitagawa_validacion_endes.xlsx'
SEMILLA = 42
N_BOOT = 2000

d = pd.read_excel(RUTA, sheet_name='DATOS')
dep = d[(d.ambito != 'Perú') & d.n_muestra.notna() & d.prevalencia.notna()]

hojas = {}

# ── Parte 1. Kitagawa canónico: cambio 2015 → 2025 ────────────────────────
def panel_dos_anios(indicador, anio_fin, anio_ini):
    a = dep[(dep.indicador == indicador) & (dep.anio == anio_fin)].set_index('ambito')
    b = dep[(dep.indicador == indicador) & (dep.anio == anio_ini)].set_index('ambito')
    com = a.index.intersection(b.index)
    return pd.DataFrame({
        'departamento': com, 'Año': anio_fin,
        'r_fin': a.loc[com, 'prevalencia'].values,
        'r_ini': b.loc[com, 'prevalencia'].values,
        'n_fin': a.loc[com, 'n_muestra'].values,
        'n_ini': b.loc[com, 'n_muestra'].values})

print("=" * 76)
print("  PARTE 1 — Cambio 2015→2025 descompuesto por departamento")
print("=" * 76)
filas_p1, detalle_p1 = [], []
for ind in ('ANEMIA', 'DCI'):
    P = panel_dos_anios(ind, 2025, 2015)
    kd = KitagawaDecomposer(
        P, 'departamento', 'r_fin', 'r_ini',
        count_A_col='n_fin', count_B_col='n_ini',
        group_A_label=f'{ind} 2025', group_B_label=f'{ind} 2015',
        random_state=SEMILLA).run()
    assert kd.check_identity(verbose=False)['passed']
    bs = kd.bootstrap(n_boot=N_BOOT, rate_uncertainty='binomial',
                      rate_scale=100, conf_level=0.95, random_state=SEMILLA)
    print(f"\n── {ind} ──  n(2025)={P.n_fin.sum():.0f}  n(2015)={P.n_ini.sum():.0f}")
    for comp in ('D_total', 'C_tasa', 'C_estructura'):
        r = bs[bs.componente == comp].iloc[0]
        print(f"  {comp:14s}{r.estimacion:9.4f} pp  EE={r.ee_boot:6.4f}  "
              f"IC95% [{r.ic_inf:7.4f}, {r.ic_sup:7.4f}]  "
              f"≠0: {'sí' if r.significativo else 'no'}")
        filas_p1.append(dict(indicador=ind, **r.to_dict()))
    print(f"  recomposición muestral = "
          f"{100*kd.results_.C_estructura.sum()/kd.results_.C_total.sum():.1f}% del cambio")
    detalle_p1.append(kd.results_.assign(indicador=ind))

hojas['P1_componentes_IC'] = pd.DataFrame(filas_p1)
hojas['P1_detalle_departamento'] = pd.concat(detalle_p1, ignore_index=True)

# ── Parte 2. Defecto 1: compare_years sobre panel real ────────────────────
w = (dep.pivot_table(index=['ambito', 'anio'], columns='indicador',
                     values=['prevalencia', 'n_muestra']).dropna().reset_index())
w.columns = ['ambito', 'anio', 'n_ANEMIA', 'n_DCI', 'p_ANEMIA', 'p_DCI']

kp = KitagawaDecomposer(w, 'ambito', 'p_ANEMIA', 'p_DCI',
                        count_A_col='n_ANEMIA', count_B_col='n_DCI',
                        year_col='anio', group_A_label='Anemia',
                        group_B_label='DCI', random_state=SEMILLA).run()

print("\n" + "=" * 76)
print("  PARTE 2 — Defecto 1: cierre de compare_years (2015→2025)")
print("=" * 76)
exacta = kp.compare_years(2015, 2025, round_output=False)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    leg15 = kp.compare_years(2015, 2025, base_year=2015, method='legacy',
                             round_output=False)
    leg25 = kp.compare_years(2015, 2025, base_year=2025, method='legacy',
                             round_output=False)
comp = pd.DataFrame({
    'v2.1.0 (exacta)':   [exacta[k].sum() for k in
                          ('delta_total', 'efecto_tasa', 'efecto_estructura', 'residuo')],
    'v2.0.0 base=2015':  [leg15[k].sum() for k in
                          ('delta_total', 'efecto_tasa', 'efecto_estructura', 'residuo')],
    'v2.0.0 base=2025':  [leg25[k].sum() for k in
                          ('delta_total', 'efecto_tasa', 'efecto_estructura', 'residuo')],
}, index=['delta_total', 'efecto_tasa', 'efecto_estructura', 'residuo'])
print(comp.round(4).to_string())
hojas['P2_compare_years'] = comp.round(6).reset_index(names='termino')
hojas['P2_por_departamento'] = exacta

# ── Parte 3. Defecto 2: ventanas bienales ─────────────────────────────────
v = dep[(dep.indicador == 'ANEMIA') & dep.anio.isin([2024, 2025])].merge(
    dep[(dep.indicador == 'DCI') & dep.anio.isin([2024, 2025])],
    on=['ambito', 'anio'], suffixes=('_a', '_d')).assign(Año=2025)

def corre(modo):
    return KitagawaDecomposer(v, 'ambito', 'prevalencia_a', 'prevalencia_d',
                              count_A_col='n_muestra_a', count_B_col='n_muestra_d',
                              rate_weighting=modo).run()

kw, ks = corre('counts'), corre('simple')
cmpw = pd.DataFrame({'tasa_ponderada': kw.results_.set_index('Estrato').tasa_A,
                     'tasa_simple_v200': ks.results_.set_index('Estrato').tasa_A})
cmpw['diferencia_pp'] = cmpw.tasa_ponderada - cmpw.tasa_simple_v200
print("\n" + "=" * 76)
print("  PARTE 3 — Defecto 2: ponderación con ventanas bienales (2024+2025)")
print("=" * 76)
print(cmpw.reindex(cmpw.diferencia_pp.abs().sort_values(ascending=False).index)
      .head(6).round(4).to_string())
print(f"\n  Brecha agregada  ponderada={kw.results_.C_total.sum():.4f}  "
      f"simple={ks.results_.C_total.sum():.4f}  "
      f"dif={kw.results_.C_total.sum()-ks.results_.C_total.sum():+.4f} pp")
hojas['P3_ponderacion'] = cmpw.round(6).reset_index()

with pd.ExcelWriter(SALIDA, engine='openpyxl') as xw:
    for nombre, tabla in hojas.items():
        tabla.to_excel(xw, sheet_name=nombre[:31], index=False)
print(f"\nResultados guardados en {SALIDA}")
