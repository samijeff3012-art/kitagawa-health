# -*- coding: utf-8 -*-
"""Suite de verificación formal — KitagawaHealth v2.1.0."""
import sys, warnings
import numpy as np, pandas as pd
sys.path.insert(0, '/home/claude')
import matplotlib
matplotlib.use('Agg')
from kitagawa_health import KitagawaDecomposer, DataError

TOL = 1e-9
ok = fail = 0

def check(nombre, cond, detalle=""):
    global ok, fail
    if cond:
        ok += 1;  print(f"  [OK  ] {nombre} {detalle}")
    else:
        fail += 1; print(f"  [FALLA] {nombre} {detalle}")

def datos(seed=7, n_estratos=8, anios=(2020, 2021, 2022), filas_por_estrato=1):
    rng = np.random.default_rng(seed)
    f = []
    for y in anios:
        for i in range(n_estratos):
            for _ in range(filas_por_estrato):
                f.append(dict(Anio=y, Causas=f"Causa {i}",
                              tA=rng.uniform(5, 60), tB=rng.uniform(5, 60),
                              nA=int(rng.integers(50, 900)),
                              nB=int(rng.integers(50, 900))))
    return pd.DataFrame(f)

def build(df, **kw):
    return KitagawaDecomposer(df, "Causas", "tA", "tB",
                              count_A_col="nA", count_B_col="nB",
                              year_col="Anio", **kw).run()

print("=" * 66)
print("  PRUEBAS DE PROPIEDADES — KitagawaHealth v2.1.0")
print("=" * 66)

kd = build(datos())

print("\n-- Bloque 1: identidad de Kitagawa (debía seguir intacta) --")
r = kd.results_
check("1 identidad por estrato",
      (r.C_tasa + r.C_estructura - r.C_total).abs().max() < TOL)
check("2 descomposicion iguala brecha ponderada",
      abs(r.groupby("Anio").C_total.sum().sub(
          r.groupby("Anio").brecha_tasa_real.sum()).abs().max()) < TOL)
check("3 proporciones suman 1",
      (r.groupby("Anio").prop_A.sum() - 1).abs().max() < TOL)

print("\n-- Bloque 2: DEFECTO 1 — cierre de compare_years --")
c = kd.compare_years(2020, 2022, round_output=False)
res = abs(c.delta_total.sum() - c.efecto_tasa.sum() - c.efecto_estructura.sum())
check("4 cierre exacto", res < TOL, f"(residuo={res:.2e})")

sumas = []
for base in (2020, 2021, 2022):
    cb = kd.compare_years(2020, 2022, base_year=base, round_output=False)
    sumas.append(cb.efecto_tasa.sum())
check("5 invariancia al ano base", np.ptp(sumas) < TOL,
      f"(rango={np.ptp(sumas):.2e})")

crev = kd.compare_years(2022, 2020, round_output=False)
m  = c.set_index("Estrato")[["efecto_tasa", "efecto_estructura"]]
mr = crev.set_index("Estrato")[["efecto_tasa", "efecto_estructura"]].loc[m.index]
check("6 antisimetria al invertir los anios",
      (m + mr).abs().to_numpy().max() < TOL)

c_id = kd.compare_years(2021, 2021, round_output=False)
check("7 comparar un anio consigo mismo da cero",
      c_id[["delta_total", "efecto_tasa", "efecto_estructura"]].abs()
      .to_numpy().max() < TOL)

# transitividad: (2020->2021) + (2021->2022) == (2020->2022) en delta_total
a = kd.compare_years(2020, 2021, round_output=False).delta_total.sum()
b = kd.compare_years(2021, 2022, round_output=False).delta_total.sum()
check("8 delta_total es transitivo entre anios", abs(a + b - c.delta_total.sum()) < TOL)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cl = kd.compare_years(2020, 2022, method="legacy", round_output=False)
res_legacy = abs(cl.delta_total.sum() - cl.efecto_tasa.sum() - cl.efecto_estructura.sum())
check("9 method='legacy' reproduce el defecto v2.0.0 (control)",
      res_legacy > 1e-3, f"(residuo v2.0.0={res_legacy:.4f})")

print("\n-- Bloque 3: DEFECTO 2 — ponderacion de tasas por conteos --")
# Caso construido: un estrato, dos filas de exposicion muy distinta
d2 = pd.DataFrame([
    dict(Anio=2020, Causas="X", tA=90.0, tB=10.0, nA=10,   nB=100),
    dict(Anio=2020, Causas="X", tA=10.0, tB=10.0, nA=1000, nB=100),
])
kw_ = build(d2, rate_weighting="counts")
esperado = (90*10 + 10*1000) / 1010
check("10 tasa de estrato ponderada = suma(r*n)/suma(n)",
      abs(kw_.results_.tasa_A.iloc[0] - esperado) < 1e-9,
      f"(obtenido={kw_.results_.tasa_A.iloc[0]:.4f}, esperado={esperado:.4f})")
ks_ = build(d2, rate_weighting="simple")
check("11 rate_weighting='simple' conserva el comportamiento v2.0.0",
      abs(ks_.results_.tasa_A.iloc[0] - 50.0) < 1e-9,
      f"(obtenido={ks_.results_.tasa_A.iloc[0]:.4f})")
check("12 'auto' usa conteos cuando estan disponibles",
      abs(build(d2).results_.tasa_A.iloc[0] - esperado) < 1e-9)

# con una sola fila por estrato, ponderar no debe cambiar nada
d1 = datos(seed=11)
check("13 con una fila por estrato, ponderado == simple",
      np.allclose(build(d1, rate_weighting="counts").results_.tasa_A.to_numpy(),
                  build(d1, rate_weighting="simple").results_.tasa_A.to_numpy()))

# la identidad sigue cumpliendose con multiples filas por estrato
kmr = build(datos(seed=3, filas_por_estrato=4))
rm = kmr.results_
check("14 identidad se mantiene con multiples filas por estrato",
      (rm.C_tasa + rm.C_estructura - rm.C_total).abs().max() < TOL)

print("\n-- Bloque 4: inferencia por bootstrap --")
kb = build(datos(seed=21), random_state=123)
bs = kb.bootstrap(n_boot=300, rate_uncertainty="binomial", rate_scale=100,
                  conf_level=0.95)
check("15 el bootstrap devuelve resultados por anio y componente",
      set(bs.componente) == {"C_tasa", "C_estructura", "D_total"} and len(bs) == 9)
check("16 los errores estandar son positivos", (bs.ee_boot > 0).all())
sub = bs[bs.componente == "C_tasa"]
check("17 el IC contiene la estimacion puntual",
      ((sub.ic_inf <= sub.estimacion) & (sub.estimacion <= sub.ic_sup)).all())
b1 = kb.bootstrap(n_boot=200, random_state=999)
b2 = kb.bootstrap(n_boot=200, random_state=999)
check("18 el bootstrap es reproducible con la misma semilla",
      np.allclose(b1.ee_boot.to_numpy(), b2.ee_boot.to_numpy()))
bfix = kb.bootstrap(n_boot=200, rate_uncertainty="none", random_state=5)
check("19 con tasas fijas el EE es menor que con tasas aleatorias",
      bfix[bfix.componente == "C_tasa"].ee_boot.mean() <
      bs[bs.componente == "C_tasa"].ee_boot.mean())
bp = kb.bootstrap(n_boot=200, rate_uncertainty="poisson", rate_scale=1000,
                  random_state=5)
check("20 el modelo de tasas poisson corre y da EE positivos",
      (bp.ee_boot > 0).all())
bst = kb.bootstrap(n_boot=200, by_stratum=True, random_state=5)
check("21 by_stratum entrega IC por estrato",
      bst[bst.Estrato != "TOTAL"].shape[0] == 3 * 8 * 2)

# el bootstrap debe negarse a operar sin conteos
dprop = datos(seed=4).assign(pA=lambda d: d.nA / d.groupby("Anio").nA.transform("sum"),
                             pB=lambda d: d.nB / d.groupby("Anio").nB.transform("sum"))
kp = KitagawaDecomposer(dprop, "Causas", "tA", "tB", prop_A_col="pA",
                        prop_B_col="pB", year_col="Anio").run()
try:
    kp.bootstrap(n_boot=10); sin_conteos = False
except DataError:
    sin_conteos = True
check("22 el bootstrap falla explicitamente sin conteos", sin_conteos)

print("\n-- Bloque 5: validaciones y compatibilidad --")
try:
    build(datos(), rate_weighting="ponderado"); mal = False
except DataError:
    mal = True
check("23 rate_weighting invalido se rechaza", mal)
chk = kd.check_identity(verbose=False)
check("24 check_identity() reporta que todo pasa", chk["passed"])
check("25 la API previa sigue existiendo",
      hasattr(kd, "summary_table") and hasattr(kd, "annual_summary")
      and hasattr(kd, "plot_map") and hasattr(kd, "export_results"))
kd.export_results("/tmp/_t.xlsx")
hojas = pd.ExcelFile("/tmp/_t.xlsx").sheet_names
check("26 la exportacion a Excel conserva sus hojas", len(hojas) >= 3, f"({hojas})")
kb.export_results("/tmp/_t2.xlsx")
check("27 la exportacion suma la hoja de inferencia",
      "Inferencia_Bootstrap" in pd.ExcelFile("/tmp/_t2.xlsx").sheet_names)

# v2.1.0b: el bootstrap debe rechazar pesos ficticios
dfake = datos(seed=8).assign(nA=1.0, nB=1.0)
kf = build(dfake)
try:
    kf.bootstrap(n_boot=10); rechazo = False
except DataError:
    rechazo = True
check("28 el bootstrap rechaza conteos ficticios (todos = 1)", rechazo)

print("\n" + "=" * 66)
print(f"  RESULTADO: {ok} pruebas pasan, {fail} fallan")
print("=" * 66)
sys.exit(1 if fail else 0)
