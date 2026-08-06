# Registro de cambios — KitagawaHealth

Este proyecto sigue [Versionado Semántico](https://semver.org/lang/es/).

---

## [2.1.0] — 2026-08

### ADVERTENCIA DE REPRODUCIBILIDAD

**Esta versión cambia resultados numéricos, no solo la implementación.**
Quien ejecute la v2.1.0 sobre los mismos datos con que corrió la v2.0.0
obtendrá cifras distintas en `compare_years()` y, si hay varias filas por
estrato, también en la descomposición principal. Los valores de la v2.1.0
son los correctos; los de la v2.0.0 estaban afectados por los defectos que
se detallan abajo. Todo resultado ya calculado o difundido con `compare_years()`
de la v2.0.0 debe recalcularse.

Para reproducir resultados publicados con la versión anterior se conserva
`compare_years(..., method='legacy')`, que replica la fórmula defectuosa y
emite una advertencia al usarla.

### Corregido

- **`compare_years()` no cerraba.** La fórmula anterior fijaba las
  proporciones y las tasas de un año base y omitía el término de
  interacción, de modo que `efecto_tasa + efecto_estructura` no sumaba el
  `delta_total` que la propia función reportaba. En datos reales de la ENDES
  (26 departamentos, 2015 vs 2025) el residuo alcanzaba el 10,4 % del cambio
  total y el efecto de estructura quedaba subestimado por un factor de 2,6.
  Además el resultado dependía del año base elegido, hasta el punto de
  invertir el signo del efecto de estructura.

  La v2.1.0 aplica la identidad exacta del producto
  `a₂b₂ − a₁b₁ = Δa·(b₁+b₂)/2 + Δb·(a₁+a₂)/2` por separado a cada grupo:

      efecto_tasa       =  Δr_A·(p_A1+p_A2)/2 − Δr_B·(p_B1+p_B2)/2
      efecto_estructura =  Δp_A·(r_A1+r_A2)/2 − Δp_B·(r_B1+r_B2)/2

  La suma reproduce `delta_total` con error de coma flotante, el resultado
  no depende de ninguna base y es antisimétrico al invertir los años.
  Se añadió una columna `residuo` para que el cierre sea verificable a simple
  vista. El parámetro `base_year` solo tiene efecto con `method='legacy'`.

- **Agregación de tasas sin ponderar.** Cuando había varias filas por
  estrato, las tasas se promediaban de forma simple aun teniendo los conteos
  disponibles, lo que sobrepondera las filas de menor exposición. Ahora la
  tasa del estrato es `Σ(r·n)/Σ(n)`. Con ventanas bienales de la ENDES la
  diferencia llega a 0,39 pp a nivel departamental. Controlado por el nuevo
  parámetro `rate_weighting` (`'auto'`, `'counts'`, `'simple'`);
  `'simple'` reproduce el comportamiento de la v2.0.0.

### Añadido

- **`bootstrap()`** — errores estándar e intervalos de confianza para los
  componentes de tasa y de estructura. Remuestreo multinomial de la
  composición combinado con cuatro modelos de incertidumbre de las tasas:
  `'binomial'` (prevalencias), `'poisson'` (tasas de incidencia o
  mortalidad), `'normal'` (errores estándar provistos por el usuario) y
  `'none'`. Devuelve estimación puntual, EE, IC percentil, sesgo y una
  bandera de significación; con `by_stratum=True` también por estrato.
  Rechaza explícitamente los conteos ausentes o implausibles en vez de
  producir una precisión inexistente.
- **`check_identity()`** — verificación formal de las identidades
  algebraicas del software, con reporte imprimible.
- Los conteos agregados se conservan en `results_`, requisito del bootstrap.
- Cuarta hoja `Inferencia_Bootstrap` en `export_results()`.
- Parámetros `rate_weighting` y `random_state` en `KitagawaDecomposer` y en
  `KitagawaLifecycle`.
- Atributo `__version__`.
- Suite de 28 pruebas de propiedades (`test_kitagawa_v210.py`), incluido un
  control invertido que confirma que la prueba de cierre detecta el defecto
  de la v2.0.0.

### Sin cambios

La identidad central `D_total = C_tasa + C_estructura` de la descomposición
de un solo periodo era correcta en la v2.0.0 y no se modificó. Tampoco
cambian la API pública, los gráficos, el mapa de burbujas ni el formato de
exportación.

---

## [2.0.0] — 2026-05

- Corrección en `KitagawaLifecycle`: cuando `rate_A_col == rate_B_col` la
  brecha resultaba siempre cero.
- Descomposición generalizada para cualquier par de grupos y cualquier
  estratificador.
- Mapa de burbujas con coordenadas departamentales incrustadas, sin
  geopandas ni conexión a internet.
- Exportación a Excel con tres hojas.
- Registro de Derechos de Autor de Software, INDECOPI (Perú).
