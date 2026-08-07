# KitagawaHealth v2.1.0

> **Herramienta generalizada de descomposición de Kitagawa para el análisis de brechas en indicadores cuantitativos de salud entre dos grupos cualesquiera.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20291586.svg)](https://doi.org/10.5281/zenodo.20291586)

---

## ¿Qué es KitagawaHealth?

**KitagawaHealth** es un módulo Python de código abierto que implementa la descomposición de Kitagawa (1955) de forma completamente generalizada para el análisis de brechas en indicadores de salud pública.

Separa el diferencial total entre dos grupos en:

- **Componente de tasa (C_T):** parte de la brecha que se debe a que una causa o evento es más letal o frecuente en un grupo que en el otro.
- **Componente de estructura (C_E):** parte que se debe a que un grupo concentra sus casos en causas distintas a las del otro grupo.

A diferencia de implementaciones existentes, KitagawaHealth:

- Acepta **cualquier par de grupos**: hombres vs mujeres, región A vs B, urbano vs rural, nivel educativo alto vs bajo, año base vs año actual, etc.
- Acepta **cualquier estratificador**: causas de muerte, diagnósticos, grupos de edad, ocupación, servicio de salud, etc.
- Entrega **errores estándar e intervalos de confianza** para ambos componentes.
- Incluye **mapa geográfico integrado** sin necesidad de internet ni geopandas.
- Detecta automáticamente **errores de configuración** antes de ejecutar.
- Funciona en **Google Colab** sin dependencias externas problemáticas.

---

## Novedades de la v2.1.0

- **Descomposición exacta de dos periodos.** `compare_years()` ahora cumple `efecto_tasa + efecto_estructura = delta_total` y su resultado no depende del año base. La fórmula de la v2.0.0 omitía el término de interacción.
- **Agregación de tasas ponderada por exposición** cuando hay varias filas por estrato, mediante el nuevo parámetro `rate_weighting`.
- **Inferencia por bootstrap** — errores estándar e intervalos de confianza para los componentes de tasa y de estructura, con `bootstrap()`.
- **Verificación formal** de las identidades algebraicas del software, con `check_identity()`.
- **Suite de 28 pruebas de propiedades** en `test_kitagawa_v210.py`.

> ### Aviso de reproducibilidad
>
> Esta versión **cambia resultados numéricos**, no solo la implementación. Quien ejecute la v2.1.0 sobre los mismos datos con que corrió la v2.0.0 obtendrá cifras distintas en `compare_years()` y, si hay varias filas por estrato, también en la descomposición principal. Los valores de la v2.1.0 son los correctos.
>
> Para reproducir resultados publicados con la versión anterior se conserva `compare_years(..., method='legacy')`, que replica la fórmula defectuosa y emite una advertencia al usarla.
>
> Detalles completos en [CHANGELOG.md](CHANGELOG.md).

---

## Autores

| Nombre | Institución |
|--------|-------------|
| Cesar Jefferson Samillan Vasquez | Universidad Nacional Toribio Rodríguez de Mendoza |
| Gladys Bernardita León Montoya | Universidad Nacional Toribio Rodríguez de Mendoza |
| Rosa Ysabel Bazán Valque | Universidad Nacional Toribio Rodríguez de Mendoza |
| Mercedes Acosta Román | Universidad Nacional Autónoma de Tayacaja Daniel Hernández Morillo |

**Contacto:** cesar.samillan@untrm.edu.pe

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/samijeff3012-art/kitagawa-health.git
cd kitagawa-health

# Instalar dependencias obligatorias
pip install -r requirements.txt

# Dependencia opcional para mapas con polígonos
pip install geopandas
```

En **Google Colab**:

```python
!pip install pandas numpy statsmodels matplotlib openpyxl
```

Para ver el módulo en funcionamiento sin datos propios:

```bash
python example_usage.py
```

---

## Inicio rápido

```python
import pandas as pd
from kitagawa_health import KitagawaDecomposer

# Cargar datos
df = pd.read_excel("mis_datos.xlsx")

# Crear el analizador
kd = KitagawaDecomposer(
    data          = df,
    stratum_col   = "Causas",
    rate_A_col    = "Tasa ajustada hombres",
    rate_B_col    = "Tasa ajustada mujeres",
    count_A_col   = "N° hombres",
    count_B_col   = "N° mujeres",
    group_A_label = "Hombres",
    group_B_label = "Mujeres",
    row_filter    = {"Región": "Nacional"},
)

# Ejecutar
kd.run()

# Ver resultados
print(kd.summary_table(top_n=10))

# Gráficos
kd.plot_decomposition()
kd.plot_strata_ranked(top_n=15)
kd.plot_map(year=2019)

# Exportar
kd.export_results("resultados.xlsx")
```

**Salida esperada:**

```
  Año  D_total_real  C_tasa_total  C_estru_total  pct_tasa  pct_estructura
 2008        9.1325       14.3517        -5.2192     157.1           -57.1
 2019       10.9172       13.3911        -2.4739     122.7           -22.7
 2020      199.4605      154.1335        45.3271      77.3            22.7
 2022       16.3578       17.0832        -0.7255     104.4            -4.4
```

---

## Verificación formal

Antes de interpretar cualquier resultado conviene comprobar que las identidades algebraicas se cumplen sobre los datos usados:

```python
kd.check_identity()
```

Verifica, para cada año, que `C_tasa + C_estructura = C_total` estrato por estrato, que la descomposición reproduce la brecha ponderada observada y que las proporciones de cada grupo suman uno; y, si hay dos o más años, que `compare_years()` cierra y es antisimétrico.

```
──────────────────────────────────────────────────────────────
  VERIFICACIÓN FORMAL — KitagawaHealth v2.1.0
──────────────────────────────────────────────────────────────
  [OK  ] identidad_por_estrato              error máx = 0.000e+00
  [OK  ] descomposicion_iguala_brecha       error máx = 3.553e-15
  [OK  ] prop_A_suman_1                     error máx = 0.000e+00
  [OK  ] prop_B_suman_1                     error máx = 0.000e+00
  [OK  ] cierre_compare_years               error máx = 1.776e-15
  [OK  ] antisimetria_compare_years         error máx = 0.000e+00
──────────────────────────────────────────────────────────────
  RESULTADO: TODAS LAS PRUEBAS PASAN
──────────────────────────────────────────────────────────────
```

---

## Comparación entre dos años

Descompone el **cambio** de la brecha entre dos periodos, separando cuánto proviene del movimiento de las tasas y cuánto de la recomposición de la estructura.

```python
cambio = kd.compare_years(2008, 2022)
print(cambio.head(10))
kd.plot_compare_years(2008, 2022)

# El cierre es verificable en la propia salida
print(cambio.residuo.sum())   # ~ 1e-15
```

La especificación aplica la identidad exacta del producto por separado a cada grupo, de modo que:

```
efecto_tasa       =  Δr_A·(p_A1+p_A2)/2  −  Δr_B·(p_B1+p_B2)/2
efecto_estructura =  Δp_A·(r_A1+r_A2)/2  −  Δp_B·(r_B1+r_B2)/2

efecto_tasa + efecto_estructura = delta_total     (identidad exacta)
```

Esto garantiza tres propiedades: **cierre** (los dos efectos suman el cambio total), **invariancia** (no hay año base que elegir) y **antisimetría** (invertir los años cambia solo el signo).

El parámetro `base_year` se conserva por compatibilidad, pero **solo tiene efecto con** `method='legacy'`, que reproduce la fórmula defectuosa de la v2.0.0 y emite una advertencia. Úselo únicamente para reproducir resultados publicados con esa versión.

---

## Inferencia por bootstrap

La descomposición de Kitagawa es una identidad algebraica exacta, pero sus insumos —tasas y composición— se estiman con error. `bootstrap()` propaga esa incertidumbre a los componentes.

```python
bs = kd.bootstrap(
    n_boot           = 2000,
    rate_uncertainty = 'binomial',   # prevalencias en porcentaje
    rate_scale       = 100,
    conf_level       = 0.95,
    by_stratum       = False,
    random_state     = 42,
)
print(bs)
```

Remuestrea la composición de cada grupo con una multinomial y las tasas según el modelo elegido:

| `rate_uncertainty` | Cuándo usarlo | `rate_scale` típico |
|---|---|---|
| `'binomial'` | La tasa es una prevalencia o proporción | 100 (%) o 1 |
| `'poisson'` | Tasa de incidencia o mortalidad por N unidades | 1 000 o 100 000 |
| `'normal'` | Se dispone de errores estándar propios (`se_A_col`, `se_B_col`) | — |
| `'none'` | Solo se quiere la incertidumbre de la composición | — |

Devuelve estimación puntual, error estándar, intervalo percentil, sesgo del bootstrap y una bandera `significativo` que indica si el intervalo excluye el cero. Con `by_stratum=True` entrega además el desglose por estrato.

**Requiere conteos reales.** Si faltan `count_A_col` y `count_B_col`, o si los conteos son pesos de relleno, el método falla con un mensaje explícito en vez de producir una precisión inexistente.

---

## Análisis por ciclo de vida

```python
from kitagawa_health import KitagawaLifecycle

lc = KitagawaLifecycle(
    data       = df,
    row_filter = {"Región": "Nacional"},
)
lc.run()
lc.plot_lifecycle_comparison()
lc.plot_lifecycle_bars(year=2019)
```

---

## Mapa geográfico (sin internet)

```python
# Funciona sin geopandas ni conexión a internet
# Coordenadas de 25 departamentos del Perú incrustadas
kd.plot_map(
    component = "C_total",
    year      = 2019,
    title     = "Brecha H-M por Departamento (2019)",
    save_path = "mapa_brecha_2019.png"
)
```

---

## Función de conveniencia en una línea

```python
from kitagawa_health import analyze_health_gap

kd = analyze_health_gap(
    data          = df,
    stratum_col   = "Causas",
    rate_A_col    = "Tasa ajustada hombres",
    rate_B_col    = "Tasa ajustada mujeres",
    count_A_col   = "N° hombres",
    count_B_col   = "N° mujeres",
    group_A_label = "Hombres",
    group_B_label = "Mujeres",
    row_filter    = {"Región": "Nacional"},
    top_n         = 15,
    export_path   = "resultados.xlsx",
)
```

---

## Estructura del repositorio

```
kitagawa-health/
├── kitagawa_health.py          ← Módulo principal
├── example_usage.py            ← Ejemplo ejecutable con datos sintéticos
├── test_kitagawa_v210.py       ← Suite de 28 pruebas de propiedades
├── validacion_endes_v210.py    ← Validación con datos reales de la ENDES
├── requirements.txt            ← Dependencias Python
├── CHANGELOG.md                ← Registro de cambios entre versiones
├── CITATION.cff                ← Metadatos de citación
├── .zenodo.json                ← Metadatos del depósito en Zenodo
├── LICENSE.txt                 ← Licencia MIT
└── README.md                   ← Este archivo
```

---

## Pruebas

```bash
python test_kitagawa_v210.py
```

Ejecuta 28 pruebas de propiedades agrupadas en cinco bloques: identidad de Kitagawa, cierre e invariancia de `compare_years()`, ponderación de tasas, inferencia por bootstrap, y validaciones de entrada y compatibilidad con la API previa. Incluye un **control invertido** que confirma que la prueba de cierre efectivamente detecta el defecto de la v2.0.0, de modo que un resultado favorable no puede deberse al azar.

---

## Componentes principales

| Componente | Tipo | Descripción |
|---|---|---|
| `KitagawaDecomposer` | Clase principal | Descomposición completa. Acepta cualquier par de grupos y cualquier estratificador. |
| `KitagawaLifecycle` | Clase especializada | Análisis por grupos del ciclo de vida. Detecta configuración incorrecta automáticamente. |
| `KitagawaAnalyzer` | Alias | Apunta a `KitagawaDecomposer` para compatibilidad. |
| `analyze_health_gap()` | Función | Pipeline completa en una línea. |
| `DataError` | Excepción | Error en datos de entrada — mensaje claro y accionable. |
| `FilterError` | Excepción | Filtros que dejan el DataFrame vacío. |

### Métodos de `KitagawaDecomposer`

| Método | Descripción |
|---|---|
| `run()` | Ejecuta la descomposición y llena `results_`. |
| `summary_table(top_n)` | Estratos ordenados por contribución a la brecha. |
| `annual_summary()` | Totales por año con el peso relativo de cada componente. |
| `compare_years(y1, y2)` | Descomposición exacta del cambio entre dos años. |
| `check_identity()` | Verificación formal de las identidades algebraicas. |
| `bootstrap()` | Errores estándar e intervalos de confianza. |
| `export_results(path)` | Exporta a Excel: detalle, resumen anual, top estratos e inferencia. |
| `plot_decomposition()`, `plot_strata_ranked()`, `plot_temporal_evolution()`, `plot_compare_years()`, `plot_map()` | Figuras. |

---

## Parámetros principales

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `data` | DataFrame | Requerido | Datos de entrada. |
| `stratum_col` | str | Requerido | Columna de estratificación (causas, diagnósticos, edad, etc.). |
| `rate_A_col` | str | Requerido | Tasa del grupo A. |
| `rate_B_col` | str | Requerido | Tasa del grupo B. Debe ser **diferente** de `rate_A_col`. |
| `count_A_col` | str | None | Conteo del grupo A. Si None, usar `prop_A_col`. |
| `count_B_col` | str | None | Conteo del grupo B. |
| `prop_A_col` | str | None | Proporción directa del grupo A (datos pre-agregados). |
| `prop_B_col` | str | None | Proporción directa del grupo B. |
| `group_A_label` | str | `'Grupo A'` | Etiqueta del grupo A en tablas y gráficos. |
| `group_B_label` | str | `'Grupo B'` | Etiqueta del grupo B en tablas y gráficos. |
| `year_filter` | int/list | None | Filtrar por uno o varios años. |
| `row_filter` | dict | None | Filtros adicionales. Ej: `{'Región': 'Lima'}`. |
| `exclude_strata` | list | None | Estratos a excluir del cálculo. |
| `rate_weighting` | str | `'auto'` | Agregación de tasas cuando hay varias filas por estrato: `'auto'` pondera por los conteos si existen, `'counts'` los exige, `'simple'` reproduce el promedio sin ponderar de la v2.0.0. |
| `random_state` | int | None | Semilla para el bootstrap, para resultados reproducibles. |

**Sobre `rate_weighting`.** En la descomposición de Kitagawa, `p_i` es la composición de la población o exposición del grupo en el estrato `i`, y `r_i` la tasa en ese estrato. Los conteos deben ser, por tanto, poblaciones o exposiciones, no eventos. Con esa lectura, la tasa del estrato es `Σ(r·n)/Σ(n)`, que es la que efectivamente se observaría al fusionar las filas.

---

## Fórmula de Kitagawa

**Un solo periodo, brecha entre dos grupos:**

```
D_total = C_T + C_E

C_T = Σ [ (r_A_i - r_B_i) × (p_A_i + p_B_i) / 2 ]   ← componente de tasa
C_E = Σ [ (p_A_i - p_B_i) × (r_A_i + r_B_i) / 2 ]   ← componente de estructura

D_real = Σ(r_A_i × p_A_i) - Σ(r_B_i × p_B_i)        ← brecha real ponderada
```

**Dos periodos, cambio de la brecha (v2.1.0):**

```
efecto_tasa       = Σ [ Δr_A_i × (p_A_i1 + p_A_i2)/2 − Δr_B_i × (p_B_i1 + p_B_i2)/2 ]
efecto_estructura = Σ [ Δp_A_i × (r_A_i1 + r_A_i2)/2 − Δp_B_i × (r_B_i1 + r_B_i2)/2 ]
```

Donde `r_A_i` y `r_B_i` son las tasas del estrato `i` para los grupos A y B, `p_A_i` y `p_B_i` las proporciones de cada grupo en ese estrato, y los subíndices 1 y 2 los dos periodos comparados.

---

## Dependencias

| Biblioteca | Versión mínima | Función |
|---|---|---|
| pandas | ≥ 1.3.0 | Manipulación de DataFrames y exportación Excel |
| numpy | ≥ 1.21.0 | Operaciones numéricas y bootstrap |
| statsmodels | ≥ 0.13.0 | Verificación estadística |
| matplotlib | ≥ 3.4.0 | Gráficos y mapa de burbujas |
| openpyxl | ≥ 3.0.0 | Exportación a .xlsx |
| geopandas | ≥ 0.10.0 | Opcional — mapas con polígonos |

---

## Cómo citar

```bibtex
@software{kitagawa_health_2026,
  author    = {Samillan Vasquez, Cesar Jefferson and
               León Montoya, Gladys Bernardita and
               Bazán Valque, Rosa Ysabel and
               Acosta Román, Mercedes},
  title     = {KitagawaHealth: descomposición de Kitagawa para brechas
               en indicadores de salud},
  year      = {2026},
  version   = {2.1.0},
  doi       = {10.5281/zenodo.20291586},
  url       = {https://doi.org/10.5281/zenodo.20291586}
}
```

El DOI anterior es el **DOI concepto**, que siempre resuelve a la última versión publicada. En un artículo o en cualquier contexto donde importe la reproducibilidad exacta, cite en cambio el **DOI de la versión** que aparece en la página del depósito correspondiente en Zenodo.

El archivo [`CITATION.cff`](CITATION.cff) permite a GitHub y a los gestores bibliográficos generar la cita automáticamente.

---

## Referencia

Kitagawa EM. (1955). Components of a difference between two rates.
*Journal of the American Statistical Association*, 50(272), 1168–1194.

Das Gupta P. (1993). *Standardization and decomposition of rates: a user's manual.*
U.S. Bureau of the Census, Current Population Reports, Series P23-186.

Efron B, Tibshirani RJ. (1993). *An introduction to the bootstrap.* Chapman & Hall.

---

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE.txt).
Copyright © 2026 Samillan Vasquez, León Montoya, Bazán Valque, Acosta Román.
