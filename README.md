# KitagawaHealth v2.1.0

> **Herramienta generalizada de descomposición de Kitagawa para el análisis de brechas en indicadores cuantitativos de salud entre dos grupos cualesquiera.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20291586.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## ¿Qué es KitagawaHealth?

**KitagawaHealth** es un módulo Python de código abierto que implementa la descomposición de Kitagawa (1955) de forma completamente generalizada para el análisis de brechas en indicadores de salud pública.

Separa el diferencial total entre dos grupos en:

- **Componente de tasa (C_T):** parte de la brecha que se debe a que una causa o evento es más letal o frecuente en un grupo que en el otro.
- **Componente de estructura (C_E):** parte que se debe a que un grupo concentra sus casos en causas distintas a las del otro grupo.

A diferencia de implementaciones existentes, KitagawaHealth:

- Acepta **cualquier par de grupos**: hombres vs mujeres, región A vs B, urbano vs rural, nivel educativo alto vs bajo, año base vs año actual, etc.
- Acepta **cualquier estratificador**: causas de muerte, diagnósticos, grupos de edad, ocupación, servicio de salud, etc.
- Incluye **mapa geográfico integrado** sin necesidad de internet ni geopandas.
- Detecta automáticamente **errores de configuración** antes de ejecutar.
- Funciona en **Google Colab** sin dependencias externas problemáticas.

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

## Comparación entre dos años

```python
# Descomponer el cambio en la brecha entre 2008 y 2022
cambio = kd.compare_years(year1=2008, year2=2022)
print(cambio.head(10))
kd.plot_compare_years(year1=2008, year2=2022)
```

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
├── kitagawa_health.py     ← Módulo principal
├── example_usage.py       ← Ejemplos con datos sintéticos
├── requirements.txt       ← Dependencias Python
├── LICENSE.txt            ← Licencia MIT
└── README.md              ← Este archivo
```

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

---

## Fórmula de Kitagawa

```
D_total = C_T + C_E

C_T = Σ [ (r_A_i - r_B_i) × (p_A_i + p_B_i) / 2 ]   ← componente de tasa
C_E = Σ [ (p_A_i - p_B_i) × (r_A_i + r_B_i) / 2 ]   ← componente de estructura

D_real = Σ(r_A_i × p_A_i) - Σ(r_B_i × p_B_i)        ← brecha real ponderada
```

Donde `r_A_i` y `r_B_i` son las tasas del estrato `i` para los grupos A y B,
y `p_A_i` y `p_B_i` son las proporciones de cada grupo en ese estrato.

---

## Dependencias

| Biblioteca | Versión mínima | Función |
|---|---|---|
| pandas | ≥ 1.3.0 | Manipulación de DataFrames y exportación Excel |
| numpy | ≥ 1.21.0 | Operaciones numéricas |
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
  title     = {KitagawaHealth: A generalized Kitagawa decomposition tool
               for quantitative health gap analysis},
  year      = {2026},
  version   = {2.0.0},
  doi       = {10.5281/zenodo.20291587},
  url       = {https://doi.org/10.5281/zenodo.20291587}
}
```

---

## Referencia

Kitagawa EM. (1955). Components of a difference between two rates.
*Journal of the American Statistical Association*, 50(272), 1168–1194.

---

## Licencia

Este proyecto está bajo la licencia [MIT](LICENSE.txt).  
Copyright © 2026 Samillan Vasquez, León Montoya, Bazán Valque, Acosta Román.
