# -*- coding: utf-8 -*-
"""
kitagawa_health.py
==================
KitagawaHealth v2.0.0

Herramienta generalizada de descomposición de Kitagawa para el análisis
de brechas en indicadores cuantitativos de salud entre dos grupos cualesquiera.

Permite comparar:
  - Hombres vs Mujeres (brecha de género)
  - Región A vs Región B (desigualdad territorial)
  - Año base vs año actual (evolución temporal)
  - Nivel educativo alto vs bajo (gradiente socioeconómico)
  - Cualquier par de grupos definidos por el investigador

Separa el diferencial total en:
  - Componente de TASA      : diferencias en qué tan letales/frecuentes
                              son las causas/estratos para cada grupo.
  - Componente de ESTRUCTURA: diferencias en la distribución de casos
                              entre causas/estratos para cada grupo.

Autores  : Cesar Jefferson Samillan Vasquez
           Mercedes Acosta Román
           Gladys Bernardita León Montoya
           Rosa Ysabel Bazán Valque
Versión  : 2.0.0
Licencia : MIT
DOI      : [To be assigned via Zenodo]

Referencias
-----------
Kitagawa EM. (1955). Components of a difference between two rates.
Journal of the American Statistical Association, 50(272), 1168–1194.
"""

import logging
import warnings
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# ── Logger profesional ────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s [%(levelname)s] KitagawaHealth: %(message)s',
    datefmt= '%H:%M:%S',
)
logger = logging.getLogger('kitagawa_health')


# ── Excepciones específicas ───────────────────────────────────────────────────
class DataError(Exception):
    """Error en la estructura o contenido de los datos de entrada."""
    pass

class FilterError(Exception):
    """Error al aplicar filtros — no quedan datos tras el filtrado."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 1 — CLASE GENÉRICA (núcleo reutilizable)
# ═══════════════════════════════════════════════════════════════════════════════

class KitagawaDecomposer:
    """
    Clase genérica de descomposición de Kitagawa para dos grupos cualesquiera.

    Acepta cualquier par de grupos (A y B) y cualquier columna de estratificación,
    lo que permite aplicar el método a brechas de género, territorio, nivel
    educativo, ocupación, cobertura de servicios, etc.

    Fórmula de Kitagawa:
        D_total   = C_T + C_E
        C_T = Σ [ (r_A_i - r_B_i) * (p_A_i + p_B_i) / 2 ]
        C_E = Σ [ (p_A_i - p_B_i) * (r_A_i + r_B_i) / 2 ]

    donde:
        r_A_i = tasa/indicador del grupo A en el estrato i
        r_B_i = tasa/indicador del grupo B en el estrato i
        p_A_i = proporción del grupo A en el estrato i
        p_B_i = proporción del grupo B en el estrato i

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con los datos del indicador de salud.
    stratum_col : str
        Columna de estratificación — el "denominador" de la descomposición.
        Ejemplos: 'Causas', 'Edad', 'Ocupación', 'Diagnóstico', 'Servicio'.
    rate_A_col : str
        Columna con la tasa/indicador del grupo A.
    rate_B_col : str
        Columna con la tasa/indicador del grupo B.
    count_A_col : str or None
        Columna con el conteo/frecuencia del grupo A. Si None y prop_A_col
        está disponible, se usan las proporciones directamente.
    count_B_col : str or None
        Columna con el conteo/frecuencia del grupo B.
    prop_A_col : str or None
        Columna con la proporción directa del grupo A (0-1 o 0-100).
        Si se provee junto con count_A_col, tiene prioridad.
    prop_B_col : str or None
        Columna con la proporción directa del grupo B.
    year_col : str
        Columna de tiempo/año (default: 'Año').
    group_A_label : str
        Etiqueta descriptiva del grupo A (default: 'Grupo A').
    group_B_label : str
        Etiqueta descriptiva del grupo B (default: 'Grupo B').
    year_filter : int or list or None
        Año(s) a analizar. Si None, usa todos.
    row_filter : dict or None
        Filtros adicionales como dict {columna: valor_o_lista}.
        Ejemplo: {'Región': 'Lima', 'Causa': ['Diabetes', 'EPOC']}
    exclude_strata : list or None
        Lista de estratos a excluir del análisis.

    Examples
    --------
    Brecha de género en mortalidad:

    >>> kd = KitagawaDecomposer(
    ...     data         = df,
    ...     stratum_col  = 'Causas',
    ...     rate_A_col   = 'Tasa ajustada hombres',
    ...     rate_B_col   = 'Tasa ajustada mujeres',
    ...     count_A_col  = 'N° hombres',
    ...     count_B_col  = 'N° mujeres',
    ...     group_A_label= 'Hombres',
    ...     group_B_label= 'Mujeres',
    ...     row_filter   = {'Región': 'Nacional'},
    ... )
    >>> kd.run()
    >>> print(kd.summary_table())

    Brecha entre dos regiones (Lima vs Huancavelica):

    >>> # Preparar datos con columnas de tasa y conteo por región
    >>> kd = KitagawaDecomposer(
    ...     data          = df_pivoted,
    ...     stratum_col   = 'Causas',
    ...     rate_A_col    = 'Tasa_Lima',
    ...     rate_B_col    = 'Tasa_Huancavelica',
    ...     count_A_col   = 'N_Lima',
    ...     count_B_col   = 'N_Huancavelica',
    ...     group_A_label = 'Lima',
    ...     group_B_label = 'Huancavelica',
    ... )
    >>> kd.run()

    Usando proporciones directas (cuando los datos ya vienen en %):

    >>> kd = KitagawaDecomposer(
    ...     data        = df,
    ...     stratum_col = 'Diagnóstico',
    ...     rate_A_col  = 'Tasa_Urbano',
    ...     rate_B_col  = 'Tasa_Rural',
    ...     prop_A_col  = 'Prop_Urbano',
    ...     prop_B_col  = 'Prop_Rural',
    ... )
    """

    def __init__(
        self,
        data          : pd.DataFrame,
        stratum_col   : str,
        rate_A_col    : str,
        rate_B_col    : str,
        count_A_col   : Optional[str]  = None,
        count_B_col   : Optional[str]  = None,
        prop_A_col    : Optional[str]  = None,
        prop_B_col    : Optional[str]  = None,
        year_col      : str            = 'Año',
        group_A_label : str            = 'Grupo A',
        group_B_label : str            = 'Grupo B',
        year_filter   : Optional[Union[int, List[int]]] = None,
        row_filter    : Optional[Dict]  = None,
        exclude_strata: Optional[List]  = None,
    ):
        self._validate_inputs(data, stratum_col, rate_A_col, rate_B_col,
                              count_A_col, count_B_col, prop_A_col, prop_B_col,
                              year_col)
        self.data           = data.copy()
        self.stratum_col    = stratum_col
        self.rate_A_col     = rate_A_col
        self.rate_B_col     = rate_B_col
        self.count_A_col    = count_A_col
        self.count_B_col    = count_B_col
        self.prop_A_col     = prop_A_col
        self.prop_B_col     = prop_B_col
        self.year_col       = year_col
        self.group_A_label  = group_A_label
        self.group_B_label  = group_B_label
        self.year_filter    = year_filter
        self.row_filter     = row_filter or {}
        self.exclude_strata = exclude_strata or []
        self.results_       = pd.DataFrame()
        self.summary_       = pd.DataFrame()
        logger.info(f"KitagawaDecomposer iniciado: '{group_A_label}' vs '{group_B_label}' "
                    f"| estrato: '{stratum_col}'")

    # ── Validación ────────────────────────────────────────────────────────────
    def _validate_inputs(self, data, stratum_col, rate_A_col, rate_B_col,
                         count_A_col, count_B_col, prop_A_col, prop_B_col, year_col):
        if not isinstance(data, pd.DataFrame):
            raise DataError("'data' debe ser un pandas DataFrame.")

        # Verificar columnas requeridas
        required = [stratum_col, rate_A_col, rate_B_col, year_col]
        for col in required:
            if col not in data.columns:
                raise DataError(
                    f"Columna requerida '{col}' no encontrada. "
                    f"Columnas disponibles: {list(data.columns)}"
                )

        # FIX 3: Verificar que las columnas opcionales que se pasan
        # realmente existen en el DataFrame (fallo inmediato y claro)
        optional_pairs = [
            (count_A_col, 'count_A_col'),
            (count_B_col, 'count_B_col'),
            (prop_A_col,  'prop_A_col'),
            (prop_B_col,  'prop_B_col'),
        ]
        for col_val, col_name in optional_pairs:
            if col_val is not None and col_val not in data.columns:
                raise DataError(
                    f"La columna '{col_val}' especificada en '{col_name}' "
                    f"no se encontró en el DataFrame. "
                    f"Columnas disponibles: {list(data.columns)}"
                )

        # Verificar que hay al menos conteos O proporciones completas
        has_counts = bool(count_A_col and count_B_col)
        has_props  = bool(prop_A_col  and prop_B_col)
        if not has_counts and not has_props:
            raise DataError(
                "Se requiere al menos uno de:\n"
                "  (a) count_A_col + count_B_col  — columnas de conteo/frecuencia\n"
                "  (b) prop_A_col  + prop_B_col   — columnas de proporción directa\n"
                "Nota: si usa prop_A_col/prop_B_col, los datos deben estar "
                "pre-agregados al nivel (year_col, stratum_col)."
            )

    # ── Filtrado ──────────────────────────────────────────────────────────────
    def _filter_data(self) -> pd.DataFrame:
        df = self.data.copy()

        # Filtros de fila (dict)
        for col, val in self.row_filter.items():
            if col not in df.columns:
                logger.warning(f"Columna de filtro '{col}' no encontrada — ignorada.")
                continue
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]

        # Filtro de año
        if self.year_filter is not None:
            years = [self.year_filter] if isinstance(self.year_filter, int) \
                    else self.year_filter
            df = df[df[self.year_col].isin(years)]

        # Excluir estratos
        if self.exclude_strata:
            df = df[~df[self.stratum_col].isin(self.exclude_strata)]

        if df.empty:
            raise FilterError(
                "No quedan datos después de aplicar los filtros. "
                f"Verifica row_filter={self.row_filter}, "
                f"year_filter={self.year_filter}, "
                f"exclude_strata={self.exclude_strata}"
            )
        logger.info(f"Datos filtrados: {len(df)} filas | "
                    f"{df[self.year_col].nunique()} años | "
                    f"{df[self.stratum_col].nunique()} estratos")
        return df

    # ── Cálculo de proporciones ───────────────────────────────────────────────
    def _compute_proportions(self, agg: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula proporciones p_A y p_B.

        Reglas:
        - Si se proveen proporciones directas (prop_A_col / prop_B_col):
            * Los datos DEBEN estar pre-agregados al nivel (year_col, stratum_col).
            * Se usa 'first' al agrupar, NO 'mean', para evitar el promedio
              incorrecto de proporciones cuando hay múltiples filas por estrato.
            * Se normaliza a 0-1 si los valores están en porcentaje (>1).
        - Si se proveen conteos: se calcula prop = count / total.
        """
        if self.prop_A_col and self.prop_A_col in agg.columns and \
           self.prop_B_col and self.prop_B_col in agg.columns:
            # FIX 2: usar las proporciones directamente (ya vienen de 'first'
            # en la agregación de _kitagawa_single)
            agg['prop_A'] = agg[self.prop_A_col]
            agg['prop_B'] = agg[self.prop_B_col]
            # Normalizar si están en porcentaje
            if agg['prop_A'].max() > 1.5:
                agg['prop_A'] = agg['prop_A'] / 100.0
                agg['prop_B'] = agg['prop_B'] / 100.0
            # Verificar que sumen ~1 (advertencia si no)
            sum_A = agg['prop_A'].sum()
            sum_B = agg['prop_B'].sum()
            if not (0.95 <= sum_A <= 1.05):
                logger.warning(
                    f"Las proporciones del grupo A suman {sum_A:.3f} (esperado ~1.0). "
                    "Verifica que los datos estén pre-agregados al nivel "
                    "(year_col, stratum_col) cuando usas prop_A_col/prop_B_col."
                )
            logger.debug("Usando proporciones directas (pre-agregadas).")
        else:
            total_A = agg['count_A'].sum()
            total_B = agg['count_B'].sum()
            if total_A == 0 or total_B == 0:
                raise DataError(
                    "El total de conteos es 0 para uno de los grupos. "
                    "Verifica los filtros y las columnas de conteo."
                )
            agg['prop_A'] = agg['count_A'] / total_A
            agg['prop_B'] = agg['count_B'] / total_B
        return agg

    # ── Núcleo de Kitagawa ────────────────────────────────────────────────────
    def _kitagawa_single(self, df_sub: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica la descomposición de Kitagawa sobre un subconjunto de datos
        (típicamente un año).

        Regla de agregación:
        - Tasas    → 'mean'  (promedio si hay múltiples filas por estrato)
        - Conteos  → 'sum'   (sumar si hay múltiples filas por estrato)
        - Proporciones directas → 'first' (deben venir pre-agregadas;
          promediar proporciones produce resultados incorrectos)

        Returns pd.DataFrame con columnas:
            Estrato, tasa_A, tasa_B, brecha_tasa_real,
            prop_A, prop_B, C_tasa, C_estructura, C_total
        """
        agg_dict = {
            'tasa_A': (self.rate_A_col, 'mean'),
            'tasa_B': (self.rate_B_col, 'mean'),
        }
        if self.count_A_col and self.count_A_col in df_sub.columns:
            agg_dict['count_A'] = (self.count_A_col, 'sum')
        if self.count_B_col and self.count_B_col in df_sub.columns:
            agg_dict['count_B'] = (self.count_B_col, 'sum')
        # FIX 2: proporciones directas → 'first', no 'mean'
        if self.prop_A_col and self.prop_A_col in df_sub.columns:
            agg_dict[self.prop_A_col] = (self.prop_A_col, 'first')
        if self.prop_B_col and self.prop_B_col in df_sub.columns:
            agg_dict[self.prop_B_col] = (self.prop_B_col, 'first')

        agg = df_sub.groupby(self.stratum_col).agg(**agg_dict).reset_index()
        agg = agg.rename(columns={self.stratum_col: 'Estrato'})
        agg = self._compute_proportions(agg)

        # ── Fórmula de Kitagawa ──────────────────────────────────────────────
        # C_T = (r_A - r_B) * (p_A + p_B) / 2
        agg['C_tasa']       = (agg['tasa_A'] - agg['tasa_B']) * \
                               (agg['prop_A'] + agg['prop_B']) / 2
        # C_E = (p_A - p_B) * (r_A + r_B) / 2
        agg['C_estructura'] = (agg['prop_A'] - agg['prop_B']) * \
                               (agg['tasa_A'] + agg['tasa_B']) / 2
        agg['C_total']      = agg['C_tasa'] + agg['C_estructura']

        # Brecha real ponderada (más precisa que la diferencia simple de tasas)
        # D_real = Σ(r_A * p_A) - Σ(r_B * p_B)
        agg['contrib_A'] = agg['tasa_A'] * agg['prop_A']
        agg['contrib_B'] = agg['tasa_B'] * agg['prop_B']
        agg['brecha_tasa_real'] = agg['contrib_A'] - agg['contrib_B']

        return agg[['Estrato', 'tasa_A', 'tasa_B', 'brecha_tasa_real',
                    'prop_A', 'prop_B', 'C_tasa', 'C_estructura', 'C_total']]

    # ── API Pública ───────────────────────────────────────────────────────────
    def run(self) -> 'KitagawaDecomposer':
        """
        Ejecuta la descomposición de Kitagawa para todos los años disponibles.

        Returns
        -------
        self (para encadenamiento)
        """
        df    = self._filter_data()
        years = sorted(df[self.year_col].unique())
        all_results = []

        for year in years:
            df_y = df[df[self.year_col] == year]
            try:
                res = self._kitagawa_single(df_y)
                if not res.empty:
                    res[self.year_col] = year
                    all_results.append(res)
            except (DataError, ZeroDivisionError) as e:
                logger.warning(f"Año {year} omitido: {e}")

        if not all_results:
            raise DataError("No se pudieron calcular resultados para ningún año. "
                            "Verifica los datos y los filtros.")

        self.results_ = pd.concat(all_results, ignore_index=True)

        # Resumen anual
        self.summary_ = self.results_.groupby(self.year_col).agg(
            D_total_real  = ('brecha_tasa_real', 'sum'),
            D_total_kita  = ('C_total',          'sum'),
            C_tasa_total  = ('C_tasa',           'sum'),
            C_estru_total = ('C_estructura',     'sum'),
        ).reset_index()

        self.summary_['pct_tasa']       = (
            self.summary_['C_tasa_total'] /
            self.summary_['D_total_kita'].replace(0, np.nan) * 100
        ).round(1)
        self.summary_['pct_estructura'] = (
            self.summary_['C_estru_total'] /
            self.summary_['D_total_kita'].replace(0, np.nan) * 100
        ).round(1)

        logger.info(f"Análisis completado: {len(years)} años | "
                    f"{self.results_['Estrato'].nunique()} estratos")
        return self

    def compare_years(
        self,
        year1: int,
        year2: int,
        base_year: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Descompone el cambio en la brecha entre dos años usando Kitagawa.

        Fija las proporciones del año base para aislar el efecto de cambio
        en tasas (efecto período) del cambio en estructura poblacional.

        Parameters
        ----------
        year1 : int
            Año inicial (referencia).
        year2 : int
            Año final (comparación).
        base_year : int or None
            Año cuyas proporciones se usan como base fija.
            Si None, usa year1.

        Returns
        -------
        pd.DataFrame con columnas:
            Estrato, brecha_y1, brecha_y2, delta_total,
            efecto_tasa, efecto_estructura
        """
        if self.results_.empty:
            raise RuntimeError("Ejecuta .run() primero.")

        base = base_year or year1
        years_available = self.results_[self.year_col].unique()
        for y in [year1, year2, base]:
            if y not in years_available:
                raise DataError(f"Año {y} no disponible. "
                                f"Años disponibles: {sorted(years_available)}")

        r1   = self.results_[self.results_[self.year_col] == year1].set_index('Estrato')
        r2   = self.results_[self.results_[self.year_col] == year2].set_index('Estrato')
        base_r = self.results_[self.results_[self.year_col] == base].set_index('Estrato')

        # Alinear índices
        common = r1.index.intersection(r2.index).intersection(base_r.index)
        r1, r2, base_r = r1.loc[common], r2.loc[common], base_r.loc[common]

        comp = pd.DataFrame(index=common)
        comp['brecha_y1']       = r1['C_total']
        comp['brecha_y2']       = r2['C_total']
        comp['delta_total']     = comp['brecha_y2'] - comp['brecha_y1']

        # Efecto tasa: cambio en tasas con estructura fija al año base
        comp['efecto_tasa']      = (
            (r2['tasa_A'] - r2['tasa_B']) -
            (r1['tasa_A'] - r1['tasa_B'])
        ) * (base_r['prop_A'] + base_r['prop_B']) / 2

        # Efecto estructura: cambio en estructura con tasas fijas al año base
        comp['efecto_estructura'] = (
            (r2['prop_A'] - r2['prop_B']) -
            (r1['prop_A'] - r1['prop_B'])
        ) * (base_r['tasa_A'] + base_r['tasa_B']) / 2

        result = comp.reset_index().rename(columns={'index': 'Estrato'})
        result = result.sort_values('delta_total', ascending=False)

        logger.info(f"compare_years: {year1} vs {year2} (base={base}) | "
                    f"Δ total={result['delta_total'].sum():.4f}")
        return result.round(4)

    def summary_table(
        self,
        top_n : int = 10,
        year  : Optional[int] = None,
        sort_by: str = 'C_total',
    ) -> pd.DataFrame:
        """
        Tabla resumen de los componentes de Kitagawa por estrato.

        Parameters
        ----------
        top_n : int
            Número de estratos a mostrar (default: 10).
        year : int or None
            Si se especifica, filtra por ese año. Si None, usa el promedio.
        sort_by : str
            Columna para ordenar: 'C_total', 'C_tasa', 'C_estructura',
            'brecha_tasa_real' (default: 'C_total').

        Returns
        -------
        pd.DataFrame
        """
        if self.results_.empty:
            raise RuntimeError("Ejecuta .run() primero.")

        if year is not None:
            df = self.results_[self.results_[self.year_col] == year].copy()
        else:
            df = self.results_.groupby('Estrato').agg(
                tasa_A          = ('tasa_A',           'mean'),
                tasa_B          = ('tasa_B',           'mean'),
                brecha_tasa_real= ('brecha_tasa_real', 'mean'),
                C_tasa          = ('C_tasa',           'mean'),
                C_estructura    = ('C_estructura',     'mean'),
                C_total         = ('C_total',          'mean'),
            ).reset_index()

        cols_rename = {
            'tasa_A'          : f'Tasa {self.group_A_label}',
            'tasa_B'          : f'Tasa {self.group_B_label}',
            'brecha_tasa_real': 'Brecha real ponderada',
        }
        df = df.rename(columns=cols_rename)

        valid_sort = [c for c in [sort_by, 'C_total'] if c in df.columns]
        df = df.sort_values(valid_sort[0], ascending=False).head(top_n)

        return df.round(4).reset_index(drop=True)

    def annual_summary(self) -> pd.DataFrame:
        """Retorna el resumen anual de componentes totales."""
        if self.summary_.empty:
            raise RuntimeError("Ejecuta .run() primero.")
        return self.summary_.round(4)

    def export_results(self, path: str = 'kitagawa_results.xlsx'):
        """Exporta resultados completos, resumen anual y top causas a Excel."""
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            self.results_.round(4).to_excel(
                writer, sheet_name='Detalle_por_Estrato', index=False)
            self.summary_.round(4).to_excel(
                writer, sheet_name='Resumen_Anual', index=False)
            self.summary_table(top_n=20).to_excel(
                writer, sheet_name='Top_Estratos', index=False)
        logger.info(f"Resultados exportados a: {path}")

    # ═══════════════════════════════════════════════════════════════════════
    #  SECCIÓN 2 — VISUALIZACIONES
    # ═══════════════════════════════════════════════════════════════════════

    def plot_decomposition(
        self,
        title     : str   = None,
        figsize   : tuple = (12, 5),
        style     : str   = 'grayscale',
        save_path : str   = None,
    ):
        """
        Barras apiladas de C_tasa y C_estructura por año, más porcentajes.

        Parameters
        ----------
        title : str or None
            Título del gráfico.
        figsize : tuple
            Tamaño de la figura.
        style : str
            Estilo de matplotlib (default: 'grayscale').
        save_path : str or None
            Ruta para guardar a 300 dpi.
        """
        if self.summary_.empty:
            raise RuntimeError("Ejecuta .run() primero.")

        plt.style.use(style)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        s = self.summary_.copy()
        x = s[self.year_col].values

        # Panel izq: barras apiladas
        ax1 = axes[0]
        ax1.bar(x, s['C_tasa_total'],  0.6,
                label='Componente de Tasa',      color='black',   alpha=0.85)
        ax1.bar(x, s['C_estru_total'], 0.6,
                bottom=s['C_tasa_total'],
                label='Componente de Estructura', color='gray',    alpha=0.75)
        ax1.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax1.set_xlabel('Año', fontsize=9)
        ax1.set_ylabel('Contribución a la brecha', fontsize=9)
        ax1.set_title('Componentes de Kitagawa por año',
                      fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2, axis='y')
        ax1.tick_params(axis='x', rotation=45)

        # Panel der: % contribución
        ax2 = axes[1]
        ax2.plot(x, s['pct_tasa'],       marker='o', color='black',
                 linewidth=1.8, label='% Componente Tasa')
        ax2.plot(x, s['pct_estructura'], marker='s', color='gray',
                 linewidth=1.8, linestyle='--', label='% Componente Estructura')
        ax2.axhline(50, color='lightgray', linestyle=':', linewidth=1)
        ax2.set_xlabel('Año', fontsize=9)
        ax2.set_ylabel('Contribución relativa (%)', fontsize=9)
        ax2.set_title('Contribución relativa de cada componente',
                      fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.tick_params(axis='x', rotation=45)

        default_title = (f'Descomposición de Kitagawa: '
                         f'{self.group_A_label} vs {self.group_B_label}')
        fig.suptitle(title or default_title, fontsize=12, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada: {save_path}")
        plt.show()

    def plot_strata_ranked(
        self,
        year      : Optional[int] = None,
        top_n     : int   = 15,
        title     : str   = None,
        figsize   : tuple = (13, 7),
        style     : str   = 'grayscale',
        save_path : str   = None,
    ):
        """
        Barras horizontales de estratos ordenados por C_total con desglose
        en C_tasa (negro) y C_estructura (gris).

        Parameters
        ----------
        year : int or None
            Año a graficar. Si None, usa el promedio.
        top_n : int
            Número de estratos a mostrar (default: 15).
        title : str or None
            Título del gráfico.
        figsize : tuple
            Tamaño de la figura.
        style : str
            Estilo de matplotlib.
        save_path : str or None
            Ruta para guardar a 300 dpi.
        """
        if self.results_.empty:
            raise RuntimeError("Ejecuta .run() primero.")

        if year is not None:
            df = self.results_[self.results_[self.year_col] == year].copy()
            year_label = str(year)
        else:
            df = self.results_.groupby('Estrato').agg(
                C_tasa      = ('C_tasa',       'mean'),
                C_estructura= ('C_estructura', 'mean'),
                C_total     = ('C_total',      'mean'),
            ).reset_index()
            year_label = 'Promedio'

        df = df.sort_values('C_total', ascending=True).tail(top_n)

        plt.style.use(style)
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(df))

        ax.barh(y, df['C_tasa'],      0.6, color='black', alpha=0.85,
                label='C. Tasa')
        ax.barh(y, df['C_estructura'], 0.6, left=df['C_tasa'],
                color='gray', alpha=0.75, label='C. Estructura')

        for i, (ct, ce) in enumerate(zip(df['C_tasa'], df['C_estructura'])):
            total = ct + ce
            ax.text(total + abs(total) * 0.01 + 0.05, i,
                    f'{total:.2f}', va='center', ha='left', fontsize=7.5)

        ax.axvline(0, color='black', linewidth=0.8)
        labels = [s[:55] + '…' if len(s) > 55 else s for s in df['Estrato']]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f'Contribución a la brecha '
                      f'{self.group_A_label}−{self.group_B_label}', fontsize=9)
        ax.set_title(
            title or (f'Top {top_n} estratos — Descomposición Kitagawa ({year_label})\n'
                      f'{self.group_A_label} vs {self.group_B_label}'),
            fontsize=11, fontweight='bold'
        )
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2, axis='x')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada: {save_path}")
        plt.show()

    def plot_temporal_evolution(
        self,
        top_n     : int   = 5,
        component : str   = 'C_total',
        title     : str   = None,
        figsize   : tuple = (12, 6),
        style     : str   = 'grayscale',
        save_path : str   = None,
    ):
        """
        Evolución temporal de C_total (o C_tasa / C_estructura) para los
        top_n estratos con mayor valor promedio.

        Parameters
        ----------
        top_n : int
            Número de estratos a graficar (default: 5).
        component : str
            Columna a graficar: 'C_total', 'C_tasa' o 'C_estructura'.
        title : str or None
            Título.
        figsize : tuple
            Tamaño.
        style : str
            Estilo matplotlib.
        save_path : str or None
            Ruta para guardar a 300 dpi.
        """
        if self.results_.empty:
            raise RuntimeError("Ejecuta .run() primero.")

        top_strata = (
            self.results_.groupby('Estrato')[component]
            .mean().sort_values(ascending=False)
            .head(top_n).index.tolist()
        )
        df = self.results_[self.results_['Estrato'].isin(top_strata)]

        plt.style.use(style)
        linestyles = ['-', '--', ':', '-.', (0, (3,1,1,1))]
        markers    = ['o', 's', '^', 'D', 'v']
        grays      = ['black', '#404040', '#606060', '#808080', '#A0A0A0']

        fig, ax = plt.subplots(figsize=figsize)
        for i, strat in enumerate(top_strata):
            sub   = df[df['Estrato'] == strat].sort_values(self.year_col)
            short = strat[:50] + '…' if len(strat) > 50 else strat
            ax.plot(sub[self.year_col], sub[component],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    color=grays[i % len(grays)],
                    linewidth=1.8, markersize=5, label=short)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Año', fontsize=9)
        ax.set_ylabel(f'{component}', fontsize=9)
        ax.set_title(
            title or (f'Evolución temporal — {component} — Top {top_n} estratos\n'
                      f'{self.group_A_label} vs {self.group_B_label}'),
            fontsize=11, fontweight='bold'
        )
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada: {save_path}")
        plt.show()

    def plot_compare_years(
        self,
        year1     : int,
        year2     : int,
        top_n     : int   = 15,
        title     : str   = None,
        figsize   : tuple = (13, 7),
        style     : str   = 'grayscale',
        save_path : str   = None,
    ):
        """
        Gráfico de barras horizontales del Δ de brecha entre year1 y year2,
        con desglose en efecto_tasa y efecto_estructura.

        Parameters
        ----------
        year1, year2 : int
            Años a comparar.
        top_n : int
            Número de estratos.
        title : str or None
            Título.
        figsize : tuple
            Tamaño.
        style : str
            Estilo matplotlib.
        save_path : str or None
            Ruta para guardar a 300 dpi.
        """
        df = self.compare_years(year1, year2)
        df = df.sort_values('delta_total', ascending=True).tail(top_n)

        plt.style.use(style)
        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(df))

        ax.barh(y, df['efecto_tasa'],      0.6,
                color='black', alpha=0.85, label='Efecto Tasa')
        ax.barh(y, df['efecto_estructura'], 0.6,
                left=df['efecto_tasa'],
                color='gray',  alpha=0.75, label='Efecto Estructura')

        for i, val in enumerate(df['delta_total']):
            ax.text(val + abs(val) * 0.01 + 0.05, i,
                    f'{val:+.2f}', va='center', ha='left', fontsize=7.5)

        ax.axvline(0, color='black', linewidth=0.8)
        labels = [s[:55] + '…' if len(s) > 55 else s for s in df['Estrato']]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(f'Δ brecha {year1}→{year2}', fontsize=9)
        ax.set_title(
            title or (f'Cambio en la brecha {self.group_A_label}−{self.group_B_label}\n'
                      f'{year1} → {year2}'),
            fontsize=11, fontweight='bold'
        )
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.2, axis='x')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada: {save_path}")
        plt.show()

    def plot_map(
        self,
        year              : Optional[int] = None,
        component         : str   = 'C_total',
        geojson_url       : str   = None,
        map_region_filter : Optional[List[str]] = None,
        title             : str   = None,
        cmap              : str   = 'RdYlGn_r',
        figsize           : tuple = (9, 11),
        save_path         : str   = None,
    ):
        """
        Mapa de burbujas del componente seleccionado por departamento del Perú.

        No requiere geopandas ni conexión a internet. Las coordenadas de los
        25 departamentos están incrustadas directamente en el código.

        Si se provee geojson_url, intenta usar geopandas para un mapa
        coroplético con polígonos. Si falla (sin internet, sin geopandas),
        cae automáticamente al mapa de burbujas.

        Parameters
        ----------
        year : int or None
            Año a mapear. Si None, usa el promedio de todos los años.
        component : str
            Métrica a visualizar: 'C_total', 'C_tasa' o 'C_estructura'.
        geojson_url : str or None
            URL de un GeoJSON externo para mapa coroplético (opcional).
            Si se omite o falla, se usa el mapa de burbujas automáticamente.
        map_region_filter : list or None
            Lista de regiones a incluir (independiente de los filtros
            usados en el análisis).
        title : str or None
            Título del mapa.
        cmap : str
            Colormap de matplotlib (default: 'RdYlGn_r').
        figsize : tuple
            Tamaño de la figura en pulgadas.
        save_path : str or None
            Ruta para guardar la figura a 300 dpi.
        """
        if self.results_.empty:
            raise RuntimeError("Ejecuta .run() primero.")

        # ── Coordenadas incrustadas (lon, lat) de 25 departamentos del Perú ──
        # Fuente: Instituto Geográfico Nacional del Perú (dominio público)
        PERU_COORDS = {
            'AMAZONAS'        : (-77.87, -4.87),
            'ANCASH'          : (-77.53, -9.52),
            'APURIMAC'        : (-73.08, -13.63),
            'AREQUIPA'        : (-72.45, -15.84),
            'AYACUCHO'        : (-74.22, -13.16),
            'CAJAMARCA'       : (-78.50, -6.83),
            'CALLAO'          : (-77.13, -12.06),
            'CUSCO'           : (-72.08, -13.53),
            'HUANCAVELICA'    : (-74.97, -12.79),
            'HUANUCO'         : (-76.23, -9.93),
            'HUÁNUCO'         : (-76.23, -9.93),  # alias con tilde
            'ICA'             : (-75.53, -14.08),
            'JUNIN'           : (-75.20, -11.16),
            'JUNÍN'           : (-75.20, -11.16),
            'LA LIBERTAD'     : (-78.15, -7.88),
            'LAMBAYEQUE'      : (-79.90, -6.70),
            'LIMA'            : (-76.77, -11.63),
            'LIMA METROPOLITANA': (-77.03, -12.05),
            'LIMA PROVINCIA'  : (-76.80, -11.80),
            'LORETO'          : (-74.42, -4.57),
            'MADRE DE DIOS'   : (-70.81, -11.77),
            'MOQUEGUA'        : (-70.93, -16.19),
            'NACIONAL'        : (-75.00, -9.50),  # centroide nacional
            'PASCO'           : (-75.50, -10.68),
            'PIURA'           : (-80.20, -5.20),
            'PUNO'            : (-70.01, -15.84),
            'SAN MARTIN'      : (-76.85, -6.96),
            'SAN MARTÍN'      : (-76.85, -6.96),
            'TACNA'           : (-70.25, -17.60),
            'TUMBES'          : (-80.45, -3.57),
            'UCAYALI'         : (-74.32, -9.53),
        }

        # ── Calcular métrica por región ───────────────────────────────────────
        df_raw = self.data.copy()
        if self.year_filter is not None:
            yrs = [self.year_filter] if isinstance(self.year_filter, int) \
                  else self.year_filter
            df_raw = df_raw[df_raw[self.year_col].isin(yrs)]
        if year is not None:
            df_raw = df_raw[df_raw[self.year_col] == year]
        if self.exclude_strata:
            df_raw = df_raw[~df_raw[self.stratum_col].isin(self.exclude_strata)]

        # Detectar columna de región automáticamente
        region_candidates = [c for c in df_raw.columns
                             if any(k in c.upper() for k in
                                    ['REGION', 'REGIÓN', 'DEPTO', 'DEPARTAMENTO'])]
        if not region_candidates:
            logger.warning("No se encontró columna de región para el mapa.")
            return
        region_col = region_candidates[0]

        if map_region_filter:
            df_raw = df_raw[df_raw[region_col].isin(map_region_filter)]

        region_vals = []
        for reg in df_raw[region_col].unique():
            sub = df_raw[df_raw[region_col] == reg]
            try:
                res = self._kitagawa_single(sub)
                region_vals.append({
                    'region'      : reg,
                    'C_total'     : float(res['C_total'].sum()),
                    'C_tasa'      : float(res['C_tasa'].sum()),
                    'C_estructura': float(res['C_estructura'].sum()),
                })
            except Exception:
                pass

        if not region_vals:
            logger.warning("No se pudieron calcular valores por región.")
            return

        df_map = pd.DataFrame(region_vals)

        # ── Intentar mapa coroplético con geopandas (si hay URL externa) ──────
        if geojson_url:
            try:
                import geopandas as gpd
                gdf = gpd.read_file(geojson_url)
                name_field = next(
                    (c for c in gdf.columns
                     if any(k in c.upper()
                            for k in ['NAME', 'NOMBRE', 'NOMB', 'DEP'])),
                    gdf.columns[0]
                )
                gdf['_key']    = gdf[name_field].str.upper().str.strip()
                df_map['_key'] = df_map['region'].str.upper().str.strip()
                gdf = gdf.merge(df_map[['_key', component]], on='_key', how='left')

                fig, ax = plt.subplots(1, 1, figsize=figsize)
                gdf[gdf[component].isna()].plot(
                    ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)
                if gdf[component].notna().any():
                    gdf[gdf[component].notna()].plot(
                        ax=ax, column=component, cmap=cmap,
                        edgecolor='white', linewidth=0.5, legend=True,
                        legend_kwds={
                            'label'      : component,
                            'orientation': 'horizontal', 'shrink': 0.5,
                        }
                    )
                year_label = str(year) if year else 'Promedio'
                ax.set_title(
                    title or (f'Kitagawa — {component} | '
                              f'{self.group_A_label} vs {self.group_B_label}'
                              f' ({year_label})'),
                    fontsize=12, fontweight='bold', pad=12
                )
                ax.set_axis_off()
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Mapa coroplético guardado: {save_path}")
                plt.show()
                return
            except Exception as e:
                logger.warning(
                    f"No se pudo usar geopandas/GeoJSON ({e}). "
                    "Usando mapa de burbujas integrado."
                )

        # ── Mapa de burbujas (sin geopandas, sin internet) ────────────────────
        df_map['_key'] = df_map['region'].str.upper().str.strip()
        df_map['lon']  = df_map['_key'].map(lambda k: PERU_COORDS.get(k, (np.nan, np.nan))[0])
        df_map['lat']  = df_map['_key'].map(lambda k: PERU_COORDS.get(k, (np.nan, np.nan))[1])
        df_map = df_map.dropna(subset=['lon', 'lat'])

        # Excluir 'Nacional' del mapa de burbujas (es un centroide artificial)
        df_map = df_map[df_map['_key'] != 'NACIONAL']

        if df_map.empty:
            logger.warning("Ninguna región pudo mapearse con las coordenadas disponibles.")
            return

        vals    = df_map[component].values
        vmin    = vals.min()
        vmax    = vals.max()
        # Normalizar tamaño de burbujas (mín 80, máx 800)
        if vmax != vmin:
            sizes = 80 + (vals - vmin) / (vmax - vmin) * 720
        else:
            sizes = np.full(len(vals), 300)

        plt.style.use('grayscale')
        fig, ax = plt.subplots(figsize=figsize)

        # Fondo simple del Perú (rectángulo aproximado)
        ax.set_xlim(-82, -68)
        ax.set_ylim(-19, -0.5)
        ax.set_facecolor('#F0F4F8')
        ax.set_aspect('equal')

        # Líneas de referencia sutiles
        for lon in range(-82, -68, 2):
            ax.axvline(lon, color='white', linewidth=0.3, alpha=0.5)
        for lat in range(-18, 0, 2):
            ax.axhline(lat, color='white', linewidth=0.3, alpha=0.5)

        # Burbujas coloreadas por valor
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        cmap_obj  = cm.get_cmap(cmap)
        norm      = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors_arr = [cmap_obj(norm(v)) for v in vals]

        sc = ax.scatter(
            df_map['lon'], df_map['lat'],
            s=sizes, c=vals,
            cmap=cmap, vmin=vmin, vmax=vmax,
            alpha=0.85, edgecolors='white', linewidths=0.8, zorder=3
        )

        # Etiquetas de región abreviadas
        for _, row in df_map.iterrows():
            short = row['region'][:10]
            val   = row[component]
            ax.annotate(
                f"{short}\n{val:+.1f}",
                xy=(row['lon'], row['lat']),
                xytext=(3, 3), textcoords='offset points',
                fontsize=6.5, color='#222222',
                zorder=4
            )

        # Barra de color
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02, aspect=20)
        cbar.set_label(component, fontsize=9)

        year_label = str(year) if year else 'Promedio'
        ax.set_title(
            title or (f'Kitagawa — {component}\n'
                      f'{self.group_A_label} vs {self.group_B_label} ({year_label})'),
            fontsize=11, fontweight='bold', pad=10
        )
        ax.set_xlabel('Longitud', fontsize=8)
        ax.set_ylabel('Latitud', fontsize=8)
        ax.tick_params(labelsize=7)

        # Nota informativa
        fig.text(0.5, 0.01,
                 'Mapa de burbujas — coordenadas incrustadas (IGN Perú) | '
                 'Tamaño proporcional al valor absoluto',
                 ha='center', fontsize=7, color='gray')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Mapa de burbujas guardado: {save_path}")
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — SUBCLASE ESPECIALIZADA: ANÁLISIS POR CICLO DE VIDA
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 3 — SUBCLASE ESPECIALIZADA: ANÁLISIS POR CICLO DE VIDA
# ═══════════════════════════════════════════════════════════════════════════════

class KitagawaLifecycle:
    """
    Análisis de Kitagawa estratificado por grupos del ciclo de vida.

    Para cada grupo etario, instancia un KitagawaDecomposer independiente
    que compara el grupo A (ej. hombres) vs el grupo B (ej. mujeres)
    usando las columnas de tasa y conteo específicas de cada grupo etario.

    Corrección crítica respecto a versiones anteriores:
    - En la versión anterior, rate_A_col == rate_B_col por grupo etario,
      lo que hacía que la brecha siempre fuera cero. INCORRECTO.
    - En esta versión, cada grupo etario tiene columnas DIFERENTES para
      el grupo A y el grupo B (ej. tasa_hombres_adultos vs tasa_mujeres_adultos).
    - Si la base de datos no tiene columnas de tasa separadas por sexo Y
      grupo etario, se usa KitagawaDecomposer directamente con filtros
      por grupo etario como row_filter.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame con datos de mortalidad.
    lifecycle_config : dict
        Diccionario que mapea nombre del grupo etario a parámetros de
        KitagawaDecomposer. Cada entrada debe tener:
            rate_A_col  : columna de tasa del grupo A en ese estrato etario
            rate_B_col  : columna de tasa del grupo B (DIFERENTE de rate_A_col)
            count_A_col : columna de conteo del grupo A (opcional)
            count_B_col : columna de conteo del grupo B (opcional)
        Si None, usa DEFAULT_CONFIG (base de mortalidad peruana con tasas
        por grupo etario y conteos totales — la comparación es de la tasa
        total del grupo etario entre causas, no H vs M dentro del grupo).
    group_A_label : str
        Etiqueta del grupo A (default: 'Grupo A').
    group_B_label : str
        Etiqueta del grupo B (default: 'Grupo B').
    year_col : str
        Columna de año (default: 'Año').
    stratum_col : str
        Columna de estratificación (default: 'Causas').
    row_filter : dict or None
        Filtros adicionales aplicados a todos los grupos.

    Notes
    -----
    Si tu base de datos tiene columnas separadas por sexo Y grupo etario
    (ej. 'Tasa hombres niños', 'Tasa mujeres niños'), defínelas en
    lifecycle_config:

        config = {
            'Niños': {
                'rate_A_col' : 'Tasa hombres niños',
                'rate_B_col' : 'Tasa mujeres niños',
                'count_A_col': 'N° hombres niños',
                'count_B_col': 'N° mujeres niños',
            },
            ...
        }

    Si tu base solo tiene tasas por grupo etario sin separación por sexo
    (como la base de mortalidad peruana incluida), KitagawaLifecycle mide
    la distribución de causas dentro de cada grupo etario — útil para
    comparar el perfil de causas entre grupos. Para comparar H vs M,
    usa KitagawaDecomposer directamente con row_filter.

    Examples
    --------
    Uso con la base peruana (tasas por grupo etario, sin separación por sexo):

    >>> lc = KitagawaLifecycle(df, row_filter={'Región': 'Nacional'})
    >>> lc.run()
    >>> lc.plot_lifecycle_comparison()

    Uso con base que tiene tasas separadas por sexo y grupo etario:

    >>> config = {
    ...     'Adultos': {
    ...         'rate_A_col' : 'Tasa hombres adultos',
    ...         'rate_B_col' : 'Tasa mujeres adultos',
    ...         'count_A_col': 'N hombres adultos',
    ...         'count_B_col': 'N mujeres adultos',
    ...     }
    ... }
    >>> lc = KitagawaLifecycle(df, lifecycle_config=config,
    ...                         group_A_label='Hombres', group_B_label='Mujeres')
    >>> lc.run()
    """

    # DEFAULT_CONFIG para la base peruana:
    # Usa la tasa del grupo etario como tasa_A y los conteos totales
    # del grupo. Mide la distribución de causas dentro del grupo etario.
    # NOTA: no compara H vs M dentro del grupo (la base no tiene esas columnas).
    DEFAULT_CONFIG = {
        'Niños': {
            'rate_A_col'  : 'Tasa ajustada niños',
            'rate_B_col'  : 'Tasa ajustada  total',   # total como referencia
            'count_A_col' : 'N° niños',
            'count_B_col' : 'N° TOTAL',
        },
        'Adolescentes': {
            'rate_A_col'  : 'Tasa ajustada adolescentes',
            'rate_B_col'  : 'Tasa ajustada  total',
            'count_A_col' : 'N° adolescente',
            'count_B_col' : 'N° TOTAL',
        },
        'Jóvenes': {
            'rate_A_col'  : 'Tasa ajustada joven',
            'rate_B_col'  : 'Tasa ajustada  total',
            'count_A_col' : 'N° joven',
            'count_B_col' : 'N° TOTAL',
        },
        'Adultos': {
            'rate_A_col'  : 'Tasa ajustada adultos',
            'rate_B_col'  : 'Tasa ajustada  total',
            'count_A_col' : 'N° adulto',
            'count_B_col' : 'N° TOTAL',
        },
        'Adultos Mayores': {
            'rate_A_col'  : 'Tasa ajustada adultos mayores',
            'rate_B_col'  : 'Tasa ajustada  total',
            'count_A_col' : 'N° adultos mayores',
            'count_B_col' : 'N° TOTAL',
        },
    }

    def __init__(
        self,
        data             : pd.DataFrame,
        lifecycle_config : Optional[Dict] = None,
        group_A_label    : str = 'Grupo Etario',
        group_B_label    : str = 'Total',
        year_col         : str = 'Año',
        stratum_col      : str = 'Causas',
        row_filter       : Optional[Dict] = None,
    ):
        self.data               = data.copy()
        self.lifecycle_config   = lifecycle_config or self.DEFAULT_CONFIG
        self.group_A_label      = group_A_label
        self.group_B_label      = group_B_label
        self.year_col           = year_col
        self.stratum_col        = stratum_col
        self.row_filter         = row_filter or {}
        self.analyzers_         = {}
        self.lifecycle_summary_ = {}

        # Advertir si algún grupo tiene rate_A_col == rate_B_col (brecha = 0)
        for gname, cfg in self.lifecycle_config.items():
            if cfg.get('rate_A_col') == cfg.get('rate_B_col'):
                logger.warning(
                    f"Grupo '{gname}': rate_A_col == rate_B_col "
                    f"('{cfg['rate_A_col']}'). La brecha de tasa siempre "
                    "será cero. Provee columnas DIFERENTES para los dos grupos."
                )

    def run(self) -> 'KitagawaLifecycle':
        """Ejecuta KitagawaDecomposer para cada grupo etario."""
        for group_name, cfg in self.lifecycle_config.items():
            rate_a = cfg.get('rate_A_col', '')
            rate_b = cfg.get('rate_B_col', '')

            # Verificar existencia de columnas
            missing = [c for c in [rate_a, rate_b]
                       if c and c not in self.data.columns]
            if missing:
                logger.warning(f"Grupo '{group_name}' omitido: "
                               f"columnas no encontradas: {missing}")
                continue

            kd = KitagawaDecomposer(
                data          = self.data,
                stratum_col   = self.stratum_col,
                rate_A_col    = rate_a,
                rate_B_col    = rate_b,
                count_A_col   = cfg.get('count_A_col'),
                count_B_col   = cfg.get('count_B_col'),
                prop_A_col    = cfg.get('prop_A_col'),
                prop_B_col    = cfg.get('prop_B_col'),
                year_col      = self.year_col,
                group_A_label = f"{group_name} ({self.group_A_label})",
                group_B_label = f"{group_name} ({self.group_B_label})",
                row_filter    = self.row_filter,
            )
            try:
                kd.run()
                self.analyzers_[group_name] = kd

                # Resumen: tasa ponderada por año para el grupo A
                summary = kd.results_.groupby(self.year_col).apply(
                    lambda x: (x['tasa_A'] * x['prop_A']).sum()
                ).reset_index(name='tasa_prom_A')
                summary['tasa_prom_B'] = kd.results_.groupby(
                    self.year_col
                ).apply(lambda x: (x['tasa_B'] * x['prop_B']).sum()).values
                self.lifecycle_summary_[group_name] = summary
                logger.info(f"Ciclo de vida — '{group_name}': OK")
            except Exception as e:
                logger.warning(f"Grupo '{group_name}' error: {e}")

        return self

    def plot_lifecycle_comparison(
        self,
        title     : str   = 'Tasas de Mortalidad por Ciclo de Vida',
        figsize   : tuple = (12, 6),
        style     : str   = 'grayscale',
        save_path : str   = None,
    ):
        """Líneas temporales de la tasa ponderada por grupo etario."""
        if not self.lifecycle_summary_:
            raise RuntimeError("Ejecuta .run() primero.")

        plt.style.use(style)
        linestyles = ['-', '--', ':', '-.', (0, (3,1,1,1))]
        markers    = ['o', 's', '^', 'D', 'v']

        fig, ax = plt.subplots(figsize=figsize)
        for i, (group, df_g) in enumerate(self.lifecycle_summary_.items()):
            col = 'tasa_prom_A' if 'tasa_prom_A' in df_g.columns else 'tasa_prom'
            ax.plot(df_g[self.year_col], df_g[col],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    linewidth=1.8, markersize=5, label=group)

        ax.set_xlabel('Año', fontsize=9)
        ax.set_ylabel('Tasa ponderada (por 100,000 hab.)', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_lifecycle_bars(
        self,
        year      : Optional[int] = None,
        title     : str   = 'Componentes de Kitagawa por Grupo Etario',
        figsize   : tuple = (10, 6),
        style     : str   = 'grayscale',
        save_path : str   = None,
    ):
        """
        Barras agrupadas de C_tasa y C_estructura por grupo etario.
        """
        if not self.analyzers_:
            raise RuntimeError("Ejecuta .run() primero.")

        records = []
        for group, kd in self.analyzers_.items():
            s = kd.annual_summary()
            if year is not None:
                row = s[s[self.year_col] == year]
            else:
                row = s.mean(numeric_only=True).to_frame().T
            if not row.empty:
                records.append({
                    'Grupo'       : group,
                    'C_tasa'      : float(row['C_tasa_total'].iloc[0]),
                    'C_estructura': float(row['C_estru_total'].iloc[0]),
                })

        df_bars = pd.DataFrame(records)
        order   = [g for g in self.DEFAULT_CONFIG if g in df_bars['Grupo'].values]
        df_bars = df_bars.set_index('Grupo').loc[order].reset_index()

        plt.style.use(style)
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(df_bars))
        w = 0.35

        ax.bar(x - w/2, df_bars['C_tasa'],       w,
               color='black', alpha=0.85, label='C. Tasa')
        ax.bar(x + w/2, df_bars['C_estructura'], w,
               color='gray',  alpha=0.75, label='C. Estructura')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_xticks(x)
        ax.set_xticklabels(df_bars['Grupo'], rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Componente de Kitagawa', fontsize=9)
        year_label = str(year) if year else 'Promedio'
        ax.set_title(f'{title} ({year_label})', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — ALIAS PARA COMPATIBILIDAD Y FUNCIÓN DE CONVENIENCIA
# ═══════════════════════════════════════════════════════════════════════════════

# Alias: KitagawaAnalyzer apunta a KitagawaDecomposer para compatibilidad
KitagawaAnalyzer = KitagawaDecomposer


def analyze_health_gap(
    data          : pd.DataFrame,
    stratum_col   : str,
    rate_A_col    : str,
    rate_B_col    : str,
    count_A_col   : Optional[str]  = None,
    count_B_col   : Optional[str]  = None,
    prop_A_col    : Optional[str]  = None,
    prop_B_col    : Optional[str]  = None,
    group_A_label : str  = 'Grupo A',
    group_B_label : str  = 'Grupo B',
    row_filter    : Optional[Dict] = None,
    year_filter   : Optional[Union[int, List[int]]] = None,
    top_n         : int  = 15,
    plot          : bool = True,
    export_path   : Optional[str] = None,
    **kwargs
) -> KitagawaDecomposer:
    """
    Función de conveniencia de una línea para KitagawaHealth.

    Ejecuta toda la pipeline: instancia, run(), imprime resumen y
    opcionalmente grafica y exporta.

    Parameters
    ----------
    data : pd.DataFrame
        Datos de entrada.
    stratum_col : str
        Columna de estratificación.
    rate_A_col, rate_B_col : str
        Columnas de tasas de los grupos A y B.
    count_A_col, count_B_col : str or None
        Columnas de conteos.
    prop_A_col, prop_B_col : str or None
        Columnas de proporciones directas.
    group_A_label, group_B_label : str
        Etiquetas de los grupos.
    row_filter : dict or None
        Filtros de fila.
    year_filter : int or list or None
        Filtro de años.
    top_n : int
        Número de estratos en el ranking (default: 15).
    plot : bool
        Mostrar gráficos (default: True).
    export_path : str or None
        Ruta para exportar resultados a Excel.
    **kwargs
        Argumentos adicionales para KitagawaDecomposer.

    Returns
    -------
    KitagawaDecomposer
        Objeto con todos los resultados accesibles.

    Examples
    --------
    >>> kd = analyze_health_gap(
    ...     data          = df,
    ...     stratum_col   = 'Causas',
    ...     rate_A_col    = 'Tasa ajustada hombres',
    ...     rate_B_col    = 'Tasa ajustada mujeres',
    ...     count_A_col   = 'N° hombres',
    ...     count_B_col   = 'N° mujeres',
    ...     group_A_label = 'Hombres',
    ...     group_B_label = 'Mujeres',
    ...     row_filter    = {'Región': 'Nacional'},
    ...     top_n         = 10,
    ... )
    """
    kd = KitagawaDecomposer(
        data          = data,
        stratum_col   = stratum_col,
        rate_A_col    = rate_A_col,
        rate_B_col    = rate_B_col,
        count_A_col   = count_A_col,
        count_B_col   = count_B_col,
        prop_A_col    = prop_A_col,
        prop_B_col    = prop_B_col,
        group_A_label = group_A_label,
        group_B_label = group_B_label,
        row_filter    = row_filter,
        year_filter   = year_filter,
        **kwargs
    )
    kd.run()

    print("=" * 65)
    print(f"KITAGAWA-HEALTH — {group_A_label} vs {group_B_label}")
    print("=" * 65)
    print("\n── Resumen anual ──")
    print(kd.annual_summary().to_string(index=False))
    print(f"\n── Top {top_n} estratos (promedio) ──")
    print(kd.summary_table(top_n=top_n).to_string(index=False))

    if plot:
        kd.plot_decomposition()
        kd.plot_strata_ranked(top_n=top_n)
        kd.plot_temporal_evolution()

    if export_path:
        kd.export_results(export_path)

    return kd
