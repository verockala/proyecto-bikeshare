# =============================================================
#  PROYECTO FINAL DE PROBABILIDAD
#  Universidad de Carabobo — FACYT
#  Departamento de Computación — Asignatura: Probabilidad
#
#  Título:   Modelos Lineales Generalizados:
#            Regresión de Poisson en Datos de Bikeshare
#
#  Autoras:   Veronicka Herrera, Alexa Perdomo, Victoria Alvarado
#  Profesor: José Marcano
#  Fecha:    Marzo 2026
#
#  Descripción:
#  Este script implementa y compara dos modelos de regresión
#  para predecir la demanda horaria de bicicletas en Washington
#  D.C. (2011-2012):
#    1. Regresión Lineal Múltiple (modelo base)
#    2. Regresión de Poisson (GLM) — modelo propuesto
#
#  El objetivo es demostrar que la Regresión de Poisson es
#  superior para datos de conteo discreto, conectando los
#  conceptos de la distribución de Poisson vistos en clase
#  con su aplicación en Machine Learning.
#
#  Repositorio: https://github.com/verockala/proyecto-bikeshare
# =============================================================


# =============================================================
#  SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS
# =============================================================
# numpy  → operaciones numéricas y vectoriales
# pandas → manejo de dataframes (tablas de datos)
# matplotlib / seaborn → visualizaciones y gráficos
# statsmodels → ajuste de modelos estadísticos (OLS y GLM)
# sklearn → métricas de error (RMSE, MAE)
# scipy → distribución de Poisson teórica para comparación
# warnings → suprimir advertencias no críticas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Configuración global de gráficos ──────────────────────────
# Paleta de colores consistente en todas las figuras
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

plt.rcParams.update({
    "figure.dpi": 150,           # resolución de figuras
    "axes.spines.top": False,    # eliminar borde superior
    "axes.spines.right": False,  # eliminar borde derecho
    "font.family": "DejaVu Sans",
})


# =============================================================
#  SECCIÓN 1: CARGA DE DATOS
# =============================================================
# El dataset Bikeshare proviene del libro ISLR (James et al.,
# 2021) y contiene 8,645 observaciones horarias de alquiler
# de bicicletas en Washington D.C. durante 2011 y 2012.
#
# Variables principales:
#   bikers     → variable respuesta: conteo total de alquileres
#   mnth       → mes del año (categórica, 1-12)
#   hr         → hora del día (categórica, 0-23)
#   weathersit → situación climática (categórica)
#   temp       → temperatura normalizada (continua, 0-1)
#   workingday → 1 si es día laboral, 0 si es feriado/fin de semana

print("=" * 60)
print("  SECCIÓN 1: CARGA DE DATOS")
print("=" * 60)

try:
    # Intento 1: cargar desde ISLP (librería oficial del libro ISLR)
    from ISLP import load_data
    df = load_data('Bikeshare')
    print("✓ Dataset cargado desde ISLP (librería oficial ISLR).")
except ImportError:
    # Intento 2: si ISLP no está instalado, usar UCI ML Repository
    print("ISLP no disponible. Cargando desde UCI ML Repository...")
    from ucimlrepo import fetch_ucirepo
    bike_repo = fetch_ucirepo(id=275)
    df = pd.concat([bike_repo.data.features, bike_repo.data.targets], axis=1)
    df.rename(columns={'cnt': 'bikers'}, inplace=True)
    print("✓ Dataset cargado desde UCI ML Repository.")

# Información general del dataset
print(f"\nDimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Columnas disponibles: {list(df.columns)}")
print("\nPrimeras 5 filas del dataset:")
print(df.head())
print("\nEstadísticas descriptivas de la variable respuesta 'bikers':")
print(df['bikers'].describe())

# Convertir variables categóricas al tipo 'category' de pandas
# Esto es importante para que statsmodels las trate como factores
# y genere automáticamente las variables dummy en los modelos
cat_cols = ['mnth', 'hr', 'weathersit']
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')


# =============================================================
#  SECCIÓN 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================
# El EDA tiene como objetivo:
#   a) Evaluar si 'bikers' se asemeja a una distribución Poisson
#   b) Identificar patrones por hora del día y clima
#   c) Detectar diferencias entre días laborales y feriados
#
# Concepto clave: en la distribución de Poisson, la media y la
# varianza son iguales (E[Y] = Var[Y] = λ). Si la varianza
# observada supera ampliamente a la media, hay "sobre-dispersión",
# lo cual es común cuando la muestra mezcla distintos regímenes.

print("\n" + "=" * 60)
print("  SECCIÓN 2: ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 60)

# Crear figura con 4 subgráficos
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis Exploratorio — Bikeshare Washington D.C.",
             fontsize=14, fontweight='bold')

# ── Subgráfico 1: Histograma + Curva Poisson teórica ──────────
# Superponemos la distribución de Poisson teórica con λ = media
# observada para evaluar el ajuste marginal (sin covariables)
ax = axes[0, 0]
lam = df['bikers'].mean()           # estimador de λ = media muestral
x_range = np.arange(0, df['bikers'].max() + 1)
poisson_pmf = stats.poisson.pmf(x_range, lam)  # P(Y=y | λ)

ax.hist(df['bikers'], bins=60, density=True, color=PALETTE[0],
        alpha=0.7, label='Datos observados', edgecolor='white', linewidth=0.3)
ax.plot(x_range, poisson_pmf, color=PALETTE[1], linewidth=2,
        label=f'Poisson(λ={lam:.1f}) teórica')
ax.set_title("Distribución de Bicicletas Alquiladas")
ax.set_xlabel("bikers (conteo por hora)")
ax.set_ylabel("Densidad / Probabilidad")
ax.legend(fontsize=9)

# Calcular índice de dispersión: Var/Media
# Si ≈ 1 → Poisson pura; si >> 1 → sobre-dispersión
var_b  = df['bikers'].var()
mean_b = df['bikers'].mean()
print(f"\nMedia de bikers:    {mean_b:.2f}")
print(f"Varianza de bikers: {var_b:.2f}")
print(f"Índice de dispersión (Var/Media): {var_b/mean_b:.2f}")
print("→ Valor >> 1 indica sobre-dispersión marginal.")
print("  Esto es normal al mezclar horas pico y madrugada.")

# ── Subgráfico 2: Promedio de alquileres por hora del día ─────
# Permite identificar los picos de demanda (commute laboral)
ax = axes[0, 1]
avg_hr = df.groupby(df['hr'].astype(str))['bikers'].mean()

# Ordenar horas numéricamente (0 a 23)
try:
    avg_hr.index = avg_hr.index.astype(int)
    avg_hr = avg_hr.sort_index()
except:
    pass

ax.bar(range(len(avg_hr)), avg_hr.values, color=PALETTE[2],
       alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(avg_hr)))
ax.set_xticklabels(range(len(avg_hr)), fontsize=7)
ax.set_title("Promedio de Alquileres por Hora del Día")
ax.set_xlabel("Hora (0 – 23)")
ax.set_ylabel("Promedio de bikers")

peak_h = avg_hr.idxmax()
print(f"\nHora pico de mayor demanda: {peak_h}h "
      f"({avg_hr.max():.1f} bicicletas promedio)")

# ── Subgráfico 3: Promedio por condición climática ────────────
# Permite cuantificar el impacto del clima en la tasa λ esperada
ax = axes[1, 0]
avg_weather = df.groupby('weathersit')['bikers'].mean()

# Mapeo de etiquetas (compatible con ISLP que usa strings)
weather_labels_int = {1: "Despejado", 2: "Nublado",
                      3: "Lluvia ligera", 4: "Mal tiempo"}
weather_labels_str = {
    'clear': "Despejado", 'cloudy': "Nublado",
    'misty': "Nublado", 'light rain': "Lluvia ligera",
    'heavy rain': "Mal tiempo", 'light rainsnow': "Lluvia ligera"
}

def get_weather_label(k):
    """Convierte claves numéricas o de texto a etiquetas legibles."""
    try:
        return weather_labels_int.get(int(k), str(k))
    except (ValueError, TypeError):
        return weather_labels_str.get(str(k).lower().strip(), str(k))

keys = [get_weather_label(k) for k in avg_weather.index]
bars = ax.bar(keys, avg_weather.values, color=PALETTE[:len(keys)],
              alpha=0.85, edgecolor='white')
ax.set_title("Promedio de Alquileres por Clima")
ax.set_xlabel("Condición climática")
ax.set_ylabel("Promedio de bikers")
ax.tick_params(axis='x', labelsize=8)

# Etiquetas de valor sobre cada barra
for bar, val in zip(bars, avg_weather.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)

# ── Subgráfico 4: Boxplot día laboral vs fin de semana ────────
# Los boxplots muestran la distribución completa (mediana, IQR,
# valores atípicos) para comparar ambos tipos de día
ax = axes[1, 1]
groups = [df.loc[df['workingday'] == 0, 'bikers'].values,
          df.loc[df['workingday'] == 1, 'bikers'].values]
bp = ax.boxplot(groups, patch_artist=True,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], [PALETTE[3], PALETTE[4]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticklabels(['Fin de semana / Feriado', 'Día laboral'])
ax.set_title("Distribución: Laboral vs No Laboral")
ax.set_ylabel("bikers")

# Guardar figura (sin plt.show() para no pausar la ejecución)
plt.tight_layout()
plt.savefig("eda_bikeshare.png", bbox_inches='tight')
plt.close()
print("\n[✓ Guardado] eda_bikeshare.png")


# =============================================================
#  SECCIÓN 3: REGRESIÓN LINEAL MÚLTIPLE (MODELO BASE)
# =============================================================
# Ajustamos primero un modelo de regresión lineal clásico para
# establecer una línea base de comparación y evidenciar sus
# limitaciones con datos de conteo:
#
#   bikers_i = β0 + β1*mnth + β2*hr + β3*weathersit
#              + β4*temp + β5*workingday + ε_i
#
# Problema conceptual: la regresión lineal asume que Y puede
# tomar cualquier valor real (incluyendo negativos), lo cual
# es imposible para una variable de conteo como 'bikers'.
#
# Usamos smf.ols() con notación de fórmula tipo R.
# C(var) indica que la variable es categórica → genera dummies
# tomando la primera categoría como referencia automáticamente.

print("\n" + "=" * 60)
print("  SECCIÓN 3: REGRESIÓN LINEAL MÚLTIPLE (Modelo Base)")
print("=" * 60)

formula_lm = "bikers ~ C(mnth) + C(hr) + C(weathersit) + temp + workingday"
lm_model = smf.ols(formula_lm, data=df).fit()

# Agregar predicciones al dataframe
df['pred_lm'] = lm_model.fittedvalues

# Conteo de predicciones negativas (falla principal del modelo)
neg_preds = (df['pred_lm'] < 0).sum()

print(f"\nR² del modelo lineal:        {lm_model.rsquared:.4f}")
print(f"RMSE del modelo lineal:      {np.sqrt(mean_squared_error(df['bikers'], df['pred_lm'])):.2f}")
print(f"MAE del modelo lineal:       {mean_absolute_error(df['bikers'], df['pred_lm']):.2f}")
print(f"\nPredicciones NEGATIVAS:      {neg_preds}")
print("→ Falla conceptual: bikers no puede ser negativo.")

# Mostrar ejemplos de predicciones negativas en madrugada
madrugada_mask = (df['hr'].astype(str).isin(['0','1','2','3','4','5']))
neg_mad = df.loc[madrugada_mask & (df['pred_lm'] < 0),
                 ['hr', 'bikers', 'pred_lm']]
if len(neg_mad) > 0:
    print("\nEjemplos de predicciones negativas en la madrugada:")
    print(neg_mad.head(8).to_string(index=False))

# ── Gráficos de limitaciones del modelo lineal ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Regresión Lineal Múltiple — Limitaciones",
             fontsize=13, fontweight='bold')

# Gráfico izquierdo: observado vs predicho por hora
# Si el modelo fuera perfecto, ambas curvas se superpondrían
ax = axes[0]
avg_obs = df.groupby(df['hr'].astype(str))['bikers'].mean()
avg_lm  = df.groupby(df['hr'].astype(str))['pred_lm'].mean()
try:
    avg_obs.index = avg_obs.index.astype(int); avg_obs = avg_obs.sort_index()
    avg_lm.index  = avg_lm.index.astype(int);  avg_lm  = avg_lm.sort_index()
except: pass

ax.plot(range(len(avg_obs)), avg_obs.values, 'o-',
        label='Observado', color=PALETTE[0])
ax.plot(range(len(avg_lm)),  avg_lm.values,  's--',
        label='Predicho (LM)', color=PALETTE[1])
ax.axhline(0, color='red', linewidth=1.5, linestyle=':',
           label='y = 0 (límite físico)')
ax.set_xticks(range(len(avg_obs)))
ax.set_xticklabels(range(len(avg_obs)), fontsize=7)
ax.set_xlabel("Hora del día")
ax.set_ylabel("Promedio de bikers")
ax.set_title("Observado vs Predicho por Hora")
ax.legend(fontsize=9)

# Gráfico derecho: residuos vs valores ajustados
# Un buen modelo debe mostrar residuos aleatorios sin patrón.
# La forma de "embudo" indica heterocedasticidad (varianza no constante),
# violando los supuestos de la regresión lineal clásica.
ax = axes[1]
ax.scatter(df['pred_lm'], lm_model.resid, alpha=0.2, s=5, color=PALETTE[2])
ax.axhline(0, color='red', linewidth=1.5, linestyle='--')
ax.set_xlabel("Valores ajustados")
ax.set_ylabel("Residuos")
ax.set_title("Residuos vs Ajustados — Heterocedasticidad")

plt.tight_layout()
plt.savefig("lm_limitaciones.png", bbox_inches='tight')
plt.close()
print("\n[✓ Guardado] lm_limitaciones.png")


# =============================================================
#  SECCIÓN 4: REGRESIÓN DE POISSON (GLM)
# =============================================================
# La Regresión de Poisson es un Modelo Lineal Generalizado (GLM)
# que asume Y_i ~ Poisson(λ_i) con función de enlace logarítmica:
#
#   log(λ_i) = β0 + β1*x_i1 + ... + βp*xip
#   λ_i = exp(X_i^T · β)   → siempre positivo
#
# Ventajas sobre la regresión lineal para datos de conteo:
#   ✓ Predicciones siempre ≥ 0 (garantía matemática)
#   ✓ Distribución correcta para datos discretos no negativos
#   ✓ La varianza crece con la media (realista para conteos)
#   ✓ Interpretación multiplicativa de coeficientes
#
# statsmodels ajusta el modelo por Máxima Verosimilitud usando
# el algoritmo IRLS (Iteratively Reweighted Least Squares).
# La log-verosimilitud a maximizar es:
#   ℓ(β) = Σ [ -exp(X_i^T β) + y_i(X_i^T β) - ln(y_i!) ]

print("\n" + "=" * 60)
print("  SECCIÓN 4: REGRESIÓN DE POISSON (GLM)")
print("=" * 60)

formula_glm = "bikers ~ C(mnth) + C(hr) + C(weathersit) + temp + workingday"

# sm.families.Poisson() especifica la distribución de la familia
# exponencial. El enlace por defecto es logarítmico (log link).
poisson_model = smf.glm(
    formula_glm,
    data=df,
    family=sm.families.Poisson()
).fit()

# Agregar predicciones (λ_i estimado para cada observación)
df['pred_glm'] = poisson_model.fittedvalues

# Verificar que no hay predicciones negativas (debe ser 0 siempre)
neg_preds_glm = (df['pred_glm'] < 0).sum()

# Calcular métricas de error para ambos modelos
rmse_lm  = np.sqrt(mean_squared_error(df['bikers'], df['pred_lm']))
rmse_glm = np.sqrt(mean_squared_error(df['bikers'], df['pred_glm']))
mae_lm   = mean_absolute_error(df['bikers'], df['pred_lm'])
mae_glm  = mean_absolute_error(df['bikers'], df['pred_glm'])

print(f"\n{'Métrica':<30} {'Lineal':>10} {'Poisson':>10}")
print("-" * 52)
print(f"{'RMSE':<30} {rmse_lm:>10.2f} {rmse_glm:>10.2f}")
print(f"{'MAE':<30} {mae_lm:>10.2f} {mae_glm:>10.2f}")
print(f"{'Predicciones negativas':<30} {neg_preds:>10} {neg_preds_glm:>10}")
print(f"\nLog-Verosimilitud (Poisson): {poisson_model.llf:.2f}")
print(f"AIC (Poisson):               {poisson_model.aic:.2f}")
print(f"Devianza nula:               {poisson_model.null_deviance:.2f}")
print(f"Devianza residual:           {poisson_model.deviance:.2f}")
print(f"Reducción de devianza:       "
      f"{(1 - poisson_model.deviance/poisson_model.null_deviance)*100:.1f}%")


# =============================================================
#  SECCIÓN 5: INTERPRETACIÓN DE COEFICIENTES
# =============================================================
# En la Regresión de Poisson, los coeficientes β se interpretan
# en escala MULTIPLICATIVA sobre λ gracias al enlace log:
#
#   log(λ) = β0 + β1*x1 + ...
#   → Si x_j aumenta en 1 unidad:
#     λ_nuevo = exp(β_j) · λ_actual
#
# exp(β_j) > 1 → efecto positivo (más alquileres)
# exp(β_j) < 1 → efecto negativo (menos alquileres)
# exp(β_j) = 1 → sin efecto (β_j = 0)

print("\n" + "=" * 60)
print("  SECCIÓN 5: INTERPRETACIÓN DE COEFICIENTES")
print("=" * 60)

coefs = poisson_model.params

# ── Coeficiente de temperatura ────────────────────────────────
beta_temp = coefs['temp']
exp_temp  = np.exp(beta_temp)
print(f"\n[TEMPERATURA]")
print(f"  β = {beta_temp:.4f}")
print(f"  exp(β) = {exp_temp:.4f}")
print(f"  Interpretación: Pasar de temp=0 a temp=1 (rango completo)")
print(f"  multiplica la tasa esperada por {exp_temp:.3f}.")
print(f"  Es decir, la demanda aumenta un {(exp_temp-1)*100:.1f}% "
      f"al máximo de temperatura.")

# ── Coeficiente de día laboral ────────────────────────────────
beta_wd = coefs['workingday']
exp_wd  = np.exp(beta_wd)
direction = "más" if exp_wd > 1 else "menos"
print(f"\n[DÍA LABORAL]")
print(f"  β = {beta_wd:.4f}")
print(f"  exp(β) = {exp_wd:.4f}")
print(f"  Interpretación: En días laborales se esperan "
      f"{abs(exp_wd-1)*100:.1f}% {direction}")
print(f"  alquileres vs fines de semana (otras vars constantes).")

# ── Top 5 horas con mayor coeficiente ────────────────────────
# Las dummies de hr se comparan contra hr=0 (madrugada) como referencia
hr_coefs = {k: v for k, v in coefs.items() if 'C(hr)' in k}
hr_sorted = sorted(hr_coefs.items(), key=lambda x: x[1], reverse=True)
print(f"\n[TOP 5 HORAS — mayor efecto multiplicativo vs hora 0]")
for k, v in hr_sorted[:5]:
    hora = k.replace('C(hr)[T.', '').replace(']', '')
    print(f"  Hora {hora:>2}h → β={v:.4f}, exp(β)={np.exp(v):.2f}x "
          f"más que en hora 0")

# ── Top 3 horas con menor coeficiente ────────────────────────
print(f"\n[TOP 3 HORAS — menor efecto multiplicativo vs hora 0]")
for k, v in hr_sorted[-3:]:
    hora = k.replace('C(hr)[T.', '').replace(']', '')
    print(f"  Hora {hora:>2}h → β={v:.4f}, exp(β)={np.exp(v):.3f}x")


# =============================================================
#  SECCIÓN 6: GRÁFICOS COMPARATIVOS
# =============================================================
# Comparamos visualmente ambos modelos para evidenciar la
# superioridad de la Regresión de Poisson en datos de conteo

print("\n" + "=" * 60)
print("  SECCIÓN 6: GRÁFICOS COMPARATIVOS")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparación: Regresión Lineal vs Regresión de Poisson",
             fontsize=13, fontweight='bold')

# Calcular promedios por hora para cada modelo
avg_glm = df.groupby(df['hr'].astype(str))['pred_glm'].mean()
try:
    avg_glm.index = avg_glm.index.astype(int)
    avg_glm = avg_glm.sort_index()
except: pass

# ── Gráfico izquierdo: predicción promedio por hora ───────────
# Muestra qué tan bien captura cada modelo el patrón temporal
ax = axes[0]
ax.plot(range(len(avg_obs)), avg_obs.values, 'o-',
        label='Observado', color=PALETTE[0], linewidth=2)
ax.plot(range(len(avg_lm)),  avg_lm.values,  's--',
        label='Lineal', color=PALETTE[1], linewidth=1.5, alpha=0.8)
ax.plot(range(len(avg_glm)), avg_glm.values, '^-',
        label='Poisson', color=PALETTE[2], linewidth=2)
ax.axhline(0, color='red', linewidth=1, linestyle=':')
ax.set_xticks(range(len(avg_obs)))
ax.set_xticklabels(range(len(avg_obs)), fontsize=7)
ax.set_xlabel("Hora del día")
ax.set_ylabel("Promedio de bikers")
ax.set_title("Predicción promedio por hora")
ax.legend(fontsize=9)

# ── Gráfico derecho: métricas de error ───────────────────────
# Barras comparativas de RMSE y MAE — valores menores son mejores
ax = axes[1]
met = pd.DataFrame({
    'Modelo': ['Lineal', 'Poisson'],
    'RMSE':   [rmse_lm,  rmse_glm],
    'MAE':    [mae_lm,   mae_glm]
})
x = np.arange(2)
w = 0.35
bars_rmse = ax.bar(x - w/2, met['RMSE'], w, label='RMSE',
                   color=PALETTE[0], alpha=0.85)
bars_mae  = ax.bar(x + w/2, met['MAE'],  w, label='MAE',
                   color=PALETTE[2], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['Regresión Lineal', 'Regresión Poisson'])
ax.set_ylabel("Error (menor es mejor)")
ax.set_title("Métricas de Error Comparativas")
ax.legend()

# Agregar etiquetas de valor sobre las barras
for bar in bars_rmse:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{bar.get_height():.1f}',
            ha='center', fontsize=9)
for bar in bars_mae:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{bar.get_height():.1f}',
            ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("comparacion_modelos.png", bbox_inches='tight')
plt.close()
print("[✓ Guardado] comparacion_modelos.png")


# =============================================================
#  SECCIÓN 7: TABLA RESUMEN DE COEFICIENTES
# =============================================================
# Generamos una tabla comparativa de los coeficientes más
# relevantes para incluir en el informe LaTeX

print("\n" + "=" * 60)
print("  SECCIÓN 7: TABLA RESUMEN DE COEFICIENTES")
print("=" * 60)

key_vars = ['temp', 'workingday']
tabla = pd.DataFrame({
    'Variable': key_vars,
    'β (Lineal)':      [lm_model.params.get(k, float('nan'))
                        for k in key_vars],
    'β (Poisson)':     [coefs.get(k, float('nan'))
                        for k in key_vars],
    'exp(β) Poisson':  [np.exp(coefs.get(k, 0))
                        for k in key_vars],
    'p-valor (Poisson)':[poisson_model.pvalues.get(k, float('nan'))
                         for k in key_vars],
}).round(4)

print("\nCoeficientes de variables continuas/binarias:")
print(tabla.to_string(index=False))
print("\nNota: Actualizar los valores en el LaTeX con los de esta tabla.")


# =============================================================
#  SECCIÓN 8: RESUMEN FINAL
# =============================================================
print("\n" + "=" * 60)
print("  RESUMEN FINAL DEL PROYECTO")
print("=" * 60)
print(f"  Total de observaciones:              {len(df):,}")
print(f"  Hora pico de demanda:                {peak_h}h")
print(f"  Clima más favorable:                 Despejado")
print(f"  RMSE Lineal:                         {rmse_lm:.2f}")
print(f"  RMSE Poisson:                        {rmse_glm:.2f}")
print(f"  Mejora en RMSE:                      "
      f"{(rmse_lm - rmse_glm)/rmse_lm*100:.1f}%")
print(f"  Predicciones negativas (Poisson):    0")
print(f"  Factor multiplicativo por temp:      {np.exp(coefs['temp']):.3f}x")
print(f"  Factor multiplicativo workingday:    {np.exp(coefs['workingday']):.3f}x")
print(f"\n  Imágenes generadas:")
print(f"    ✓ eda_bikeshare.png")
print(f"    ✓ lm_limitaciones.png")
print(f"    ✓ comparacion_modelos.png")
print(f"\n  Repositorio: https://github.com/verockala/proyecto-bikeshare")
print("\n" + "=" * 60)
print("  ✓ Script ejecutado exitosamente.")
print("=" * 60)
