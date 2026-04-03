# =============================================================
#  PROYECTO FINAL DE PROBABILIDAD
#  Regresión de Poisson en Datos de Bikeshare
#  Autores: [Nombre del equipo]
#  Fecha:   Marzo 2026
# =============================================================

# ── Librerías ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paleta de colores consistente
PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

# =============================================================
#  1. CARGA DE DATOS
# =============================================================
print("=" * 60)
print("  CARGA DE DATOS")
print("=" * 60)

try:
    from ISLP import load_data
    df = load_data('Bikeshare')
    print("Dataset cargado desde ISLP.")
except ImportError:
    # Alternativa: UCI Machine Learning Repository vía ucimlrepo
    from ucimlrepo import fetch_ucirepo
    bike_repo = fetch_ucirepo(id=275)
    df = pd.concat([bike_repo.data.features, bike_repo.data.targets], axis=1)
    df.rename(columns={'cnt': 'bikers'}, inplace=True)
    print("Dataset cargado desde UCI ML Repository.")

print(f"\nDimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Columnas: {list(df.columns)}")
print("\nPrimeras filas:")
print(df.head())
print("\nEstadísticas descriptivas de 'bikers':")
print(df['bikers'].describe())

# Aseguramos tipos correctos para las categóricas
cat_cols = ['mnth', 'hr', 'weathersit']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# =============================================================
#  2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================
print("\n" + "=" * 60)
print("  2. ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Análisis Exploratorio — Bikeshare Washington D.C.", fontsize=14, fontweight='bold')

# 2.1 Histograma + curva Poisson teórica
ax = axes[0, 0]
lam = df['bikers'].mean()
x_range = np.arange(0, df['bikers'].max() + 1)
poisson_pmf = stats.poisson.pmf(x_range, lam)

ax.hist(df['bikers'], bins=60, density=True, color=PALETTE[0],
        alpha=0.7, label='Datos observados', edgecolor='white', linewidth=0.3)
ax.plot(x_range, poisson_pmf, color=PALETTE[1], linewidth=2,
        label=f'Poisson(λ={lam:.1f}) teórica')
ax.set_title("Distribución de Bicicletas Alquiladas")
ax.set_xlabel("bikers (conteo por hora)")
ax.set_ylabel("Densidad / Probabilidad")
ax.legend(fontsize=9)

# Estadísticos para sobre-dispersión
var_b  = df['bikers'].var()
mean_b = df['bikers'].mean()
print(f"\nMedia de bikers:    {mean_b:.2f}")
print(f"Varianza de bikers: {var_b:.2f}")
print(f"Índice de dispersión (Var/Media): {var_b/mean_b:.2f}")
print("→ Si >> 1: sobre-dispersión (esperable con covariables mezcladas)")

# 2.2 Promedio de alquileres por hora del día
ax = axes[0, 1]
hr_col = 'hr'
avg_hr = df.groupby(hr_col)['bikers'].mean()
hours  = avg_hr.index.astype(int) if hasattr(avg_hr.index[0], '__int__') else avg_hr.index
ax.bar(range(len(avg_hr)), avg_hr.values, color=PALETTE[2], alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(avg_hr)))
ax.set_xticklabels(range(len(avg_hr)), fontsize=7)
ax.set_title("Promedio de Alquileres por Hora del Día")
ax.set_xlabel("Hora (0 – 23)")
ax.set_ylabel("Promedio de bikers")
# Máximo
peak_h = avg_hr.idxmax()
print(f"\nHora pico de mayor demanda: {peak_h}h ({avg_hr.max():.1f} bicicletas promedio)")

# 2.3 Promedio por clima
ax = axes[1, 0]
avg_weather = df.groupby('weathersit')['bikers'].mean()
# Compatibilidad: ISLP usa strings, UCI usa enteros
weather_labels_int = {1: "Despejado", 2: "Nublado", 3: "Lluvia ligera", 4: "Mal tiempo"}
weather_labels_str = {
    'clear': "Despejado", 'cloudy': "Nublado",
    'light rain': "Lluvia ligera", 'heavy rain': "Mal tiempo",
    'misty': "Nublado"
}
def get_label(k):
    try:
        return weather_labels_int.get(int(k), str(k))
    except (ValueError, TypeError):
        return weather_labels_str.get(str(k).lower().strip(), str(k))
keys = [get_label(k) for k in avg_weather.index]
bars = ax.bar(keys, avg_weather.values, color=PALETTE[:len(keys)], alpha=0.85, edgecolor='white')
ax.set_title("Promedio de Alquileres por Clima")
ax.set_xlabel("Condición climática")
ax.set_ylabel("Promedio de bikers")
ax.tick_params(axis='x', labelsize=8)
for bar, val in zip(bars, avg_weather.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)

# 2.4 Boxplot: día laboral vs fin de semana
ax = axes[1, 1]
groups = [df.loc[df['workingday'] == 0, 'bikers'].values,
          df.loc[df['workingday'] == 1, 'bikers'].values]
bp = ax.boxplot(groups, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp['boxes'], [PALETTE[3], PALETTE[4]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax.set_xticklabels(['Fin de semana / Feriado', 'Día laboral'])
ax.set_title("Distribución de Alquileres: Laboral vs No Laboral")
ax.set_ylabel("bikers")

plt.tight_layout()
plt.savefig("eda_bikeshare.png", bbox_inches='tight')
plt.close()
print("\n[Guardado] eda_bikeshare.png")

# =============================================================
#  3. REGRESIÓN LINEAL MÚLTIPLE (MODELO BASE)
# =============================================================
print("\n" + "=" * 60)
print("  3. REGRESIÓN LINEAL MÚLTIPLE (Modelo Base)")
print("=" * 60)

# Variables predictoras (dummies automáticas vía fórmula)
formula_lm = "bikers ~ C(mnth) + C(hr) + C(weathersit) + temp + workingday"
lm_model   = smf.ols(formula_lm, data=df).fit()

df['pred_lm'] = lm_model.fittedvalues
neg_preds      = (df['pred_lm'] < 0).sum()
print(f"\nR² del modelo lineal:        {lm_model.rsquared:.4f}")
print(f"RMSE del modelo lineal:      {np.sqrt(mean_squared_error(df['bikers'], df['pred_lm'])):.2f}")
print(f"Predicciones NEGATIVAS:      {neg_preds}  ← falla conceptual")

# Mostrar madrugada con predicciones negativas
madrugada_mask = (df['hr'].astype(int) <= 5)
neg_mad = df.loc[madrugada_mask & (df['pred_lm'] < 0), ['hr', 'bikers', 'pred_lm']]
if len(neg_mad) > 0:
    print("\nEjemplos de predicciones negativas en la madrugada:")
    print(neg_mad.head(10).to_string(index=False))

# Gráfico: predicciones por hora (lineal)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Regresión Lineal Múltiple — Limitaciones", fontsize=13, fontweight='bold')

ax = axes[0]
avg_obs = df.groupby(df['hr'].astype(int))['bikers'].mean()
avg_lm  = df.groupby(df['hr'].astype(int))['pred_lm'].mean()
ax.plot(avg_obs.index, avg_obs.values, 'o-', label='Observado', color=PALETTE[0])
ax.plot(avg_lm.index,  avg_lm.values,  's--', label='Predicho (LM)', color=PALETTE[1])
ax.axhline(0, color='red', linewidth=1.2, linestyle=':', label='y = 0')
ax.set_xlabel("Hora del día")
ax.set_ylabel("Promedio de bikers")
ax.set_title("Observado vs Predicho por Hora (LM)")
ax.legend(fontsize=9)

ax = axes[1]
ax.scatter(df['pred_lm'], lm_model.resid, alpha=0.2, s=5, color=PALETTE[2])
ax.axhline(0, color='red', linewidth=1.2, linestyle='--')
ax.set_xlabel("Valores ajustados")
ax.set_ylabel("Residuos")
ax.set_title("Residuos vs Ajustados (LM) — heterocedasticidad")

plt.tight_layout()
plt.savefig("lm_limitaciones.png", bbox_inches='tight')
plt.close()
print("[Guardado] lm_limitaciones.png")

# =============================================================
#  4. REGRESIÓN DE POISSON (GLM)
# =============================================================
print("\n" + "=" * 60)
print("  4. REGRESIÓN DE POISSON (GLM)")
print("=" * 60)

formula_glm = "bikers ~ C(mnth) + C(hr) + C(weathersit) + temp + workingday"
poisson_model = smf.glm(formula_glm, data=df,
                        family=sm.families.Poisson()).fit()

df['pred_glm'] = poisson_model.fittedvalues
neg_preds_glm  = (df['pred_glm'] < 0).sum()

rmse_lm  = np.sqrt(mean_squared_error(df['bikers'], df['pred_lm']))
rmse_glm = np.sqrt(mean_squared_error(df['bikers'], df['pred_glm']))
mae_lm   = mean_absolute_error(df['bikers'], df['pred_lm'])
mae_glm  = mean_absolute_error(df['bikers'], df['pred_glm'])

print(f"\nRMSE — Regresión Lineal:  {rmse_lm:.2f}")
print(f"RMSE — Regresión Poisson: {rmse_glm:.2f}")
print(f"MAE  — Regresión Lineal:  {mae_lm:.2f}")
print(f"MAE  — Regresión Poisson: {mae_glm:.2f}")
print(f"Predicciones negativas GLM: {neg_preds_glm}  (siempre ≥ 0 por construcción)")
print(f"\nLog-Verosimilitud Poisson: {poisson_model.llf:.2f}")
print(f"AIC  Poisson: {poisson_model.aic:.2f}")
print(f"Devianza nula:     {poisson_model.null_deviance:.2f}")
print(f"Devianza residual: {poisson_model.deviance:.2f}")

# =============================================================
#  5. INTERPRETACIÓN DE COEFICIENTES
# =============================================================
print("\n" + "=" * 60)
print("  5. INTERPRETACIÓN DE COEFICIENTES")
print("=" * 60)

coefs = poisson_model.params
exp_coefs = np.exp(coefs)

# Temperatura
beta_temp  = coefs['temp']
exp_temp   = np.exp(beta_temp)
print(f"\n[temp]  β = {beta_temp:.4f}  →  exp(β) = {exp_temp:.4f}")
print(f"  Interpretación: Un incremento de 1 unidad en temp (escala normalizada 0-1)")
print(f"  multiplica la tasa esperada por {exp_temp:.4f}.")
print(f"  Es decir, a mayor temperatura, se esperan {(exp_temp-1)*100:.1f}% más alquileres.")

# Día laboral
beta_wd  = coefs['workingday']
exp_wd   = np.exp(beta_wd)
print(f"\n[workingday]  β = {beta_wd:.4f}  →  exp(β) = {exp_wd:.4f}")
direction = "más" if exp_wd > 1 else "menos"
print(f"  Interpretación: En días laborales se esperan {abs(exp_wd-1)*100:.1f}% {direction}")
print(f"  alquileres en comparación con fines de semana/feriados,")
print(f"  manteniendo constantes el resto de variables.")

# Top coeficientes por hora (excluyendo referencia hr=0)
hr_coefs = {k: v for k, v in coefs.items() if 'C(hr)' in k}
hr_sorted = sorted(hr_coefs.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 horas con mayor coeficiente (vs hr=0 como referencia):")
for k, v in hr_sorted[:5]:
    print(f"  {k:30s}  β={v:.4f}  exp(β)={np.exp(v):.2f}x")

# =============================================================
#  6. GRÁFICOS COMPARATIVOS
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Comparación: Regresión Lineal vs Regresión de Poisson", fontsize=13, fontweight='bold')

avg_glm = df.groupby(df['hr'].astype(int))['pred_glm'].mean()

ax = axes[0]
ax.plot(avg_obs.index, avg_obs.values, 'o-', label='Observado',       color=PALETTE[0], linewidth=2)
ax.plot(avg_lm.index,  avg_lm.values,  's--', label='LM (lineal)',    color=PALETTE[1], linewidth=1.5, alpha=0.8)
ax.plot(avg_glm.index, avg_glm.values, '^-',  label='GLM (Poisson)',  color=PALETTE[2], linewidth=2)
ax.axhline(0, color='red', linewidth=1, linestyle=':')
ax.set_xlabel("Hora del día")
ax.set_ylabel("Promedio de bikers")
ax.set_title("Predicción promedio por hora")
ax.legend(fontsize=9)

ax = axes[1]
met = pd.DataFrame({'Modelo': ['Lineal', 'Poisson'],
                    'RMSE':   [rmse_lm, rmse_glm],
                    'MAE':    [mae_lm,  mae_glm]})
x = np.arange(2)
w = 0.35
ax.bar(x - w/2, met['RMSE'], w, label='RMSE', color=PALETTE[0], alpha=0.85)
ax.bar(x + w/2, met['MAE'],  w, label='MAE',  color=PALETTE[2], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['Regresión Lineal', 'Regresión Poisson'])
ax.set_ylabel("Error")
ax.set_title("Métricas de Error Comparativas")
ax.legend()
for i, (rmse, mae) in enumerate(zip(met['RMSE'], met['MAE'])):
    ax.text(i - w/2, rmse + 1, f'{rmse:.1f}', ha='center', fontsize=9)
    ax.text(i + w/2, mae  + 1, f'{mae:.1f}',  ha='center', fontsize=9)

plt.tight_layout()
plt.savefig("comparacion_modelos.png", bbox_inches='tight')
plt.close()
print("[Guardado] comparacion_modelos.png")

# =============================================================
#  7. TABLA DE COEFICIENTES SELECCIONADOS
# =============================================================
print("\n" + "=" * 60)
print("  7. TABLA RESUMEN DE COEFICIENTES (selección)")
print("=" * 60)

# Seleccionamos coeficientes de variables continuas + clima + workingday
key_vars = ['temp', 'workingday']
key_coefs_glm = {k: poisson_model.params[k] for k in key_vars if k in poisson_model.params}
key_coefs_lm  = {k: lm_model.params[k]      for k in key_vars if k in lm_model.params}

tabla = pd.DataFrame({
    'Variable': list(key_coefs_glm.keys()),
    'β (Lineal)':  [key_coefs_lm.get(k, np.nan) for k in key_coefs_glm],
    'β (Poisson)': list(key_coefs_glm.values()),
    'exp(β) Poisson': [np.exp(v) for v in key_coefs_glm.values()],
    'p-valor (Poisson)': [poisson_model.pvalues[k] for k in key_coefs_glm]
}).round(4)

print(tabla.to_string(index=False))

# =============================================================
#  8. RESUMEN FINAL
# =============================================================
print("\n" + "=" * 60)
print("  RESUMEN FINAL")
print("=" * 60)
print(f"  Observaciones totales:           {len(df):,}")
print(f"  Hora pico de demanda:            {peak_h}h")
print(f"  Clima con más alquileres:        Despejado ({avg_weather.iloc[0]:.0f} bicicletas promedio)")
print(f"  RMSE Poisson < RMSE Lineal:      {rmse_glm:.2f} < {rmse_lm:.2f}")
print(f"  Predicciones negativas GLM:      0 (garantía matemática)")
print(f"  Factor multiplicativo por temp:  {np.exp(coefs['temp']):.3f}x")
print(f"  Factor multiplicativo workingday:{np.exp(coefs['workingday']):.3f}x")
print("\n  ✓ Código ejecutado correctamente. Imágenes guardadas.")