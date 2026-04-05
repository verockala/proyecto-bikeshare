# Proyecto Final de Probabilidad — Regresión de Poisson en Datos de Bikeshare

Análisis estadístico del sistema de bicicletas compartidas de Washington D.C. usando Regresión Lineal y Regresión de Poisson (GLM), como proyecto final del curso de Probabilidad.

## Descripción

Se modela la demanda horaria de bicicletas (`bikers`) en función de variables climáticas y temporales. El objetivo principal es demostrar por qué la Regresión de Poisson es más adecuada que la Regresión Lineal para datos de conteo.

## Dataset

- **Fuente:** `Bikeshare` del paquete ISLP (o UCI ML Repository como alternativa)
- **Dimensiones:** 8,645 observaciones × 15 columnas
- **Variable respuesta:** `bikers` — conteo de bicicletas alquiladas por hora
- **Predictoras:** mes, hora del día, condición climática, temperatura, día laboral

## Resultados principales

| Modelo | RMSE | MAE | Predicciones negativas |
|---|---|---|---|
| Regresión Lineal | 76.33 | 57.44 | 833 |
| Regresión de Poisson | 69.02 | 46.87 | 0 |

- **Hora pico de demanda:** 17h (349.7 bicicletas promedio)
- **Efecto temperatura:** multiplica la demanda esperada por **2.19x**
- **Días laborales:** 1.5% más alquileres que fines de semana
- La Regresión de Poisson garantiza predicciones no negativas por construcción matemática

## Gráficos generados

| Archivo | Contenido |
|---|---|
| `eda_bikeshare.png` | Distribución, demanda por hora, clima y tipo de día |
| `lm_limitaciones.png` | Predicciones negativas y heterocedasticidad del modelo lineal |
| `comparacion_modelos.png` | Comparación de predicciones y métricas de error |

## Requisitos

```
pip install ISLP statsmodels pandas numpy matplotlib seaborn scikit-learn
```

## Ejecución

```bash
python proyecto_bikeshare.py
```

> En Windows, si hay errores de codificación de caracteres:
> ```powershell
> $env:PYTHONIOENCODING='utf-8'; python proyecto_bikeshare.py
> ```

## Estructura del repositorio

```
proyecto-bikeshare/
├── proyecto_bikeshare.py    # Código principal
├── document.tex             # Informe en LaTeX
├── document.pdf             # Informe compilado
├── eda_bikeshare.png
├── lm_limitaciones.png
├── comparacion_modelos.png
└── README.md
```

## Autores

Proyecto desarrollado para el curso de Probabilidad — Marzo 2026.
