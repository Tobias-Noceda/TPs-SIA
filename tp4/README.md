# TP4 - Unsupervised learning

> [!NOTE]  
> Los comandos son relativos a la carpeta de este archivo (`/tp4`)

## Instalación

```bash
uv sync --all-packages
```

## Uso

```bash
uv run tp4 -c res/config/kohonen.json -o res/output/kohonen.npz --plot
```

El método se elige con la clave `method` del archivo de configuración:
 - kohonen
 - pca
 - oja
 - hopfield

Opciones:
 - `-c`, `--config`: archivo de configuración (obligatorio)
 - `-o`, `--output`: guarda el modelo y los resultados en un `.npz` (opcional)
 - `-p`, `--plot`: muestra los gráficos del método

Hay ejemplos de configuraciones en `res/config/`.
El json está tipado con un schema para conocer los parámetros válidos.

Datasets disponibles en `res/datasets/`:
 - `creadores.csv`: creadores de contenido (Kohonen, PCA, Oja)
 - `letters/`: letras 5×5 (Hopfield)
 - `hopfield/`: códigos QR 37×37 (Hopfield)

## Análisis exploratorio

```bash
uv run eda -c res/config/pca.json -o res/output/eda
```

Analiza el dataset de la configuración: guarda `summary.csv` y gráficos en `plots/`, e imprime
chequeos de faltantes, duplicados, columnas derivadas y pares muy correlacionados (`--threshold`, default 0.9).
