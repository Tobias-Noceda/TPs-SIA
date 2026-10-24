# TP4 — Métodos de aprendizaje no supervisado

## Enunciado

Fuente: `assignment/SIATP42026.pdf`. Teoría en `theory/` (Clase 14.1–16).

### Ejercicio 1 — Campañas de marketing (creadores de contenido)

Una agencia elige creadores solo por `followers` y las campañas no rinden. Clientes: refugios de
perros (consultas/adopciones), ropa de fútbol, belleza (maquillaje) y materiales de arte (ventas).
Agrupar creadores considerando perfil comercial, audiencia y contenido.

- Dataset: `res/datasets/creadores.csv`. **50 creadores**, `creator_id`, `name` + 31 variables numéricas,
  sin faltantes ni duplicados. Pocos datos → grillas chicas (6×6 deja ~40% de neuronas muertas).
- **Columnas derivadas exactas** (redundantes, candidatas a descartar): `price_ars` = 1500 · `price_usd`;
  `total_engagement_per_1000` = likes + comments + saves + shares; `estimated_argentina_reach` =
  `average_reach` · `argentina_audience_pct` / 100 (redondeado a entero); `organic_posts_pct` = 100 − `sponsored_posts_pct`.
- **Composiciones**: los 4 `topic_*` suman 100 en cada fila y los 4 `content_*` también → con los 4 de
  cada grupo en las entradas hay 2 dependencias lineales exactas (autovalores ≈ 0 en PC26/PC27).
  `average_reach` ~ `average_video_views` r = 0.997 (PC25, λ ≈ 0.001); alcance, impresiones, vistas,
  followers y precio forman un bloque con r ≥ 0.9; también video/short/trend posts (r ≥ 0.95).
- Partes:
  1. **EDA + PCA**: elegir variables, justificar reducción de dimensionalidad, interpretar componentes (biplot PC1/PC2). ✅ implementado (falta decidir el set final de variables)
  2. **Kohonen**: agrupar creadores con todas sus variables; variar hiperparámetros y justificar la
     elección; analizar grilla, distribución de creadores y matriz U; interpretar regiones
     (¿coinciden con PC1/PC2? ¿qué indica una distancia alta en la matriz U?). ✅ implementado
  3. **Oja**: índice de creadores; interpretarlo; ¿elegir siempre el máximo?; comparar con PC1. ⏳
  4. **Decisiones de campaña**: fútbol (alcance / descubrimiento / profundidad educativa), comparar
     con selección por `followers`, mantenimiento del sistema (nuevas campañas, creadores, variables,
     actualización mensual).
- Foco: métodos no supervisados + interpretación; no análisis de marketing exhaustivo.

### Ejercicio 2 — Memoria asociativa (Hopfield) ⏳

- 2.1: 5 QR (menú, vinos, cócteles, postres, especiales) en `datos/hopfield` → matrices binarias
  37×37; almacenarlos, consultar con ruido de distintos niveles, analizar iteraciones y energía por
  paso, limitaciones.
- 2.2 (recomendado primero): letras 5×5 en ±1; almacenar 4, análisis de ortogonalidad y capacidad,
  patrón muy ruidoso → estado espurio.
- QR en `res/datasets/hopfield/`.
- Activación: `sign(h)` conservando el estado anterior si `h = 0` (la `step` de tp3 da {0,1}, no sirve).

## Ejecución

Desde `tp4/` (paths de datasets relativos al cwd):

```bash
uv sync --all-packages   # ¡no `uv sync` a secas desde la raíz: desinstala los demás TPs del venv!
uv run tp4 --config res/config/kohonen.json [--output res/output/kohonen.npz] [--plot]
uv run kohonen-sweep -c res/config/kohonen.json [--grid-sizes 4 6 8 10] [--learning-rates 0.1 0.5 0.9] \
    [--radii 1 k/2 k] [--seeds 3] [--epochs N]                     # → res/output/kohonen_sweep/runs/*.npz
uv run kohonen-sweep-analysis [-i res/output/kohonen_sweep]        # → summary.csv + plots/sweep.png
uv run tp4 --config res/config/pca.json [--output res/output/pca.npz] [--plot]
uv run eda --config res/config/pca.json [-o res/output/eda] [--threshold 0.9]
```

PCA imprime autovalores (λ, % explicado, acumulado, `*` = componentes conservadas), cargas de PC1/PC2
ordenadas por |carga| y los 5 creadores más altos/bajos en PC1; `--output` guarda autovalores,
autovectores, matriz de correlación, scores de todas las PCs, media/desvío, etiquetas y columnas (para
comparar con Kohonen y Oja); `--plot` muestra scree, heatmap de cargas, biplot PC1/PC2 e índice PC1.
EDA guarda `summary.csv` (describe + faltantes + CV + asimetría) y `plots/` (boxplots crudo y
estandarizado, matriz de correlación, histogramas) e imprime chequeos sobre **todas** las columnas
numéricas: faltantes, duplicados, constantes, columnas derivadas, composiciones, pares |r| ≥ umbral
y dependencias lineales exactas entre las entradas.

`tp4` imprime QE, TE, neuronas muertas y los creadores por neurona; `--output` guarda pesos (normalizados
y desnormalizados), posiciones, ganadoras, coordenadas, etiquetas, hits, matriz U y resultados por
época; `--plot` muestra hits + matriz U, QE/TE por época y un *component plane* por variable.
`res/output/` está en `.gitignore`.

## Configuración

JSON validado contra `res/config/config.schema.json` (`config.py`: `load_config(path)` →
`parse_config(dict)`, jsonschema Draft 2020-12, `_describe` resuelve `$ref` locales).

| Clave | Valores |
|---|---|
| `method` | `kohonen` \| `pca` (obligatorio): decide qué bloques se exigen y qué corre `main` |
| `pca.components` | `{ criterion: kaiser }` (default, λ > 1) \| `{ criterion: variance, value ∈ (0, 1] }` (menor q con acumulado ≥ value) \| `{ criterion: fixed, value: q }` |
| `pca.orientation` | `max_loading_positive` (default: la carga de mayor módulo de cada autovector es positiva) \| `none` |
| `kohonen.grid_size` | k ≥ 2 (grilla k × k) |
| `kohonen.topology` | `rectangular` (default, 4 vecinos con R = 1) \| `hexagonal` (6 vecinos) |
| `kohonen.similarity` | `euclidean` (default) \| `exponential` (`1 − e^{−d²}`: mismo ganador que euclídea) |
| `kohonen.neighborhood` | `{ name: step }` (default, regla de Kohonen: todo d ≤ R se actualiza igual) \| `{ name: gaussian }` (`e^{−d²/2R²}`) |
| `kohonen.initialization` | `samples` (default, muestras al azar → evita neuronas muertas) \| `uniform` (U[min, max] por variable) |
| `learning_rate`, `radius` | schedule `{ type, value, hyperparameters?: { final } }`: `constant`; `inverse` (`value/(i+1)`, piso `final`); `linear` y `exponential` (de `value` a `final`, `final` obligatorio). `i` = iteración global de `epochs · P` |
| `epochs` | pasadas por el dataset, una muestra por iteración (teoría: ~500 · n iteraciones) |
| `seed` | opcional: `RandomState(MT19937(SeedSequence(seed)))` para pesos iniciales y orden |
| `dataset` | `path`, `normalize` (solo kohonen, prohibido en pca: `z_score` default \| `min_max` \| `unit_length` \| `none`), `columns.input` (lista explícita), `columns.label` (opcional, p. ej. `name`) |

## Arquitectura

```
src/tp4/
├── __init__.py                     # main + create_model (seed → parse → normalize → train)
├── config.py                       # load_config / parse_config (despacha por method)
├── interfaces/unsupervised_network.py  # UnsupervisedNetwork[R]: train(inputs, epochs) -> Tuple[R, ...], predict
├── interfaces/projection.py        # Projection: fit(inputs), project(inputs) -> (P, q)
├── models/kohonen.py               # Kohonen(UnsupervisedNetwork[EpochResult])
├── models/pca.py                   # PCA(Projection): pasos del procedimiento de Clase 15.2 p.32
├── functions/
│   ├── normalization/              # z_score (copia de tp3 + wrapper), min_max, unit_length, identity
│   ├── similarity/                 # euclidean, exponential
│   ├── neighborhood/               # step, gaussian
│   ├── schedule/                   # constant, inverse, linear, exponential (factories value, final)
│   ├── initialization/             # samples, uniform
│   ├── topology/                   # rectangular, hexagonal (posiciones con vecinos a distancia 1)
│   ├── metrics/                    # quantization_error, topographic_error, u_matrix, hits
│   ├── statistics/correlation.py   # Zᵀ Z / P
│   ├── linear_algebra/             # sorted_eigen (eigh, descendente), orientation (max_loading_positive, align)
│   └── components/                 # kaiser, variance, fixed (ComponentSelection)
├── parsers/dataset_parser.py       # CSV numérico → X (P, n) + etiquetas
├── visualization/kohonen.py        # draw_grid (polígonos por neurona), plot_map, plot_components, plot_training
├── visualization/pca.py            # plot_heatmap (divergente), plot_scree, plot_biplot, plot_index
├── experiments/kohonen_sweep.py    # barrido k × η₀ × R₀ × semillas (paralelo)
├── analysis/kohonen_sweep.py       # summary.csv + plots/sweep.png
├── analysis/eda.py                 # EDA del dataset de una config
└── types/                          # Float, Samples, GridPositions/Distances, *Function aliases,
    └── config/                     #   Topology, EpochResult + TypedDicts/enums de la config
```

## Decisiones de diseño

- **Reuso de tp3 = copia** (no se importa entre TPs): `Float`, `z_score`, esqueleto de `config.py`,
  parser numérico recortado, patrón de semilla. No aplican `Neuron`, activaciones, inicializaciones
  (son para backprop), optimizadores, pérdidas ni clasificación.
- **Sin función de activación**: aprendizaje competitivo, la salida es la ganadora `argmin d(x, Wj)`.
- **`SimilarityFunction` devuelve disimilitud (P, neuronas)**: menor = más cerca, así el ganador es
  siempre `argmin` y la misma función sirve para entrenar (P = 1) y para métricas (todo el set).
- **Posiciones en el plano** (no índices): vecinos a distancia 1 en ambas topologías, así el radio,
  la vecindad, la matriz U (vecinos = distancia ≤ 1) y el error topográfico no dependen de la topología.
  La neurona `j` es fila `j // k`, columna `j % k`.
- **Online, una muestra por iteración**, orden barajado por época; η(i) y R(i) con la iteración global.
- Un schedule de R puede bajar de 1 (con `step`, R < 1 actualiza solo la ganadora): se controla con `final`.
- Métricas por época: QE (media ‖x − W_bmu‖, euclídea siempre) y TE (fracción con 1ª y 2ª BMU no
  adyacentes). Neuronas muertas = hits 0.
- **Normalización devuelve (normalize, denormalize)**: los *component planes* se muestran en las
  unidades originales; `unit_length` no es invertible (denormalize = None).
- **PCA sigue Clase 15.2 p.32 paso a paso** en `PCA.fit`: X → estandarizar (`fit_z_score`, desvío
  poblacional) → Sx = Zᵀ Z / P (= `np.corrcoef`) → `eigh` → orden descendente → V → Y = Z V. El modelo
  estandariza solo, por eso `dataset.normalize` está prohibido con `method: pca`.
- El signo de un autovector es arbitrario (p.46 "autovector opuesto"): `max_loading_positive` lo fija
  para que sea reproducible; para comparar Oja vs PC1 usar `align(v, reference)`.
- `scores` proyecta sobre todas las PCs; `project` solo sobre las q elegidas.
- El slide escribe `argmin e^{−‖x−w‖²}` para la exponencial: es argmax; se implementó como
  disimilitud `1 − e^{−d²}`.

## Estado

- Kohonen verificado con datos sintéticos (3 blobs gaussianos 3D): cada blob cae en una región
  contigua y disjunta, la matriz U marca los bordes, QE 1.2 → 0.2, misma semilla ⇒ mismos pesos.
  Funciona en ambas topologías y con vecindad step o gaussiana.
- `res/config/kohonen.json` usa las 27 variables no derivadas (sin las 4 derivadas exactas ni
  id/name). Ajustar tras el EDA (1.1).
- Dataset real, 6×6 / η 0.5→0.01 / R 6→1 / 500 épocas: QE 2.59, TE 0.02, 15/36 muertas (~2 s).
  Barrido k ∈ {3..6} × η₀ ∈ {0.1, 0.5, 0.9} × R₀ ∈ {1, k/2, k} × 3 semillas: QE baja siempre con k
  (más neuronas por creador), así que elegir k mirando también neuronas muertas y TE, no solo QE.
- PCA verificado: ejemplo de p.28 (λ = 6, 3, 2; V = p.30 salvo signo); sobre creadores Sx == `np.corrcoef`,
  Σλ = p, cov(Y) = diag(λ), λ = s²/P del SVD de Z, V ortonormal.
- PCA con las 27 entradas: PC1 49.7%, PC2 9.2%; Kaiser conserva 6 (81%). PC1 opone tamaño/formato
  viral (alcance, impresiones, vistas, followers, precio, trend/video/short, entretenimiento; −) a
  profundidad (palabras de caption, saves, comments, visitas al perfil, watch time, tutorial/review; +).
  PC1 alto: cuentas de perros; bajo: fútbol/maquillaje masivos. PC2: showcase + arte (+) vs audiencia
  argentina + perros (−).
- Pendiente: decidir variables finales (sacar 1 `topic_*` y 1 `content_*`, y redundantes del bloque de
  alcance), Oja (1.3), decisiones (1.4), Hopfield (ej. 2).
