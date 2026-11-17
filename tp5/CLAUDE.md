# TP5 — Deep Learning: autoencoders

## Enunciado

Fuente: `assignment/SIATP5.pdf`. Teoría en `theory/` (Clase 17 Deep Learning, 18 Capsule, 19 Autoencoders).

### Ejercicio 1 — Autoencoders sobre `font.h`

- Dataset: `res/datasets/font.h`, 32 glifos de 7×5 binarios (`` ` `` a `~` + `DEL`, que es todo 1).
- **a) Autoencoder básico** ✅ implementado
  1. Arquitectura encoder/decoder con **espacio latente 2-D**.
  2. Estudiar arquitecturas y técnicas de optimización hasta aprender los 32 patrones con **≤ 1 píxel
     incorrecto** (o justificar por qué solo un subconjunto).
  3. Gráfico 2-D de los patrones en el espacio latente.
  4. Generar una letra nueva que no esté en el set de entrenamiento.
- **b) Denoising autoencoder** ✅ implementado: arquitectura justificada; distorsionar entradas con distintos
  niveles de ruido y estudiar la capacidad de eliminarlo (entrada ruidosa, salida esperada = original).

### Ejercicio 2 — VAE ✅ implementado

Dataset nuevo, esquema variacional (reparametrization trick) y generar una muestra nueva que se juzgue
perteneciente al dataset. Datasets en el repo (`res/datasets/`, ver README):

- `simpsons32/`: 30 personajes de The Simpsons API (no Kaggle) recortados a 32×32 + 10 aumentos por
  personaje (`<personaje>_NN.png`) = 330 PNG; `manifest.json` con origen y parámetros de cada aumento.
  Lo usa `res/config/ej2/simpsons.json`.
- `escudos30/`, `escudos150/`: escudos de clubes argentinos 64×64 (PNG modo P, 31 y 155), sin manifest.
  Los usa `res/config/ej2/escudos.json` (stretch a 32×32), pero su `dataset.path` es `res/datasets/escudos`,
  que **no existe**: apuntarlo a `escudos30` o `escudos150`.
- `flags.json` y el dataset de banderas se descartaron.

## Ejecución

Desde `tp5/` (paths relativos al cwd):

```bash
uv sync --all-packages   # ¡no `uv sync` a secas desde la raíz!
uv run tp5 --config res/config/ej1a/autoencoder.json [--output res/output/run.npz]
# Explorador web del espacio latente: entrenar + guardar + abrir
uv run tp5 -c res/config/ej1a/autoencoder.json -o res/output/ej1a/autoencoder.npz -i
# ... o con pesos ya entrenados (sin reentrenar)
uv run tp5 --load res/output/ej1a/autoencoder.npz -i [--port 8000] [--host 127.0.0.1] [--no-browser]
uv run tp5 --config res/config/ej1a/baseline.json             # 35-2-35 con GD: no llega
# Ej. 1b: denoising autoencoder (entrenar + guardar + abrir; la página de ruido está en /denoise)
uv run tp5 -c res/config/ej1b/denoising.json -o res/output/ej1b/denoising.npz -i
uv run tp5 --load res/output/ej1b/denoising.npz -i               # http://localhost:8000/denoise
# Generador de patrones ruidosos (archivos que /denoise carga)
uv run noisy --sample a --noise flip --level 0.15 [--seed 7] [--count 5] [--dataset res/datasets/font.h] [--output-dir res/output/ej1/noisy]
# Configs "best" (ReLU) de 1a y 1b
uv run tp5 -c res/config/ej1a/best.json -o res/output/ej1a/best.npz -i
uv run tp5 -c res/config/ej1b/best.json -o res/output/ej1b/best.npz -i
# Ej. 2: VAE (datasets ya en res/datasets/)
uv run tp5 -c res/config/ej2/simpsons.json -o res/output/ej2/simpsons.npz -i
uv run tp5 -c res/config/ej2/simpsons_variants.json -o res/output/ej2/simpsons_variants.npz -i
uv run tp5 -c res/config/ej2/escudos.json -o res/output/ej2/escudos.npz -i
# Armar un dataset nuevo de imágenes (herramienta)
uv run prepare-images -s <carpeta> -o res/datasets/<nombre> -n 50 --size 32 --fit crop [--classes ...]
```

- `main` carga la config, despacha por `autoencoder.type` (`RUNNERS`), entrena e imprime épocas,
  pérdida, máx. píxeles incorrectos, patrones aprendidos y cuáles no. `--output` guarda `.npz`.
- **Save image** (1a.4, solo a mano desde el explorador, nada se guarda solo): decodifica el `z` actual y
  guarda en `--generated-dir` (default `res/output/ej1/generated/`) figuras listas para el informe con el
  **nombre que se escribe en la página** (obligatorio; letras, dígitos, espacios y `. _ - + ( ) ,`; nunca
  sobrescribe: si existe, pide otro): `{nombre}.png` (salida tal cual, con escala de valores) y
  `{nombre}_binarized.png` (umbral 0.5,
  solo si el dataset es binario: la fuente sí, el RGB del ej. 2 no). Cada figura muestra la imagen con
  grilla de píxeles y al lado lo necesario para replicarla: coordenadas, muestra más cercana, tipo,
  arquitectura y activaciones, pérdida, optimizador, batch, inicialización, semilla, condiciones de corte
  (`epochs`, `epsilon`, objetivo de píxeles, `stop_at_goal`), resultado del entrenamiento y dataset.
- **`-i/--interactive`**: explorador web (`web/`), servidor `http.server` de la stdlib en localhost con el
  modelo en memoria; todo decode pasa por Python (`Autoencoder.decode`), sin reentrenar. `--load` reconstruye
  la corrida desde un `.npz` de `--output` (config guardada + pesos) y reemplaza a `--config` (son
  excluyentes). La página: mapa del latente (2 ejes elegibles, click en un patrón = cargar su código,
  click/arrastre = mover esos 2 ejes, grilla decodificada opcional sobre esos ejes con el resto fijo),
  un slider por cada `z_k` (sirve para n > 2), salida del decoder (cruda o binarizada) + patrón más
  cercano y badge "new", tabla ordenable de códigos y error de reconstrucción. URL opcional:
  `?x=0&y=2&grid=1&size=15&binarize=1&sample=4`.
- **`/denoise`** (misma sesión `-i`, sirve con cualquier AE: con el básico es la línea de base): elegir la
  muestra limpia (salida esperada), ruido `flip | salt_pepper | gaussian` con nivel y semilla (las mismas
  funciones de Python que en el entrenamiento; default = ruido de entrenamiento; cambiar nivel conserva la
  semilla, "New noise" sortea otra), click en píxeles de la entrada ruidosa para invertirlos a mano.
  Muestra original / ruidosa / salida (+ binarizada) con píxeles mal (o MAE), código latente y muestra más
  cercana. "All samples" aplica el mismo ruido a las 32 (tabla + resumen: error medio entrada → salida,
  recuperadas ≤ tolerancia). **Save image**: figura con nombre (mismas reglas que 1a.4) en
  `--generated-dir`, con ruido de entrada, errores y modelo. **Archivos ruidosos**: lista del
  `--noisy-dir` + selector de archivo local, valida forma de imagen y que la etiqueta esté en el dataset,
  ◀ ▶ entre variantes; "Save noisy pattern" guarda la entrada actual (editada o no) en el mismo formato.
  URL opcional: `?sample=a&type=flip&level=0.2&seed=7&all=1&binarize=1&file=<nombre>.json`.
- **`prepare-images`** (herramienta): de una carpeta grande arma un dataset chico en `--output` (nunca
  sobrescribe). Clases = subcarpetas con imágenes (`--classes` elige algunas); si no hay subcarpetas, la
  carpeta es plana (p. ej. una bandera por país) y es un solo grupo. `--names` filtra por nombre de archivo,
  `-n` elige al azar por clase (con `--seed`), `--size N|HxW`, `--fit crop|pad|stretch`, `--min-side`.
  Escribe PNG (transparencia sobre blanco) + `manifest.json` con lo necesario para rehacerlo.
- **VAE en el explorador** (`/`): botón **Sample from p(z)** (z ~ N(0, I) sorteado en el servidor,
  `POST /api/sample {count, seed}`: la generación del 2c), elipses de 1 σ(x) de cada muestra en el mapa
  (`σ(x)`), rangos de sliders que incluyen ±3 del prior, e **Interpolation** (para todo AE): tira de 9
  decodificaciones entre los códigos de dos muestras + slider `t`. URL: `?sigma=1&from=3&to=7`. Las imágenes
  RGB se dibujan tal cual (`ink` solo para font.h); cercanía y "new" por MAE; Save image sin binarizada.
- **`noisy`** (herramienta, no corrida de modelo): `--sample` tiene que ser una etiqueta del dataset (o
  `#índice`; si no, error listando las válidas). Escribe `res/output/ej1/noisy/{ruido}-{YYYYmmdd-HHMMSS}.json`
  (nunca sobrescribe: sufijo `-2`…) con `dataset, label, index, image_shape, noise {type, level, seed},
  original, noisy [variantes], created`. JSON y no `.h`: gaussiano y RGB (ej. 2) son continuos.

## Configuración

JSON validado contra `res/config/config.schema.json` (`config.py`: `load_config(path)` →
`parse_config(dict)`, igual que tp4).

| Clave | Valores |
|---|---|
| `autoencoder.type` | `basic` \| `denoising` \| `variational` (la última capa del encoder tiene **2·k neuronas lineales**: μ y log Σ; el decoder recibe k) |
| `autoencoder.encoder.arquitecture` | capas sin la de entrada; la **última es el espacio latente**. Cada capa: lista de neuronas `{ activation_function }` (estilo tp3) **o** forma compacta `{ neurons, activation_function }` |
| `autoencoder.decoder.arquitecture` | ídem; su entrada es el latente y la última capa debe tener tantas neuronas como píxeles (35) |
| `activation_function` | `{ name: linear\|tanh\|logistic\|relu, hyperparameters?: { beta?, L? } }` |
| `loss_function` | `mse` (default, media) \| `hse` (½ Σ) \| `bce` (entropía cruzada binaria media, para salida logística) |
| `learning_rate` | `{ type: constant\|momentum\|adaptative\|adam, value, hyperparameters? }` (como tp3) |
| `epochs`, `batch_size`, `epsilon`, `seed`, `initialization` | como tp3 (`random` \| `xavier`) |
| `goal_pixels` | opcional: un patrón está aprendido con ≤ goal_pixels píxeles mal (binarizando en 0.5). **Sin él no hay objetivo**: el entrenamiento corta solo por `epsilon` (o `epochs`) y "aprendido" = 0 píxeles mal |
| `stop_at_goal` | default `true`: corta cuando todos los patrones están aprendidos (solo si hay `goal_pixels`) |
| `noise` | **obligatorio con `denoising`, prohibido con el resto**: `{ type: flip\|salt_pepper\|gaussian, level ∈ [0, 1], copies ≥ 1 (default 1) }`. `flip`: invierte cada píxel con prob. `level`; `salt_pepper`: lo pone en 0 o 1 al azar con prob. `level`; `gaussian`: suma N(0, level²) y recorta a [0, 1]. `copies` = versiones ruidosas por patrón y época |
| `variational` | solo con `variational` (opcional): `{ beta }` = λ de J = L + λ KL, default 1 |
| `dataset` | `{ type: font (default) \| images, path, size: [h, w], channels: 1\|3 (default 3), fit: crop\|pad\|stretch (default stretch) }`; `size/channels/fit` solo con `images` (`size` obligatorio). Con `images` no se permiten `goal_pixels` ni `stop_at_goal` (no hay píxeles "mal" en una foto) |

Configs: `res/config/ej1a/autoencoder.json` (35-25-15-**2**-15-25-35, tanh oculta, latente lineal,
salida logística, BCE, Adam 0.001, batch 32, xavier) y `baseline.json` (35-2-35, MSE, GD 0.5).
`res/config/ej2/simpsons.json`: 32×32×3 crop (3072 entradas), 3072-512-128-**2·8**(lineal) /
8-128-512-3072 (logística), BCE, Adam 0.001, batch 16, 2000 épocas, β 1 (latente 8: retratos tienen mucha
variación). `simpsons_variants.json`: mismo dataset, ReLU, latente **2·2**, 300 épocas.
`escudos.json`: 32×32×3 stretch, ReLU, latente **2·2**, 1500 épocas (path roto, ver arriba).
`epsilon` 1e-4 casi nunca se alcanza: la BCE con objetivos continuos no llega a 0 (cortan por épocas).
`res/config/ej1b/denoising.json`: 35-30-20-**10**-20-30-35 (tanh, latente lineal, salida logística), BCE,
Adam 0.001, batch 32, xavier, ruido flip 0.1 × 4 copias, 20000 épocas, epsilon 1e-3, goal 1 px.
`ej1a/best.json`: 35-30-20-10-**2**-10-20-30-35 ReLU, BCE, Adam 0.01, batch 8, 20000 épocas.
`ej1b/best.json`: 35-30-20-10-**10**-10-20-30-35 ReLU, flip 0.2 × 4, Adam 0.001, batch 8, 8000 épocas.
Ambos `best.json` traen `stop_at_goal` sin `goal_pixels`: no tiene efecto (cortan por `epsilon`/épocas).

## Arquitectura

```
src/tp5/
├── __init__.py                 # main: config → RUNNERS[type] (o --load) → report → [--output] → [-i] (sin --plot)
├── config.py                   # load_config / parse_config (schema + resolución de funciones)
├── runners/basic.py            # build_networks, build_basic_autoencoder, train_run(config, builder), run_basic
├── runners/denoising.py        # build_denoising_autoencoder, run_denoising
├── runners/variational.py      # build_variational_autoencoder (kl_weight = β / píxeles), run_variational
├── tools/prepare_images.py     # script `prepare-images`: subconjunto chico y redimensionado de una carpeta de imágenes
├── tools/noisy.py              # script `noisy`: patrones ruidosos de una muestra del dataset → JSON
├── interfaces/autoencoder.py   # Autoencoder (ABC): encode / decode / reconstruct / train(inputs, expected, epochs)
├── models/
│   ├── neuron.py, layer.py     # copiados de tp3 (W (in, out), bias aparte, grupos de activación)
│   ├── multi_layer_perceptron.py  # forward (cachea) / backward(∂E/∂O) -> ∂E/∂input / step / predict
│   ├── basic_autoencoder.py    # encoder MLP + decoder MLP entrenados como una sola red; hooks _epoch_data / _evaluation_data
│   ├── denoising_autoencoder.py  # entrada ruidosa nueva cada época (copies), métricas sobre un set ruidoso fijo
│   └── variational_autoencoder.py  # μ/log Σ, reparametrization trick, J = L + λ KL; encode = μ; sample(n) del prior
├── functions/
│   ├── activation/             # linear, tanh, logistic, relu (tp3)
│   ├── loss/                   # hse, mse (tp3) + binary_cross_entropy
│   ├── learning_rate/          # constant, momentum, adaptative, adam (tp3)
│   ├── initialization/         # random, xavier (tp3)
│   ├── noise/                  # flip, salt_pepper, gaussian: (RandomState, Samples, level) -> Samples
│   └── pixel_error.py          # píxeles incorrectos por patrón
├── parsers/font_parser.py      # font.h → (32, 35) en {0, 1}, etiquetas, forma (7, 5)
├── parsers/image_folder.py     # parse_images (carpeta recursiva → (N, h·w·c) en [0, 1]), fit_image (crop/pad/stretch)
├── parsers/dataset.py          # load_dataset(config["dataset"]) por tipo (font | images), describe_dataset
├── persistence/run_file.py     # save_run (.npz con config, history, latente, reconstrucciones, pesos), load_run, load_parameters
├── persistence/generated.py    # run_metadata(run), save_generated(model, z, ...) → figura normal (+ binarizada), save_denoising
├── persistence/noisy_file.py   # save_noisy / list_noisy / load_noisy (archivos de patrones ruidosos)
├── web/
│   ├── explorer.py             # LatentExplorer: dataset, decode, generate, noise, reconstruct, archivos ruidosos, save_denoising
│   ├── server.py               # GET / y /denoise, /static/*, /api/dataset, /api/noisy[/{nombre}];
│   │                           # POST /api/decode, /api/generate, /api/noise, /api/reconstruct, /api/noisy, /api/save-denoising
│   └── static/                 # common.js (helpers), index.html + app.js, denoise.html + denoise.js, style.css (sin CDN)
├── visualization/glyphs.py     # draw_glyph (píxeles incorrectos en rojo)
├── visualization/generated.py  # plot_generated: imagen + datos de replicación (API OO de matplotlib, sin pyplot); describe_model
├── visualization/denoising.py  # plot_denoising: original / ruidosa / salida (+ binarizada) + datos de replicación
└── types/                      # Float, Samples, EpochResult, TrainingRun, *Function + config/ (enums y TypedDicts)
```

## Decisiones de diseño

- **Reuso de tp3 = copia** (no se importa entre TPs): `Neuron`, `Layer`, activaciones, pérdidas,
  optimizadores, inicializaciones, esqueleto de config. Se quitaron step/sigmoid y la clasificación.
- **MLP partido en `forward` / `backward` / `step`**: `backward` devuelve ∂E/∂entrada, así el decoder
  retropropaga hasta el latente y el encoder sigue desde ahí. Es lo que permite encadenar redes (DAE:
  misma red con entrada ruidosa; VAE: capa de muestreo entre ambas) sin reescribir backprop.
  Verificado contra gradiente numérico (diferencia relativa ≈ 1e-7 con mse/hse/bce, tanh/relu).
- **Métricas al final de cada época** con los pesos finales (`reconstruct` de los 32 patrones): el
  criterio de corte se refiere al modelo devuelto. `EpochResult = (época, pérdida, máx. píxeles mal,
  patrones ≤ goal_pixels)`.
- **Pixel error**: salida binarizada en 0.5 vs. patrón original.
- **BCE con salida logística**: ∂E/∂h = 2β(O − ζ) (se cancela O(1 − O)), sin el gradiente que se apaga
  cuando una sigmoide satura del lado equivocado; con MSE ese término sí aparece.
- **Latente lineal** por default: con tanh/logistic el latente queda acotado y los patrones se
  amontonan en los bordes.
- **Explorador reutilizable**: la página solo conoce "imágenes (h, w, c) aplanadas channels-last con valores
  en [0, 1]" (`ImageShape`, `TrainingRun.shape`); grises con `ink` (1 = negro, glifos) o RGB (c = 3). Para el
  dataset RGB del ej. 2 alcanza con armar un `LatentExplorer(model, inputs, labels, ImageShape(h, w, 3), ink=False)`.
  Distancia al más cercano: píxeles distintos binarizando en 0.5 si el dataset es binario, MAE si no.
  Rangos de los sliders: rango de los códigos de entrenamiento ± 25%. Decodes del navegador: siempre el
  último pedido, uno por vez (`latestOnly`).
- `persistence.run_file.load_run`: reconstruye el modelo (`build_basic_autoencoder` + `load_parameters`);
  `.npz` viejos con `shape` (h, w) se leen como 1 canal.
- Persistencia: el `.npz` guarda la config original (y `goal_pixels = -1` si no hay objetivo).
- **DAE** (`DenoisingAutoencoder(BasicAutoencoder)`): mismas redes/pérdida/optimizador; solo cambian dos hooks
  del loop. `_epoch_data`: entrada = ruido nuevo sobre `copies` repeticiones de los patrones, salida
  esperada = patrones limpios (la red nunca ve dos veces la misma corrupción). `_evaluation_data`: set
  ruidoso **fijo** (sorteado una vez, 32 × copies) contra los limpios, así pérdida, máx. píxeles, "aprendidos",
  `epsilon` y `goal_pixels` miden **denoising**. `report` imprime además el resultado con entradas limpias.
- **VAE** (`VariationalAutoencoder(BasicAutoencoder)`, slides 72–89): p(z) = N(0, I), q(z|x) = N(μ, Σ)
  diagonal; el encoder da μ y log Σ (log Σ recortado a ±10). Entrenamiento: z = μ + ε ⊙ exp(½ log Σ),
  `decoder.backward` da ∂L/∂z; el decoder se actualiza solo con la reconstrucción, el encoder con la
  suma ∂L/∂z + λ ∂KL/∂μ y ∂L/∂z · ε σ/2 + λ ∂KL/∂log Σ. Verificado contra gradiente numérico (≈ 1e-9).
  **Escala del KL**: mse/bce son medias sobre píxeles, así que el KL de cada muestra se pesa con
  `kl_weight = β / píxeles` (β = 1 ⇔ ELBO estándar: reconstrucción sumada + KL); con hse (suma) es β.
  Métricas de cada época determinísticas (z = μ), así `epsilon` no depende de ε; `loss_terms` guarda
  (reconstrucción, KL medio) por época (en el `.npz`). `report` muestra J, ambos términos, MAE y media/desvío
  de μ y σ medio por dimensión (σ ≈ 1 con μ ≈ 0 = dimensión sin usar).
- `BasicAutoencoder` tiene hooks `_check_sizes`, `_train_batch`, `_epoch_loss`, `_epoch_data`,
  `_evaluation_data`: DAE y VAE solo pisan los suyos, el loop es uno.
- **Arquitectura del DAE (1b.1)**: latente de 10 y no de 2: el objetivo es robustez, no un mapa 2-D;
  un cuello de botella más ancho deja separar los 32 patrones con margen para que una entrada con
  algunos píxeles mal caiga cerca del código correcto. El explorador igual funciona con n dimensiones.

## Estado

- `autoencoder.json` (seed 42, sin `goal_pixels`, `epsilon` 1e-3): 32/32 con 0 píxeles mal en 5639 épocas
  (~2 s); con `goal_pixels: 1` corta en 1359 épocas. `baseline.json`: 4/32,
  máx. 11 píxeles tras 20000 épocas.
- `--plot` eliminado (commit 5ecb95e): las figuras salen del explorador (*Save image*).
- `denoising.json`: 20000 épocas (~32 s); set ruidoso fijo (128): pérdida 0.0153, máx. 7 px, 125/128 con
  ≤ 1 px; entradas limpias 32/32. Mismo ruido flip (10 semillas × 32 muestras), píxeles mal en la salida,
  DAE vs. básico (`ej1a/autoencoder.npz`): 0.05 → 0.02 vs 7.3; 0.1 → 0.18 vs 9.3; 0.2 → 1.9 vs 12.7;
  0.3 → 6.5 vs 14.8.
- VAE probado de punta a punta (`prepare-images`, entrenamiento, `--load`, explorador, figuras).
  `simpsons.json` con `simpsons32` (330 muestras): ~4 s/época → 2000 épocas ≈ 2 h 15 min. Smoke test de 2
  épocas OK (2026-11-17).
- `res/output/` (gitignoreado) tiene corridas de barridos del VAE (`runs/beta__{ae,0.1,1,5,25}__s0.npz`,
  ReLU, latente 2, 1500 épocas) y figuras en `plots-ej2/` (β, dimensión latente, KL por dimensión,
  interpolación, muestras). El código que las generó (`src/tp5/analysis/`, `src/tp5/experiments/`) **no
  está en el repo**: esas carpetas solo tienen `__pycache__`.
- `README.md` del TP: instalación, uso por ejercicio, explorador, config y datasets (en castellano).
- Pendiente: estudio de arquitecturas/optimizadores para el informe (1a.2), si no se hizo fuera del repo;
  corregir `dataset.path` de `escudos.json`; recuperar/commitear los scripts de barridos si se quieren
  reproducir.
