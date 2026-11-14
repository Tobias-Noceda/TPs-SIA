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
- **b) Denoising autoencoder** (pendiente): arquitectura justificada; distorsionar entradas con distintos
  niveles de ruido y estudiar la capacidad de eliminarlo (entrada ruidosa, salida esperada = original).

### Ejercicio 2 — VAE (pendiente)

Dataset nuevo (p. ej. emojis), esquema variacional (reparametrization trick) y generar una muestra nueva
que se juzgue perteneciente al dataset.

## Ejecución

Desde `tp5/` (paths relativos al cwd):

```bash
uv sync --all-packages   # ¡no `uv sync` a secas desde la raíz!
uv run tp5 --config res/config/ej1a/autoencoder.json [--output res/output/run.npz] [--plot [--plot-dir res/output/plots]]
# Explorador web del espacio latente: entrenar + guardar + abrir
uv run tp5 -c res/config/ej1a/autoencoder.json -o res/output/ej1a/autoencoder.npz -i
# ... o con pesos ya entrenados (sin reentrenar)
uv run tp5 --load res/output/ej1a/autoencoder.npz -i [--port 8000] [--host 127.0.0.1] [--no-browser]
uv run tp5 --config res/config/ej1a/baseline.json             # 35-2-35 con GD: no llega
uv run ej1a-sweep [-s 5] [-e 20000] [--studies architecture optimizer ...] [-w N] [--force]
uv run ej1a-analysis [-i res/output/ej1a]
```

- `main` carga la config, despacha por `autoencoder.type` (`RUNNERS`), entrena e imprime épocas,
  pérdida, máx. píxeles incorrectos, patrones aprendidos y cuáles no. `--output` guarda `.npz`;
  **`--plot` es lo único que importa matplotlib** (`visualization.autoencoder.plot_run`, backend `Agg`):
  guarda los PNG en `--plot-dir` (default `res/output/plots`); sin el flag solo entrena. `main` no abre
  ventanas: `plot_run(run)` sin `save_dir` (`plt.show()`) queda en el código pero no se usa (el backend
  interactivo crasheaba con GLib `xsettings ... antialiasing`).
- **`-i/--interactive`**: explorador web (`web/`), servidor `http.server` de la stdlib en localhost con el
  modelo en memoria; todo decode pasa por Python (`Autoencoder.decode`), sin reentrenar. `--load` reconstruye
  la corrida desde un `.npz` de `--output` (config guardada + pesos) y reemplaza a `--config` (son
  excluyentes). La página: mapa del latente (2 ejes elegibles, click en un patrón = cargar su código,
  click/arrastre = mover esos 2 ejes, grilla decodificada opcional sobre esos ejes con el resto fijo),
  un slider por cada `z_k` (sirve para n > 2), salida del decoder (cruda o binarizada) + patrón más
  cercano y badge "new", tabla ordenable de códigos y error de reconstrucción. URL opcional:
  `?x=0&y=2&grid=1&size=15&binarize=1&sample=4`.
- `ej1a-sweep`: estudio de **un factor por vez** sobre la config base (`STUDIES`): arquitectura,
  optimizador, pérdida, `batch_size`, activación latente, activación oculta, inicialización; semillas
  `0…s−1`; corridas en paralelo a `res/output/ej1a/runs/{study}__{variant}__s{seed}.npz` (saltea las
  existentes salvo `--force`). Los experimentos nunca grafican.
- `ej1a-analysis`: `summary.csv` (por corrida), `aggregate.csv` (por estudio/variante: tasa de éxito,
  mediana de épocas al objetivo, píxeles, tiempo) y `plots/`: `{study}_curves` (pérdida y máx. píxeles,
  mediana + banda min–max), `{study}_summary`, `hardest_patterns` (en las corridas que no llegan) y
  `best/` (mejor corrida reconstruida desde el `.npz`: history, reconstructions, latent_space,
  latent_grid, interpolation).

## Configuración

JSON validado contra `res/config/config.schema.json` (`config.py`: `load_config(path)` →
`parse_config(dict)`, igual que tp4).

| Clave | Valores |
|---|---|
| `autoencoder.type` | `basic` (implementado) \| `denoising` \| `variational` (reservados: `main` lanza `NotImplementedError`) |
| `autoencoder.encoder.arquitecture` | capas sin la de entrada; la **última es el espacio latente**. Cada capa: lista de neuronas `{ activation_function }` (estilo tp3) **o** forma compacta `{ neurons, activation_function }` |
| `autoencoder.decoder.arquitecture` | ídem; su entrada es el latente y la última capa debe tener tantas neuronas como píxeles (35) |
| `activation_function` | `{ name: linear\|tanh\|logistic\|relu, hyperparameters?: { beta?, L? } }` |
| `loss_function` | `mse` (default, media) \| `hse` (½ Σ) \| `bce` (entropía cruzada binaria media, para salida logística) |
| `learning_rate` | `{ type: constant\|momentum\|adaptative\|adam, value, hyperparameters? }` (como tp3) |
| `epochs`, `batch_size`, `epsilon`, `seed`, `initialization` | como tp3 (`random` \| `xavier`) |
| `goal_pixels` | default 1: un patrón está aprendido con ≤ goal_pixels píxeles mal (binarizando en 0.5) |
| `stop_at_goal` | default `true`: corta cuando todos los patrones están aprendidos |
| `dataset.path` | archivo `font.h` |

Configs: `res/config/ej1a/autoencoder.json` (35-25-15-**2**-15-25-35, tanh oculta, latente lineal,
salida logística, BCE, Adam 0.001, batch 32, xavier) y `baseline.json` (35-2-35, MSE, GD 0.5).

## Arquitectura

```
src/tp5/
├── __init__.py                 # main: config → RUNNERS[type] → report → [--output] → [--plot]
├── config.py                   # load_config / parse_config (schema + resolución de funciones)
├── runners/basic.py            # build_basic_autoencoder(config, input_size, random), run_basic(config) -> TrainingRun
├── interfaces/autoencoder.py   # Autoencoder (ABC): encode / decode / reconstruct / train(inputs, expected, epochs)
├── models/
│   ├── neuron.py, layer.py     # copiados de tp3 (W (in, out), bias aparte, grupos de activación)
│   ├── multi_layer_perceptron.py  # forward (cachea) / backward(∂E/∂O) -> ∂E/∂input / step / predict
│   └── basic_autoencoder.py    # encoder MLP + decoder MLP entrenados como una sola red
├── functions/
│   ├── activation/             # linear, tanh, logistic, relu (tp3)
│   ├── loss/                   # hse, mse (tp3) + binary_cross_entropy
│   ├── learning_rate/          # constant, momentum, adaptative, adam (tp3)
│   ├── initialization/         # random, xavier (tp3)
│   └── pixel_error.py          # píxeles incorrectos por patrón
├── parsers/font_parser.py      # font.h → (32, 35) en {0, 1}, etiquetas, forma (7, 5)
├── persistence/run_file.py     # save_run (.npz con config, history, latente, reconstrucciones, pesos), load_parameters
├── web/
│   ├── explorer.py             # LatentExplorer: dataset() y decode(z) en JSON, genérico en ImageShape (h, w, c)
│   ├── server.py               # serve(): GET / y /static/*, GET /api/dataset, POST /api/decode {z: [[...]]}
│   └── static/                 # index.html, app.js, style.css (sin dependencias ni CDN)
├── visualization/
│   ├── glyphs.py               # draw_glyph (píxeles incorrectos en rojo)
│   ├── autoencoder.py          # history, reconstructions, latent, latent_grid, interpolation; plot_run
│   └── sweep.py                # curvas / resumen por estudio, hardest_patterns
├── experiments/ej1a_sweep.py   # script `ej1a-sweep`
├── analysis/ej1a.py            # script `ej1a-analysis`
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
- **Generación**: `latent_grid` decodifica una grilla regular sobre la caja del latente (+10%) e
  `interpolation` el segmento entre dos patrones (por default, vecinos en el latente con mayor distancia
  de Hamming). Cada glifo se titula con el patrón de entrenamiento más cercano y la distancia; en azul
  los que están a > goal_pixels de todos (= letra nueva).
- **Explorador reutilizable**: la página solo conoce "imágenes (h, w, c) aplanadas channels-last con valores
  en [0, 1]" (`ImageShape`, `TrainingRun.shape`); grises con `ink` (1 = negro, glifos) o RGB (c = 3). Para el
  dataset RGB del ej. 2 alcanza con armar un `LatentExplorer(model, inputs, labels, ImageShape(h, w, 3), ink=False)`.
  Distancia al más cercano: píxeles distintos binarizando en 0.5 si el dataset es binario, MAE si no.
  Rangos de los sliders: rango de los códigos de entrenamiento ± 25%. Decodes del navegador: siempre el
  último pedido, uno por vez (`latestOnly`).
- `persistence.run_file.load_run`: reconstruye el modelo (`build_basic_autoencoder` + `load_parameters`);
  `.npz` viejos con `shape` (h, w) se leen como 1 canal.
- Persistencia: el `.npz` guarda la config original; `analysis` reconstruye el modelo con
  `build_basic_autoencoder` + `load_parameters` (reconstrucciones idénticas a las guardadas).

## Estado

- `autoencoder.json` (seed 42): **32/32 con ≤ 1 píxel** en 1476 épocas (~1 s). `baseline.json`: 4/32,
  máx. 11 píxeles tras 20000 épocas.
- Barrido corto (2 semillas × 2000 épocas): llegan al objetivo 35-25-15-2 y 35-30-20-10-2 (no 35-2,
  35-10-2 ni 35-20-2); Adam 0.01 es el más rápido (~190 épocas), GD 0.5 no llega, momentum sí; BCE 2/2
  vs. MSE 1/2; latente tanh/logistic no llega; oculta logistic no llega (relu y tanh sí); batch 1 llega
  en menos épocas pero tarda más en tiempo real.
- Pendiente: barrido completo (`ej1a-sweep` con 5 semillas y 20000 épocas) para el informe; ej. 1b
  (`denoising`) y ej. 2 (`variational`): agregar runner a `RUNNERS` + claves en el esquema.
