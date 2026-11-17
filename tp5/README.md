# TP5 - Deep Learning: Autoencoders

> [!NOTE]
> Los comandos son relativos a la carpeta de este archivo (`/tp5`)

Implementación desde cero (numpy) de tres autoencoders:

- **Ej. 1a — Autoencoder básico** sobre `font.h` (32 glifos de 7×5) con espacio latente 2-D.
- **Ej. 1b — Denoising autoencoder**: entrada con ruido, salida esperada = patrón limpio.
- **Ej. 2 — Autoencoder variacional (VAE)** sobre imágenes RGB, con reparametrization trick y generación
  de muestras nuevas desde el prior.

## Instalación

```bash
uv sync --all-packages
```

> [!WARNING]
> No usar `uv sync` a secas desde la raíz del repo: desinstala los paquetes de los otros TPs del venv.

## Uso

```bash
uv run tp5 -c <config.json> [-o <salida.npz>] [-i]
uv run tp5 -l <salida.npz> -i
```

Opciones:

- `-c`, `--config`: archivo de configuración (obligatorio salvo con `--load`)
- `-o`, `--output`: guarda config, historial, códigos latentes, reconstrucciones y pesos en un `.npz`
- `-l`, `--load`: carga una corrida guardada con `--output` en vez de entrenar (excluyente con `--config`)
- `-i`, `--interactive`: abre el explorador web del espacio latente con el modelo entrenado
- `--host`, `--port`: dirección del explorador (default `127.0.0.1:8000`)
- `--no-browser`: no abre el navegador automáticamente
- `--generated-dir`: dónde guarda las figuras el botón *Save image* (default `res/output/ej1/generated/`)
- `--noisy-dir`: carpeta de patrones ruidosos que lista/guarda `/denoise` (default `res/output/ej1/noisy/`)

Al terminar el entrenamiento se imprime: épocas, pérdida, máximo de píxeles incorrectos, patrones aprendidos
y cuáles no (para el VAE: J, reconstrucción, KL y estadísticas de μ y σ por dimensión).

### Ej. 1a — Autoencoder básico

```bash
uv run tp5 -c res/config/ej1a/autoencoder.json -o res/output/ej1a/autoencoder.npz -i
uv run tp5 -c res/config/ej1a/best.json -o res/output/ej1a/best.npz -i
uv run tp5 -c res/config/ej1a/baseline.json   # 35-2-35 con descenso por gradiente: no llega
```

`best.json`: 35-30-20-10-**2**-10-20-30-35, ReLU en las ocultas, latente lineal, salida logística, BCE,
Adam (0.01), batch 8, xavier, hasta 20000 épocas.

`autoencoder.json`: 35-25-15-**2**-15-25-35, tanh en las ocultas, latente lineal, salida logística, BCE,
Adam (0.001), batch 32, inicialización xavier. Aprende los 32 patrones con 0 píxeles incorrectos en ~5600
épocas (~2 s). `baseline.json` (35-2-35, MSE, GD 0.5) queda en 4/32 con hasta 11 píxeles mal.

### Ej. 1b — Denoising autoencoder

```bash
uv run tp5 -c res/config/ej1b/denoising.json -o res/output/ej1b/denoising.npz -i
uv run tp5 -c res/config/ej1b/best.json -o res/output/ej1b/best.npz -i
# luego abrir http://localhost:8000/denoise
```

`best.json`: 35-30-20-10-**10**-10-20-30-35 con ReLU, ruido `flip` 0.2 × 4 copias, Adam 0.001, batch 8,
8000 épocas.

`denoising.json`: 35-30-20-**10**-20-30-35 (latente de 10: se busca robustez, no un mapa 2-D), ruido
`flip` 0.1 con 4 copias ruidosas nuevas por patrón y época, 20000 épocas (~30 s).

Píxeles incorrectos promedio en la salida (ruido flip, 10 semillas × 32 patrones):

| Nivel de ruido | DAE | AE básico |
|---|---|---|
| 0.05 | 0.02 | 7.3 |
| 0.10 | 0.18 | 9.3 |
| 0.20 | 1.9 | 12.7 |
| 0.30 | 6.5 | 14.8 |

Generar archivos de patrones ruidosos (los carga la página `/denoise`):

```bash
uv run noisy --sample a --noise flip --level 0.15 [--seed 7] [--count 5]
```

`--noise` es `flip | salt_pepper | gaussian`; `--sample` es una etiqueta del dataset (`a`, `DEL`, ...) o `#índice`.
Se escribe un JSON en `res/output/ej1/noisy/` (nunca sobrescribe).

### Ej. 2 — VAE

```bash
uv run tp5 -c res/config/ej2/simpsons.json -o res/output/ej2/simpsons.npz -i
uv run tp5 -c res/config/ej2/simpsons_variants.json -o res/output/ej2/simpsons_variants.npz -i
uv run tp5 -c res/config/ej2/escudos.json -o res/output/ej2/escudos.npz -i
```

Todas: imágenes 32×32×3 (3072 entradas), encoder 3072-512-128-**2·k** (μ y log σ², lineal), decoder
k-128-512-3072 (logística), BCE, Adam 0.001, batch 16, β = 1.

| Config | Dataset | Latente k | Ocultas | Épocas |
|---|---|---|---|---|
| `simpsons.json` | `simpsons32` (crop) | 8 | tanh | 2000 |
| `simpsons_variants.json` | `simpsons32` (crop) | 2 | ReLU | 300 |
| `escudos.json` | `escudos` (stretch) | 2 | ReLU | 1500 |

Con `simpsons32` (330 imágenes) cada época tarda ~4 s.

#### Preparar un dataset de imágenes

```bash
uv run prepare-images -s <carpeta original> -o res/datasets/<nombre> -n 50 --size 32 --fit crop [--classes a b c]
```

Las subcarpetas son clases (`--classes` elige algunas); si no hay subcarpetas, la carpeta es un solo grupo.
`-n` elige al azar por clase (`--seed`), `--size N|HxW`, `--fit crop|pad|stretch`, `--names` filtra por
nombre y `--min-side` descarta imágenes chicas. Escribe PNG (transparencia sobre blanco) y un
`manifest.json` para reproducirlo. Nunca sobrescribe.

## Explorador web (`-i`)

Servidor local (stdlib, sin dependencias externas) con el modelo en memoria; todo decode lo hace Python.

- **`/`** — mapa del espacio latente (2 ejes elegibles; click en un patrón carga su código, click/arrastre
  mueve el punto), grilla decodificada opcional, un slider por dimensión, salida del decoder (cruda o
  binarizada) con el patrón más cercano y un badge *new* si no coincide con ninguno, tabla de códigos y error
  de reconstrucción, interpolación entre dos muestras. Con un VAE además: **Sample from p(z)** (z ~ N(0, I))
  y elipses de 1σ de cada muestra.
- **`/denoise`** — elegir una muestra, tipo de ruido, nivel y semilla; invertir píxeles a mano; ver
  original / ruidosa / salida; aplicar el mismo ruido a todas las muestras; cargar y guardar archivos ruidosos.
- **Save image** — guarda la imagen actual en `--generated-dir` con el nombre escrito en la página, junto con
  todos los datos necesarios para replicarla (coordenadas, arquitectura, optimizador, semilla, dataset, ...).

Parámetros opcionales en la URL: `/?x=0&y=2&grid=1&size=15&binarize=1&sample=4&sigma=1&from=3&to=7`,
`/denoise?sample=a&type=flip&level=0.2&seed=7&all=1&binarize=1`.

## Configuración

Ejemplos en `res/config/` (`ej1a/`, `ej1b/`, `ej2/`). El JSON se valida contra
`res/config/config.schema.json`, que documenta todos los parámetros válidos.

| Clave | Valores |
|---|---|
| `autoencoder.type` | `basic` \| `denoising` \| `variational` |
| `autoencoder.encoder/decoder.arquitecture` | capas sin la de entrada; la última del encoder es el latente (`2·k` neuronas lineales en el VAE: μ y log σ²). Cada capa: `{ neurons, activation_function }` |
| `activation_function` | `{ name: linear \| tanh \| logistic \| relu, hyperparameters?: { beta?, L? } }` |
| `loss_function` | `mse` (default) \| `hse` (½ Σ) \| `bce` (para salida logística) |
| `learning_rate` | `{ type: constant \| momentum \| adaptative \| adam, value, hyperparameters? }` |
| `epochs`, `batch_size`, `epsilon`, `seed` | condiciones de corte, tamaño de batch, semilla |
| `initialization` | `random` \| `xavier` |
| `goal_pixels`, `stop_at_goal` | patrón aprendido con ≤ `goal_pixels` píxeles mal; corta al aprender todos (solo font) |
| `noise` | solo con `denoising`: `{ type: flip \| salt_pepper \| gaussian, level ∈ [0, 1], copies }` |
| `variational` | solo con `variational`: `{ beta }` (peso del término KL, default 1) |
| `dataset` | `{ type: font \| images, path, size: [h, w], channels: 1 \| 3, fit: crop \| pad \| stretch }` |

## Datasets

En `res/datasets/`:

- `font.h`: 32 glifos binarios de 7×5 (ej. 1).
- `simpsons32/`: 30 personajes de [The Simpsons API](https://thesimpsonsapi.com) recortados a 32×32, más 10
  versiones aumentadas de cada uno (flip, rotación, zoom, desplazamiento, color); 330 imágenes. El
  `manifest.json` detalla el origen y las transformaciones.
- `escudos30/`, `escudos150/`: escudos de clubes argentinos de 64×64 (31 y 155 imágenes); `escudos.json`
  los redimensiona a 32×32.
