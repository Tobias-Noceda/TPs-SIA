# TP2 — Algoritmos Genéticos: aproximación de imágenes con triángulos

## Enunciado

### Ejercicio 1 (teórico, **no se implementa**)

Pensar cómo se implementaría, con Algoritmos Genéticos, un programa que tome una imagen
cuadrada y la represente lo mejor posible en un mapa de NxN caracteres ASCII
(referencia: <http://www.nicassio.it/daniele/AsciiArtGenetic/>).

### Ejercicio 2 (el que se implementa)

Implementar un **compresor de imágenes** peculiar: un motor de Algoritmos Genéticos que
recibe una imagen y logra la mejor aproximación posible a ella dibujando **triángulos**
sobre un canvas de color uniforme (blanco u otro).

- Parámetros del problema (no confundir con hiperparámetros): **la imagen** y **la cantidad
  de triángulos**.
- Los triángulos son de **color uniforme** pero pueden ser **traslúcidos** (RGBA, HSLA, …).
- Consejo del enunciado: empezar con imágenes simples — banderas, siluetas, pictogramas,
  señales de tránsito, logos, iconos, mapas esquemáticos, símbolos, emojis.

**Input**

- Imagen.
- Cantidad de triángulos.
- Hiperparámetros de la implementación de AG.

**Output**

- Imagen generada.
- Enumeración de triángulos (posición, color, …).
- Métricas para el análisis y defensa de la implementación (fitness, error, generaciones, …).

**Implementar y resolver**

- Todos los métodos de **selección** vistos en clase: Elite, Ruleta, Universal, Boltzmann,
  Torneos (**ambas versiones**: determinístico y probabilístico), Ranking.
- **Ambas estrategias de supervivencia**: Aditiva (`N + K` compiten) y Exclusiva
  (los `K` hijos reemplazan, con relleno si `K < N`).
- Decidir y justificar **criterio(s) de corte** (máximo de generaciones, estructura, contenido, …).
- Justificar la **estructura del individuo** y la **función de aptitud**.
- Al menos **2 métodos de cruza** de: un punto, dos puntos, uniforme, anular.
- Al menos **2 métodos de mutación** de: Gen, MultiGen, Uniforme, No Uniforme.

**Restricción dura del enunciado**: se pueden usar librerías externas para el **manejo de
imágenes**, pero **NO** para la implementación de los Algoritmos Genéticos.
→ Pillow / numpy / scikit-image solo para abrir, rasterizar y comparar imágenes.
Selección, cruza, mutación, supervivencia y el loop generacional van escritos a mano.

**Entregable**: código fuente, presentación, y un README explicando cómo ejecutar.

**Preguntas a responder ANTES de experimentar** (guía para la presentación)

- ¿Cómo evalúo mi aproximación al dibujo?
- ¿Qué es un individuo en este problema? ¿Cuáles son sus genes?
- ¿Qué es el fitness en este problema?
- ¿Cómo mutar un individuo?
- ¿Cómo cruzar individuos? ¿Esa cruza genera descendencia con buenas chances de mejorar?
- ¿Cuál sería la versión más simple de esto?
- ¿Qué tipo de imagen, y sobre todo cómo afecta la cantidad de triángulos a la performance,
  si quiero evaluar rápido el motor de AG?
- ¿Alcanza con implementar PARCIALMENTE los requisitos para evaluar el motor?

**Opcionales**: cantidad de triángulos como cota máxima + error mínimo como corte; otros
polígonos u óvalos `(x, y, rx, ry, θ)`; otros métodos de cruza/mutación; exploración de
los horizontes de los AG vistos en clase.

## Modelado actual

- **Gen** = un `Polygon` (hoy `Triangle`): 3 vértices + `Color` RGBA con alfa continuo.
- **Individuo** = `Sequence[Polygon]` de largo fijo `n` (`type Individual`), sin clase propia,
  igual que en TP1 (tuplas → hashables → cacheables).
- **Cromosoma** = la secuencia de polígonos; los puntos de cruza caen **entre polígonos**,
  no dentro de un polígono.
- **Fenotipo** = el render: canvas del color de fondo dominante de la imagen + polígonos
  dibujados en orden con alfa (`FitnessCalculator._get_individual_render`).
- **Fitness** = similitud del render contra la imagen de referencia, normalizado a [0, 1]
  (1 = mejor). `IndividualFitness` empaqueta `(individual, score)` y es `total_ordering`
  por score, así todo el resto ordena con `sorted(..., reverse=True)`.

## Arquitectura

```
tp2/
├── res/
│   ├── config.json          # hiperparámetros (se pasa con -hp)
│   ├── config.schema.json   # JSON Schema del anterior
│   └── images/banderas/     # 12 banderas PNG (ar, br, cl, cz, dj, gb-sct, gy, jp, mc, np, pl, sc)
└── src/
    ├── tp2/
    │   ├── __init__.py                  # CLI + loop generacional (entrypoint `tp2`)
    │   ├── rt_vis.py                    # RealTimeVis: publica el mejor individuo por shared memory
    │   ├── types/
    │   │   ├── coords.py                # type Coords = Tuple[int, int]
    │   │   ├── color.py                 # Color(r, g, b, a: float 0-1) + pillow() -> RGBA 0-255
    │   │   └── individual.py            # type Individual = Sequence[Polygon]
    │   ├── interfaces/
    │   │   ├── polygon.py               # ABC Polygon: vertices, color, generate_random, mutate_*
    │   │   └── algorithms/
    │   │       ├── selection.py         # type Selection = (Sequence[IndividualFitness], Dict) -> Tuple[...]
    │   │       ├── breeding.py          # type Breeding = (Tuple[Ind, Ind]) -> Tuple[Ind, Ind]
    │   │       ├── mutation.py          # type Mutation = (IndividualFitness, float) -> Individual
    │   │       ├── survival.py          # type Survival = (N, K) -> Tuple[IndividualFitness, ...]
    │   │       └── fitness.py           # ABC FitnessCalculator + render/carga de imagen compartidos
    │   ├── models/
    │   │   ├── triangle.py              # Triangle(Polygon)
    │   │   ├── polygon_generator.py     # PolygonGenerator.generate_many_polygons(type, lt, rb, count)
    │   │   └── individual_fitness.py    # IndividualFitness(individual, score), ordenable por score
    │   └── algorithms/
    │       ├── selection/
    │       │   ├── _wheel.py         # helper compartido: ruleta acumulada (no es un Selection)
    │       │   ├── elite.py
    │       │   ├── roulette.py
    │       │   ├── universal.py
    │       │   ├── ranking.py
    │       │   ├── boltzmann.py
    │       │   ├── deterministic_tournament.py
    │       │   └── probabilistic_tournament.py
    │       ├── breeding/{single_point,double_point,uniform,anular}.py
    │       ├── mutation/{gene,uniform,complete}_mutation.py
    │       ├── survival/{additive,exclusive}.py
    │       └── fitness/{delta,ssim,overlap,mixed,fake}.py
    └── visualizer/
        ├── __init__.py                  # ventana pygame (entrypoint `visualizer`)
        └── stdio.py                     # escribe polígonos a mano en la shm, para debug
```

## Ejecución

Comandos relativos a la raíz del workspace:

```bash
uv sync

# motor (todos los flags son obligatorios)
uv run --package tp2 tp2 -i tp2/res/images/banderas/ar.png -p 50 -n 100 -c 500 \
    -hp tp2/res/config.json -o out.png

# visualizador, en otra terminal (lee la shared memory "visualizer_state")
uv run --package tp2 visualizer
```

> **Blocker de entorno**: `pygame 2.6.1` no tiene wheel para Python 3.14, así que uv lo compila
> desde el sdist y falla con `RuntimeError: Unable to run "sdl-config"`. Sin los headers de SDL
> instalados no se puede sincronizar el paquete `tp2`. Opciones: instalar `libsdl2-dev` y
> compañía, subir a una versión de pygame con wheels para 3.14, o mover `pygame` a un extra
> opcional para que el motor corra sin el visualizador.

| Flag | Significado |
|---|---|
| `-i/--image` | imagen objetivo |
| `-p/--poligons` | cantidad de triángulos por individuo |
| `-n/--population` | tamaño de población |
| `-c/--cycles` | generaciones |
| `-hp/--hyper` | JSON de hiperparámetros |
| `-o/--output` | nombre de archivo de salida (**declarado pero nunca usado**) |

`res/config.json` tiene `generation` (contador que `main` pisa en cada ciclo, lo lee Boltzmann)
y una sección por método de selección. Todas exponen `select` (cuántos individuos selecciona,
K), que el schema fuerza a múltiplo de 2 porque después se aparean de a pares; boltzmann suma
`t0`/`tc`/`k`, `deterministic_tournament` suma `m` y `probabilistic_tournament` suma
`threshold`.

## Loop actual (`src/tp2/__init__.py`)

1. `DeltaEFitnessCalculator.for_image(args.image)` → `(fitness, size, background)`.
2. Población inicial: `args.population` individuos de `args.n` triángulos random dentro de `size`.
3. Por generación: `elite_selection` → `gene_mutation(rate=0.5)` sobre los seleccionados →
   `single_point_crossover` sobre pares consecutivos → fitness de los hijos →
   `additive_survival(population, hijos)` → `vis.update(mejor)`.
4. Corte: solo por `args.cycles` (`tqdm.trange`).

Los algoritmos concretos están **hardcodeados** en `main`; no se eligen por config.

## Decisiones de diseño

- **Los algoritmos son funciones, no clases.** Cada familia tiene un type alias en
  `interfaces/algorithms/` y las implementaciones son funciones puras con esa firma.
  La excepción es fitness, que sí es una ABC porque tiene estado (imagen de referencia, tamaño,
  color de fondo).
- **`Selection` recibe el dict de config entero** en vez de parámetros posicionales, para que
  cada método lea sus propios hiperparámetros (Boltzmann necesita `generation`, torneo su `M`, etc.).
- **`Mutation` recibe `IndividualFitness`, no `Individual`**, porque las mutaciones son
  *adaptativas*: `mutate_with_fitness(score)` escala la magnitud del cambio con `(1 - fitness)`
  — fitness alto muta poco, fitness bajo muta mucho. Es la variante "no uniforme" del enunciado
  pero indexada por fitness en vez de por generación.
- **`Polygon` es abstracto** para permitir el opcional de otros polígonos/óvalos sin tocar el motor.
  `PolygonGenerator` recibe `type[Polygon]`.
- **Caché agresiva de fitness y render**: `@cache` en `calculate_fitness` y en
  `_get_individual_render`. Por eso los individuos deben ser **tuplas de dataclasses frozen**
  (hashables) y los `FitnessCalculator` son `frozen=True, eq=False` (hash por identidad).
- **Fondo = color dominante de la imagen** (`_get_dominant_color`, cuantización adaptativa a
  256 colores sobre un resize 150x150), no blanco fijo: arranca mucho más cerca del objetivo.
- **Visualización desacorplada por shared memory**: el motor escribe JSON del mejor individuo en
  `/dev/shm/visualizer_state`, el proceso pygame lo lee a 30 fps. Ningún import compartido.

## Calculadoras de fitness

| Módulo | Métrica | Estado |
|---|---|---|
| `delta.py` | ΔE CIEDE2000 medio en espacio Lab → `1 / (1 + mean(ΔE))` | la que usa `main`; la más fiel a percepción humana |
| `ssim.py` | `structural_similarity` de scikit-image | **compara escalas distintas**: la referencia viene normalizada a [0,1] y el render sale en 0-255 |
| `overlap.py` | `1 - max(solapamiento) / n` | **no mira la imagen de referencia**: penaliza triángulos apilados, nada más. Solo sirve combinada |
| `mixed.py` | `0.2 * overlap + 0.8 * delta` | combinación fija, el 0.2 está hardcodeado |
| `fake.py` | devuelve 0.5 siempre | placeholder para testear el loop sin costo de render |

## Estado del código / qué falta

**Selección** — los 7 métodos del enunciado implementados: elite, ruleta, universal, ranking,
boltzmann, torneo determinístico y torneo probabilístico. Ruleta / universal / ranking /
boltzmann comparten `_wheel.spin(population, weights, rolls)` y solo aportan cómo construyen
pesos y tiradas. Cada método lee sus hiperparámetros de su propia sección en `config.json`.
**Falta elegirlos por config**: `main` sigue hardcodeando `elite_selection`.

**Cruza** — las 4 implementadas. `uniform_crossover` devuelve **listas**, no tuplas, y las
listas no son hashables → revienta la `@cache` de `calculate_fitness`. Mismo bug en
`uniform_mutation` y `complete_mutation`. Hoy no se nota porque `main` usa
`single_point_crossover` + `gene_mutation`, que sí devuelven tuplas.

**Mutación** — `gene_mutation` (Gen), `uniform_mutation` (por gen con prob. `rate`),
`complete_mutation` (una tirada, muta todos). Mapearlas explícitamente a los nombres del
enunciado (Gen / MultiGen / Uniforme / No Uniforme) y decidir cuál falta.

**Supervivencia** — ambas listas. `exclusive_survival` rellena con los mejores de `N`
cuando `K < N`.

**Criterio de corte** — solo cantidad de generaciones. El enunciado pide decidir y justificar
criterios (estructura, contenido, fitness objetivo, estancamiento). El opcional pide además
error mínimo como corte.

**Output** — no se implementó nada de lo que pide el enunciado: `-o` se ignora, no se guarda
la imagen generada, no se enumeran los triángulos y no se emiten métricas por generación
(fitness máx/medio/mín, diversidad, tiempo). Sin esto no hay análisis para la presentación.

**Configuración** — todo hardcodeado en `main`: `mutation_rate = 0.5`, el `time.sleep(10)` de
debug antes del loop, y la elección de selección/cruza/mutación/supervivencia/fitness.
Deberían venir de `config.json`. El schema ya tiene una sección por método de selección y
`generation` sin cota superior (antes la forzaba a `[0, 0]`, lo cual rompía Boltzmann).

**`RealTimeVis` con 4096 bytes de shared memory** — cada polígono serializa a ~90 bytes de
JSON, así que pasados unos ~40 triángulos tira `ValueError: State is too large for shared memory`.
Subir `SHM_SIZE` en `rt_vis.py` **y** en `visualizer/stdio.py` (la constante y el protocolo
están duplicados entre los dos archivos).

**`fake.py` devuelve la tupla en otro orden** (`(calc, lt, (500,500))` en vez de
`(calc, size, color)`), rompe el contrato de `for_image`.

**README desactualizado** — documenta `uv run tp2 idk` y "algoritmos disponibles: idk".
El entregable exige un README que explique cómo ejecutar.
