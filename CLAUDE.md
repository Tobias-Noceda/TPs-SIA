# TPs-SIA

Repositorio con los trabajos prácticos de la materia **Sistemas de Inteligencia Artificial** (ITBA).

## Organización

Cada TP vive en su propia carpeta `tpN`, donde `N` es el número de trabajo práctico (`tp1`, `tp2`, ...).

```
/                        # workspace uv raíz
├── CLAUDE.md            # este archivo: convenciones generales
├── pyproject.toml
├── src/tps_sia/         # paquete stub del raíz, sin uso real
├── tp1/
│   ├── CLAUDE.md        # contexto/enunciado específico del TP
│   ├── pyproject.toml
│   ├── res/boards/      # recursos de entrada (niveles JSON)
│   └── src/tp1/
│       ├── interfaces/  # ABCs genéricas
│       ├── types/       # type aliases
│       ├── models/      # implementaciones concretas
│       └── algorithms/  # algoritmos genéricos
└── tpN/                 # misma estructura
```

Reglas:

- **No mezclar código entre TPs.** Cada `tpN` es un paquete Python independiente.
- Cada `tpN` tiene su propio `CLAUDE.md` con el contexto del problema (enunciado, restricciones, decisiones de diseño). Leerlo antes de trabajar en ese TP.
- Al crear un TP nuevo: agregar la carpeta a `[tool.uv.workspace] members` en el `pyproject.toml` raíz y crear su `CLAUDE.md`.

## Stack

- Python `>=3.14` (ver `.python-version`).
- Gestor de dependencias y workspace: **uv**.
- Dependencias por TP declaradas en el `pyproject.toml` de cada carpeta, no en el raíz.

Comandos:

```bash
uv sync                                              # instalar dependencias del workspace
uv run --package tp1 tp1 tp1/res/boards/board1.json  # ejecutar el entrypoint del TP
```

Cada TP declara su entrypoint en `[project.scripts]` de su `pyproject.toml` (`tp1 = "tp1:main"`).

## Convenciones de código

- Layout `src/`: el código de cada TP va en `tpN/src/tpN/`.
- Type hints en todo el código, usando sintaxis de generics de PEP 695 (`def f[S](...)`, `type Alias[S] = ...`, `class C[S](Base[S])`).
- Separación por capas dentro de cada TP:
  - `interfaces/` — abstracciones (ABCs genéricas, parametrizadas por el tipo de estado).
  - `types/` — type aliases (`type Coords = Tuple[int, int]`, `type Heuristic[S] = Callable[[S], int]`).
  - `models/` — implementaciones concretas de las interfaces para el problema del TP.
  - `algorithms/` — algoritmos, escritos contra las abstracciones de `interfaces/`, nunca contra tipos concretos.
- Los algoritmos genéricos deben quedar reutilizables entre TPs: parametrizar por el tipo de estado (`[S]`) en lugar de acoplarlos al problema puntual.
- Citar la fuente en un comentario cuando se implementa un algoritmo conocido (ej. `# @link https://en.wikipedia.org/wiki/A*_search_algorithm`).
- Documentar en el docstring/comentario los requisitos que el algoritmo impone sobre `S` (ej. hashable, comparable).
