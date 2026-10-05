# TP3 — Perceptrón simple y multicapa

## Enunciado

Fuente: `theory/Enunciado TP3.pdf`. Implementar **perceptrón simple** (escalón, lineal, no
lineal) y **perceptrón multicapa** con retropropagación, y usarlos para resolver los
ejercicios 1–3 para la empresa ficticia *CompanyX*.

### Ejercicio de validación (NO se presenta)

Sirve para validar las implementaciones con datos simples. Muy recomendado.

| Herramienta      | Problema                                                                  | Dataset                                         | Estado                                                            |
| ---------------- | ------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| Simple escalón  | AND, entradas`{{−1,1},{1,−1},{−1,−1},{1,1}}` → `{−1,−1,−1,1}` | `tests/and.csv`                               | ⚠️ el CSV está en**0/1**, el enunciado usa **±1** |
| Simple lineal    | ~50 muestras de una función lineal (ej.`y = x`)                        | `tests/linear.csv` (`y = 2x`, x ∈ [1, 50]) | ✅                                                                |
| Simple no lineal | ~50 muestras de una no lineal (ej.`y = tanh(x)`)                        | `tests/tanh.csv`                              | ✅                                                                |
| Multicapa        | XOR, entradas`{{−1,1},{1,−1},{−1,−1},{1,1}}` → `{1,1,−1,−1}`   | `tests/xor.csv` (±1)                         | ✅                                                                |

- XOR con arquitecturas **`[2, 2, 1]`** (`config_xor.json`) y **`[2, 3, 2, 1]`**
  (`config_xor_2321.json`). La notación incluye la capa de entrada; en `arquitecture` se
  escriben solo las capas siguientes.
- Se recomienda hacer los cálculos a mano para esas arquitecturas y **comparar el multicapa
  con el simple escalón en XOR** (el escalón no puede: XOR no es linealmente separable).

### Ejercicio 1 — Fraude (*Knowledge Distillation*)

*BigModel* estima la probabilidad de fraude pero es caro. Construir un *TinyModel* que lo imite.

- Dataset: `res/datasets/fraud_dataset.csv` (7500 filas; `fraud_mini_dataset.csv` = 19 filas
  para pruebas). Documentación en `res/datasets/fraud_dataset_documentation.pdf`.
- **Target = `big_model_fraud_probability`** ∈ [0, 1] (regresión de probabilidad, se destila
  BigModel). **`flagged_fraud` es ground truth y NO debe usarse para entrenar**; sí sirve para
  evaluar y elegir el umbral.
- Modelos: **solo perceptrón simple lineal vs. no lineal** con activación adecuada
  (salida en [0, 1] → logística).
- **Parte A — aprendizaje** (usando **todas** las muestras):
  a) ¿hay underfitting? b) ¿hay saturación de capacidades? c) ¿cuál se elige para el estudio
  de generalización, según su potencial de aprendizaje?
- **Parte B — generalización** con el perceptrón elegido:
  a) métricas de evaluación elegidas y por qué; b) estrategia de partición del dataset
  (holdout, k-fold, …) y cómo se elige el mejor conjunto de entrenamiento;
  c) mejor modelo + **recomendación de umbral de detección de fraude**.
- **Explorar el dataset antes**: documentación de cada columna, rangos, composición,
  limpieza. Escalas muy distintas (`timestamp` ≈ 1.7e9, `device_screen_resolution` ≈ 1e6)
  → normalizar.
- Opcionales: ReLU en el no lineal (práctico); features nuevos / descartables (teórico);
  calibración (teórico).

### Ejercicio 2 — Dígitos con perceptrón multicapa

- Clasificación de dígitos 0–9. `digits.csv` (≈12.4k) para aprendizaje **y** ajuste de
  hiperparámetros; `digits_test.csv` (≈2.5k) = "producción", solo para generalización final.
- Columnas `label`, `image` (vector 28×28 = 784 floats en [0, 1] serializado como string).
  Loader en `parsers/digit_dataset_loader.py`. Salida natural: 10 neuronas (one-hot) + argmax.
- Preguntas: (a) ¿cómo evalúo el desempeño? (b) ¿qué variantes pruebo?
- **Mínimo a analizar**: variantes de **tasa de aprendizaje**, de **arquitectura** y de
  **mecanismo de optimización** (ej. gradiente descendente, momentum, Adam). Se pueden
  variar otros hiperparámetros.

### Ejercicio 3 — Dígitos, accuracy ≥ 98%

- Nuevo dataset `more_digits.csv` (≈15.7k). Objetivo: **accuracy ≥ 98%** (sobre
  `digits_test.csv`, mismo criterio que el ej. 2).
- (a) mejor resultado obtenido; (b) técnicas usadas para mejorar respecto del ej. 2;
  (c) otros factores que influyeron en el cambio (más datos, distribución, …).
- Opcionales (ej. 2 y 3): robustez a ruido (ej. gaussiano sobre el set de test);
  interpretabilidad (métodos de atribución).

**Aclaración del enunciado**: no arrancar opcionales antes de terminar lo obligatorio.

### Funcionalidades recomendadas

- Operaciones **matriciales** para performance.
- **Reportar progreso** al correr (ya: `tqdm`).
- **Configuración extensible** y almacenable (ya: JSON + schema).
- **Guardar y cargar modelos** (junto con su config) para continuar entrenamientos largos.
- **Separar experimentos de análisis**: guardar por experimento loss por época,
  hiperparámetros, tiempos, etc., y graficar/tabular aparte.

## Ejecución

Desde `tp3/` (los paths de datasets en las configs son relativos al cwd):

```bash
uv run tp3 --config res/config/tests/config_and.json
uv run tp3 --config res/config/tests/config_linear.json
uv run tp3 --config res/config/tests/config_tanh.json
uv run tp3 --config res/config/tests/config_xor.json        # MLP 2-2-1
uv run tp3 --config res/config/tests/config_xor_2321.json   # MLP 2-3-2-1
uv run tp3 --config res/config/config_fraud.json            # ej. 1, simple logístico normalizado
```

Sin ventana de matplotlib: `MPLBACKEND=Agg uv run tp3 ...`.

## Configuración

JSON validado contra `res/config/config.schema.json` (`config.py:load_config`, jsonschema
Draft 2020-12). Claves:

| Clave                              | Valores                                                                                                                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `perceptron.type`                | `single_layer` → requiere `activation_function`; `multi_layer` → requiere `arquitecture`                                                                          |
| `perceptron.activation_function` | `{ name: step\|linear\|tanh\|logistic, hyperparameters?: { beta?, L? } }`                                                                                                    |
| `perceptron.arquitecture`        | lista de capas; cada capa es lista de neuronas`{ activation_function: {...} }`. **La capa de entrada no se escribe**: su tamaño es `len(datasets.columns.input)` |
| `loss_function`                  | `hse` (½ Σ, la de la teoría) o `mse` (media)                                                                                                                         |
| `learning_rate`, `epochs`      | número, entero                                                                                                                                                             |
| `batch_size`                     | entero ≥ 1, opcional (default `1`): `1` = online, tamaño del train = batch, intermedio = mini-batch                                                                         |
| `datasets`                       | `train`, `test`, `validation` (CSV) + `columns.input` / `columns.output` / `columns.expected`                                                                     |
| `datasets.normalize`             | bool, opcional (default `false`): z-score por columna de entrada con media/desvío del **train**, reaplicado a test y validación                                      |

`load_config` reemplaza los nombres por tuplas `(función, derivada)` y `type` por `PerceptronType`.
`main` despacha por `PerceptronType`.

## Arquitectura

```
src/tp3/
├── __init__.py                  # main: carga config, entrena, testea, plotea, predice
├── config.py                    # load_config: validación por schema + resolución de funciones
├── interfaces/perceptron.py     # Perceptron[OT]: train / test / predict
├── models/
│   ├── neuron.py                # Neuron: activation + derivative
│   ├── simple_perceptron.py     # SimplePerceptron(Perceptron[Float], Neuron), mini-batch
│   ├── layer.py                 # Layer: weights (in, out), bias (out,), fowards / delta / update
│   └── multi_layer_perceptron.py# MultiLayerPerceptron: lista de Layer, backprop mini-batch
├── functions/
│   ├── activation/              # step, linear, non_linear/{tanh, logistic} (+ derivadas)
│   └── loss/                    # half_squared_error, mean_squared_error (+ derivadas vectoriales)
├── parsers/                     # dataset_parser (CSV numérico → X, y), digit_dataset_loader
└── types/                       # Float, PerceptronInput, LossFunction, ActivationFunction,
    └── config/                  #   EpochResult + TypedDicts/enums de la config
```

## Decisiones de diseño

- **Pesos como matriz `(input_size, n_neuronas)`**, `W[k, j]` conecta la entrada `k` con la
  neurona `j`. Forward: `h = V @ W + b`. Backward: `delta @ W.T` = `Σ_i δ_i w_ij`.
- **Bias separado de `W`** (no como columna `x0 = 1`). Equivalente a la teoría; la
  actualización es por neurona: `b -= η δ`.
- **Convención de signo**: el MLP usa el gradiente `∂E/∂O = O − ζ` y resta (`-=`); el
  perceptrón simple usa `(ζ − O)` y suma (`+=`). Son la misma regla.
- **Derivadas de la pérdida devuelven un vector** (una componente por neurona de salida).
- **Step con derivada 1**: truco para que la regla general se reduzca a Rosenblatt en el
  perceptrón simple. **No usar step en el MLP**: su derivada real es 0 y backprop no aprende.
- **Guarda de divergencia**: ambos `train` lanzan `FloatingPointError` si la pérdida deja de ser
  finita (síntoma típico: `η` demasiado grande para la escala de los datos).
- **XOR en ±1** para que tanh trabaje en todo su rango.

### Vectorización y mini-batch

- **Todo opera sobre matrices `(muestras, features)`**: `train` / `predict` reciben `X` 2-D
  (`PerceptronInputs`) y `predict` devuelve un `ndarray` (`(N,)` simple, `(N, salidas)` MLP).
- **`Layer.update` usa `data.T @ delta`** = `Σ_μ outer(V^μ, δ^μ)` (teoría p.54), **promediado
  sobre el batch** para que `η` no dependa de `batch_size`. Con `batch_size = 1` reproduce
  exactamente el online anterior (verificado: mismos pesos con semilla y orden fijo).
- **Se baraja el orden de las muestras en cada época** (`np.random.permutation`).
- **Activación por neurona, aplicada por grupo**: `Layer` agrupa las neuronas que comparten
  `(f, f')` y aplica cada función una vez sobre todas sus columnas. `ActivationFunctions.functions`
  está cacheada por `(nombre, beta, L)` para que neuronas configuradas igual compartan objeto.
- **Activaciones y derivadas deben aceptar arrays** (step usa `np.where`, derivadas constantes
  `np.ones_like`).
- **Derivada de MSE por muestra**: divide por la cantidad de salidas (`shape[-1]`), no por el
  tamaño del batch; el promedio sobre el batch lo hace `update`.
- **Error de época = `loss_function` sobre las predicciones hechas durante la época**. Con
  `hse` es la suma por muestra de siempre; con `mse` es la media sobre todo el set (antes era
  la suma de las MSE por muestra).

## Estado del código

Funciona (verificado):

- AND (step), lineal (`η = 1e-3`), tanh, XOR 2-2-1 y XOR 2-3-2-1 convergen.
- Backprop verificado contra gradiente numérico (diferencia ≈ 1e-11).
- Experimento XOR, 20 semillas × 10k épocas: 2-2-1 resuelve 14/20 (η=0.1) y 18/20 (η=0.05);
  2-3-2-1 resuelve 20/20 en ambos. Con `η = 0.5` y datos ±1, 2-2-1 cae a 7/20 por saturación.

Pendiente por ejercicio (detalle en `theory/possible_improvements.md`):

- **Validación**: `tests/and.csv` en 0/1 → pasarlo a ±1 como el enunciado. Falta una config
  de XOR con perceptrón simple escalón para la comparación pedida.
- **Ej. 1**:
  - `config_fraud.json`: logística, η = 0.01, `batch_size` 32, 200 épocas, `normalize: true`,
    train y test = dataset completo (parte A). Sin normalizar, la logística satura y predice 1.0
    para todo. HSE final ≈ 40.8 (MSE ≈ 0.011 vs. BigModel); lineal queda en MSE ≈ 0.026.
  - `config_multi.json` apunta a datasets inexistentes (y es MLP, pero el ej. 1 pide solo
    perceptrón simple).
  - Falta partición train/test (holdout / k-fold), métricas (MSE/MAE vs. BigModel; con
    `flagged_fraud`: precision, recall, F1, ROC/PR) y barrido de umbral.
- **Ej. 2/3**:
  - El parser CSV numérico no lee `digits.csv` (columna `image` es string) → integrar
    `digit_dataset_loader` + one-hot de `label`.
  - Falta salida multi-clase (10 neuronas + argmax) y accuracy / matriz de confusión.
  - Faltan **otros optimizadores** (momentum, Adam) — requisito mínimo del ej. 2.
  - Parsear `digits.csv` tarda ~10 s (`ast.literal_eval` por fila); entrenar una época con
    784-32-16-10 y `batch_size = 32` tarda ~0.06 s.
- **Transversal**:
  - `main` imprime la pérdida como "accuracy" y no evalúa validación.
  - Sin semilla, `epsilon` hardcodeado.
  - Falta guardar/cargar modelos y separar registro de experimentos del análisis
    (recomendación del enunciado).
  - Falta citar las fuentes de los algoritmos con `# @link` (convención del `CLAUDE.md` raíz).
