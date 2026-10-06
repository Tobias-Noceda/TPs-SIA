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

Estudio del ej. 1 parte A (lineal vs. logística, barrido de η × semillas sobre todo el dataset):

```bash
./run1.sh [épocas=1000] [semillas=5] [batch_size=32]   # ~20 s con 16 cores
```

Experimentos (`experiments/ej1_learning.py`, entrypoint `ej1-learning`) guardan corridas crudas en
`res/output/ej1_learning/runs/*.npz` + `reference.npz` (target, `flagged_fraud`, PCA). El análisis
(`analysis/ej1_learning.py`, `ej1-learning-analysis`) arma `summary.csv` y `plots/`: `loss_curves`,
`best_comparison`, `pc1_curves`, `distributions`, `pred_vs_target`.
Configs base en `res/config/ej1/{linear,logistic}.json` (HSE, normalizado).

Estudio del ej. 1 parte B (generalización de la logística, k-fold estratificado):

```bash
./run1_generalization.sh [épocas=1000] [semillas=3] [folds=5] [η=0.01] [batch_size=32]   # ~10 s
```

`experiments/ej1_generalization.py` (`ej1-generalization`): folds estratificados por `flagged_fraud`
(solo para partir, nunca para entrenar); **z-score ajustado con el train de cada fold** y reaplicado
a su test; curvas de train/test en MSE (HSE es suma y los tamaños difieren); métricas con
`perceptron.test` + `fraud_classification` + `functions/accuracy.py`. `analysis/ej1_generalization.py`
arma `summary.csv` y `plots/`: `generalization_curves`, `fold_metrics`, `fold_f1`, `threshold_metrics`. Mejor
conjunto de entrenamiento = fold con mayor F1 de test (desempate: menor MSE de test).
`fraud_classification` acepta `threshold` (default 0.85 = criterio con el que BigModel marca
`flagged_fraud`); `classification/confusion.py` cuenta las clasificaciones en un `TestResult`.

Estudio del ej. 2 (MLP en dígitos):

```bash
./run2.sh [estudios...]   # ~4 min con 16 cores (estudios + análisis)
```

Spec en `res/config/ej2/studies.json`: `base` (784-64-32-10, oculta tanh, salida sigmoide, HSE, Adam η = 1e-3,
batch 32, 40 épocas, pesos U(−0.5, 0.5)) + `studies` cuyas variantes pisan campos de la base: `optimizer` (η × {GD,
momentum, Adam, η adaptativo}, 5 η por optimizador), `architecture`, `hidden_activation` (ReLU / tanh / logística, 5 semillas).
`experiments/ej2_digits.py` (`ej2-digits`) cachea los CSV parseados en `data.npz`. **Cada corrida entrena con todo
`digits.csv`** (decisión del equipo: sin partición, aunque falten los 8 y haya pocos 5) y, ya entrenada, se evalúa
en `digits_test.csv`; guarda por época pérdida y accuracy de train y accuracy de test. `analysis/ej2_digits.py`
(`ej2-digits-analysis`): `summary.csv` + `plots/`: `lr_loss_curves` (pérdida por época de GD sin
optimizador, una curva por η), `lr_optimizer`, `optimizer_curves`, `architecture`,
`hidden_activation`, `confusion_test` (de la configuración con mayor accuracy media de test). "Dígito perdido" =
dígito presente en el entrenamiento con recall de test < 0.1 (el 8 no cuenta).

Estudio del ej. 3 (784-256-10 logística, objetivo ≥ 98%):

```bash
./run3.sh [estudios...]   # ~17 min con 16 cores (54 corridas de 40 épocas)
```

Reusa `ej2-digits` con `res/config/ej3/studies.json`. Extensiones del experimento: `train` (spec o variante) puede ser
lista de CSV → se concatenan **sin imágenes repetidas** (`digits.csv` y `more_digits.csv` comparten 3689; la unión tiene
24501), cache `data_<hash>.npz` por conjunto; por variante `weight_initialization` (`uniform` = U(−0.5, 0.5) default |
`xavier`, `functions/initialization/`), `augment_shift` (desplazamiento aleatorio ±k px por imagen y época,
`functions/augmentation/shift.py`) y `oversample` (sobremuestreo de clases minoritarias por época,
`functions/augmentation/oversample.py`). Las métricas de train se miden sobre las imágenes originales.
`analysis/ej3_digits.py` (`ej3-digits-analysis`): `summary.csv` + `plots/`: `dataset`, `dataset_recall`,
`lr_optimizer`, `optimizer_curves`, `techniques`, `confusion_test`.

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
| `optimizer`                      | opcional (MLP), default gradiente descendente: `{ name: gradient_descent\|momentum\|adam\|adaptive, hyperparameters?: { momentum?, beta1?, beta2?, epsilon?, a?, b?, k? } }`. `adaptive` = η adaptativo de la teoría (ec. 8.13), requiere `a`; ajusta η al final de cada época vía `Optimizer.end_epoch(loss)` |
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
│   ├── multi_layer_perceptron.py# MultiLayerPerceptron: lista de Layer, backprop mini-batch
│   └── optimizers/              # gradient_descent, momentum, adam, adaptive_learning_rate (uno por Layer)
├── functions/
│   ├── activation/              # step, linear, non_linear/{tanh, logistic, sigmoid, relu} (+ derivadas)
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
  - Parte A (`run1.sh`, 1000 épocas, batch 32): lineal MSE ≈ 0.0261 / HSE ≈ 97.7 (R² 0.715, ~6% de salidas
    fuera de [0, 1]); logística MSE ≈ 0.0109 / HSE ≈ 40.8 (R² 0.881). Ambas se aplanan independientemente de η
    (saturación de capacidad) muy por encima de 0 (underfitting, mayor en la lineal).
  - `config_multi.json` apunta a datasets inexistentes (y es MLP, pero el ej. 1 pide solo
    perceptrón simple).
  - Parte B (`run1_generalization.sh`, 3 × 5-fold): train MSE ≈ test MSE ≈ 0.0109 (sin
    overfitting). Umbral 0.85: accuracy 0.964, precision 0.815, recall 0.895, F1 0.853 (± ~0.02).
    Máx. F1 medio en umbral 0.89 (F1 0.871). Falta decidir mejor modelo + umbral recomendado (c).
- **Ej. 2/3**:
  - **`digits.csv` no tiene ningún 8 y tiene pocos 5 (271) a propósito** (lo aclaró el equipo): el ej. 2
    debe dar bajo en test y mejorar en el ej. 3 con `more_digits.csv`. Techo en test sin predecir 8s = 0.903.
  - **Salida sigmoide + error cuadrático puede perder dígitos**: una salida se satura en ≈0 al principio
    (σ' ≈ 1e-5) y no se recupera. Con pesos U(−0.5, 0.5) pasa con oculta ReLU (activaciones no acotadas → h
    grande): 5/5 semillas pierden un dígito (0, 3 o 5; test 0.755 ± 0.03); con tanh o logística 0/5. Por eso
    la base usa tanh.
  - Resultados (todo `digits.csv` → test, 40 épocas, 3 semillas, media): mejor η por optimizador: GD 3 (0.856),
    momentum 0.3 (0.856), Adam 0.003 (0.855). Con η más grande fallan: GD η = 10 (0.10), momentum η = 1 (0.54).
    Los tres llegan a ≈0.856: el optimizador cambia la escala de η útil (Adam ~1000× menor que GD) y la
    estabilidad (GD η = 3 tiene picos de pérdida), no el techo. η adaptativo (a = η₀/10, b = 0.5, k = 3): mejor η₀ = 3 (0.854),
    η₀ = 10 falla (0.14); no elimina los picos (reacciona al final de la época, cuando el pico ya ocurrió, y el +a
    vuelve a subir η hasta la zona inestable: η final 2.25 ± 1.31). Arquitectura: 784-10 0.834, 784-16-10 0.822,
    784-64-10 0.852, 784-256-10 0.863 (mejor), 784-256-128-10 y 784-128-64-32-10 0.859, 784-64-32-10 0.849.
    Train ≈ 0.99 desde 64 neuronas. Activación: logística 0.857, tanh 0.849.
  - Mejor (784-256-10, tanh, Adam 1e-3): test 0.863 ± 0.000; sin contar los 8 0.957 (techo 0.903). Recall del 8
    = 0 (≈1/3 → 3), del 5 = 0.84; el resto ≥ 0.94.
  - **Ej. 3** (`run3.sh`, 40 épocas, 3 semillas, test final): mejor = unión + Adam 1e-3 + Xavier + desplazamientos ±1:
    **0.9844 ± 0.0009** (mejor semilla 0.9856, 3/3 ≥ 0.98); GD η = 1 con lo mismo 0.9810 ± 0.0010. GD η = 3 falla con
    logística (0.32 ± 0.31: salidas saturadas; con U(−0.5, 0.5) 0.70); η = 0.3 0.976, η = 0.1 0.944. Adam 3e-4 0.979,
    3e-3 0.977. Ablación (Adam, unión): ninguna 0.965, Xavier 0.969, desplaz. ±1 0.980, ambas 0.984, ±2 0.982,
    + sobremuestreo 0.983 (no ayuda). Datos (Adam, con técnicas): digits 0.884, more_digits 0.979, unión 0.984; sin
    técnicas 0.867 / 0.961 / 0.965. Recall del 8 0.96, del 5 0.97 (resto ≥ 0.98). Train ≈ 0.996: el límite es
    generalización, por eso rinde la augmentation y no la capacidad.
  - Parsear `digits.csv` tarda ~8 s (`ast.literal_eval` por fila); por eso el experimento cachea.
- **Transversal**:
  - `main` imprime la pérdida como "accuracy" y no evalúa validación.
  - Sin semilla, `epsilon` hardcodeado.
  - Falta guardar/cargar modelos y separar registro de experimentos del análisis
    (recomendación del enunciado).
  - Falta citar las fuentes de los algoritmos con `# @link` (convención del `CLAUDE.md` raíz).
