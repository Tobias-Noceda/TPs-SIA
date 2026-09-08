# TP2 - Genetic algorithms

> [!NOTE]  
> Los comandos son relativos a la carpeta de este archivo (`/tp2`)

## Instalación

```bash
uv sync
```

## Uso

```bash
tp2 -i path -p polygons -n individuals -c cycles --config res/config.json -o res/output/result.json
head -n4 res/output/result.json
visualizer res/output/result.json
```

Los algorítmos e hiperparámetros se editan desde el archivo de configuración.
El json está tipado con un schema para conocer los parámetros válidos.

Imágenes disponibles en `res/images/`
