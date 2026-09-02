# agentesAGV — Simulación multiagente de AGVs en un almacén

Servidor **Python** de una simulación multiagente de AGVs (vehículos de guiado automático)
que se mueven dentro de un almacén.

Python es el dueño de **toda** la lógica: la simulación, los agentes y el aprendizaje.
Unity es solo el cliente visual y lo desarrolla otra persona en otro repo.

> En este repo **no** se escribe nada de C# ni de Unity.

Estado actual: **fase 1, andamiaje**. Todavía no hay lógica de simulación; los subcomandos
existen pero solo avisan que no están implementados.

## Contrato PULL

La comunicación con Unity es **PULL**: Unity pide, Python responde. Python nunca empuja datos
por su cuenta y Unity nunca calcula nada, solo dibuja lo que recibe.

1. Unity abre un socket **TCP** contra `127.0.0.1:5000`.
2. Unity envía la línea `GET_STATE\n`.
3. Python responde con **una sola línea** de JSON terminada en `\n`, con el estado completo
   de la simulación en ese momento.

```
Unity  ──────  "GET_STATE\n"  ─────▶  Python
Unity  ◀────  "{...json...}\n"  ────  Python
```

Reglas del contrato:

- Encoding `utf-8`, mensajes delimitados por salto de línea (`\n`).
- La respuesta es **una** línea: el JSON no lleva saltos de línea internos.
- El estado es completo en cada respuesta, no incremental. Unity no guarda historia.
- La forma exacta del JSON se define en la fase del servidor.

Los valores del contrato viven en `python/config.py` (`HOST`, `PORT`, `ENCODING`,
`CMD_GET_STATE`), no sueltos por el código.

## Estructura

```
agentesAGV/
├── python/
│   ├── config.py       constantes (red, ticks, escala de Unity, semilla)
│   ├── logs.py         configuración del logging
│   ├── main.py         CLI con argparse
│   └── models/         AGVs, almacén y Q-Learning (siguientes fases)
├── results/            salidas de las corridas (no se versionan)
├── tests/              tests con unittest
├── requirements.txt
└── README.md
```

## Requisitos

Python **3.10 o superior**. No hay dependencias que instalar: todo es librería estándar.

```bash
python3 --version
```

> En macOS normalmente el comando es `python3`. Si en tu sistema `python` apunta a Python 3,
> puedes usar `python` en todos los ejemplos de abajo.

## Uso

```bash
python3 python/main.py --help
```

| Subcomando | Qué hace |
|---|---|
| `serve` | Levanta el servidor TCP y atiende las peticiones `GET_STATE` de Unity |
| `simulate` | Corre la simulación sin servidor, útil para probar la lógica sola |
| `train` | Entrena los agentes con Q-Learning |
| `evaluate` | Evalúa una política ya entrenada |
| `benchmark` | Mide el rendimiento de la simulación |

```bash
python3 python/main.py serve
python3 python/main.py simulate
python3 python/main.py train
python3 python/main.py evaluate
python3 python/main.py benchmark
```

### Logs

Todo sale por `stderr` con el módulo `logging`, nunca con `print`. Con `--verbose` (o `-v`)
se activa el nivel `DEBUG`. La bandera funciona antes o después del subcomando:

```bash
python3 python/main.py --verbose serve
python3 python/main.py serve --verbose
```

## Tests

```bash
python3 -m unittest discover -s tests -t . -v
```

Cada módulo se puede importar y probar **sin levantar el servidor**.

## Reglas del proyecto

- Python 3.10+, type hints en todas las funciones públicas y docstrings cortos.
- Sin dependencias pesadas: nada de gym, stable-baselines ni torch.
  El Q-Learning se implementa a mano con diccionarios.
- Nada de lógica de negocio dentro de `server.py`: el servidor solo traduce sockets a llamadas.
- Cada módulo debe poder importarse y probarse por separado, sin levantar el servidor.
- Logging con el módulo `logging`, nunca con `print`.
