# agentesAGV — Simulación multiagente de AGVs en un almacén

Servidor **Python** de una simulación multiagente de AGVs (vehículos de guiado automático)
que se mueven dentro de un almacén.

Python es el dueño de **toda** la lógica: la simulación, los agentes y el aprendizaje.
Unity es solo el cliente visual y lo desarrolla otra persona en otro repo.

> En este repo **no** se escribe nada de C# ni de Unity.

Estado actual: **fase 2 terminada**. La comunicación PULL funciona de extremo a extremo, pero
con datos falsos: `serve` levanta el servidor real y responde con un AGV que avanza en línea
recta. La fase 2 añade el **mapa lógico** (`python/graph.py` y `python/maps/`), que es el
espacio que comparten Python y Unity, con el subcomando `map` para verlo y validarlo. Todavía
no hay simulación, ni A\*, ni Q-Learning; `simulate`, `train`, `evaluate` y `benchmark` siguen
avisando que no están implementados.

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
- **Una línea entra, una línea sale.** Siempre, incluso si el comando es desconocido o la línea
  venía vacía: así el cliente nunca pierde el emparejamiento entre lo que pide y lo que recibe.
- La respuesta es **una** línea: el JSON no lleva saltos de línea internos.
- El estado es completo en cada respuesta, no incremental. Unity no guarda historia.
- El comando no distingue mayúsculas de minúsculas y se admite `\r\n`.

### Comandos

| Comando | Qué hace | Respuesta |
|---|---|---|
| `GET_STATE` | Pide el estado actual | El snapshot completo |
| `RESET` | Reinicia la simulación | `{"ok":true}` |
| `PING` | Comprueba que el servidor vive | `{"ok":true}` |

Un comando desconocido **no** cierra la conexión, responde y sigue:

```
-> BASURA\n
<- {"error":"unknown_command","command":"BASURA"}\n
```

### Formato del snapshot

Este formato está **congelado**. En fases futuras solo se le *agregan* campos; los que ya
existen no cambian de nombre ni de tipo.

```json
{"step":1,"agents":[{"id":1,"x":0.25,"y":0.0,"z":0.0,"rotation":0.0,"state":"moving"}]}
```

| Campo | Tipo | Qué es |
|---|---|---|
| `step` | int | Número de paso de la simulación, empieza en 1 |
| `agents[].id` | int | Identificador del AGV |
| `agents[].x/y/z` | float | Posición ya en coordenadas de Unity |
| `agents[].rotation` | float | Giro en grados sobre el eje vertical |
| `agents[].state` | str | Qué está haciendo el AGV |

### Coordenadas

La simulación piensa en un plano `(px, py)`; Unity usa Y como eje vertical. La conversión es
**una sola función**, `protocol.to_unity()`, y no se repite en ningún otro sitio:

```
unity_x = px * UNITY_SCALE
unity_y = 0.0                # la altura la aplica Unity con el prefab
unity_z = py * UNITY_SCALE
```

| Eje de Python | Eje de Unity | Cómo sale |
|---|---|---|
| `px`, el ancho del almacén | `x` | `px * UNITY_SCALE` |
| — | `y`, el vertical | siempre `0.0`: la altura la aplica Unity con el prefab |
| `py`, el fondo del almacén | **`z`** | `py * UNITY_SCALE` |

Lo importante es la última fila: **la Y de Python se convierte en la Z de Unity**, porque en
Unity el eje vertical es la Y y en la simulación no hay altura, solo el plano del suelo.

**La escala.** Una unidad lógica es **un metro** y `UNITY_SCALE` vale **`1.0`**, así que hoy los
números de las coordenadas lógicas y los de Unity coinciden. Cambiar `UNITY_SCALE` en
`config.py` cambia **todas** las coordenadas exportadas, las del snapshot y las del mapa: todo
pasa por `protocol.to_unity()` y en ningún sitio se guarda una copia ya convertida.

Los valores del contrato viven en `python/config.py` (`HOST`, `PORT`, `ENCODING`,
`CMD_GET_STATE`, `CMD_RESET`, `CMD_PING`, `UNITY_SCALE`, `MAPS_DIR`, `DEFAULT_MAP`), no sueltos
por el código.

## Mapa lógico

Python y Unity tienen que hablar del **mismo** sitio, así que el almacén es un grafo: los nodos
son puntos donde un AGV puede estar y las aristas son tramos por los que puede pasar, con su
costo. `python/graph.py` es el dueño del mapa, y `to_unity_dict()` lo exporta con las
coordenadas ya convertidas para que quien monte la escena pueda generarla desde aquí.

```bash
python3 python/main.py map --name warehouse   # el almacén (es el de por defecto)
python3 python/main.py map --name simple      # el grafo de 6 nodos de la guía
```

Imprime la cabecera, los nodos con sus coordenadas lógicas **y** las de Unity, las aristas con
su costo, y el resultado de `validate()`. Sale con código 1 si el mapa no es válido.

### Los dos mapas

`simple` es el grafo de 6 nodos de la guía, para pruebas rápidas.

`warehouse` tiene 13 nodos con forma de pasillos: dos corredores horizontales (`S1`–`S6` al sur,
`N1`–`N6` al norte), cuatro conexiones verticales y un **cuello de botella** en `G`.

```
N1──N2──N3            N4──N5──N6      y = 8
 │       │  ╲        ╱  │       │
 │       │    ▶ G ◀     │       │     y = 4
 │       │  ╱        ╲  │       │
S1──S2──S3            S4──S5──S6      y = 0
 x=0     4   8   12   16   20   24
```

`G` es un **nodo de articulación**: es la única unión entre la zona izquierda y la derecha, así
que toda ruta que cruce el almacén pasa por él a la fuerza y quitarlo parte el grafo en dos. De
ahí salen los escenarios de congestión de las fases siguientes.

> El costo de una arista **no** tiene por qué ser la distancia entre sus nodos. En `simple`,
> `A(0,0) → D(0,3)` mide 3 pero cuesta 4: un pasillo puede ser lento sin ser largo. Por eso
> `validate()` nunca compara el costo con la geometría.

### Editar mapas sin tocar código

Los mapas viven en `python/maps/*.json` y se cargan con `graph.load_graph(ruta)`. El fichero
guarda **solo las coordenadas lógicas**: las de Unity son derivadas y dependen de `UNITY_SCALE`,
así que congelarlas ahí sería guardar una copia condenada a quedarse vieja.

```json
{
  "name": "simple",
  "directed": false,
  "positions": {"A": [0.0, 0.0], "B": [2.0, 0.0]},
  "adjacency": {"A": {"B": 2.0, "D": 4.0}, "B": {"A": 2.0}}
}
```

Si el fichero no existe, `map` tira del mapa que `graph.py` lleva dentro y lo avisa por el log.

### `validate()`

Revienta con un `GraphError` que junta **todos** los problemas en un solo mensaje, en vez de
parar en el primero, para poder arreglar un mapa mal editado de una pasada.

| Comprueba | Qué caza |
|---|---|
| Posiciones | Un nodo sin posición, o una posición de un nodo que no existe |
| Aristas | Que apunten a nodos reales, y que ningún nodo tenga una arista a sí mismo |
| Costos | Nada negativo, ni infinito, ni `NaN` |
| Simetría | En un grafo no dirigido, que cada tramo exista en los dos sentidos y valga lo mismo |
| Conectividad | Que desde cualquier nodo se llegue a todos los demás |

Un grafo se puede declarar `directed=True` para pasillos de un solo sentido: entonces la
asimetría es legítima y lo que se exige es poder **ir y volver** (conectividad fuerte). Los dos
mapas del repo son no dirigidos.

## Estructura

```
agentesAGV/
├── python/
│   ├── config.py       constantes (red, ticks, escala de Unity, semilla)
│   ├── logs.py         configuración del logging
│   ├── protocol.py     el contrato: comandos, serialización y coordenadas
│   ├── server.py       servidor TCP y la simulación falsa de la fase 1
│   ├── graph.py        el mapa lógico: grafo, validación y carga desde JSON
│   ├── main.py         CLI con argparse
│   ├── maps/           los mapas en JSON (simple.json, warehouse.json)
│   └── models/         AGVs, almacén y Q-Learning (siguientes fases)
├── results/            salidas de las corridas (no se versionan)
├── tests/              tests con unittest, y el cliente falso de Unity
├── requirements.txt
└── README.md
```

El servidor recibe la simulación por **inyección de dependencia**: `serve_forever()` acepta
cualquier objeto con `get_snapshot()` y `reset()` (el `Protocol` está declarado en
`protocol.Simulation`). En la fase 1 se le pasa `server.FakeSimulation`; cambiarla por la
simulación de verdad es una línea de `main.py`.

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
| `serve` | Levanta el servidor TCP y atiende las peticiones de Unity |
| `map` | Muestra el mapa lógico del almacén y lo valida |
| `simulate` | Corre la simulación sin servidor, útil para probar la lógica sola |
| `train` | Entrena los agentes con Q-Learning |
| `evaluate` | Evalúa una política ya entrenada |
| `benchmark` | Mide el rendimiento de la simulación |

```bash
python3 python/main.py serve                          # 127.0.0.1:5000
python3 python/main.py serve --port 5055              # otro puerto
python3 python/main.py serve --host 0.0.0.0 --port 5055
```

`Ctrl+C` cierra limpio, y también un `kill` (SIGTERM). Con un cliente conectado tarda unos
milisegundos: los hilos de los clientes son *daemon*, no bloquean la salida.

> **macOS y el puerto 5000.** El receptor de AirPlay se queda con `*:5000`. El servidor
> igual consigue abrir `127.0.0.1:5000` porque es una dirección más específica, pero si algo
> se comporta raro, apágalo en Ajustes → General → AirDrop y Handoff → Receptor de AirPlay, o
> usa `--port`.

### Logs

Todo sale por `stderr` con el módulo `logging`, nunca con `print`. Con `--verbose` (o `-v`)
se activa el nivel `DEBUG`, que en el servidor imprime cada petición con su respuesta. La
bandera funciona antes o después del subcomando:

```bash
python3 python/main.py --verbose serve
python3 python/main.py serve --verbose
```

## Tests

```bash
python3 -m unittest discover -s tests -t . -v
```

### Cliente falso de Unity

`tests/fake_unity_client.py` hace de Unity mientras Unity no existe: se conecta, pide
`GET_STATE` a un ritmo fijo, muestra lo que recibe, valida que cada respuesta sea JSON con la
forma del contrato y comprueba que `step` va creciendo. Sale con código 1 si algo falla.

```bash
python3 python/main.py serve --port 5055 &
python3 tests/fake_unity_client.py --port 5055 --seconds 60 --rate 10
python3 tests/fake_unity_client.py --port 5055 --seconds 3 -v   # muestra cada respuesta
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--host` / `--port` | los de `config.py` | Contra qué servidor |
| `--seconds` | `10` | Cuánto dura la corrida |
| `--rate` | `config.TICK_RATE` (10) | Peticiones por segundo |
| `--label` | vacío | Distingue varios clientes a la vez en el log |
| `-v` | apagado | Muestra todas las respuestas, no una por segundo |

Al terminar imprime un resumen con las peticiones enviadas, los errores de JSON, de forma y de
red, y las latencias mín/media/p95/máx.

## Reglas del proyecto

- Python 3.10+, type hints en todas las funciones públicas y docstrings cortos.
- Sin dependencias pesadas: nada de gym, stable-baselines ni torch.
  El Q-Learning se implementa a mano con diccionarios.
- Nada de lógica de negocio dentro de `server.py`: el servidor solo traduce sockets a llamadas.
  `FakeSimulation` es la excepción temporal de la fase 1 y desaparece cuando llegue la
  simulación de verdad en `python/models/`.
- Cada módulo debe poder importarse y probarse por separado, sin levantar el servidor.
- Logging con el módulo `logging`, nunca con `print`.
