# agentesAGV — Simulación multiagente de AGVs en un almacén

Servidor **Python** de una simulación multiagente de AGVs (vehículos de guiado automático)
que se mueven dentro de un almacén.

Python es el dueño de **toda** la lógica: la simulación, los agentes y el aprendizaje.
Unity es solo el cliente visual y lo desarrolla otra persona en otro repo.

> En este repo **no** se escribe nada de C# ni de Unity.

Estado actual: **fase 5 terminada**. Los AGVs ya no se atraviesan: la fase 5 añade la detección
de conflictos y la política base (`python/conflicts.py`), parte el tick en dos fases y saca los
números de cada corrida en `snapshot["stats"]`. La política es **intercambiable**, así que el
Q-Learning de la fase 8 entra sin tocar el motor.

Antes: la fase 3 trajo el pathfinding (`python/astar.py`), el agente (`python/agent.py`) y la
simulación (`python/simulation.py`), y estrenó el subcomando `simulate`. La `FakeSimulation` de
la fase 1 ha desaparecido. Todavía no hay aprendizaje: `train`, `evaluate` y `benchmark` siguen
avisando que no están implementados.

> El baseline **no** está para funcionar bien. Está para funcionar siempre igual y dejar números
> que medir: gana el AGV de id menor y punto, así que se atasca en el cuello de botella. Sin esa
> referencia no habría con qué comparar el Q-Learning después.

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
{"step":10,"agents":[{"id":1,"x":9.4,"y":0.0,"z":1.4,"rotation":45.0,"state":"moving",
 "node":"S3","next_node":"G","path":["S1","S2","S3","G","N4","N5","N6"],"task":1}]}
```

(en el cable va en **una sola línea**; aquí está partido para que se lea)

| Campo | Tipo | Desde | Qué es |
|---|---|---|---|
| `step` | int | fase 1 | Número de paso de la simulación, empieza en 1 |
| `agents[].id` | int | fase 1 | Identificador del AGV |
| `agents[].x/y/z` | float | fase 1 | Posición ya en coordenadas de Unity |
| `agents[].rotation` | float | fase 1 | Giro en grados sobre el eje vertical |
| `agents[].state` | str | fase 1 | `idle`, `moving`, `waiting` o `done` |
| `agents[].node` | str | fase 3 | Nodo en el que está, o del que acaba de salir |
| `agents[].next_node` | str \| null | fase 3 | Hacia dónde va ahora, `null` si ya llegó |
| `agents[].path` | list[str] | fase 3 | La ruta entera, para poder pintarla |
| `agents[].task` | int \| null | fase 3 | Id de la tarea que lleva |
| `agents[].wait_time` | int | fase 5 | Ticks **acumulados** que lleva cediendo el paso |
| `stats` | object | fase 5 | Los números de la corrida, ver abajo |

La **posición va interpolada** entre `node` y `next_node`: un AGV a mitad de un tramo manda la
mitad de camino, no el nodo de destino. Así Unity puede mover el prefab sin teletransportes.

> Los campos de las fases 3 y 5 son *añadidos*: los de la fase 1 conservan nombre y tipo, y
> `JsonUtility` de Unity ignora lo que no conoce, así que un cliente de la fase 1 sigue
> funcionando sin tocarle una línea.

#### `stats`

| Campo | Tipo | Qué es |
|---|---|---|
| `run` | int | Número de corrida; sube en cada `reset()` |
| `policy` | str | La política activa (`baseline` por ahora) |
| `conflicts` | int | Conflictos detectados en esta corrida |
| `conflicts_by_type` | object | Desglose: `vertex`, `edge`, `following`, `congestion` |
| `deadlocks` | int | Atascos de la **sesión**; no se borra al reiniciar |
| `waiting` | int | Cuántos AGVs están cediendo el paso ahora mismo |
| `total_wait_time` | int | Suma del `wait_time` de todos |
| `finished_reason` | str \| null | `"deadlock"` si la corrida murió atascada |

Cuando una corrida muere en deadlock, el servidor entrega **una vez** el snapshot del atasco
(con `finished_reason` puesto) y en la petición siguiente arranca otra corrida: `step` vuelve a 1
y `run` sube. Es la única situación en la que `step` no crece, y se reconoce por `run`.

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

## Rutas con A\*

`python/astar.py` calcula la ruta más barata entre dos nodos. `f(n) = g(n) + h(n)`, con `g` el
costo real acumulado y `h` la distancia euclidiana entre posiciones. El heap desempata por
**nombre de nodo**, así que dos corridas sobre el mismo mapa devuelven siempre la misma ruta.

```python
astar.astar(grafo, "A", "F")            # ['A', 'B', 'E', 'F']
astar.path_cost(grafo, ruta)            # 7.0
astar.astar(grafo, "A", "Atlantida")    # None, nunca una excepción
```

**El factor de la heurística.** A\* solo garantiza la ruta óptima si `h` nunca sobreestima lo que
falta, y aquí `h` es geometría mientras que el costo no lo es: nada impide un mapa donde un tramo
cueste *menos* que su longitud. Por eso `h` se escala por
`factor = min(1.0, min(costo/longitud))` sobre todas las aristas: así `costo(u,v) >= factor *
dist(u,v)` para cada tramo y, por desigualdad triangular, `h` nunca pasa del costo real. En los
dos mapas del repo el factor sale **1.0**, así que hoy `h` es la euclidiana tal cual.

**Penalizaciones.** El tercer argumento encarece nodos y tramos sin tocar el mapa, que es el
gancho del REROUTE de fases posteriores. Como solo suman, la heurística sigue siendo admisible.

```python
astar.astar(grafo, "A", "F", {"E": 5.0})           # penaliza entrar en E
astar.astar(grafo, "A", "F", {("B", "E"): 5.0})    # penaliza cruzar ese tramo
```

Al nodo de partida no se le cobra penalización: el AGV ya estaba ahí, no *entra* en él. En un
grafo no dirigido `(a, b)` y `(b, a)` son el mismo tramo; con `directed=True` no.

## Los agentes y la simulación

Cada `Agent` es dueño de **su** ruta: `assign_task()` se queda con una copia de lo que devuelve
A\*, nunca con la misma lista que otro agente. Sus campos (`id`, `current_node`, `target_node`,
`path`, `path_index`, `state`, `wait_time`, `task`, `progress`) son los que salen en el snapshot.

`Simulation` mueve a todos en cada `tick()`. Un AGV tarda **`cost(a,b)` ticks** en cruzar un
tramo, y su `progress` avanza `1/cost` por tick: un tramo de costo 4 se cruza en 4 ticks y uno de
5.7 en 6, porque el tick que pasa del 1.0 es el que llega.

`get_snapshot()` **avanza un paso** y devuelve el estado, igual que en la fase 1: la petición es
la que mueve el mundo, Python nunca empuja nada por su cuenta. `reset()` vuelve al paso cero de
forma determinista con `config.RANDOM_SEED`, así que dos corridas dan exactamente lo mismo.

Sin ruta posible el agente se queda `idle` y la simulación sigue: que dos zonas del almacén estén
incomunicadas es un estado normal del mapa, no un error del programa.

## Conflictos y política base

### El tick va en dos fases

Las dos dentro del **mismo** paso: declarar la intención no puede costar un tick extra, o un AGV
solo tardaría el doble en cruzar el almacén y las medidas de la fase 3 dejarían de valer.

```
FASE A   cada AGV parado en un nodo declara a cuál quiere entrar
FASE B   se detectan los conflictos -> la política decide quién cede -> se mueve
```

El conflicto se ve **antes** de mover a nadie: la detección trabaja sobre las intenciones y sobre
la ocupación tal como estaban al empezar el tick, así que el perdedor de un choque termina el
paso exactamente donde empezó.

### Reserva doble: un nodo, un AGV

El movimiento es continuo, así que a media travesía el `current_node` de un AGV sigue siendo el
nodo del que salió. Por eso el que cruza `X → Y` **retiene los dos** y suelta `X` solo al llegar:

```
tick 3:   AGV 1  ------>------   progreso 0.4
          X                 Y
   occupancy:  X -> 1,  Y -> 1

   AGV 2 quiere entrar en X  ->  FOLLOWING CONFLICT  ->  waiting
```

`occupancy` es `nodo -> un solo agent_id`, y esa es la invariante del almacén. Lo contrario sí
vale: un AGV puede tener dos nodos, un nodo nunca tiene dos AGVs.

La consecuencia es que el **following está prohibido**: nadie entra en el nodo que otro está
dejando hasta que lo suelta. Es una decisión, no un descuido. Cuesta throughput y provoca
deadlocks, pero deja la invariante comprobable directamente sobre `current_node`, sin trucos, y
un baseline que se atasca es exactamente lo que hace falta para que la fase 8 tenga qué mejorar.

### Los cuatro conflictos

| Tipo | Qué es |
|---|---|
| `vertex` | Dos o más AGVs quieren el mismo nodo. El que ya está encima cuenta como uno más |
| `edge` | Cruce de frente: A va de X a Y mientras B va de Y a X |
| `following` | A quiere entrar en el nodo que B está dejando. No se permite |
| `congestion` | Un AGV pasa de `CONFLICT_WAIT_THRESHOLD` esperando, o hay `CONGESTION_ZONE_AGENTS` esperando en una zona (un nodo y sus vecinos) |

La congestión se cuenta **solo en el tick en que se cruza el umbral**. Un atasco de cincuenta
ticks es un conflicto, no cincuenta, o el número dejaría de significar algo. Los otros tres sí se
cuentan cada tick: mientras el choque siga ahí, sigue siendo un choque.

### La política

`resolve_baseline()` es toda la inteligencia que hay: **gana el AGV de id menor**, el resto pasa a
`waiting` y suma un tick a su `wait_time`. Es pura — dice quién gana, no toca a nadie; quien
aplica el cambio de estado es el motor.

Por debajo de la política hay un **gate físico**: diga lo que diga, en un nodo ocupado no se
entra. Eso hace la invariante inviolable venga la política que venga, incluida la que aprenda la
fase 8.

En un cruce de frente el baseline se ve tal como es: nombra ganador al de id menor, pero el
perdedor sigue sentado en el nodo destino, así que el gate frena **también al ganador** y los dos
se quedan esperando hasta el deadlock. El baseline no sabe deshacer eso. Para eso está el
Q-Learning.

`Simulation` recibe la política por el constructor, igual que el servidor recibe la simulación:

```python
simulacion = Simulation(grafo, 6, policy=MiPolitica())
```

Cualquier objeto con `name` y `decide(agent, local_state) -> "go" | "wait"` vale
(`conflicts.Policy`). `local_state` es deliberadamente **local**: el nodo en el que está, a dónde
quiere ir, lo que lleva esperando y quién le ganó este tick. Ni el mapa entero ni las rutas de los
demás; si pudiera mirarlo todo aprendería una política centralizada, que es otro problema.

### Deadlock

Si en `config.DEADLOCK_TICKS` ticks seguidos no avanza **ningún** AGV activo, la corrida se marca
como muerta (`finished_reason = "deadlock"`), se cuenta en `stats.deadlocks` y `simulate` para.
Que hayan llegado todos no es un atasco: sin AGVs activos no hay deadlock.

Una simulación colgada para siempre no es un resultado experimental, es un bug de la corrida.

## El entorno de Q-Learning

Fase 6: aquí se **define** el problema de aprendizaje; entrenarlo es la fase 7. Todo vive en
`python/qlearning.py`, y los números que se ajustan, en `config.py`.

### Q-Learning no sustituye a A\*

El pathfinding lo sigue resolviendo A\*: quién dice por dónde se va de `S1` a `N6` es
`astar.astar()`, igual que en la fase 3. Lo que se aprende es mucho más chico: **qué hacer AHORA**
cuando la ruta que ya tengo me mete en un conflicto.

Si el estado fuera la ruta entera, el espacio explotaría. En `warehouse` hay 13 nodos, y solo las
posiciones de 6 AGVs ya son 13⁶ = 4.826.809 estados, sin contar rutas ni destinos. Con el estado
local son **72**.

### El estado: cinco enteros, discreto y local

`get_local_state(agent, simulation) -> tuple` devuelve **siempre** una tupla de cinco enteros,
hasheable y con cada campo en su rango: es la clave de la Q-table.

| Campo | Valores | Qué pregunta |
|---|---|---|
| `next_node_occupied` | 0/1 | ¿hay alguien en el nodo al que voy a entrar? |
| `edge_conflict` | 0/1 | ¿alguien viene de frente por mi siguiente arista? |
| `queue_ahead` | 0/1/2 | ¿cuántos AGVs esperan en mis 2 nodos siguientes? (saturado en 2) |
| `distance_bucket` | 0/1/2 | ¿cuánto me falta? cerca / medio / lejos |
| `has_priority` | 0/1 | ¿soy el id menor de los que estamos en conflicto? |

```
2 × 2 × 3 × 3 × 2 = 72 estados × 3 acciones = 216 celdas de Q(s, a)
```

```bash
python3 python/qlearning.py      # imprime el desglose por el log
```

`distance_bucket` cuenta **nodos que faltan de la ruta**, nunca distancia euclidiana: en un almacén
dos nodos pueden estar pegados y tener medio pasillo de por medio, así que la geometría mentiría
sobre lo que falta de verdad. Los cortes son `DISTANCE_NEAR_NODES` y `DISTANCE_MID_NODES`.

Ni coordenadas continuas, ni el mapa completo, ni las rutas de los demás: cinco preguntas sobre lo
que este AGV tiene delante.

### Las acciones

| Acción | Qué hace |
|---|---|
| `ADVANCE` | Avanzar al siguiente nodo del path que trazó A\* |
| `WAIT` | Quedarse un tick |
| `REROUTE` | Recalcular A\* penalizando el nodo/tramo congestionado (`astar.Penalties`) |

Ninguna acción elige un nodo: la ruta la traza A\*. `REROUTE` **no mueve** al AGV en el mismo tick
—cuando la política decide, el motor ya fijó la intención en la fase A—, así que la ruta nueva
entra en vigor en el siguiente y hacia el motor un `REROUTE` se traduce a `wait`. Tampoco pasa por
`Agent.assign_task()`, que reiniciaría `wait_time`: solo toca `path`, `path_index` y `progress`.

`config.ENABLE_REROUTE` decide si la política puede **elegir** `REROUTE`; con el flag apagado
quedan `ADVANCE` y `WAIT`, por si la fase 7 converge antes con dos acciones. El flag no cambia la
Q-table, que guarda siempre las tres: encenderlo después no obliga a migrar ningún fichero.

### La recompensa

Una sola función, `reward(event)`, y los seis números en `config.py` para poder ajustarlos sin
buscar por el código:

| Evento | Valor | Cuándo |
|---|---|---|
| `TASK_COMPLETE` | +100 | el AGV llegó a su destino |
| `PROGRESS` | +2 | **`path_index` subió**, no que se moviera por el mapa |
| `WAIT` | -1 | se quedó un tick parado |
| `CONFLICT` | -20 | intentó entrar donde había choque |
| `DEADLOCK` | -50 | la corrida murió atascada |
| `USELESS_REROUTE` | -3 | recalculó **sin salir más barato y sin esquivar un conflicto real** |

Un evento mal escrito lanza `ValueError` en vez de devolver 0.0: un premio invisible se busca
durante días. Lo de `USELESS_REROUTE` lo decide `is_useless_reroute()`, que compara por costo
(`astar.path_cost`), no por número de nodos.

### La Q-table

`dict[tuple, dict[Action, float]]` con `defaultdict`: un estado nuevo nace con sus acciones a cero,
así que la fase 7 puede preguntar por cualquier estado sin comprobar antes si existe. `save(path)` y
`load(path)` en JSON:

```json
{
  "format": "agv-qtable/1",
  "state_fields": ["next_node_occupied", "edge_conflict", "queue_ahead",
                   "distance_bucket", "has_priority"],
  "actions": ["advance", "wait", "reroute"],
  "q": { "0|1|2|1|0": {"advance": 1.5, "wait": -0.25, "reroute": 0.0} }
}
```

La clave es la tupla de estado con los campos en el orden de `state_fields`, separados por `|`
(JSON no admite tuplas como clave, y así se lee de un vistazo). `state_fields` va escrito en el
fichero a propósito: sin él, una tabla guardada hoy y leída después de reordenar los campos
seguiría cargando, y aprendería sobre estados equivocados sin avisar.

### La política

`QLearningPolicy` cumple el mismo contrato que la baseline (`name` + `decide(agent, local_state)`),
así que entra por el constructor **sin tocar `simulation.py`**:

```python
politica = qlearning.QLearningPolicy()
simulacion = Simulation(grafo, 6, policy=politica)
politica.bind(simulacion)      # para que vea el estado completo
```

En la fase 6 no aprende: elige la mejor acción de una tabla que está a ceros, o sea que siempre
avanza (con `epsilon > 0` explora, con un generador sembrado para que la corrida siga siendo
reproducible). Lo que se demuestra ahora no es lo que decide, es que se enchufa. Sin `bind()`
sigue funcionando, pero saca el estado del `LocalState` del motor y ahí `queue_ahead` es una
aproximación; avisa una vez por el log.

## Estructura

```
agentesAGV/
├── python/
│   ├── config.py       constantes (red, ticks, Unity, semilla, umbrales)
│   ├── logs.py         configuración del logging
│   ├── protocol.py     el contrato: comandos, serialización y coordenadas
│   ├── server.py       servidor TCP, solo transporte
│   ├── graph.py        el mapa lógico: grafo, validación y carga desde JSON
│   ├── astar.py        pathfinding con A\* y penalizaciones
│   ├── agent.py        el AGV: ruta, estado y tarea
│   ├── conflicts.py    conflictos, ocupación y la política base
│   ├── simulation.py   el almacén en marcha: agentes, ticks y snapshot
│   ├── qlearning.py    el entorno de Q-Learning: estado, acciones y recompensa
│   ├── main.py         CLI con argparse
│   ├── maps/           los mapas en JSON (simple.json, warehouse.json)
│   └── models/         Q-Learning (siguientes fases)
├── results/            salidas de las corridas (no se versionan)
├── tests/              tests con unittest, y el cliente falso de Unity
├── requirements.txt
└── README.md
```

El servidor recibe la simulación por **inyección de dependencia**: `serve_forever()` acepta
cualquier objeto con `get_snapshot()` y `reset()` (el `Protocol` está declarado en
`protocol.Simulation`). Desde la fase 3 se le pasa una `simulation.Simulation`, y en `server.py`
no queda ni una línea de lógica del almacén.

## Requisitos

Python **3.10 o superior**. No hay dependencias que instalar: todo es librería estándar.

```bash
python3 --version
```

> En macOS normalmente el comando es `python3`. Si en tu sistema `python` apunta a Python 3,
> puedes usar `python` en todos los ejemplos de abajo.

> **Ojo con el Python del sistema en macOS.** El `/usr/bin/python3` que trae macOS es 3.9 y **no
> vale**: el proyecto usa `X | Y` en las anotaciones, que es 3.10+. Con `brew install python@3.12`
> tendrás uno en `/opt/homebrew/bin/python3.12`, y ese es el que hay que usar en todos los
> comandos de abajo si tu `python3` sigue siendo el del sistema.

## Uso

```bash
python3 python/main.py --help
```

| Subcomando | Qué hace |
|---|---|
| `serve` | Levanta el servidor TCP y atiende las peticiones de Unity |
| `map` | Muestra el mapa lógico del almacén y lo valida |
| `simulate` | Corre la simulación sin servidor e imprime paso a paso qué hace el AGV |
| `train` | Entrena los agentes con Q-Learning |
| `evaluate` | Evalúa una política ya entrenada |
| `benchmark` | Mide el rendimiento de la simulación |

```bash
python3 python/main.py serve                          # 127.0.0.1:5000
python3 python/main.py serve --port 5055              # otro puerto
python3 python/main.py serve --host 0.0.0.0 --port 5055
python3 python/main.py serve --map simple             # sirve el otro mapa
python3 python/main.py serve --agents 6               # con tráfico, para ver conflictos
```

Con `--agents 1` (el defecto) no hay con quién chocar y `stats.conflicts` sale siempre 0. Para
ver la fase 5 en marcha hacen falta varios AGVs.

### `simulate`

Corre la simulación **sin servidor** y cuenta por el log lo que hace el AGV en cada paso: útil
para probar la lógica sola, sin Unity y sin sockets.

```bash
python3 python/main.py simulate --map warehouse --agents 1 --steps 100 --headless
python3 python/main.py simulate --map simple --from A --to F --headless
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--map` | `config.DEFAULT_MAP` | Mapa por el que moverse |
| `--agents` | `1` | Cuántos AGVs correr; no puede haber más que nodos en el mapa |
| `--steps` | `100` | Tope de pasos; corta antes si llegan todos |
| `--headless` | apagado | Corre sin servidor, que es el único modo por ahora |
| `--from` / `--to` | la ruta del mapa | Origen y destino (`simple`: `A→F`; `warehouse`: `S1→N6`) |

```
AGV 1: S1 -> N6 | costo 27.4 | S1 -> S2 -> S3 -> G -> N4 -> N5 -> N6
paso   1 | AGV 1 | moving  | S1   -> S2   |  25% | tramo 0/6 | espera   0 | tarea 1
...
paso  28 | AGV 1 | done    | N6   -> -    |   0% | tramo 6/6 | espera   0 | tarea 1
```

Con varios AGVs salen además los conflictos y el resumen de la corrida:

```
paso  19 | CONFLICTO edge       | AGV 4, 6 | G <-> N3
paso  20 | CONFLICTO vertex     | AGV 1, 2, 4, 5, 6 | G
deadlock en el paso 28: 20 ticks seguidos sin que avance nadie
--- resumen ---
final       : deadlock, nadie avanzo en 20 ticks seguidos
conflictos  : 61 (vertex 28, edge 28, following 0, congestion 5)
espera total: 128 ticks entre todos
AGV 4      : waiting en G, tramo 0/3, 28 ticks esperando
```

Sale con `0` si la corrida fue bien, `1` si algún AGV se quedó sin ruta y `2` si el mapa o los
nodos que le pasaste no existen. Un **deadlock sale con `0`**: que el baseline se atasque es un
resultado experimental válido, no un fallo del programa. La razón del final va en el resumen.

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
python3 -m unittest tests.test_astar -v          # solo la fase 3
python3 -m unittest tests.test_conflicts -v      # solo la fase 5
python3 -m unittest tests.test_qlearning -v      # solo la fase 6
```

| Fichero | Qué cubre |
|---|---|
| `test_config.py` | Las constantes del proyecto |
| `test_logs.py` | La configuración del logging |
| `test_protocol.py` | El contrato: comandos, serialización y coordenadas |
| `test_server.py` | El servidor TCP contra un socket de verdad |
| `test_graph.py` | El mapa lógico: grafo, `validate()` y ficheros |
| `test_main.py` | El CLI |
| `test_astar.py` | La fase 3: A\*, penalizaciones, `Agent`, `Simulation` y el snapshot |
| `test_conflicts.py` | La fase 5: conflictos, política base, invariante y deadlock |
| `test_qlearning.py` | La fase 6: estado, acciones, recompensa, Q-table y la política |

`test_astar.py` compara A\* contra una **búsqueda exhaustiva** en los dos mapas (los 186 pares
ordenados de nodos), comprueba que cada par consecutivo de una ruta es una arista de verdad, y
valida el snapshot con el mismo `validar_snapshot()` que usa el cliente falso de Unity.

`test_qlearning.py` comprueba que el espacio de estados son 72 (y falla si pasa de 500), que
`get_local_state()` devuelve la misma tupla de cinco enteros en 200 ticks con 6 AGVs, y que la
política se intercambia con la baseline sin tocar el motor.

`test_conflicts.py` monta a propósito un cruce de frente y demuestra que se detecta como
`edge conflict`, y corre **500 ticks con 6 AGVs** comprobando en cada tick que no hay dos en el
mismo nodo. También prueba que una política temeraria que siempre dice `go` sigue sin poder
romper la invariante: para eso está el gate físico.

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
  La simulación entra por inyección de dependencia y vive en `python/simulation.py`.
- Cada módulo debe poder importarse y probarse por separado, sin levantar el servidor.
- Logging con el módulo `logging`, nunca con `print`.
