# agentesAGV — Simulación multiagente de AGVs en un almacén

Servidor **Python** de una simulación multiagente de AGVs (vehículos de guiado automático)
que se mueven dentro de un almacén.

Python es el dueño de **toda** la lógica: la simulación, los agentes y el aprendizaje.
Unity es solo el cliente visual y lo desarrolla otra persona en otro repo.

> En este repo **no** se escribe nada de C# ni de Unity.

Estado actual: **fase 9 terminada**. Ya no hay que fiarse de la intuición: `python/metrics.py` y
`main.py benchmark` **miden** las dos políticas semilla a semilla bajo condiciones idénticas y
escriben `results/baseline.csv`, `results/qlearning.csv` y `results/comparison.json`. El resultado
real, con sus matices, está en [La fase 9](#la-fase-9-medir-en-vez-de-suponer) — y no es un triunfo
limpio del Q-Learning.

Antes: en la **fase 8** las dos piezas empezaron a correr juntas en el mismo tick: **A\* dice
por dónde y el Q-Learning dice qué conviene hacer ahora**. `main.py serve --policy qlearning`
levanta el almacén con la Q-table entrenada, la acción de cada AGV (`advance` / `wait` /
`reroute`) sale en el snapshot para que Unity la pinte, y `SET_MODE` cambia de política en caliente
sin reiniciar nada. El motor sigue siendo la autoridad —una acción es una intención, no una
garantía— y ahora además desatasca: **cero deadlocks** en las dos políticas, donde antes moría el
91 % de los episodios del baseline. Los detalles, en [La fase 8](#la-fase-8-a-dice-por-dónde-q-learning-dice-qué-hacer-ahora).
La fase 7 puso el bucle que entrena la Q-table (`python/main.py train`, mil episodios sin
servidor y sin Unity) sobre el entorno que definió la fase 6 (estado local de 72 valores, tres
acciones y la recompensa). Los números están en [El entrenamiento](#el-entrenamiento). La fase 5 añadió la detección de conflictos y la política base (`python/conflicts.py`),
partió el tick en dos fases y sacó los números de cada corrida en `snapshot["stats"]`; la política
es **intercambiable**, así que el Q-Learning entró sin tocar el motor. La fase 3 trajo el
pathfinding (`python/astar.py`), el agente (`python/agent.py`) y la simulación
(`python/simulation.py`), y estrenó el subcomando `simulate`. Ya no queda andamiaje: los seis
subcomandos hacen algo.

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
| `SET_MODE baseline\|qlearning` | Cambia de política **en caliente** y arranca una corrida limpia | `{"ok":true,"mode":"qlearning","run":3}` |

`SET_MODE` es el único comando con argumento, y ninguno de sus finales cierra la conexión:

```
-> SET_MODE turbo\n
<- {"error":"bad_mode","command":"SET_MODE","mode":"turbo","modes":["baseline","qlearning"]}\n
```

| Error | Cuándo |
|---|---|
| `bad_mode` | El modo no existe, o no venía ninguno |
| `set_mode_failed` | El modo existe pero no se pudo montar (falta la Q-table, típicamente) |
| `mode_not_supported` | Esta simulación no sabe cambiar de política |

Cambiar de modo **siempre reinicia**, aunque el modo pedido sea el que ya estaba: media corrida con
una política y media con otra no es una corrida de ninguna de las dos, y sus números no dirían
nada. Sube `run`, `step` vuelve a 1 y lo único que sobrevive es `stats.deadlocks`, que cuenta los
de la sesión.

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
| `agents[].action` | str | fase 8 | Lo que **eligió** hacer: `advance`, `wait` o `reroute` |
| `agents[].blocked` | bool | fase 8 | Eligió `advance` y el motor **no le dejó** pasar |
| `mode` | str | fase 8 | La política activa: `baseline` o `qlearning` |

`action` y `blocked` van juntos a propósito: uno es lo que el AGV **quiso** y el otro lo que el
motor **le concedió**. Que puedan no coincidir es la fase 8 entera. Un AGV a media travesía está
ejecutando un avance, así que su acción es `advance`; el que ya llegó o no tiene ruta sale como
`wait`.

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
| `actions` | object | Decisiones de la corrida por tipo: `{"advance":n,"wait":n,"reroute":n}` |
| `forced` | int | Veces que el motor tuvo que desatascar a la fuerza |
| `penalties` | int | Penalizaciones de ruta vivas ahora mismo |

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
python3 python/main.py map --name grid        # la rejilla 4×4 de la fase 10
```

Imprime la cabecera, los nodos con sus coordenadas lógicas **y** las de Unity, las aristas con
su costo, y el resultado de `validate()`. Sale con código 1 si el mapa no es válido.

### Los tres mapas

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

`grid` (fase 10) es lo contrario exacto: una rejilla 4×4 de 16 nodos y 24 tramos, **todos del
mismo coste**, sin un solo punto de articulación.

```
A4──B4──C4──D4      y = 12
 │   │   │   │
A3──B3──C3──D3      y = 8
 │   │   │   │
A2──B2──C2──D2      y = 4
 │   │   │   │
A1──B1──C1──D1      y = 0
x=0  4   8   12
```

Entre dos nodos cualesquiera hay **varias rutas de coste mínimo**, así que penalizar uno le deja a
A\* una alternativa igual de buena en vez de un rodeo. Es la condición para que el `REROUTE` del
Q-Learning pueda aportar algo, y sin ese mapa no habría con qué comparar el cuello de botella del
`warehouse`: es el par D/E de [la fase 10](#la-fase-10-cinco-escenarios-y-dónde-el-q-learning-sí-sirve).

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

Elige la mejor acción de la Q-table con epsilon-greedy, con un generador sembrado para que la
corrida siga siendo reproducible. Con la tabla recién creada todo empata a cero y siempre avanza;
lo que la llena es el entrenamiento de la fase 7. Sin `bind()` sigue funcionando, pero saca el
estado del `LocalState` del motor y ahí `queue_ahead` es una aproximación; avisa una vez por el
log, y **para entrenar hay que atarla**.

## El entrenamiento

Fase 7: el bucle que llena la Q-table. Vive en `python/qlearning.py`, debajo del entorno, y se
lanza desde `main.py`.

```bash
python3 python/main.py train    --map warehouse --agents 4 --episodes 1000 --seed 42
python3 python/main.py evaluate --map warehouse --agents 4 --model python/models/q_table.json
```

**Sin servidor y sin Unity.** Mil episodios son ~130.000 ticks: meter un socket en medio
multiplicaría el tiempo por el ping y no le daría al algoritmo ni un dato más de los que ya tiene.
Unity entra después, con `serve`, a ver correr lo aprendido. La corrida entera tarda unos 15 s.

### La actualización

```
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
```

Es `QTable.update()`, y `Trainer` es quien la llama. `terminal=True` pone el término del futuro a
cero: cuando el AGV ya llegó, detrás no hay nada que valorar, y sin eso el +100 de la llegada se
contaría dos veces. Un episodio cortado por el tope de ticks **no** es terminal: el mundo seguía,
el que se acabó fue el reloj del experimento.

| Parámetro | `config.py` | Valor | Por qué |
|---|---|---|---|
| `alpha` | `ALPHA` | 0.2 | Cuánto pesa lo nuevo frente a lo que ya sabía |
| `gamma` | `GAMMA` | 0.95 | Casi 1: el +100 está al final de la ruta, no en el tick siguiente |
| `epsilon` | `EPSILON_START` → `EPSILON_END` | 1.0 → 0.05 | Exponencial, `eps <- max(END, eps * DECAY)` |
| `EPSILON_DECAY` | | 0.995 | Toca el suelo sobre el episodio 600 |
| `EPISODES` | | 1000 | |
| `MAX_STEPS_PER_EPISODE` | | 200 | Un tramo cuesta 4-8 ticks y una ruta entera ~30 |

El decaimiento es exponencial y no lineal porque hace falta explorar mucho al principio, con la
tabla a ceros, y cada vez menos después. El suelo no se quita nunca: una política que deja de
explorar del todo no vuelve a corregir un estado que aprendió mal.

### Una sola Q-table para todos los AGVs

Los N agentes comparten **la misma tabla**: todos leen de ella y todos escriben en ella (política
homogénea). No es un atajo:

- **Un AGV es intercambiable con otro.** El estado es local y no lleva el id dentro
  (`has_priority` dice si soy el menor, no quién soy), así que lo que aprende el AGV 3 sobre
  "hay alguien delante y no tengo prioridad" vale igual para el 1.
- **Cada episodio produce N veces más experiencia.** Con 4 agentes la tabla ve ~4× transiciones
  por episodio que con políticas separadas, y son 72 estados: se llenan en decenas de episodios
  en vez de en miles. La corrida de 1000 episodios hace **111.079 actualizaciones** sobre
  31 estados.
- **Cambiar el número de AGVs no invalida el modelo.** Se entrena con 4 y se evalúa con 6 sin
  reentrenar, porque la tabla no está indexada por agente.

Lo que se pierde es la especialización (no puede haber un AGV "agresivo" y otro "cauto"), y en un
almacén de AGVs idénticos eso no es una pérdida.

### Cuándo se cobra la recompensa

Un AGV **decide solo cuando está parado en un nodo**. Si elige `ADVANCE` cruza un tramo que cuesta
entre 4 y 8 ticks, y durante esos ticks no vuelve a decidir nada. Así que la transición no se
cierra en el mismo tick: se le va sumando la recompensa hasta que el AGV vuelve a decidir o
termina.

```
tick 12  decide ADVANCE en S2 ─┐
tick 13   cruzando             │  todo esto es consecuencia
tick 14   cruzando             │  de la decisión del tick 12
tick 15   cruzando             │
tick 16   llega a S3  +2  ─────┘  se cierra: (s, ADVANCE, +2, s')
tick 16  decide ADVANCE en S3 ...
```

Si la transición se cerrara en el tick 12, **todo `ADVANCE` valdría 0** y no habría nada que
aprender. Los eventos, todos con su precio en `config.py`:

| Evento | Cuándo se cobra |
|---|---|
| `PROGRESS` +2 | `path_index` subió: cruzó un tramo entero |
| `TASK_COMPLETE` +100 | entró en `done` en este tick |
| `CONFLICT` -20 | eligió `ADVANCE` teniendo un conflicto encima **y se quedó donde estaba** |
| `WAIT` -1 | no se movió, sin haberlo intentado |
| `USELESS_REROUTE` -3 | recalculó y la ruta nueva ni salía más barata ni esquivaba nada |
| `DEADLOCK` -50 | seguía en marcha cuando la corrida murió atascada |

Lo que separa `CONFLICT` de `WAIT` es **haberlo intentado**. Si el castigo cayera sobre todos los
del conflicto, la acción no cambiaría la recompensa y no habría nada que aprender; y si cayera
sobre el que **sí** pasa, el AGV con prioridad aprendería a no usarla y el almacén se pararía
entero.

### Un escenario nuevo en cada episodio

`random_routes()` sortea orígenes y destinos distintos en cada `reset()`, de un generador sembrado.
Con el reparto fijo de `Simulation._planea_rutas()` los mil episodios serían el mismo y la tabla
lo aprendería de memoria en vez de aprender a ceder el paso. Orígenes y destinos van sin repetir
por lo mismo que en la fase 5: dos AGVs en el mismo nodo rompen la invariante antes de mover nada.

### Los dos modos

| Modo | epsilon | Q-table | Escenarios |
|---|---|---|---|
| `train` | de 1.0 a 0.05 | **se actualiza** | sorteados con `--seed` |
| `evaluate` | **0**, greedy puro | **no se toca**, se carga del disco | los mismos, con la misma semilla |

`evaluate` corre además la baseline de la fase 5 sobre **los mismos escenarios**, con la misma vara
(`BaselineAdapter` decide exactamente lo que `conflicts.BaselinePolicy` y además apunta lo que
decidió, para que la recompensa se calcule igual en las dos).

### Qué sale de una corrida

| Fichero | Qué es |
|---|---|
| `python/models/q_table.json` | La Q-table **y su metadata**: mapa, agentes, hiperparámetros, semilla, fecha y visitas por estado |
| `results/training_log.csv` | Una fila por episodio, con las diez columnas de abajo |
| `results/learning_curve.png` | Cuatro paneles con media móvil, **si hay matplotlib** |

| Columna | Qué es |
|---|---|
| `episode` | Número de episodio, desde 1 |
| `epsilon` | El epsilon con el que se jugó (0 en `evaluate`) |
| `total_reward` | Suma de la recompensa de **todos** los AGVs |
| `avg_reward` | `total_reward` **por decisión tomada** |
| `conflicts` | Conflictos detectados en el episodio |
| `deadlocks` | 1 si la corrida murió atascada, 0 si no |
| `completed_tasks` | Cuántos AGVs llegaron a su destino |
| `makespan` | Tick en que llegó el último; si no llegaron todos, los ticks que duró |
| `total_wait` | Ticks perdidos cediendo el paso, entre todos |
| `states_visited` | Estados distintos en la Q-table (acumulado, tope 72) |

`avg_reward` va **por decisión** y no por agente: el episodio del principio dura 200 ticks y el del
final 60, así que dividir por el número de AGVs solo reescalaría `total_reward`.

matplotlib es opcional a propósito (el proyecto no tiene dependencias): si no está, se avisa por el
log y se sigue, con los mil episodios ya escritos en el CSV.

### Resultados

`python3 python/main.py train --map warehouse --agents 4 --episodes 1000 --seed 42`:

> Los números de abajo son los del **motor de la fase 8**, que es el que hay: penalizaciones que
> caducan y desatasco forzado. La fase 8 cambió el mundo en el que se entrena, así que la Q-table
> se volvió a entrenar entera con el mismo comando.

```
   episodios  epsilon  recompensa  r/decision  conflictos  deadlocks  completadas  makespan   espera  estados
-------------------------------------------------------------------------------------------------------------
    1-100       0.788      -277.4        0.16       105.5       0.00         3.52     129.3    140.5       36
  101-200       0.478       -75.1        2.67       106.8       0.00         3.54     121.5    132.1       36
  201-300       0.289        69.8        4.14       100.4       0.00         3.57     122.2    117.8       36
  301-400       0.175       151.8        5.60       105.2       0.00         3.47     119.8    116.2       36
  401-500       0.106       261.8        6.11        90.0       0.00         3.64     107.0    102.8       37
  501-600       0.064       223.3        4.60       106.1       0.00         3.49     119.8    120.4       37
  601-700       0.050       202.2        5.92       109.5       0.00         3.37     127.5    131.0       39
  701-800       0.050       254.4        6.30       101.6       0.00         3.61     115.4    112.4       39
  801-900       0.050       240.2        7.08        97.4       0.00         3.48     116.3    109.5       40
  901-1000      0.050       170.0        5.28       112.7       0.00         3.40     123.5    135.6       40
```

La recompensa media pasa de **-277.4 a +170.0** y deja de ser ruido sobre el episodio 300, que es
donde epsilon baja de 0.3. La columna que más cambia respecto a la fase 7 es `deadlocks`: **0.00 en
los mil episodios**, porque el desatasco de la fase 8 no deja que ninguna corrida muera atascada.

`evaluate` con 100 episodios, greedy puro, contra la baseline en los mismos escenarios:

```
metrica            q-learning    baseline   diferencia
------------------------------------------------------
recompensa             321.88     -753.28     +1075.16
completadas              3.69        3.79        -0.10
deadlocks                0.00        0.00        +0.00
makespan               106.68      107.83        -1.15
conflictos              91.12      115.68       -24.56
conflictos/tick          0.85        1.07        -0.22
espera                  92.21      146.64       -54.43
espera/tick              0.86        1.36        -0.50
```

**Y aquí está el resultado más interesante de la fase 8.** En la fase 7 la baseline entregaba 1.55
tareas de 4 porque se moría atascada en el 91 % de los episodios, y el Q-Learning ganaba de calle.
Con el desatasco del motor ya no se muere ninguna corrida, y entonces las dos entregan
prácticamente lo mismo: **3.79 la baseline y 3.69 el Q-Learning**.

No es que el Q-Learning haya empeorado: es que **quitarle los deadlocks a la baseline le quita su
único problema grave**. Embestir contra el nodo ocupado sale carísimo en recompensa (-753 contra
+322, que son los -20 de cada intento) pero deja de ser fatal en cuanto el motor garantiza que
nadie se queda trabado.

Lo que el Q-Learning sigue ganando, y por bastante, es la **calidad** del tráfico: **0.85
conflictos por tick contra 1.07** y **0.86 ticks de espera por tick contra 1.36**, con el mismo
makespan (106.7 contra 107.8). Entrega lo mismo chocando un 20 % menos y esperando un 37 % menos:
cede el paso y rodea en vez de embestir, y llega igual de rápido.

Dicho de otro modo: el desatasco del motor y la política aprendida atacan el mismo problema por dos
sitios, y con el primero puesto el segundo tiene menos que arreglar. Es justo la clase de cosa que
la fase 10 tiene que medir, y por eso importa tanto que `policy` sea la **única** variable entre
las dos corridas.

### Qué aprendió, celda a celda

La regla que sale es una sola, y es la que se buscaba: **si el nodo de delante está ocupado, no
intentes entrar**. De los 24 estados con `next_node_occupied = 1` y al menos 50 visitas detrás,
`ADVANCE` no es la mejor acción en **ninguno**, y es la peor de las tres en casi todos:

```
      estado  visitas  mejor    advance /    wait / reroute
   0|0|0|0|1    17058  advance    68.69 /   11.68 /   16.82   nada delante -> pasa
   0|0|0|1|1    14785  advance    24.06 /    5.54 /    3.22
   1|0|0|0|0    21656  reroute   -11.56 /   -4.99 /   12.23   ocupado y sin prioridad -> rodea
   1|0|0|0|1    16607  wait       -0.57 /   17.58 /    8.36   ocupado y con prioridad -> espera turno
   1|0|0|1|0    13190  reroute   -14.80 /   -3.90 /    4.12
   1|1|1|0|0     7993  reroute   -16.76 /   -2.39 /    8.29   de frente y con cola -> rodear
   1|0|1|0|0     7801  reroute   -13.40 /    1.36 /    2.81
```

`ADVANCE` va de **+68.69 con el camino libre a -16.76 con alguien de frente y cola delante**: 85
puntos de diferencia entre la misma acción en dos sitios distintos, que es exactamente lo que el
estado local tenía que poder distinguir.

De los 72 estados posibles solo se visitan **40**: los otros 32 no se dan en este mapa. El contador
de visitas va en la metadata del modelo justo para poder distinguir una fila aprendida de una que
sigue casi en el cero con el que nació: hay estados visitados 2 o 3 veces en mil episodios, y lo
que llevan dentro no es política, es ruido.

Lo que conviene mirar con lupa es **cuánto pesa `REROUTE`**: es la mejor acción en 20 de los 33
estados con datos detrás, y en una corrida de 6 AGVs se lleva la mayoría de las decisiones. Rodear
evita el choque, pero alarga la ruta, y por eso el Q-Learning gana en conflictos y en espera mucho
más de lo que gana en makespan.

### Lo que no resuelve

Los dos problemas que la fase 7 dejó abiertos los cerró la fase 8, y los cerró **en el motor**, no
en el aprendizaje: el cara a cara en un pasillo y el AGV aparcado encima del cuello de botella se
deshacen ahora con el [desatasco](#el-desatasco-el-almacén-no-se-queda-trabado). De ahí que
`deadlocks` valga 0.00 en toda la tabla de arriba, en los dos modos.

Lo que queda abierto ya no es un atasco, es **rendimiento**:

- **Con el almacén muy lleno se acaban los ticks antes que las tareas.** Con 6 AGVs en 13 nodos
  casi siempre hay alguien delante, y en diez corridas de 1000 ticks se quedan tareas sin terminar:
  **51 de 60 con `qlearning` y 49 de 60 con la baseline** (con 4 AGVs terminan las 40 y las 38). No
  es un atasco —no hay un solo deadlock, y el motor desatasca 390 veces por las 559 de la
  baseline—, es que el cuello de botella no da más de sí.
- **Un AGV que llegó sigue ocupando su nodo.** El desatasco lo aparta cuando estorba de verdad,
  pero apartarlo es moverlo a otro sitio, donde puede volver a estorbar. La solución de fondo sigue
  siendo la que ya decía la fase 7: cambiar el modelo de ocupación, o darle una tarea nueva al que
  termina en vez de dejarlo aparcado.

Al servir se juega **greedy puro** (`config.SERVE_EPSILON = 0.0`). El argumento de la fase 7 para
dejar algo de azar —sin él, dos AGVs en el mismo estado eligen lo mismo y el empate no se rompe
nunca— lo cubre ahora el desatasco, y una corrida determinista es lo que la fase 10 necesita para
poder comparar.

## La fase 8: A\* dice por dónde, Q-Learning dice qué hacer ahora

Hasta aquí las dos piezas existían por separado: A\* trazaba rutas desde la fase 3 y una Q-table
entrenada elegía acciones desde la fase 7, pero el servidor montaba siempre la baseline. La fase 8
las junta en el mismo tick.

**La idea, en una línea: A\* responde *por dónde* y el Q-Learning responde *qué conviene hacer
ahora*.** Ninguna acción elige un nodo. Ni una.

### El bucle de cada AGV en cada tick

```
1. recibe la tarea origen -> destino          Simulation._planea_rutas / routes
2. A* traza el path                           Agent.assign_task -> astar.astar
3. consulta el siguiente nodo del path        FASE A, _fase_a_intenciones
4. construye el estado local (5 enteros)      qlearning.get_local_state
5. elige ADVANCE / WAIT / REROUTE             policy.decide        <- lo unico que cambia de modo
6. si ADVANCE y es seguro -> se mueve         _puede_entrar + _empieza_travesia
7. si WAIT -> espera y acumula wait_time      _cede_el_paso
8. si REROUTE -> penaliza y A* otra vez       _recalcula
9. repetir hasta completar la tarea
```

| La pregunta | Quién la contesta | Dónde |
|---|---|---|
| ¿Por dónde se va de `S1` a `N6`? | **A\*** | `astar.astar()` |
| ¿Qué hago ahora, con la ruta que ya tengo? | **Q-Learning** | `qlearning.QLearningPolicy.decide()` |
| ¿Puedo hacerlo de verdad? | **El motor** | `Simulation._puede_entrar()` |

Los tres tramos del tick están escritos en ese orden en
`Simulation._fase_b_resuelve_y_aplica()`: detectar, decidir, aplicar. La decisión va **en medio**, y
esa es la fase entera: la política propone y el motor dispone.

### `policy` es la única variable experimental

```python
Simulation(grafo, 4, policy="baseline")
Simulation(grafo, 4, policy="qlearning", model="python/models/q_table.json")
Simulation(grafo, 4, policy=MiPolitica())   # un objeto tambien vale, como en la fase 5
```

Con `baseline` y con `qlearning` corre **exactamente el mismo motor**: mismo mapa, mismas rutas,
misma semilla, misma detección de conflictos, mismo desatasco y misma caducidad de penalizaciones.
Si cambiara algo más, comparar las dos corridas no mediría la política, y la comparación de la fase
10 no valdría nada.

Montar la política por nombre lo hace `simulation.make_policy()`. En modo `qlearning` sin Q-table
legible **lanza** en vez de servir una tabla vacía: una tabla a ceros siempre avanza, o sea que
parecería funcionar sin haber aprendido nada, y eso es peor que un error.

### La acción es una intención, no una garantía

El gate físico de la fase 5 no se toca: en un nodo ocupado no se entra, diga lo que diga la
política. Lo que la fase 8 añade es **dejarlo por escrito**. Si dos AGVs eligen `ADVANCE` al mismo
nodo, el motor aplica el desempate, uno pasa y el otro se queda donde estaba con `blocked` puesto:

```json
{"id":4,"state":"waiting","action":"advance","blocked":true, "...": "..."}
```

Ese `blocked` es lo que el entrenamiento cobra a **-20**, y así el AGV aprende que `ADVANCE` en ese
estado es mala idea.

> **A quién se le cobra el -20.** Al que eligió `ADVANCE` y se quedó donde estaba. **Al ganador
> no**, y no es un descuido: el estado local no distingue "camino libre sin rivales" de "camino
> libre y gano la disputa" —en los dos `next_node_occupied = 0` y `has_priority = 1`—, así que
> cobrárselo al ganador envenenaría la celda `0|0|0|0|1`, la de 16.906 visitas y `advance = +78.91`
> que sostiene toda la política, y el almacén se pararía entero. El que puede aprender algo del
> choque es el perdedor, que tiene `has_priority = 0` y por tanto su propia celda.

### El REROUTE penaliza, y la penalización caduca

Recalcular es mecánica del **motor**: la política solo dice `reroute` y quien encarece el mapa y
vuelve a llamar a A\* es `Simulation._recalcula()`. Antes lo hacía la propia política, y eso dejaba
al motor con una intención que apuntaba a la ruta vieja.

La tabla de penalizaciones es `astar.TemporaryPenalties`, un `Mapping` de verdad que entra en
`astar.astar()` **sin tocar una línea de A\***, y hay **una sola por almacén**: que `G` esté
congestionado es un hecho del mapa, no la opinión de un AGV.

```python
castigos.add("G", 10.0, step=12)     # suma (tope PENALTY_MAX) y arranca el reloj
castigos.expire(step=28)             # a los PENALTY_TTL ticks, G vuelve a valer lo que vale
```

**Caducan a los `config.PENALTY_TTL` (15) ticks**, y esa es la mitad importante de la idea: sin
reloj el mapa se degrada para siempre y A\* acaba esquivando pasillos que llevan cien ticks libres
solo porque una vez hubo alguien parado ahí. Como las penalizaciones solo suman, la heurística
sigue siendo admisible y A\* sigue devolviendo la ruta óptima del mapa encarecido.

Dos reglas más, las dos aprendidas a base de verlas fallar:

- **No hay reroute que esquive tu propio destino.** Si el nodo de delante es el destino, todas las
  rutas acaban ahí; recalcular solo da una vuelta larga para volver al mismo sitio. Con dos AGVs
  sentados en el destino del otro, eso es una persecución en círculo que no termina nunca.
- **No se recalcula dos veces seguidas** (`config.REROUTE_COOLDOWN`, 8 ticks). El primer recalculo
  penaliza el nodo de delante y A\* da la otra salida; el segundo penaliza esa y devuelve la
  primera. Un ir y venir que además encarece medio mapa. Durante la pausa, un `REROUTE` es esperar.

### El desatasco: el almacén no se queda trabado

`config.DEADLOCK_TICKS` (20) sigue matando la corrida si nadie avanza, pero ahora **antes** de
llegar ahí manda el motor. Dos atascos distintos, dos umbrales:

| Cuándo salta | Umbral | Qué mira |
|---|---|---|
| El almacén entero parado | `DEADLOCK_FORCE_TICKS` (8) | Nadie se movió en 8 ticks |
| Un AGV muriéndose de hambre | `STARVED_TICKS` (45) | **Ese** AGV lleva 45 ticks clavado |

El segundo umbral no es un lujo: un contador global de "no se movió nadie" no ve nunca al AGV que
lleva doscientos ticks en un rincón mientras otros dos dan vueltas por un pasillo. Va más alto
porque esperar es normal —cruzar un tramo cuesta entre 4 y 8 ticks—, y esperar cuarenta y cinco es
que nadie te va a dejar pasar nunca.

Cuando salta, la escalada tiene tres peldaños y se prueban en orden hasta que uno cambia algo:

```
1. pasa el de id menor que tenga el nodo libre     el desempate por prioridad
2. veto temporal y REROUTE al de id mayor          si de verdad hay otra ruta
3. el que estorba se aparta a un hueco libre       aunque ya haya terminado su tarea
```

- **El peldaño 2 exige que la ruta nueva esquive el nodo en disputa del todo.** Que la ruta cambie
  no basta: en un mapa con cuello de botella A\* devuelve encantado otro camino que vuelve a pasar
  por el mismo sitio ocupado, y con eso los dos AGVs se pasan el día dando vueltas alrededor del que
  estorba. Si el veto no sirve, se retira en vez de dejarlo caducar.
- **El peldaño 3 busca el hueco libre más cercano con un BFS** que atraviesa solo nodos ocupados
  —o sea, recorre la fila de AGVs atascados hasta ver dónde se acaba— y empuja al **último de la
  fila**, que es el único que cabe. La fila se acorta en un AGV por desatasco, y a la segunda o la
  tercera le toca al que estorbaba. El hueco que deja le queda **reservado** al que esperaba durante
  `config.YIELD_TICKS`: sin eso el que se aparta vuelve, gana el desempate por id menor y el atasco
  se rehace igual. Apartarse y volver es no apartarse.

Apartarse **no es un movimiento nuevo**: es una ruta de un tramo que pasa por el gate como
cualquier otra, así que la invariante "un nodo, un AGV" sigue intacta y el test de 500 ticks de la
fase 5 sigue verde. Al que ya había terminado se le mueve el destino con él y se aparca en el
hueco, que es lo que hace un AGV de verdad cuando le piden el pasillo.

Todo AGV al que el motor tuvo que forzar algo queda marcado `forced` y sale por el log; al entrenar
se le cobran los -20, sobre la acción que él eligió. La lección es esa: **quedarse todos parados
sale caro**.

Con `config.DEADLOCK_FORCE_TICKS = 0` el desatasco se apaga y el motor vuelve a ser el de la fase 5.
Es lo que usan los tests de deadlock de `test_conflicts.py`, que si no no verían nunca lo que vienen
a probar.

> **Lo que sigue sin tener arreglo.** Si el que estorba no tiene **ningún** hueco al que ir en todo
> su lado del mapa, no hay salida posible y la corrida muere como en la fase 5. Hace falta llenar
> un componente entero del grafo para llegar ahí.

### Cambiar de política en caliente

```
-> SET_MODE qlearning\n
<- {"ok":true,"mode":"qlearning","run":3}\n
```

Para la demo: se enseña la baseline atascándose en el cuello de botella, se manda `SET_MODE
qlearning` y se ve el mismo escenario con la política aprendida, sin reiniciar el servidor ni
recargar Unity. Los detalles del comando están en [Comandos](#comandos).

## La fase 9: medir en vez de suponer

Hasta aquí el proyecto tenía dos políticas y una intuición. La fase 9 pone los números:
`python/metrics.py` recoge las métricas de cada corrida y `main.py benchmark` enfrenta las dos
políticas **semilla a semilla**, escribiendo `results/baseline.csv`, `results/qlearning.csv`,
`results/comparison.json` y (si hay matplotlib) `results/comparison.png`.

```bash
python3 python/main.py benchmark --map warehouse --agents 4 --runs 20 --seeds 1-20
```

### Lo que de verdad importa: que el escenario sea el mismo

El riesgo de esta fase no es el código, es el sesgo. Si comparas una corrida fácil del baseline
contra una difícil del Q-Learning, el argumento se cae solo. Por eso el escenario se construye
**una vez por semilla, antes de que exista ninguna política**:

```python
for semilla in semillas:
    escenario = build_scenario(grafo, n_agents, n_tasks, seed=semilla)   # todavía no hay política
    for politica in politicas:
        run_once(grafo, escenario, politica)
```

Que el `build_scenario()` esté **fuera** del bucle de políticas no es cosmético: es lo que hace
imposible que una vea un trabajo distinto del que vio la otra. Las dos comparten mapa, AGVs,
orígenes, destinos, la cola de tareas entera y la semilla. Lo único que cambia es el nombre de la
política. `tests/test_metrics.py::TestMismasCondiciones` lo comprueba sobre 20 semillas, campo a
campo, y `TestElEscenarioNoEsConstante` comprueba que ese test no pasa por ser trivial.

### La cola de tareas vive en el runner

El motor de las fases 5-8 da **una** ruta por AGV y termina cuando llegan todos. Para medir
throughput hacen falta más tareas, así que la cola vive en `metrics.py` y no en `simulation.py`:
ni una línea del motor cambia, y lo que se compara sigue siendo exactamente el motor que ya
validan los tests de las fases anteriores.

La cola se sortea **entera y de golpe** desde la semilla, nunca según hace falta. Si los destinos
se sortearan al asignarlos, el orden de las extracciones dependería del orden de llegada, el orden
de llegada depende de la política, y ahí se acabó la comparación pareada.

Un detalle que cuesta ver: `Agent.assign_task()` pone `wait_time` a 0, y `wait_time` es justo la
medida con la que se comparan las políticas. El runner lo guarda y lo restaura, por la misma razón
por la que `conflicts.reroute()` evita `assign_task()` a propósito.

### Las métricas

| Métrica | Qué es |
|---|---|
| `makespan` | Ticks hasta despachar todas las tareas; si no se despacharon todas, los que duró |
| `avg_task_time` | Ticks que costó una tarea de media |
| `total_wait_time` / `wait_agv_N` | Ticks perdidos cediendo el paso, en total y por AGV |
| `conflicts_vertex/edge/following/congestion` | Los cuatro tipos, con los ceros explícitos |
| `deadlocks` | 1 si la corrida murió atascada |
| `total_distance` | Suma del coste de los tramos realmente pisados |
| `throughput` | Tareas completadas por 100 ticks |
| `reroutes` | Cuántas veces la política pidió recalcular |
| `conflicts_per_tick` / `wait_per_tick` | Las mismas, por tick |
| `all_completed` / `finished_reason` | Si la corrida terminó, y por qué paró |

La distancia se mide por **cambio de `current_node`**, nunca por `path_index`: un REROUTE pone el
índice a cero sin mover al AGV ni un metro, y contar por índice inflaría la cifra justo en la
política que más recalcula.

Y hay una trampa que el informe avisa por escrito: **los totales crudos premian al que muere
antes**. Una corrida que se atasca en el tick 30 acumula menos conflictos y menos espera que una
que corre 300 y despacha el triple. Por eso van también las tasas por tick y las de completitud.

### Resultados: 20 semillas, warehouse, 4 AGVs, 16 tareas

Con la Q-table de la fase 7 (`warehouse`, 4 AGVs, 1000 episodios, semilla 42) y tope de 800 ticks:

| Métrica | Baseline | Q-Learning | Diferencia | ¿Mejora? |
|---|---:|---:|---:|:--:|
| makespan (ticks) | 401.05 | 494.45 | +23.3 % | **NO** |
| tiempo por tarea | 74.41 | 51.71 | −30.5 % | sí |
| tareas completadas | 15.90 | 15.35 | −3.5 % | **NO** |
| throughput /100t | 4.58 | 4.74 | +3.4 % | sí |
| espera total | 845.80 | 644.95 | −23.7 % | sí |
| espera por tick | 2.23 | 1.45 | −34.9 % | sí |
| distancia recorrida | 398.12 | 588.36 | +47.8 % | **NO** |
| conflictos | 621.35 | 535.65 | −13.8 % | sí |
| conflictos por tick | 1.62 | 1.21 | −25.2 % | sí |
| &nbsp;&nbsp;de nodo | 353.50 | 336.40 | −4.8 % | sí |
| &nbsp;&nbsp;de arista | 141.10 | 101.20 | −28.3 % | sí |
| &nbsp;&nbsp;de seguimiento | 122.75 | 91.75 | −25.3 % | sí |
| &nbsp;&nbsp;de congestión | 4.00 | 6.30 | +57.5 % | **NO** |
| deadlocks | 0.00 | 0.00 | — | igual |
| reroutes | 18.15 | 484.70 | +2570.5 % | **NO** |
| corridas completas % | 90.00 | 55.00 | −38.9 % | **NO** |
| tareas despachadas % | 99.38 | 95.94 | −3.5 % | **NO** |
| **semillas ganadas** | **7** | **11** | 2 empates | |

### El veredicto honesto

**El Q-Learning no gana.** Y el caso es interesante porque los dos números que se suelen mirar
dicen cosas distintas:

- **Gana en 11 de 20 semillas** contra 7 del baseline. Cuando funciona, funciona bien: espera un
  35 % menos por tick, provoca un 25 % menos de conflictos por tick y cierra cada tarea en 30 %
  menos ticks.
- **Y aun así su makespan medio es un 23 % peor** (494 contra 401 ticks). No es que rinda peor de
  media: es que **se cuelga del todo en 9 de las 20 semillas** (1, 3, 4, 7, 12, 16, 17, 18, 19),
  que se van al tope de 800 ticks. El baseline solo llega al tope en 2. Nueve corridas al tope se
  llevan la media por delante; la mediana, que no las nota tanto, queda casi empatada: 365 contra
  340.

La causa se ve en la fila de `reroutes`: **484 recálculos de media contra 18 del baseline**, y un
47 % más de distancia recorrida. La política aprendida pide REROUTE constantemente, los AGVs dan
vueltas largas, y en las corridas malas entra en un ir y venir del que no sale — dos rutas
alternándose cada `REROUTE_COOLDOWN` ticks hasta que se acaba el reloj.

Por qué pasa, y no es un fallo del benchmark: **la Q-table se entrenó fuera de esta distribución**.
Los mil episodios de la fase 7 son de **una tarea por AGV y 200 ticks de tope**; aquí se le piden
**cuatro tareas por AGV y 800 ticks**, con AGVs que terminan y se quedan aparcados ocupando nodos.
El estado local de cinco enteros no distingue esa situación, así que la política aplica lo que
aprendió en un almacén más vacío. El experimento está bien montado; lo que falta es reentrenar en
el régimen en el que se va a evaluar.

> Esto es el resultado real, sin maquillar. Un benchmark que solo sabe dar buenas noticias no sirve
> para decidir nada.

### Dos defectos que el benchmark destapó

Medir sirvió para encontrar dos fallos **silenciosos** en cómo se sirve un modelo. Ninguno da
error: los dos dan resultados malos, que es peor.

**1. El modelo se servía fuera de su action set.** La fila de la Q-table lleva siempre las tres
acciones, aunque se entrene solo con dos (`train --no-reroute`). `make_policy()` no miraba con qué
se había entrenado y habilitaba REROUTE igualmente — y esa columna sigue **a ceros**. Como la
recompensa del almacén es casi toda negativa (−20 por chocar, −1 por cada tick esperando), **el
cero le gana a todo lo aprendido**: la política se pasaba la corrida eligiendo precisamente la
acción de la que no sabía nada.

Ahora `make_policy()` lee `metadata.hyperparameters.actions` y sirve con el action set con el que
se entrenó. Un modelo antiguo que no lo diga se sirve como antes.

**2. Una celda sin explorar valía 0, y 0 es optimista.** El mismo problema, un nivel más abajo: no
la columna entera, sino la celda `(estado, acción)` que el entrenamiento apenas tocó. Se guardaban
las visitas **por estado**, y eso no basta: un estado con 20 000 visitas puede tener una acción
probada 19 000 veces y otra tres. Ahora se guardan también **por celda** (`metadata.action_visits`)
y al servir sólo compiten las que llegan a `config.SERVE_MIN_VISITS` (30). Si ninguna llega, se
cae a todas: mejor una decisión con poco respaldo que ninguna. El filtro **no** toca la
exploración — al entrenar, la celda sin probar es justo la que hay que probar.

Los dos arreglos, medidos sobre el mismo modelo entrenado con `--no-reroute` y las mismas 20
semillas:

| | makespan | reroutes | completas |
|---|---:|---:|---:|
| baseline (referencia) | 401.1 | 18.1 | 90 % |
| sin ningún arreglo | 449.9 | 357.4 | 70 % |
| solo el arreglo 1 | **408.1** | **18.8** | **85 %** |
| solo el arreglo 3 | **408.1** | **18.8** | **85 %** |

Cada uno atrapa el fallo por su cuenta, y dan exactamente el mismo resultado: son la misma trampa
vista desde la metadata y desde los contadores.

> **Lo que estos arreglos NO hacen: mejorar el modelo que se sirve hoy.** El `q_table.json` del
> repo se entrenó con las tres acciones y con exploración de sobra en los estados frecuentes, así
> que no hay nada que enmascarar: con y sin los arreglos da los mismos 494.4 ticks. Son un cepo
> para un fallo que estaba puesto, no una mejora del rendimiento. Lo que le falta al modelo actual
> es otra cosa, y es lo de abajo.

### Lo que le falta al Q-Learning

Lo que de verdad explica los 484 reroutes no es un defecto de código, es la recompensa. Contando
eventos en 300 episodios de entrenamiento:

| evento | veces | precio | total |
|---|---:|---:|---:|
| `WAIT` | 33 375 | −1 | −33 375 |
| `CONFLICT` | 6 515 | −20 | −130 300 |
| `PROGRESS` | 6 837 | +2 | +13 674 |
| `USELESS_REROUTE` | 2 256 | −3 | −6 768 |
| `TASK_COMPLETE` | 1 285 | +100 | +128 500 |

De 5054 reroutes ejecutados, **el 55 % sale gratis** y el resto cuesta −3: coste esperado ≈ −1.3,
**una sola vez**. Esperar cuesta −1 **por tick**. Estar bloqueado cinco ticks son −5, así que
recalcular siempre sale a cuenta — y la recompensa **nunca cobra los 40 puntos de distancia extra
del rodeo**. `is_useless_reroute()` casi no salta porque `avoided_conflict` es cierto siempre que
estés bloqueado y el primer paso cambie, o sea casi siempre.

(Probado y descartado: quitar `REWARD_PROGRESS` **empeora** las cosas — 655 reroutes en vez de 484.
No es que rerutear pague; es que no cuesta.)

Quedan tres cosas para la fase 10, en orden:

1. **Cobrar el reroute por lo que cuesta**, proporcional a `cost(nueva) − cost(vieja)`, en vez de
   un −3 plano.
2. **Entrenar en el régimen en que se evalúa.** La tabla salió de 1 tarea por AGV y 200 ticks; el
   benchmark le pide 4 tareas y 800, con AGVs aparcados ocupando nodos. `metrics.build_scenario()`
   ya existe y se le puede pasar a `TrainingEnv`.
3. **Un bit de estado que vea el bucle.** El fallo observado es un ping-pong determinista entre dos
   rutas cada 21 ticks: mismo estado, misma acción, para siempre. Un `recently_rerouted` pasa el
   espacio de 72 a 144 estados y rompe la simetría.

## La fase 10: cinco escenarios, y dónde el Q-Learning sí sirve

La fase 9 midió **un** experimento y concluyó que el Q-Learning no gana. Lo que no dijo es
*dónde* falla, porque con un solo escenario no se puede saber. La fase 10 monta cinco escenarios
reproducibles que barren el rango, de un almacén casi vacío a un cuello de botella, y los corre
con las dos políticas bajo condiciones idénticas.

```bash
python3 python/main.py scenario --name C --policy qlearning --runs 20
python3 python/main.py scenario --all --runs 20      # los cinco con las dos politicas
```

| | Escenario | Mapa | AGVs | Tareas | Qué prueba |
|---|---|---|--:|--:|---|
| **A** | Baja congestión | `warehouse` | 2 | 6 | Cada AGV en su mitad: nadie se cruza con nadie |
| **B** | Congestión media | `warehouse` | 4 | 12 | Cuatro rutas que se cruzan en `S3`, dos de ellas de frente |
| **C** | Alta congestión | `warehouse` | 6 | 18 | Seis AGVs en trece nodos: el almacén por encima de su capacidad |
| **D** | Cuello de botella | `warehouse` | 4 | 16 | Toda tarea cruza `G`, que es el **único** paso entre las dos mitades |
| **E** | Rutas alternativas | `grid` | 4 | 16 | El mismo cruce constante que D, pero con rutas de **igual coste** |

### El par D/E es el experimento

Los otros tres escenarios son el contexto; la pregunta del proyecto la contesta el par D/E, que
está construido como una comparación controlada: **mismo número de AGVs, mismas tareas, misma
presión** (1.74 conflictos por tick en D contra 1.70 en E), y los dos mandan a los AGVs a cruzar
el mapa de lado a lado sin parar. Lo único que cambia es si existe una ruta alternativa.

- En **D**, `G` es punto de articulación: penalizar `G` no le da a A\* otra ruta, le da una peor
  o ninguna. El REROUTE del Q-Learning no tiene a dónde ir.
- En **E**, el mapa `grid` es una rejilla 4×4 con todos los tramos del mismo coste, así que entre
  dos nodos cualesquiera hay **varias rutas mínimas**. Penalizar un nodo devuelve una alternativa
  igual de buena.

Si "el Q-Learning rerutea mucho" es un defecto o una virtud depende de eso, y por eso hacía falta
un mapa nuevo: `simple` es un anillo de 6 nodos donde con 4 AGVs no cabe un rodeo, y `warehouse`
es justo el mapa sin alternativas.

### Qué es reproducible, y qué no

De un escenario **no se sortea nada estructural**: el mapa, cuántos AGVs hay, dónde arranca cada
uno y de qué conjunto de nodos salen las tareas están escritos a mano en la `ScenarioSpec`. Lo
único que depende de la semilla es **qué destinos concretos** tocan esta vez.

```python
spec.build(k)   # el mismo metrics.Scenario campo a campo, siempre
spec.seeds(20)  # las semillas de 20 corridas: 100, 101, ... 119
```

Por eso `diff -r` de dos invocaciones idénticas no da ni una diferencia, y por eso las 20 corridas
de un escenario no son la misma corrida 20 veces (si lo fueran, la desviación típica saldría 0 y
el test de reproducibilidad pasaría sin medir nada; `TestElEscenarioNoEsConstante` lo comprueba).

Del motor no cambia una línea. El escenario se sigue construyendo **una vez por semilla y fuera
del bucle de políticas**, que es la invariante de la fase 9: `metrics.run_comparison()` ganó un
parámetro `builder` y nada más.

### Resultados: 5 escenarios × 2 políticas × 20 semillas

Con la Q-table general del repo (`warehouse`, 4 AGVs, 1000 episodios, semilla 42). Es
`results/summary_table.csv` tal cual:

| Esc. | Escenario | AGVs | Política | Makespan | Completas | Tareas | Conf/tick | Espera/tick | Reroutes | Gana |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|:--:|
| A | Baja congestión | 2 | baseline | 50.5 | 95 % | 96.7 % | 0.10 | 0.13 | 0.3 | — |
| | | | Q-Learning | **122.8** | 85 % | 93.3 % | 0.09 | 0.13 | 37.6 | 3/20 |
| B | Congestión media | 4 | baseline | 308.7 | 85 % | 98.8 % | 1.72 | 2.17 | 10.5 | — |
| | | | Q-Learning | 294.5 | 75 % | 97.5 % | 1.14 | 1.22 | 172.7 | **15/20** |
| C | Alta congestión | 6 | baseline | 1831.8 | 25 % | 93.9 % | 2.53 | 3.51 | 171.0 | — |
| | | | Q-Learning | 1724.2 | 25 % | 94.7 % | 1.56 | 1.98 | 2450.6 | 4/20 |
| D | Cuello de botella | 4 | baseline | 497.1 | 95 % | 99.7 % | 1.74 | 2.42 | 23.3 | — |
| | | | Q-Learning | 505.5 | **70 %** | 97.2 % | 1.41 | 1.71 | 620.6 | 10/20 |
| E | Rutas alternativas | 4 | baseline | 216.5 | 100 % | 100.0 % | 1.70 | 2.29 | 6.2 | — |
| | | | Q-Learning | **103.0** | 100 % | 100.0 % | 0.71 | 0.83 | 57.1 | **20/20** |

**Dónde mejora y dónde no, sin maquillar:**

- **E (rutas alternativas): mejora, y mucho.** −52 % de makespan, −58 % de conflictos por tick,
  −64 % de espera por tick, y gana **las 20 semillas**, sin perder ni una tarea. Es el único
  escenario donde el Q-Learning gana limpio.
- **B (congestión media): mixto, y la media engaña.** Gana 15 de 20 semillas y baja los
  conflictos por tick un 34 %, pero completa menos corridas (75 % contra 85 %). Va mejor casi
  siempre y se cuelga de vez en cuando.
- **D (cuello de botella): no aporta.** Empata en makespan (+1.7 %) y **termina 25 puntos menos de
  corridas** (70 % contra 95 %), a cambio de 620 reroutes contra 23. Es la confirmación directa de
  la hipótesis: donde no hay ruta alternativa, recalcular sólo cuesta ticks.
- **C (alta congestión): empate, y el makespan no dice nada** (ver abajo). Despacha un 0.8 % más
  de tareas, con 2450 reroutes de media contra 171.
- **A (baja congestión): no aporta, y encima estorba.** Con dos AGVs que no se cruzan, el baseline
  tarda 50 ticks y el Q-Learning **123**: dos veces y media más, por 37 reroutes que no evitan
  ningún conflicto (los conflictos por tick son los mismos, 0.09 contra 0.10). No es que aprenda
  mal a ceder el paso; es que rerutea aunque no haya nadie delante.

El patrón es limpio y es el que se buscaba: **el Q-Learning gana exactamente donde el REROUTE
tiene a dónde ir (E), empata donde hay congestión pero también sitio (B, C), y pierde donde
recalcular no puede ayudar (D) o no hace falta (A).**

### Lo que C destapó: 6 AGVs no caben en 13 nodos

C tiene tope de 2000 ticks y no de 800 como los demás, y hay una razón medida. Con 6 AGVs el
almacén **se satura**: con tope 800 el baseline no completa ni una de 20 corridas y despacha el
25.6 % de las tareas.

Pero además, con cualquier tope, **15 de las 20 corridas se quedan a 1, 2 o 3 tareas del final y
ya no avanzan nunca**. Subir el tope de 2000 a 5000 no cambia ni una: siguen siendo 5 completas y
el mismo 93.9 %. La causa es el AGV que se queda sin trabajo: **aparca**, y con 6 AGVs sobre 13
nodos es casi seguro que uno acabe aparcado justo encima del último destino que queda por servir
—el mismo riesgo que ya documentaba `Simulation._planea_rutas`—.

Así que en C el makespan de esas 15 corridas es el tope que uno elija, no una medida de nada. Por
eso `scenarios.scenario_verdict()` decide por **`task_rate`** cuando la completitud baja de
`SATURATED_BELOW`, y por eso hay que leer C por trabajo despachado y no por ticks. Es un límite
del montaje, no del motor ni de las políticas: las dos lo sufren igual y el escenario sigue
siendo pareado. Sacarlo por escrito es preferible a publicar un makespan que sólo mide el tope.

### ¿Una Q-table por escenario, o una general?

Se probaron las dos. `train --scenario X` entrena en el mapa, los AGVs y las posiciones de salida
de ese escenario (`TrainingEnv` ganó un `routes_factory` opcional; sin él la fase 7 no cambia ni
un número), y escribe en `python/models/q_table_<letra>.json`.

```bash
for L in A B C D E; do python3 python/main.py train --scenario $L --episodes 1000; done
python3 python/main.py scenario --all --runs 20 --per-scenario-model --out results/per_scenario
```

| Esc. | makespan base | Q general | Q por esc. | completas base | Q gen | Q esc | reroutes base | Q gen | Q esc |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| A | 50.5 | 122.8 | 123.8 | 95 % | 85 % | 85 % | 0.3 | 37.6 | 35.4 |
| B | 308.7 | **294.5** | 311.6 | 85 % | 75 % | **85 %** | 10.5 | 172.7 | 215.7 |
| C | 1831.8 | 1724.2 | **1410.5** | 25 % | 25 % | **55 %** | 171.0 | 2450.6 | **1509.0** |
| D | 497.1 | 505.5 | **480.3** | 95 % | 70 % | **75 %** | 23.3 | 620.6 | **567.0** |
| E | 216.5 | **103.0** | 138.1 | 100 % | 100 % | 100 % | 6.2 | **57.1** | 108.8 |
| **media** | **580.9** | **550.0** | **492.9** | **80 %** | **71 %** | **80 %** | **42.2** | **667.7** | **487.2** |

**Gana la tabla por escenario, pero no en todos.** De media baja el makespan un 10 % respecto a la
general (492.9 contra 550.0) y, sobre todo, **recupera la completitud del baseline**: 80 % contra
el 71 % de la general. Donde más se nota es en los dos escenarios cargados —C pasa de 25 % a 55 %
de corridas completas, D de 70 % a 75 %—, que es justo lo que predecía el README de la fase 9:
entrenar en el régimen en que se evalúa. Donde pierde es en B y en E, y en los dos por el mismo
motivo: rerutea más (215 contra 172, 108 contra 57).

Recomendación: **una por escenario**, porque nunca hunde la completitud como la general y arregla
el escenario peor. Si sólo se puede mantener una, la general sigue siendo la mejor para E.

### Y entrenar más lo empeora

Es el dato que cierra el diagnóstico. Reentrenando B y E con 3000 episodios en vez de 1000
(`python3 python/main.py train --scenario B --episodes 3000 --model /tmp/q_B_3000.json`):

| | makespan B | reroutes B | completas B | makespan E | reroutes E |
|---|--:|--:|--:|--:|--:|
| baseline | 308.7 | 10.5 | 85 % | 216.5 | 6.2 |
| Q por escenario, 1000 ep. | **311.6** | **215.7** | **85 %** | **138.1** | **108.8** |
| Q por escenario, 3000 ep. | 406.8 | 451.4 | 65 % | 155.2 | 165.2 |

Tres veces más entrenamiento da **el doble de reroutes y peor makespan en los dos escenarios**. El
problema no es que la tabla esté poco entrenada: es que la recompensa está mal puesta, exactamente
como dejó escrito la fase 9. Rerutear cuesta −3 **una sola vez** y sólo el 45 % de las veces,
mientras que esperar cuesta −1 **por tick**; la recompensa no cobra nunca la distancia extra del
rodeo. Cuanto mejor optimiza el algoritmo esa recompensa, más rerutea, y rerutear es lo que hace
daño. Que en E ayude y en D estorbe no cambia el diagnóstico: lo confirma.

De las tres cosas que la fase 9 dejó apuntadas, la fase 10 hizo la **segunda** (entrenar en el
régimen en que se evalúa) y midió que ayuda pero no basta. La primera (cobrar el reroute
proporcional a `cost(nueva) − cost(vieja)`) es ahora la que tiene los datos más claros a favor.

### Los ficheros que salen

| Fichero | Qué lleva |
|---|---|
| `results/scenario_<letra>_<policy>.csv` | Una fila por semilla, las mismas 28 columnas que la fase 9 |
| `results/summary_table.csv` | Una fila por (escenario, política) y 33 columnas: identificación, tasas, la **media** de cada métrica y las semillas ganadas contra el baseline |

`summary_table.csv` es el que se pega en el reporte: plano, con coma y punto decimal, y se abre en
Excel tal cual.


## Estructura

```
agentesAGV/
├── python/
│   ├── config.py       constantes (red, ticks, Unity, semilla, umbrales)
│   ├── logs.py         configuración del logging
│   ├── protocol.py     el contrato: comandos, serialización y coordenadas
│   ├── server.py       servidor TCP, solo transporte
│   ├── graph.py        el mapa lógico: grafo, validación y carga desde JSON
│   ├── astar.py        pathfinding con A\* y penalizaciones (que caducan)
│   ├── agent.py        el AGV: ruta, estado y tarea
│   ├── conflicts.py    conflictos, ocupación, acciones, reroute y la política base
│   ├── simulation.py   el almacén en marcha: ticks, modos, desatasco y snapshot
│   ├── qlearning.py    Q-Learning: el entorno (fase 6) y el entrenamiento (fase 7)
│   ├── metrics.py      las métricas de una corrida y la comparación entre políticas (fase 9)
│   ├── scenarios.py    los cinco escenarios, su runner y la tabla resumen (fase 10)
│   ├── main.py         CLI con argparse
│   ├── maps/           los mapas en JSON (simple.json, warehouse.json, grid.json)
│   └── models/         los modelos entrenados (q_table.json y q_table_A..E.json)
├── results/            salidas: training_log.csv, learning_curve.png, baseline.csv,
│                       qlearning.csv, comparison.json, comparison.png,
│                       scenario_<letra>_<policy>.csv, summary_table.csv
├── tests/              tests con unittest, y el cliente falso de Unity
├── requirements.txt
└── README.md
```

`results/` **no se versiona** (está en `.gitignore`, salvo el `.gitkeep`): son salidas de corridas
y se regeneran con `train` y `scenario`. Los modelos de `python/models/` **sí**: la `q_table.json`
general y las cinco `q_table_A..E.json` por escenario, que es lo que `evaluate` y
`scenario --per-scenario-model` necesitan.

El servidor recibe la simulación por **inyección de dependencia**: `serve_forever()` acepta
cualquier objeto con `get_snapshot()` y `reset()` (el `Protocol` está declarado en
`protocol.Simulation`). Desde la fase 3 se le pasa una `simulation.Simulation`, y en `server.py`
no queda ni una línea de lógica del almacén.

## Requisitos

Python **3.10 o superior**. No hay dependencias que instalar: todo es librería estándar, el
Q-Learning incluido.

```bash
python3 --version
```

**matplotlib es opcional**, y solo para el PNG de la curva de aprendizaje. Sin él, `train` avisa
por el log y sigue: el CSV con los mil episodios se escribe igual, antes de intentar dibujar nada.

```bash
python3 -m pip install matplotlib     # opcional, solo para results/learning_curve.png
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
| `train` | Entrena la Q-table, sin servidor y sin Unity |
| `evaluate` | Carga una Q-table y la juega greedy, contra la baseline |
| `benchmark` | Enfrenta las políticas semilla a semilla y escribe `results/` |
| `scenario` | Corre los cinco escenarios de la fase 10 y escribe la tabla resumen |

```bash
python3 python/main.py serve                          # 127.0.0.1:5000
python3 python/main.py serve --port 5055              # otro puerto
python3 python/main.py serve --host 0.0.0.0 --port 5055
python3 python/main.py serve --map simple             # sirve el otro mapa
python3 python/main.py serve --agents 6               # con tráfico, para ver conflictos

# la fase 8: el almacen con la politica aprendida
python3 python/main.py serve --map warehouse --agents 4 --policy qlearning \
        --model python/models/q_table.json
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--map` | `config.DEFAULT_MAP` | Mapa a servir |
| `--host` / `--port` | `127.0.0.1:5000` | Dónde escuchar |
| `--agents` | `1` | Cuántos AGVs servir |
| `--policy` | `baseline` | `baseline` o `qlearning`. **La única variable experimental** |
| `--model` | `python/models/q_table.json` | La Q-table, solo con `--policy qlearning` |

Con `--agents 1` (el defecto) no hay con quién chocar y `stats.conflicts` sale siempre 0. Para
ver la fase 5 en marcha hacen falta varios AGVs.

`--policy qlearning` sin modelo en el disco sale con código **2** y dice con qué comando
entrenarlo. Y una vez levantado, `SET_MODE` cambia de política sin reiniciar.

### `simulate`

Corre la simulación **sin servidor** y cuenta por el log lo que hace el AGV en cada paso: útil
para probar la lógica sola, sin Unity y sin sockets.

```bash
python3 python/main.py simulate --map warehouse --agents 1 --steps 100 --headless
python3 python/main.py simulate --map simple --from A --to F --headless

# la fase 8 por el log, sin Unity y sin sockets: se ven WAIT y REROUTE, no solo ADVANCE
python3 python/main.py simulate --map warehouse --agents 6 --steps 300 --headless \
        --policy qlearning
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--map` | `config.DEFAULT_MAP` | Mapa por el que moverse |
| `--agents` | `1` | Cuántos AGVs correr; no puede haber más que nodos en el mapa |
| `--steps` | `100` | Tope de pasos; corta antes si llegan todos |
| `--headless` | apagado | Corre sin servidor, que es el único modo por ahora |
| `--from` / `--to` | la ruta del mapa | Origen y destino (`simple`: `A→F`; `warehouse`: `S1→N6`) |
| `--policy` | `baseline` | `baseline` o `qlearning`, igual que en `serve` |
| `--model` | `python/models/q_table.json` | La Q-table, solo con `--policy qlearning` |

La cuarta columna es la **acción elegida**, y el `!` de al lado dice que el motor no se la
concedió: eso es un `ADVANCE` bloqueado, o sea los -20 de la fase 8 vistos por el log.

```
AGV 1: S1 -> N6 | costo 27.4 | S1 -> S2 -> S3 -> G -> N4 -> N5 -> N6
paso   1 | AGV 1 | moving  | advance  | S1   -> S2   |  25% | tramo 0/6 | espera   0 | tarea 1
paso   5 | AGV 1 | waiting | reroute  | S2   -> S3   |   0% | tramo 0/6 | espera   1 | tarea 1
paso  19 | AGV 4 | waiting | advance! | G    -> N3   |   0% | tramo 0/3 | espera   6 | tarea 4
...
paso  28 | AGV 1 | done    | wait     | N6   -> -    |   0% | tramo 6/6 | espera   0 | tarea 1
```

Con varios AGVs salen además los conflictos y el resumen de la corrida:

```
paso  19 | CONFLICTO edge       | AGV 4, 6 | G <-> N3
paso  20 | CONFLICTO vertex     | AGV 1, 2, 4, 5, 6 | G
paso  30 | DESATASCO | el AGV 4 se aparta a N3 (le deja G al AGV 6) para descongestionar G
--- resumen ---
final       : llegaron todos
conflictos  : 61 (vertex 28, edge 28, following 0, congestion 5)
espera total: 128 ticks entre todos
acciones    : advance 96, wait 41, reroute 12 (desatascos forzados: 3)
AGV 4      : done en N3, tramo 3/3, 28 ticks esperando
```

Un **deadlock ya no sale por aquí**: desde la fase 8 el motor desatasca antes de llegar a él.

Sale con `0` si la corrida fue bien, `1` si algún AGV se quedó sin ruta y `2` si el mapa o los
nodos que le pasaste no existen. Un **deadlock sale con `0`**: que el baseline se atasque es un
resultado experimental válido, no un fallo del programa. La razón del final va en el resumen.

`Ctrl+C` cierra limpio, y también un `kill` (SIGTERM). Con un cliente conectado tarda unos
milisegundos: los hilos de los clientes son *daemon*, no bloquean la salida.

> **macOS y el puerto 5000.** El receptor de AirPlay se queda con `*:5000`. El servidor
> igual consigue abrir `127.0.0.1:5000` porque es una dirección más específica, pero si algo
> se comporta raro, apágalo en Ajustes → General → AirDrop y Handoff → Receptor de AirPlay, o
> usa `--port`.

### `train` y `evaluate`

Los dos corren **sin servidor y sin Unity** (ver [El entrenamiento](#el-entrenamiento)).

```bash
python3 python/main.py train    --map warehouse --agents 4 --episodes 1000 --seed 42
python3 python/main.py evaluate --map warehouse --agents 4 --model python/models/q_table.json
```

| Opción | Por defecto | Para qué | |
|---|---|---|---|
| `--map` | `warehouse` | Mapa sobre el que entrenar o evaluar | ambos |
| `--agents` | `TRAIN_AGENTS` (4) | Cuántos AGVs por episodio | ambos |
| `--seed` | `RANDOM_SEED` (42) | La misma semilla da la misma corrida | ambos |
| `--max-steps` | `MAX_STEPS_PER_EPISODE` (200) | Tope de ticks por episodio | ambos |
| `--model` | `python/models/q_table.json` | Dónde se escribe / de dónde se carga | ambos |
| `--episodes` | 1000 / 100 | Cuántos episodios | ambos |
| `--alpha` `--gamma` | `config.py` | La tasa de aprendizaje y el descuento | `train` |
| `--epsilon-start` `--epsilon-end` `--epsilon-decay` | `config.py` | La exploración | `train` |
| `--no-reroute` | apagado | Deja fuera `REROUTE`: solo `ADVANCE` y `WAIT` | `train` |
| `--scenario` | — | Entrena **en** un escenario de la fase 10 (A–E): su mapa, sus AGVs y sus salidas | `train` |
| `--log` | `results/training_log.csv` | El CSV por episodio (en `evaluate` es opcional) | ambos |
| `--curve` / `--no-curve` | `results/learning_curve.png` | El PNG de la curva | `train` |

`train` sale con `0` si entrenó, `2` si el mapa no existe. `evaluate` sale con `2` si el modelo no
está (y dice con qué comando entrenarlo) y con `1` si el fichero está pero es de otro formato o
tiene otros campos de estado: una Q-table cargada a ciegas sobre estados que no son los suyos no da
error nunca, da resultados malos.

Las dos imprimen la metadata del modelo, que es lo que dice **con qué se entrenó**:

```
map             : warehouse
agents          : 4
seed            : 42
episodes_run    : 1000
hyperparameters : alpha=0.2, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05, ...
states_visited  : 31
```

### `benchmark`

Enfrenta las políticas bajo condiciones idénticas y deja los números en `results/`.

```bash
# lo que pide la fase 9: 20 semillas pareadas
python3 python/main.py benchmark --map warehouse --agents 4 --runs 20 --seeds 1-20

python3 python/main.py benchmark --seeds 1-50              # más semillas, media más estable
python3 python/main.py benchmark --agents 6 --tasks 30     # más tráfico y más trabajo
python3 python/main.py benchmark --policies baseline       # solo la referencia, sin Q-table
python3 python/main.py benchmark --out /tmp/prueba --no-plots
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--map` | `warehouse` | Mapa sobre el que medir |
| `--agents` | `config.TRAIN_AGENTS` (4) | AGVs por corrida |
| `--tasks` | 4 por AGV | Tareas totales del escenario |
| `--runs` | `config.BENCHMARK_RUNS` (20) | Cuántas semillas si no se dan con `--seeds` |
| `--seeds` | — | Semillas concretas: `1-20`, `1,2,3` o `1-5,10`; manda sobre `--runs` |
| `--policies` | las dos | Qué políticas enfrentar |
| `--model` | `config.Q_TABLE_FILE` | Q-table del modo `qlearning` |
| `--max-steps` | `config.BENCHMARK_MAX_STEPS` (800) | Tope de ticks por corrida |
| `--out` | `results/` | Dónde escribir los CSV y el JSON |
| `--no-plots` | apagado | No dibuja aunque haya matplotlib |

Todas menos `--policies` se aplican **idénticas** a las dos políticas: es la condición de que la
comparación mida algo.

Salen cuatro ficheros. Los CSV llevan una fila por semilla y se abren en Excel tal cual (coma,
punto decimal, sin nada anidado); `comparison.json` lleva media, desviación típica, mediana,
mínimo y máximo de cada métrica por política, más la cabecera del experimento.

```bash
head -3 results/baseline.csv
python3 -c "import json;print(json.load(open('results/comparison.json'))['policies']['baseline']['metrics']['makespan'])"
```

matplotlib es **opcional**: sin él se avisa por el log y se sigue, porque los números ya están en
los CSV. Con él sale `results/comparison.png` con barras de makespan, conflictos y espera, y
barras de error de una desviación típica.

### `scenario`

Los cinco escenarios de la fase 10, con las dos políticas y bajo condiciones idénticas.

```bash
python3 python/main.py scenario --name C --policy qlearning --runs 20
python3 python/main.py scenario --all --runs 20              # los cinco con las dos politicas

python3 python/main.py scenario --name E --runs 50           # mas semillas, media mas estable
python3 python/main.py scenario --all --per-scenario-model \
        --out results/per_scenario                           # con la Q-table de cada escenario
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--name` | — | Escenario a correr: `A`–`E` (da igual la mayúscula) |
| `--all` | — | Los cinco seguidos. Excluyente con `--name`; hace falta uno de los dos |
| `--policy` | **las dos** | `baseline` o `qlearning`; omitido corre las dos y las compara |
| `--runs` | `config.SCENARIO_RUNS` (20) | Semillas por escenario: `seed`, `seed+1`, … |
| `--seeds` | — | Semillas concretas: `1-20`, `1,2,3` o `1-5,10`; manda sobre `--runs` |
| `--model` | `config.Q_TABLE_FILE` | La Q-table general |
| `--per-scenario-model` | apagado | Usa `python/models/q_table_<letra>.json` en vez de la general |
| `--max-steps` | el de cada escenario | Tope de ticks por corrida |
| `--out` | `results/` | Dónde escribir los CSV |
| `--no-summary` | apagado | No escribe `summary_table.csv` |

Sale con `0` aunque el Q-Learning pierda en los cinco (un resultado malo es un resultado) y con
`2` si el escenario, el mapa o la Q-table no existen. Los modelos se comprueban **todos antes de
correr nada**: con `--all`, reventar en el escenario D después de haber corrido A, B y C dejaría
`results/` a medias y la tabla resumen incompleta.

Por cada escenario imprime su ficha (mapa, AGVs, salidas, cola, qué prueba), la tabla comparativa
de la fase 9 y, al final, un veredicto por escenario:

```
A (Baja congestion): NO APORTA. makespan 122.8 contra 50.5 (+143.3%), gana 3 de 20 semillas, ...
E (Rutas alternativas): MEJORA. makespan 103.0 contra 216.5 (-52.4%), gana 20 de 20 semillas, ...
```

**Es repetible byte a byte.** Misma semilla, mismo resultado:

```bash
python3 python/main.py scenario --name D --runs 5 --out /tmp/a
python3 python/main.py scenario --name D --runs 5 --out /tmp/b
diff -r /tmp/a /tmp/b     # sin diferencias
```

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
python3 -m unittest tests.test_training -v       # solo la fase 7
python3 -m unittest tests.test_phase8 -v         # solo la fase 8
python3 -m unittest tests.test_scenarios -v      # solo la fase 10
```

| Fichero | Qué cubre |
|---|---|
| `test_config.py` | Las constantes del proyecto |
| `test_logs.py` | La configuración del logging |
| `test_protocol.py` | El contrato: comandos, serialización y coordenadas |
| `test_server.py` | El servidor TCP contra un socket de verdad |
| `test_graph.py` | El mapa lógico: grafo, `validate()` y ficheros |
| `test_main.py` | El CLI |
| `test_metrics.py` | La fase 9: escenarios pareados, métricas, CSV/JSON y el reporte |
| `test_scenarios.py` | La fase 10: los cinco escenarios, su reproducibilidad y la tabla resumen |
| `test_astar.py` | La fase 3: A\*, penalizaciones, `Agent`, `Simulation` y el snapshot |
| `test_conflicts.py` | La fase 5: conflictos, política base, invariante y deadlock |
| `test_qlearning.py` | La fase 6: estado, acciones, recompensa, Q-table y la política |
| `test_training.py` | La fase 7: Bellman, los dos modos, que aprende y que es reproducible |
| `test_phase8.py` | La fase 8: los dos modos, la acción en el snapshot, `SET_MODE` y el desatasco |

`test_astar.py` compara A\* contra una **búsqueda exhaustiva** en los dos mapas (los 186 pares
ordenados de nodos), comprueba que cada par consecutivo de una ruta es una arista de verdad, y
valida el snapshot con el mismo `validar_snapshot()` que usa el cliente falso de Unity.

`test_qlearning.py` comprueba que el espacio de estados son 72 (y falla si pasa de 500), que
`get_local_state()` devuelve la misma tupla de cinco enteros en 200 ticks con 6 AGVs, y que la
política se intercambia con la baseline sin tocar el motor.

`test_training.py` entrena **300 episodios de verdad** dentro del test y comprueba que la
recompensa de los últimos 100 supera a la de los primeros 100, que los conflictos por episodio
bajan, y que en todos los estados con datos detrás (≥50 visitas) la política aprendió a no meterse
en un nodo ocupado. Comprueba además que dos entrenamientos con la misma semilla dan la **misma
Q-table celda a celda**, que `evaluate` no toca la tabla, y que entrenar no abre ni un socket
(`server.serve_forever` y `socket.socket` mockeados y sin llamar).

`test_phase8.py` mide los cuatro criterios de aceptación de la fase tal y como están escritos: que
con la Q-table entrenada se completan todas las tareas, que salen las tres acciones y no solo
`ADVANCE`, que `SET_MODE` cambia de modo en caliente dejando la corrida limpia, y que en **10
corridas de 1000 ticks con 6 AGVs no hay un solo deadlock** —ni con `qlearning` ni con `baseline`,
porque el desatasco es del motor y no de la política—. Las diez corridas van con diez semillas
distintas: con una sola serían la misma corrida diez veces.

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
  El Q-Learning se implementa a mano con diccionarios. matplotlib es lo único opcional, y solo
  para dibujar la curva: si no está, se avisa y se sigue.
- El entrenamiento corre **sin servidor y sin Unity**. Levantarlos durante el `train` solo lo haría
  lento y no le daría al algoritmo ni un dato más.
- Nada de lógica de negocio dentro de `server.py`: el servidor solo traduce sockets a llamadas.
  La simulación entra por inyección de dependencia y vive en `python/simulation.py`.
- Cada módulo debe poder importarse y probarse por separado, sin levantar el servidor.
- Logging con el módulo `logging`, nunca con `print`.
