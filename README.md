# agentesAGV — Simulación multiagente de AGVs en un almacén

Servidor **Python** de una simulación multiagente de AGVs (vehículos de guiado automático) que se
mueven por un almacén, se disputan los pasillos y aprenden a ceder el paso.

Python es dueño de **toda** la lógica: el mapa, los agentes, el pathfinding, la detección de
conflictos y el aprendizaje. Unity es solo el cliente visual, lo desarrolla otra persona en otro
repo, y habla con Python por HTTP.

> En este repo **no** se escribe nada de C# ni de Unity.

Los AGVs recogen cajas de las estanterías y las llevan a los muelles. **Nadie les asigna el
trabajo**: las misiones se publican y cada AGV puja por las que le convienen. Y por Q-Learning
aprenden a quién le toca ceder el paso cuando se cruzan.

| | |
|---|---|
| Escribir el cliente de Unity | **[docs/PROTOCOL.md](docs/PROTOCOL.md)** — el contrato solo, autosuficiente |
| Verlo funcionar en un comando | `python3 python/main.py simulate --map grid --agents 3 --deliveries` |

---

## Índice

1. [Qué es y qué problema resuelve](#1-qué-es-y-qué-problema-resuelve)
2. [Arquitectura y contrato con Unity](#2-arquitectura-y-contrato-con-unity)
3. [Coordenadas y escala](#3-coordenadas-y-escala)
4. [Instalación y uso](#4-instalación-y-uso)
5. [Diseño del Q-Learning](#5-diseño-del-q-learning)
6. [Limitaciones y qué haría falta](#6-limitaciones-y-qué-haría-falta)
7. [Estructura del repo](#7-estructura-del-repo)

---

## 1. Qué es y qué problema resuelve

### El problema

En un almacén automatizado hay varios AGVs moviendo mercancía a la vez por los mismos pasillos.
Calcular la ruta más corta de cada uno por separado es fácil —eso es A\*— pero **no basta**: en
cuanto dos AGVs quieren el mismo pasillo hay que decidir quién pasa y quién espera, y esa decisión
se toma decenas de veces por minuto, con información local y sin tiempo para replanificar el
almacén entero.

Ese es el problema: **la coordinación, no el pathfinding.**

El mapa de pruebas lo pone a propósito difícil. `warehouse` son 13 nodos con forma de pasillos y un
**cuello de botella** en `G`:

```
N1──N2──N3            N4──N5──N6      y = 8
 │       │  ╲        ╱  │       │
 │       │    ▶ G ◀     │       │     y = 4
 │       │  ╱        ╲  │       │
S1──S2──S3            S4──S5──S6      y = 0
 x=0     4   8   12   16   20   24
```

`G` es un **nodo de articulación**: es la única unión entre las dos mitades, así que toda ruta que
cruce el almacén pasa por él a la fuerza. Quitarlo parte el grafo en dos. Ahí se concentra todo el
conflicto interesante.

### Cómo se reparte el trabajo

> **A\* responde *por dónde* se va. El Q-Learning responde *qué conviene hacer ahora*.**

Ninguna acción del aprendizaje elige un nodo: la ruta la traza siempre A\*. Lo que se aprende es
mucho más pequeño —avanzar, esperar o pedir otra ruta— y por eso cabe en una tabla de 144 estados en
vez de explotar. Está desarrollado en la [sección 5](#5-diseño-del-q-learning).

Y por encima de las dos está **el motor, que es la autoridad**: una acción es una intención, no una
garantía. En un nodo ocupado no se entra, diga lo que diga la política.

### Qué hay dentro

| Pieza | Fichero | Qué hace |
|---|---|---|
| Mapa y rutas | `python/graph.py` | El almacén como grafo, las cajas, la validación y A\* |
| AGV | `python/agent.py` | Un vehículo: su ruta, su estado, su entrega y **su puja** |
| Negociación | `python/missions.py` | Las misiones, el bus de mensajes y la subasta |
| Conflictos | `python/conflicts.py` | Los cuatro tipos de choque y la política baseline |
| Motor | `python/simulation.py` | El tick en dos fases, el gate físico y el desatasco |
| Aprendizaje | `python/qlearning.py` | Estado, acciones, recompensa, Q-table y entrenamiento |
| Transporte | `python/server.py` | El servidor HTTP y el contrato, sin lógica de almacén |

### Dos flujos, y se encadenan

El almacén mueve cajas por dos caminos:

```
1. produccion -> rack     guardar lo que sale de la linea de produccion
2. rack -> muelle         sacar del almacen lo que ya estaba guardado
```

Y **se encadenan**: en cuanto una caja de producción queda guardada en un rack, el almacén le abre
su misión de salida. Por eso 12 cajas dan 18 misiones — las 6 de producción pasan por los dos
flujos y las 6 que ya estaban en el rack solo por el segundo.

La caja **se mueve de verdad**: `graph.boxes` del mapa es el inventario inicial, y durante la
corrida cada caja tiene posición y estado propios (`WAITING_PICKUP`, `RESERVED`, `IN_TRANSIT`,
`STORED`, `DELIVERED`), que es lo que Unity necesita para pintarla donde está.

### El reparto del trabajo es una subasta

Una **misión** es una entrega entera: recoger una caja y dejarla en su destino, que es un rack o un
muelle según el flujo. El
`MissionManager` las publica al bus y **no decide nada**: no calcula distancias, no compara AGVs y
no elige ganador.

Cada AGV mira lo publicado y calcula **su propia** utilidad para cada misión:

```
U = -W_D · distancia  -  W_L · carga_de_trabajo  -  W_N · (nivel - 1)
     2.0                 20.0                       4.0
```

Los tres términos van en contra, así que gana el AGV **libre, cercano y con la caja más baja** (una
caja del nivel 2 cuesta el doble de ticks de recoger). A igualdad de utilidad gana el id menor, o la
corrida dependería del orden de un diccionario.

Un AGV lleva **una misión a la vez**: puja solo si está libre, y no vuelve a pujar hasta que deja la
caja en el muelle. Se ve entero en el bus con `--bus`:

```
[t=  1] MANAGER -> TODOS   MISSION_PUBLISHED  mision=M01 flujo=produccion -> rack caja=A1-L1 nodo=A1 destino=D1
[t=  1]   AGV-1 -> TODOS   BID                agv=1 mision=M01 utilidad=49.9
[t=  1]   AGV-2 -> TODOS   BID                agv=2 mision=M01 utilidad=21.1
[t=  1]   AGV-1 -> MANAGER ACCEPT             mision=M01 utilidad=49.9
[t=  2]   AGV-1 -> MANAGER PICKED_UP          mision=M01 caja=A1-L1
[t=  2]   AGV-2 -> TODOS   UNAVAILABLE        agv=2 motivo=moving M03
[t= 12]   AGV-1 -> MANAGER COMPLETED          mision=M01 caja=A1-L1 destino=D1
[t= 13] MANAGER -> TODOS   MISSION_PUBLISHED  mision=M13 flujo=rack -> muelle caja=A1-L1 nodo=D1 destino=B1
[t=136]   AGV-1 -> TODOS   UNAVAILABLE        agv=1 motivo=bateria insuficiente 30%
[t=136]   AGV-1 -> MANAGER CHARGING           agv=1 estacion=C4 bateria=29.6
```

La última línea del `COMPLETED` y la primera del `MISSION_PUBLISHED` siguiente son el encadenado:
la caja `A1-L1` acaba de quedarse guardada en `D1` y el almacén ya le abre su salida al muelle.

El AGV‑1 arrancaba en `A1`, que es donde está la caja de M01: puja 0.0 y gana. El AGV‑2, a 14 metros,
puja −28.84 y se lleva otra. Entre el `ACCEPT` de una misión y el de la siguiente, el AGV publica
`UNAVAILABLE` en cada paso en vez de pujar.

---

## 2. Arquitectura y contrato con Unity

### PULL: Unity pide, Python responde

```
Unity  ──────  POST /step  ─────▶  Python
Unity  ◀────  {...json...}  ────  Python
```

Python **nunca** empuja datos por su cuenta y Unity nunca calcula nada: solo dibuja lo que recibe.

> **`POST /step` avanza la simulación un paso; `GET /state` no.** El mundo se mueve porque alguien
> lo pide, así que el ritmo lo marca el cliente: llamando 10 veces por segundo, el almacén corre a
> 10 ticks/s.

Reglas del contrato:

- **HTTP** contra `127.0.0.1:5000`, JSON en los dos sentidos, `utf-8`.
- El estado es completo en cada respuesta, no incremental. Unity no guarda historia.
- Mirar y avanzar son rutas distintas, así que dos clientes ya no se roban los ticks.

### Las cinco rutas

| Ruta | Qué hace | Respuesta | Avanza el tick |
|---|---|---|:--:|
| `GET /state` | El estado actual | El snapshot completo | no |
| `GET /health` | Comprueba que el servidor vive | `{"ok":true}` | no |
| `POST /step` | Avanza un paso | El snapshot completo | **sí** |
| `POST /reset` | Reinicia la corrida | `{"ok":true}` | no |
| `POST /mode` | Cambia de política **en caliente** | `{"ok":true,"mode":"qlearning","run":3}` | no |

Se prueba entero sin Unity:

```bash
curl localhost:5000/state
curl -X POST localhost:5000/step
curl -X POST -d '{"mode":"qlearning"}' localhost:5000/mode
```

Una ruta desconocida **no** cierra nada: contesta 404 con la lista de las que hay.

```
-> GET /loquesea
<- 404 {"error":"unknown_route","path":"/loquesea","routes":[...]}

-> POST /mode {"mode":"turbo"}
<- 400 {"error":"bad_mode","mode":"turbo","modes":["baseline","qlearning"]}
```

`POST /mode` reinicia siempre, también si el modo pedido es el que ya estaba: media corrida con una
política y media con otra no es una corrida de ninguna de las dos. Sube `run` y `step` vuelve a 1;
lo único que sobrevive es `stats.deadlocks`, que cuenta los de la sesión.

### El snapshot

Copiado tal cual de una corrida real, partido aquí en varias líneas para que se lea (en el cable va
en una sola):

```json
{"step":1,
 "agents":[{"id":1,"x":1.0,"y":0.0,"z":0.0,"rotation":90.0,"state":"moving",
            "node":"S1","next_node":"S2","path":["S1","S2","S3","G","N4","N5","N6"],
            "task":1,"wait_time":0,"action":"advance","blocked":false}],
 "stats":{"run":1,"policy":"qlearning","conflicts":0,
          "conflicts_by_type":{"vertex":0,"edge":0,"following":0,"congestion":0},
          "deadlocks":0,"waiting":0,"total_wait_time":0,"finished_reason":null,
          "actions":{"advance":3,"wait":0,"reroute":0},"forced":0,"penalties":0},
 "mode":"qlearning"}
```

| Campo | Tipo | Qué es |
|---|---|---|
| `step` | int | Número de paso, empieza en 1 y siempre crece dentro de una corrida |
| `mode` | str | La política activa: `baseline` o `qlearning` |
| `agents[].id` | int | Identificador del AGV, estable toda la sesión |
| `agents[].x/y/z` | float | Posición **ya en coordenadas de Unity**, lista para un `Vector3` |
| `agents[].rotation` | float | Giro en grados sobre el eje vertical, 0-360 |
| `agents[].state` | str | `idle`, `moving`, `waiting`, `picking`, `dropping` o `done` |
| `agents[].node` | str | Nodo en el que está, o del que acaba de salir |
| `agents[].next_node` | str \| null | Hacia dónde va; `null` si ya llegó |
| `agents[].path` | list[str] | La ruta entera, para poder pintarla |
| `agents[].task` | int \| null | Id de la tarea que lleva |
| `agents[].wait_time` | int | Ticks **acumulados** cediendo el paso |
| `agents[].action` | str | Lo que **eligió** hacer: `advance`, `wait` o `reroute` |
| `agents[].blocked` | bool | Eligió `advance` y el motor **no le dejó** pasar |
| `agents[].leg` | str | `none`, `to_pick` (va a por la caja) o `to_drop` (la lleva al muelle) |
| `agents[].mission` | str \| null | Id de la misión que ganó pujando |
| `agents[].box` | str \| null | Id de la caja que tiene que recoger |
| `agents[].destination` | str \| null | Donde tiene que dejarla: un rack o un muelle |
| `agents[].carrying` | str \| null | Id de la caja que lleva **encima ahora mismo** |
| `agents[].busy` | int | Ticks que le quedan de la recogida o de la entrega |
| `agents[].battery` | float | Bateria que le queda, 0-100 |
| `boxes[]` | list | Cada caja: `id`, `node`, `level`, `status`, `mission` |
| `stats` | object | Los números de la corrida: conflictos por tipo, esperas, deadlocks, acciones |

**`action` y `blocked` son lo más útil para enseñar qué pasa**: uno es lo que el AGV quiso y el otro
lo que el motor le concedió. Un AGV con `action: "advance"` y `blocked: true` pidió pasar y no le
dejaron, que es justo el momento interesante en el cuello de botella.

**La posición va interpolada** entre `node` y `next_node`: un AGV a mitad de tramo manda la mitad
del camino, así que el prefab se mueve sin teletransportes. Un tramo se cruza en 4-8 ticks.

La especificación completa —todos los errores, el ciclo de vida del cliente, la reconexión y las
reglas de compatibilidad— está en **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

---

## 3. Coordenadas y escala

La simulación piensa en un plano `(px, py)` sin altura. Unity usa **Y como eje vertical**, así que
el segundo eje del plano va a **Z**:

| Eje de Python | Eje de Unity | Cómo sale |
|---|---|---|
| `px` — el ancho del almacén | `x` | `px * UNITY_SCALE` |
| — | `y` — el vertical | **siempre `0.0`**: la altura la aplica Unity con el prefab |
| `py` — el fondo del almacén | **`z`** | `py * UNITY_SCALE` |

> Lo importante es la última fila: **la Y de Python se convierte en la Z de Unity.**

```
unity_x = px * UNITY_SCALE
unity_y = 0.0
unity_z = py * UNITY_SCALE
```

`UNITY_SCALE` vale **`1.0`** y una unidad lógica es **un metro**, así que hoy los números coinciden.
Vive en `python/config.py`, y cambiarlo cambia **todas** las coordenadas exportadas de golpe: las
del snapshot y las del mapa. La conversión está en **una sola función** del proyecto,
`protocol.to_unity()`, y en ningún sitio se guarda una copia ya convertida.

Unity no tiene que convertir nada: `x`, `y` y `z` llegan listos.

Para montar la escena hace falta el grafo, que se exporta con las coordenadas ya convertidas:

```bash
python3 python/main.py map --name warehouse     # por consola, logicas y Unity al lado
python3 -c "import sys,json; sys.path.insert(0,'python'); import graph; \
print(json.dumps(graph.warehouse_graph().to_unity_dict(), indent=2))" > warehouse_unity.json
```

```json
{"name": "warehouse", "directed": false, "scale": 1.0,
 "nodes": [{"id": "G", "x": 12.0, "y": 0.0, "z": 4.0}],
 "edges": [{"from": "G", "to": "N3", "cost": 5.7}]}
```

Los ficheros de `python/maps/*.json` llevan las coordenadas **lógicas**, sin convertir: son la
fuente, no la exportación.

---

## 4. Instalación y uso

### Requisitos

Python **3.10 o superior**. **No hay dependencias que instalar**: todo es librería estándar, el
Q-Learning incluido.

```bash
python3 --version
```

> **Ojo con el Python del sistema en macOS.** El `/usr/bin/python3` que trae macOS es 3.9 y **no
> vale**: el proyecto usa `X | Y` en las anotaciones, que es 3.10+. Con `brew install python@3.12`
> tendrás uno en `/opt/homebrew/bin/python3.12`, y ese es el que hay que usar.

No hay dependencias, ni obligatorias ni opcionales: el proyecto corre con la librería estándar.

### Los cinco subcomandos

```bash
python3 python/main.py --help
```

| Subcomando | Qué hace |
|---|---|
| [`serve`](#serve) | Levanta el servidor HTTP y atiende a Unity |
| [`map`](#map) | Muestra el mapa lógico y lo valida |
| [`simulate`](#simulate) | Corre la simulación sin servidor, paso a paso por el log |
| [`train`](#train) | Entrena la Q-table, sin servidor y sin Unity |
| [`evaluate`](#evaluate) | Carga una Q-table y la juega greedy, contra la baseline |

#### `serve`

```bash
python3 python/main.py serve                                   # 127.0.0.1:5000, baseline
python3 python/main.py serve --agents 6 --port 5055            # con trafico
python3 python/main.py serve --map warehouse --agents 4 --policy qlearning \
        --model python/models/q_table.json
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--map` | `warehouse` | Mapa a servir |
| `--host` / `--port` | `127.0.0.1:5000` | Dónde escuchar |
| `--agents` | `1` | Cuántos AGVs |
| `--policy` | `baseline` | `baseline` o `qlearning` |
| `--model` | `python/models/q_table.json` | La Q-table, con `--policy qlearning` |

Con `--agents 1` no hay con quién chocar y `stats.conflicts` sale siempre 0; para ver conflictos
hacen falta varios. `--policy qlearning` sin modelo sale con código **2** y dice cómo entrenarlo.

`Ctrl+C` cierra limpio, y también un `kill` (SIGTERM).

> **macOS y el puerto 5000.** El receptor de AirPlay se queda con `*:5000`. El servidor igual
> consigue abrir `127.0.0.1:5000` porque es más específico, pero si algo va raro, apágalo en
> Ajustes → General → AirDrop y Handoff, o usa `--port`.

#### `map`

```bash
python3 python/main.py map --name warehouse
```

```
--- mapa warehouse ---
origen        : python/maps/warehouse.json
nodos         : 13
aristas       : 16
dirigido      : no
UNITY_SCALE   : 1.0
--- nodos: logicas (x, y) -> Unity (x, y, z) ---
G           (12, 4)  ->  (12, 0, 4)
N1           (0, 8)  ->  (0, 0, 8)
--- aristas ---
G    -- N3    costo 5.7
validate(): OK
```

Sale con código 1 si el mapa no es válido. Hay tres mapas: `warehouse` (13 nodos con cuello de
botella), `simple` (6 nodos, para pruebas rápidas) y `grid` (rejilla 4×4 con rutas alternativas).

#### `simulate`

Corre sin servidor y cuenta por el log lo que hace cada AGV en cada paso.

```bash
python3 python/main.py simulate --map warehouse --agents 1 --steps 100
python3 python/main.py simulate --map warehouse --agents 6 --steps 300 \
        --policy qlearning
```

```
AGV 1: S1 -> N6 | costo 27.4 | S1 -> S2 -> S3 -> G -> N4 -> N5 -> N6
paso   1 | AGV 1 | moving  | advance  | S1   -> S2   |  25% | tramo 0/6 | espera   0 | tarea 1
paso   5 | AGV 1 | waiting | reroute  | S2   -> S3   |   0% | tramo 0/6 | espera   1 | tarea 1
paso  19 | AGV 4 | waiting | advance! | G    -> N3   |   0% | tramo 0/3 | espera   6 | tarea 4
--- resumen ---
final       : llegaron todos
conflictos  : 61 (vertex 28, edge 28, following 0, congestion 5)
espera total: 128 ticks entre todos
acciones    : advance 96, wait 41, reroute 12 (desatascos forzados: 3)
```

La cuarta columna es la **acción elegida**, y el `!` dice que el motor no se la concedió.

Con `--deliveries` el trabajo sale de la subasta y el resumen cuenta las misiones; con `--bus`
además escupe la negociación entera, mensaje a mensaje.

```bash
python3 python/main.py simulate --map grid --agents 4 --steps 400 --deliveries
python3 python/main.py simulate --map grid --agents 3 --steps 60 --deliveries --bus
```

```
cajas       : 12 entregada(s) de 12
misiones    : 18 abiertas en total, 18 servidas, 0 en la bolsa
negociacion : 4090 mensajes en el bus, AGV 1 sirvio 4, AGV 2 sirvio 2, AGV 3 sirvio 7, AGV 4 sirvio 5
--- las misiones ---
M01  produccion -> rack caja A1-L1  A1  nivel 1 -> D1  | COMPLETED   | AGV 1
M07  rack -> muelle     caja D1-L1  D1  nivel 1 -> B1  | COMPLETED   | AGV 3
M13  rack -> muelle     caja A1-L1  D1  nivel 1 -> B1  | COMPLETED   | AGV 4
```

`M13` es `M01` encadenada: la misma caja, ya guardada en `D1`, saliendo ahora por el muelle.

Los números por AGV son las misiones que sirvió **a lo largo de la corrida**, una detrás de otra:
nunca lleva más de una a la vez.

#### `train`

Entrena la Q-table. **Sin servidor y sin Unity**: mil episodios son ~130.000 ticks y meter un
servidor en medio multiplicaría el tiempo sin darle al algoritmo ni un dato más. Tarda
unos 8 segundos.

```bash
python3 python/main.py train --map warehouse --agents 4 --episodes 1000 --seed 42
```

Escribe `python/models/q_table.json` (la tabla **y su metadata**: mapa, agentes, hiperparámetros,
semilla, fecha y visitas por estado), `results/training_log.csv` y `results/learning_curve.png`.

#### `evaluate`

Carga una Q-table y la juega **greedy puro** (epsilon 0, la tabla no se toca), contra la baseline
sobre los mismos episodios.

```bash
python3 python/main.py evaluate --map warehouse --agents 4
```

Sale con **2** si el modelo no está, y con **1** si está pero es de otro formato: una Q-table
cargada a ciegas sobre estados que no son los suyos no da error, da resultados malos.

### Logs

Todo sale por `stderr` con el módulo `logging`, nunca con `print`. Con `--verbose` (o `-v`) se
activa `DEBUG`, que en el servidor imprime cada petición con su respuesta. La bandera funciona
antes o después del subcomando.

---

## 5. Diseño del Q-Learning

### Por qué A\* y Q-Learning se reparten el trabajo así

El pathfinding lo resuelve **A\***: quién dice por dónde se va de `S1` a `N6` es `astar.astar()`.
Lo que se aprende es mucho más chico: **qué hacer AHORA** cuando la ruta que ya tengo me mete en un
conflicto.

La razón es de tamaño. Si el estado fuera "dónde está todo el mundo", el espacio explotaría: en
`warehouse` hay 13 nodos, y solo las posiciones de 6 AGVs ya son 13⁶ = **4.826.809** estados, sin
contar rutas ni destinos. Con el estado local de aquí abajo son **72**.

```
                    ┌─────────────────────────────────────┐
   ¿por dónde? ───▶ │  A*   ruta completa, 13 nodos       │ ──▶ path
                    └─────────────────────────────────────┘
                    ┌─────────────────────────────────────┐
 ¿y ahora qué? ───▶ │  Q-Learning   5 enteros, 72 estados │ ──▶ advance/wait/reroute
                    └─────────────────────────────────────┘
                    ┌─────────────────────────────────────┐
   ¿puedo? ──────▶  │  el motor: gate físico + desatasco  │ ──▶ lo que de verdad pasa
                    └─────────────────────────────────────┘
```

Hay una tercera capa y es la que manda: **el motor es la autoridad**. Una acción es una intención,
no una garantía; en un nodo ocupado no se entra venga la política que venga.

### El estado: seis enteros, discreto y local

`get_local_state(agent, simulation)` devuelve **siempre** una tupla de seis enteros, hasheable y
con cada campo en su rango. Es la clave de la Q-table.

| Campo | Valores | Qué pregunta |
|---|---|---|
| `next_node_occupied` | 0/1 | ¿hay alguien en el nodo al que voy a entrar? |
| `edge_conflict` | 0/1 | ¿alguien viene de frente por mi siguiente arista? |
| `queue_ahead` | 0/1/2 | ¿cuántos AGVs esperan en mis 2 nodos siguientes? (saturado en 2) |
| `distance_bucket` | 0/1/2 | ¿cuánto me falta? cerca / medio / lejos |
| `has_priority` | 0/1 | ¿soy el id menor de los que estamos en conflicto? |
| `carrying` | 0/1 | ¿voy cargado con una caja? |

```
2 × 2 × 3 × 3 × 2 = 72 estados × 3 acciones = 216 celdas de Q(s, a)
```

Ni coordenadas continuas, ni el mapa completo, ni las rutas de los demás: **cinco preguntas sobre
lo que este AGV tiene delante**. `distance_bucket` cuenta nodos que faltan de la ruta, nunca
distancia euclidiana: en un almacén dos nodos pueden estar pegados y tener medio pasillo de por
medio, así que la geometría mentiría.

### Las tres acciones

| Acción | Qué hace |
|---|---|
| `ADVANCE` | Avanzar al siguiente nodo del path que trazó A\* |
| `WAIT` | Quedarse un tick |
| `REROUTE` | Penalizar el nodo congestionado y pedirle otra ruta a A\* |

**Ninguna elige un nodo.** `REROUTE` no mueve al AGV en el mismo tick: encarece lo que tiene
delante en una tabla de penalizaciones que **caduca a los 15 ticks** (sin reloj, el mapa se
degradaría para siempre) y la ruta nueva entra en vigor en el tick siguiente.

### La recompensa

Una sola función, `reward(event)`, y los seis números en `config.py` para poder ajustarlos sin
buscar por el código:

| Evento | Valor | Cuándo |
|---|---|---|
| `TASK_COMPLETE` | **+100** | el AGV llegó a su destino |
| `PROGRESS` | **+2** | `path_index` subió: cruzó un tramo entero |
| `WAIT` | **−1** | se quedó un tick parado |
| `CONFLICT` | **−20** | eligió `ADVANCE` y el motor no le dejó pasar |
| `DEADLOCK` | **−50** | la corrida murió atascada |
| `USELESS_REROUTE` | **−3** | recalculó sin salir más barato y sin esquivar nada |

Lo que separa `CONFLICT` de `WAIT` es **haberlo intentado**. Y el −20 se le cobra al que se quedó
donde estaba, **no al que ganó el desempate**: el estado local no distingue "camino libre sin
rivales" de "camino libre y gano la disputa", así que cobrárselo al ganador envenenaría la celda
que sostiene toda la política y el almacén se pararía entero.

Hiperparámetros: `alpha` 0.2, `gamma` 0.95, `epsilon` de 1.0 a 0.05 con decaimiento exponencial
0.995, 1000 episodios, tope de 200 ticks por episodio. Todos en `config.py`.

**Una sola Q-table para todos los AGVs.** El estado es local y no lleva el id dentro, así que lo
que aprende el AGV 3 vale igual para el 1; cada episodio produce N veces más experiencia; y cambiar
el número de AGVs no invalida el modelo. Lo que se pierde es la especialización, y en un almacén de
AGVs idénticos eso no es una pérdida.

Cuándo se cobra cada recompensa y por qué la transición no se cierra en el mismo tick está
comentado en `python/qlearning.py`, junto al código que lo hace.

---


---

## 6. Limitaciones y qué haría falta

Ordenadas por lo que más pesa en los resultados.

### 1. La recompensa no cobra el rodeo (es la grande)

Rerutear cuesta **−3 una sola vez**, mientras que esperar cuesta **−1 por tick**, y la recompensa
**no cobra nunca la distancia extra** del rodeo. Peor: `PROGRESS` paga **+2 por cada nodo cruzado**,
así que dar la vuelta al almacén sale rentable. De ahí salen los 2450 reroutes de C y los 37 de A,
donde no hay nadie con quien chocar.

El dato que lo cierra: **entrenar más lo empeora**. Con 3000 episodios en vez de 1000, B pasa de
311 a 407 de makespan y de 215 a 451 reroutes. Cuanto mejor optimiza el algoritmo esa recompensa,
más rerutea, y rerutear es lo que hace daño.

> **Qué haría falta:** cobrar el reroute proporcional a `cost(ruta nueva) − cost(ruta vieja)` en vez
> de una constante. Es un cambio de una línea en `qlearning.reward()` más el evento nuevo, y es el
> que tiene los datos más claros a favor.

### 2. Un AGV que termina se queda aparcado

Al quedarse sin tarea, el AGV se para en su nodo y lo ocupa. Con 6 AGVs sobre 13 nodos es casi
seguro que uno acabe encima del último destino que queda por servir: en C, **15 de 20 corridas se
quedan a 1-3 tareas del final y ya no avanzan**, y subir el tope de 2000 a 5000 ticks no cambia ni
una. El desatasco del motor lo aparta cuando estorba, pero apartarlo es moverlo a otro sitio donde
puede volver a estorbar.

> **Qué haría falta:** una zona de aparcamiento fuera de los pasillos, o que el AGV sin tarea
> libere el nodo. Es cambiar el modelo de ocupación, no el aprendizaje.

### 3. El cara a cara sin hueco libre

El motor desatasca en tres peldaños (forzar el paso, forzar un reroute con veto, o apartar al que
estorba a un hueco libre), y con eso **no hay un solo deadlock** en ninguno de los resultados de
arriba. Pero si no queda **ningún** nodo libre en todo ese lado del mapa, no hay a dónde apartarse
y la corrida muere. Hace falta llenar un componente entero del grafo para llegar ahí.

### 4. Del protocolo

- **No se puede pedir el mapa por HTTP.** Hay que exportarlo con `main.py map`. Una ruta
  `GET /map` sería lo que falta para que Unity se configure entero por red.
- **No hay pausa ni control de velocidad desde el servidor.** Se para dejando de pedir `POST /step`.
- **No hay autenticación ni cifrado.** Es HTTP plano, pensado para `127.0.0.1`.

### 5. Del alcance

- El mapa **no cambia en marcha**: no hay obstáculos dinámicos ni pasillos que se cierren.
- Los AGVs no tienen batería ni tamaño: un AGV es un punto que ocupa un nodo. Carga sí llevan,
  pero solo una caja y solo con `--deliveries`; sin esa bandera siguen siendo puntos que van
  de un nodo a otro. El mapa `grid` declara estaciones de carga (`B4`, `C4`) que **nadie usa
  todavía**: son datos del mapa esperando a que exista un modelo de batería.
- No hay prioridades entre tareas ni ventanas de tiempo.
- La comparación es contra una baseline de "gana el id menor". No se ha medido contra un
  planificador centralizado tipo CBS, que sería la referencia fuerte.

---

## 7. Estructura del repo

```
agentesAGV/
├── python/
│   ├── config.py       constantes y configuracion del logging
│   ├── graph.py        el mapa: grafo, cajas, validacion, carga desde JSON y A*
│   ├── agent.py        el AGV: ruta, estado, entrega y puja
│   ├── missions.py     misiones, bus de mensajes y subasta
│   ├── conflicts.py    conflictos, ocupacion, acciones, reroute y la politica base
│   ├── simulation.py   el almacen en marcha: ticks, modos, desatasco y snapshot
│   ├── qlearning.py    Q-Learning: el entorno y el entrenamiento
│   ├── server.py       el contrato con Unity: servidor HTTP con JSON
│   ├── main.py         CLI con argparse
│   ├── maps/           los mapas en JSON (simple, warehouse, grid)
│   └── models/         las Q-tables entrenadas
├── docs/
│   └── PROTOCOL.md     el protocolo HTTP solo, para el equipo de Unity
├── results/            salidas de las corridas (no se versiona)
└── requirements.txt    vacio a proposito: el proyecto corre sin dependencias
```

`results/` **no se versiona** (está en `.gitignore`): son salidas y se regeneran. Las Q-tables de
`python/models/` **sí**, que son el modelo entrenado y lo que `evaluate` necesita.

El servidor recibe la simulación por **inyección de dependencia**: `serve_forever()` acepta
cualquier objeto con `get_snapshot()` y `reset()`, y en `server.py` no queda ni una línea de lógica
del almacén.

### Reglas del proyecto

- Python 3.10+, type hints en las funciones públicas y docstrings cortos.
- **Sin dependencias para ejecutar**: nada de gym, stable-baselines ni torch. El Q-Learning está
  implementado a mano con diccionarios.
- El entrenamiento corre **sin servidor y sin Unity**.
- Nada de lógica de negocio en `server.py`: el servidor solo traduce peticiones HTTP a llamadas.
- Cada módulo se puede importar y probar por separado.
- Logging con el módulo `logging`, nunca con `print`.
