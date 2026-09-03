# agentesAGV — Simulación multiagente de AGVs en un almacén

Servidor **Python** de una simulación multiagente de AGVs (vehículos de guiado automático) que se
mueven por un almacén, se disputan los pasillos y aprenden a ceder el paso.

Python es dueño de **toda** la lógica: el mapa, los agentes, el pathfinding, la detección de
conflictos y el aprendizaje. Unity es solo el cliente visual, lo desarrolla otra persona en otro
repo, y habla con Python por un socket TCP.

> En este repo **no** se escribe nada de C# ni de Unity.

**Estado: terminado.** 529 tests en verde, cinco escenarios reproducibles medidos con las dos
políticas, y una demo que enseña la diferencia en un comando.

| | |
|---|---|
| Escribir el cliente de Unity | **[docs/PROTOCOL.md](docs/PROTOCOL.md)** — el contrato solo, autosuficiente |
| Por qué el código es como es | [docs/DESIGN.md](docs/DESIGN.md) — el diario de las fases 2 a 10 |
| Verlo funcionar en un comando | `python3 python/demo.py` |

---

## Índice

1. [Qué es y qué problema resuelve](#1-qué-es-y-qué-problema-resuelve)
2. [Arquitectura y contrato con Unity](#2-arquitectura-y-contrato-con-unity)
3. [Coordenadas y escala](#3-coordenadas-y-escala)
4. [Instalación y uso](#4-instalación-y-uso)
5. [Diseño del Q-Learning](#5-diseño-del-q-learning)
6. [Resultados](#6-resultados)
7. [Limitaciones y qué haría falta](#7-limitaciones-y-qué-haría-falta)
8. [Estructura del repo y tests](#8-estructura-del-repo-y-tests)

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
mucho más pequeño —avanzar, esperar o pedir otra ruta— y por eso cabe en una tabla de 72 estados en
vez de explotar. Está desarrollado en la [sección 5](#5-diseño-del-q-learning).

Y por encima de las dos está **el motor, que es la autoridad**: una acción es una intención, no una
garantía. En un nodo ocupado no se entra, diga lo que diga la política.

### Qué hay dentro

| Pieza | Fichero | Qué hace |
|---|---|---|
| Mapa | `python/graph.py` | El almacén como grafo, con validación y carga desde JSON |
| Rutas | `python/astar.py` | A\* con penalizaciones temporales, que es el gancho del reroute |
| AGV | `python/agent.py` | Un vehículo: su ruta, su estado y su tarea |
| Conflictos | `python/conflicts.py` | Los cuatro tipos de choque y la política baseline |
| Motor | `python/simulation.py` | El tick en dos fases, el gate físico y el desatasco |
| Aprendizaje | `python/qlearning.py` | Estado, acciones, recompensa, Q-table y entrenamiento |
| Medición | `python/metrics.py` | Métricas pareadas de una corrida, y los CSV/JSON |
| Escenarios | `python/scenarios.py` | Los cinco escenarios reproducibles A-E |
| Transporte | `python/server.py`, `python/protocol.py` | El socket y el contrato, sin lógica de almacén |

---

## 2. Arquitectura y contrato con Unity

### PULL: Unity pide, Python responde

```
Unity  ──────  "GET_STATE\n"  ─────▶  Python
Unity  ◀────  "{...json...}\n"  ────  Python
```

Python **nunca** empuja datos por su cuenta y Unity nunca calcula nada: solo dibuja lo que recibe.

> **`GET_STATE` avanza la simulación un paso.** El mundo se mueve porque alguien pregunta, así que
> el ritmo lo marca el cliente: pidiendo 10 veces por segundo, el almacén corre a 10 ticks/s.

Reglas del contrato:

- TCP contra `127.0.0.1:5000`, codificación `utf-8`, mensajes delimitados por `\n`.
- **Una línea entra, una línea sale.** Siempre, incluso si el comando es desconocido o la línea
  venía vacía: así el cliente nunca pierde el emparejamiento.
- La respuesta es **una** línea: el JSON no lleva saltos de línea internos.
- El estado es completo en cada respuesta, no incremental. Unity no guarda historia.
- El comando no distingue mayúsculas de minúsculas y se admite `\r\n`.

> **Cuidado al leer del socket.** TCP entrega un flujo de bytes, no mensajes: un `send()` puede
> llegar partido en tres `recv()`, y tres respuestas pueden llegar pegadas en una. Hay que acumular
> en un buffer y cortar por `\n`. Está explicado con código en
> [PROTOCOL.md §1](docs/PROTOCOL.md#1-transporte).

### Los cuatro comandos

| Comando | Qué hace | Respuesta | Avanza el tick |
|---|---|---|:--:|
| `GET_STATE` | Pide el estado actual | El snapshot completo | **sí** |
| `RESET` | Reinicia la corrida | `{"ok":true}` | no |
| `PING` | Comprueba que el servidor vive | `{"ok":true}` | no |
| `SET_MODE baseline\|qlearning` | Cambia de política **en caliente** | `{"ok":true,"mode":"qlearning","run":3}` | no |

Un comando desconocido **no** cierra la conexión: responde y sigue.

```
-> BASURA algo
<- {"error":"unknown_command","command":"BASURA"}

-> SET_MODE turbo
<- {"error":"bad_mode","command":"SET_MODE","mode":"turbo","modes":["baseline","qlearning"]}
```

`SET_MODE` reinicia siempre, también si el modo pedido es el que ya estaba: media corrida con una
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
| `agents[].state` | str | `idle`, `moving`, `waiting` o `done` |
| `agents[].node` | str | Nodo en el que está, o del que acaba de salir |
| `agents[].next_node` | str \| null | Hacia dónde va; `null` si ya llegó |
| `agents[].path` | list[str] | La ruta entera, para poder pintarla |
| `agents[].task` | int \| null | Id de la tarea que lleva |
| `agents[].wait_time` | int | Ticks **acumulados** cediendo el paso |
| `agents[].action` | str | Lo que **eligió** hacer: `advance`, `wait` o `reroute` |
| `agents[].blocked` | bool | Eligió `advance` y el motor **no le dejó** pasar |
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

Para **desarrollar** (correr los tests y dibujar las curvas) hay dos opcionales:

```bash
python3 -m pip install -r requirements-dev.txt      # pytest y matplotlib
```

Sin matplotlib todo funciona: se avisa por el log y no se dibuja el PNG. Sin pytest también, con
`python3 -m unittest discover -s tests -t .`.

### Los siete subcomandos

```bash
python3 python/main.py --help
```

| Subcomando | Qué hace |
|---|---|
| [`serve`](#serve) | Levanta el servidor TCP y atiende a Unity |
| [`map`](#map) | Muestra el mapa lógico y lo valida |
| [`simulate`](#simulate) | Corre la simulación sin servidor, paso a paso por el log |
| [`train`](#train) | Entrena la Q-table, sin servidor y sin Unity |
| [`evaluate`](#evaluate) | Carga una Q-table y la juega greedy, contra la baseline |
| [`benchmark`](#benchmark) | Enfrenta las políticas semilla a semilla y escribe `results/` |
| [`scenario`](#scenario) | Corre los cinco escenarios y saca la tabla resumen |

Y aparte del CLI, la demo: [`demo.py`](#la-demo).

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
python3 python/main.py simulate --map warehouse --agents 1 --steps 100 --headless
python3 python/main.py simulate --map warehouse --agents 6 --steps 300 --headless \
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

#### `train`

Entrena la Q-table. **Sin servidor y sin Unity**: mil episodios son ~130.000 ticks y meter un
socket en medio multiplicaría el tiempo por el ping sin darle al algoritmo ni un dato más. Tarda
unos 8 segundos.

```bash
python3 python/main.py train --map warehouse --agents 4 --episodes 1000 --seed 42
python3 python/main.py train --scenario C --episodes 1000     # entrenar EN un escenario
```

Escribe `python/models/q_table.json` (la tabla **y su metadata**: mapa, agentes, hiperparámetros,
semilla, fecha y visitas por estado), `results/training_log.csv` y `results/learning_curve.png`.

#### `evaluate`

Carga una Q-table y la juega **greedy puro** (epsilon 0, la tabla no se toca), contra la baseline
sobre los mismos escenarios.

```bash
python3 python/main.py evaluate --map warehouse --agents 4
```

Sale con **2** si el modelo no está, y con **1** si está pero es de otro formato: una Q-table
cargada a ciegas sobre estados que no son los suyos no da error, da resultados malos.

#### `benchmark`

Enfrenta las políticas **semilla a semilla**: para cada semilla se construye un escenario y se
corre con las dos. Mismo mapa, mismos AGVs, mismos destinos, misma cola; lo único que cambia es la
política.

```bash
python3 python/main.py benchmark --agents 4 --runs 20
```

Escribe `results/<policy>.csv` (una fila por semilla), `results/comparison.json` con el resumen
de las dos y `results/comparison.png` con los paneles.

#### `scenario`

Los cinco escenarios de la [sección 6](#6-resultados), con las dos políticas y su tabla resumen.

```bash
python3 python/main.py scenario --name C --policy qlearning --runs 20
python3 python/main.py scenario --all --runs 20                    # los cinco, las dos
python3 python/main.py scenario --all --runs 20 --per-scenario-model
```

Escribe `results/scenario_<letra>_<policy>.csv` y `results/summary_table.csv`, que es el que se
pega en el reporte.

### La demo

```bash
python3 python/demo.py                       # escenario C con las dos politicas
python3 python/demo.py --scenario B          # otro escenario
python3 python/demo.py --rate 10             # al ritmo de Unity
python3 python/demo.py --policy qlearning    # solo una
```

Levanta el servidor de verdad, se conecta a él como cliente —el mundo avanza porque alguien
pregunta, que es el contrato— y va imprimiendo métricas cada 50 ticks. Al terminar manda
`SET_MODE` para repetir **el mismo escenario** con la otra política y compara las dos.

Unity puede conectarse al mismo puerto y mirar mientras corre.

```
--- BASELINE | escenario B: Congestion media ---
paso    50 | baseline  | tareas  0/12 | conf/tick  2.68 | espera/tick  3.44 | reroutes     2 | desatascos   4
paso   210 | baseline  | tareas 12/12 | conf/tick  1.98 | espera/tick  2.45 | reroutes     6 | desatascos  10  <- final

--- QLEARNING | escenario B: Congestion media ---
paso   112 | qlearning | tareas 12/12 | conf/tick  1.25 | espera/tick  1.20 | reroutes    99 | desatascos   1  <- final

VEREDICTO: el Q-Learning MEJORA. Makespan medio 112.0 contra 210.0 ticks (-46.7%).
```

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

### El estado: cinco enteros, discreto y local

`get_local_state(agent, simulation)` devuelve **siempre** una tupla de cinco enteros, hasheable y
con cada campo en su rango. Es la clave de la Q-table.

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

El desarrollo completo —cuándo se cobra cada recompensa, por qué la transición no se cierra en el
mismo tick, y cómo se enchufa la política sin tocar el motor— está en
[DESIGN.md](docs/DESIGN.md#el-entorno-de-q-learning).

---

## 6. Resultados

### Qué aprendió

La regla que sale de las 111.000 actualizaciones es una sola, y es la que se buscaba: **si el nodo
de delante está ocupado, no intentes entrar**. De los 24 estados con `next_node_occupied = 1` y al
menos 50 visitas detrás, `ADVANCE` no es la mejor acción en **ninguno**:

```
      estado  visitas  mejor    advance /    wait / reroute
   0|0|0|0|1    17058  advance    68.69 /   11.68 /   16.82   nada delante -> pasa
   1|0|0|0|0    21656  reroute   -11.56 /   -4.99 /   12.23   ocupado y sin prioridad -> rodea
   1|0|0|0|1    16607  wait       -0.57 /   17.58 /    8.36   ocupado y con prioridad -> espera turno
   1|1|1|0|0     7993  reroute   -16.76 /   -2.39 /    8.29   de frente y con cola -> rodear
```

`ADVANCE` va de **+68.69 con el camino libre a −16.76 con alguien de frente y cola delante**: 85
puntos de diferencia entre la misma acción en dos sitios distintos, que es exactamente lo que el
estado local tenía que poder distinguir.

### Los cinco escenarios

Cinco escenarios reproducibles que barren el rango, de un almacén casi vacío a un cuello de
botella. **De un escenario no se sortea nada estructural**: el mapa, cuántos AGVs, dónde arranca
cada uno y de qué conjunto salen las tareas están escritos a mano. Lo único que depende de la
semilla es qué destinos concretos tocan.

| | Escenario | Mapa | AGVs | Tareas | Qué prueba |
|---|---|---|--:|--:|---|
| **A** | Baja congestión | `warehouse` | 2 | 6 | Cada AGV en su mitad: nadie se cruza con nadie |
| **B** | Congestión media | `warehouse` | 4 | 12 | Cuatro rutas que se cruzan en `S3` |
| **C** | Alta congestión | `warehouse` | 6 | 18 | Seis AGVs en trece nodos: por encima de su capacidad |
| **D** | Cuello de botella | `warehouse` | 4 | 16 | Toda tarea cruza `G`, el **único** paso |
| **E** | Rutas alternativas | `grid` | 4 | 16 | El mismo cruce que D, pero con rutas de **igual coste** |

**El par D/E es el experimento.** Mismo número de AGVs, mismas tareas, misma presión (1.74
conflictos por tick en D contra 1.70 en E); lo único que cambia es si existe una ruta alternativa.
En D, `G` es punto de articulación y penalizarlo no le da a A\* otra ruta, le da una peor. En E, la
rejilla tiene varias rutas mínimas entre dos nodos cualesquiera.

### Baseline contra Q-Learning: 5 escenarios × 2 políticas × 20 semillas

Con la Q-table general del repo. Es `results/summary_table.csv` tal cual:

| Esc. | Escenario | AGVs | Política | Makespan | Completas | Tareas | Conf/tick | Espera/tick | Reroutes | Gana |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|:--:|
| A | Baja congestión | 2 | baseline | **50.5** | 95 % | 96.7 % | 0.10 | 0.13 | 0.3 | — |
| | | | Q-Learning | 122.8 | 85 % | 93.3 % | 0.09 | 0.13 | 37.6 | 3/20 |
| B | Congestión media | 4 | baseline | 308.7 | 85 % | 98.8 % | 1.72 | 2.17 | 10.5 | — |
| | | | Q-Learning | **294.5** | 75 % | 97.5 % | 1.14 | 1.22 | 172.7 | **15/20** |
| C | Alta congestión | 6 | baseline | 1831.8 | 25 % | 93.9 % | 2.53 | 3.51 | 171.0 | — |
| | | | Q-Learning | 1724.2 | 25 % | 94.7 % | 1.56 | 1.98 | 2450.6 | 4/20 |
| D | Cuello de botella | 4 | baseline | 497.1 | **95 %** | 99.7 % | 1.74 | 2.42 | 23.3 | — |
| | | | Q-Learning | 505.5 | 70 % | 97.2 % | 1.41 | 1.71 | 620.6 | 10/20 |
| E | Rutas alternativas | 4 | baseline | 216.5 | 100 % | 100.0 % | 1.70 | 2.29 | 6.2 | — |
| | | | Q-Learning | **103.0** | 100 % | 100.0 % | 0.71 | 0.83 | 57.1 | **20/20** |

**Dónde mejora y dónde no, sin maquillar:**

- **E (rutas alternativas): mejora, y mucho.** −52 % de makespan, −58 % de conflictos por tick,
  −64 % de espera por tick, y gana **las 20 semillas** sin perder una tarea. Es el único escenario
  donde el Q-Learning gana limpio.
- **B (congestión media): mixto, y la media engaña.** Gana 15 de 20 semillas y baja los conflictos
  por tick un 34 %, pero completa menos corridas (75 % contra 85 %): va mejor casi siempre y se
  cuelga de vez en cuando.
- **D (cuello de botella): no aporta.** Empata en makespan y **termina 25 puntos menos de
  corridas**, a cambio de 620 reroutes contra 23. Donde no hay ruta alternativa, recalcular solo
  cuesta ticks.
- **C (alta congestión): empate.** Despacha un 0.8 % más de tareas con 2450 reroutes contra 171.
- **A (baja congestión): estorba.** Con dos AGVs que no se cruzan, la baseline tarda 50 ticks y el
  Q-Learning **123**: rerutea aunque no haya nadie delante.

> El patrón es limpio y es el que se buscaba: **el Q-Learning gana exactamente donde el REROUTE
> tiene a dónde ir (E), empata donde hay congestión pero también sitio (B, C), y pierde donde
> recalcular no puede ayudar (D) o no hace falta (A).**

Los totales crudos (conflictos, espera) **premian al que muere antes**: una corrida que se atasca
acumula menos de todo. Por eso van las tasas por tick, y por eso `completas` sale en la tabla.

---

## 7. Limitaciones y qué haría falta

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

### 3. La saturación de C no se mide con el makespan

Por lo anterior, en esas 15 corridas el makespan **es el tope que uno elija**, no una medida de
nada. `scenarios.scenario_verdict()` decide por tareas despachadas cuando la completitud baja del
50 %, y C hay que leerlo por trabajo despachado y no por ticks. Es un límite del montaje, no del
motor: las dos políticas lo sufren igual y el escenario sigue siendo pareado.

### 4. El cara a cara sin hueco libre

El motor desatasca en tres peldaños (forzar el paso, forzar un reroute con veto, o apartar al que
estorba a un hueco libre), y con eso **no hay un solo deadlock** en ninguno de los resultados de
arriba. Pero si no queda **ningún** nodo libre en todo ese lado del mapa, no hay a dónde apartarse
y la corrida muere. Hace falta llenar un componente entero del grafo para llegar ahí.

### 5. Una Q-table por escenario ayuda, pero no basta

Entrenar en el régimen en que se evalúa (`train --scenario X`) baja el makespan un 10 % de media y
**recupera la completitud de la baseline** (80 % contra 71 % de la general): C pasa de 25 % a 55 %
de corridas completas. Pero pierde en B y en E, y por el mismo motivo de siempre: rerutea más.

### 6. Del protocolo

- **No hay forma de mirar sin avanzar el mundo**: `GET_STATE` siempre consume un tick, así que dos
  clientes conectados ven pasos distintos. Un comando `PEEK` lo resolvería, y de paso daría el modo
  pausa.
- **No se puede pedir el mapa por el socket.** Hay que exportarlo con `map`. Un `GET_MAP` sería lo
  que falta para que Unity se configure entero por red.
- **No hay autenticación ni cifrado.** Está pensado para `127.0.0.1`.

### 7. Del alcance

- El mapa **no cambia en marcha**: no hay obstáculos dinámicos ni pasillos que se cierren.
- Los AGVs no tienen batería, ni carga, ni tamaño: un AGV es un punto que ocupa un nodo.
- No hay prioridades entre tareas ni ventanas de tiempo.
- La comparación es contra una baseline de "gana el id menor". No se ha medido contra un
  planificador centralizado tipo CBS, que sería la referencia fuerte.

---

## 8. Estructura del repo y tests

```
agentesAGV/
├── python/
│   ├── config.py       constantes (red, ticks, Unity, semilla, umbrales, recompensas)
│   ├── logs.py         configuracion del logging
│   ├── protocol.py     el contrato: comandos, serializacion y coordenadas
│   ├── server.py       servidor TCP, solo transporte
│   ├── graph.py        el mapa logico: grafo, validacion y carga desde JSON
│   ├── astar.py        A* con penalizaciones temporales
│   ├── agent.py        el AGV: ruta, estado y tarea
│   ├── conflicts.py    conflictos, ocupacion, acciones, reroute y la politica base
│   ├── simulation.py   el almacen en marcha: ticks, modos, desatasco y snapshot
│   ├── qlearning.py    Q-Learning: el entorno y el entrenamiento
│   ├── metrics.py      metricas pareadas de una corrida, CSV y JSON
│   ├── scenarios.py    los cinco escenarios reproducibles A-E
│   ├── demo.py         la demostracion final
│   ├── main.py         CLI con argparse
│   ├── maps/           los mapas en JSON (simple, warehouse, grid)
│   └── models/         las Q-tables entrenadas
├── docs/
│   ├── PROTOCOL.md     el protocolo TCP solo, para el equipo de Unity
│   └── DESIGN.md       el diario de las fases: por que el codigo es como es
├── results/            salidas de las corridas (no se versiona)
├── tests/              529 tests, y el cliente falso de Unity
├── requirements.txt        vacio a proposito: el proyecto corre sin dependencias
└── requirements-dev.txt    pytest y matplotlib, solo para desarrollar
```

`results/` **no se versiona** (está en `.gitignore`): son salidas y se regeneran. Las Q-tables de
`python/models/` **sí**, que son el modelo entrenado y lo que `evaluate` necesita.

El servidor recibe la simulación por **inyección de dependencia**: `serve_forever()` acepta
cualquier objeto con `get_snapshot()` y `reset()`, y en `server.py` no queda ni una línea de lógica
del almacén. Es lo que permite que `demo.py` sirva un escenario con cola de tareas sin tocar el
servidor.

### Tests

```bash
pytest                                            # los 529
pytest tests/test_integration.py -v               # solo uno
pytest -q -k "protocol or simulation"             # por nombre
python3 -m unittest discover -s tests -t .        # sin instalar nada
```

Los tests están escritos con `unittest.TestCase` de la librería estándar **a propósito**: pytest
los ejecuta tal cual, así que los dos comandos valen y quien no quiera instalar nada sigue
teniendo el runner de siempre.

| Fichero | Qué cubre |
|---|---|
| `test_graph.py` | El mapa: validación, carga/guardado y conversión de coordenadas |
| `test_astar.py` | A\*: optimalidad contra búsqueda exhaustiva, aristas válidas, sin ruta, penalties |
| `test_agent.py` | El AGV, y que su estado es **suyo**: nadie comparte ruta con nadie |
| `test_conflicts.py` | `vertex`, `edge`, deadlock y la invariante de un AGV por nodo |
| `test_qlearning.py` | Forma del estado, actualización de la Q-table, save/load, greedy contra epsilon |
| `test_simulation.py` | Determinismo por semilla y snapshot válido en cada tick |
| `test_protocol.py` | Comandos, comando desconocido y **mensajes fragmentados en TCP** |
| `test_integration.py` | Corrida completa baseline y qlearning de punta a punta, por el socket |
| `test_server.py` | El servidor TCP contra un socket de verdad |
| `test_main.py` | El CLI y sus siete subcomandos |
| `test_training.py` | Bellman, los dos modos, que aprende y que es reproducible |
| `test_phase8.py` | Los dos modos, la acción en el snapshot, `SET_MODE` y el desatasco |
| `test_metrics.py` | Escenarios pareados, métricas, CSV/JSON y el reporte |
| `test_scenarios.py` | Los cinco escenarios, su reproducibilidad y la tabla resumen |
| `test_config.py`, `test_logs.py` | Las constantes y el logging |

Algunos son más que un test unitario: `test_astar.py` compara A\* contra una **búsqueda exhaustiva**
sobre los 186 pares ordenados de nodos de los dos mapas; `test_conflicts.py` corre **500 ticks con
6 AGVs** comprobando en cada tick que no hay dos en el mismo nodo; `test_training.py` **entrena 300
episodios de verdad** y comprueba que la recompensa sube y los conflictos bajan; `test_phase8.py`
comprueba que en **10 corridas de 1000 ticks con 6 AGVs no hay un solo deadlock**.

### Cliente falso de Unity

`tests/fake_unity_client.py` hace de Unity mientras Unity no existe: se conecta, pide `GET_STATE` a
un ritmo fijo, valida que cada respuesta cumpla el contrato y comprueba que `step` va creciendo.
Sale con código 1 si algo falla, e imprime latencias mín/media/p95/máx.

```bash
python3 python/main.py serve --port 5055 &
python3 tests/fake_unity_client.py --port 5055 --seconds 60 --rate 10
```

### Reglas del proyecto

- Python 3.10+, type hints en las funciones públicas y docstrings cortos.
- **Sin dependencias para ejecutar**: nada de gym, stable-baselines ni torch. El Q-Learning está
  implementado a mano con diccionarios.
- El entrenamiento corre **sin servidor y sin Unity**.
- Nada de lógica de negocio en `server.py`: el servidor solo traduce sockets a llamadas.
- Cada módulo se puede importar y probar por separado.
- Logging con el módulo `logging`, nunca con `print`.
