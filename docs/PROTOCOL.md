# Protocolo TCP del servidor de AGVs

Especificación completa del contrato entre el servidor Python y el cliente de Unity. **Este
fichero se basta solo**: no hace falta leer el código ni el README para escribir el cliente.

Versión del contrato: **fase 11**. Todo el JSON de este documento está copiado tal cual de una
corrida real (`serve --map warehouse --agents 3 --policy qlearning`), no está escrito a mano.

---

## 1. Transporte

| | |
|---|---|
| Protocolo | TCP |
| Dirección por defecto | `127.0.0.1:5000` (configurable con `--host` / `--port`) |
| Codificación | `utf-8` |
| Delimitador | `\n` (salto de línea). Se admite `\r\n` en lo que envía el cliente |
| Concurrencia | Un hilo por cliente. Varios clientes a la vez están permitidos |

### La regla que lo gobierna todo

> **Una línea entra, una línea sale.** Siempre, sin excepción: también cuando el comando es
> desconocido, cuando viene mal escrito o cuando la línea llega vacía.

Nunca hay respuestas de más ni de menos, así que el cliente puede emparejar cada petición con su
respuesta por orden de llegada, sin identificadores ni números de secuencia.

La respuesta es **una sola línea**: el JSON no lleva saltos de línea dentro. Un `\n` en el flujo es
siempre un final de mensaje.

### Cómo leer del socket (importante)

TCP entrega un **flujo de bytes**, no mensajes. Un `send()` del servidor puede llegar partido en
tres `recv()`, y tres respuestas pueden llegar pegadas en uno solo. Un cliente que dé por hecho
que cada `Receive()` trae exactamente una respuesta **funciona en localhost y se rompe en cuanto
hay red de por medio**.

Lo correcto es acumular en un buffer y cortar por `\n`:

```csharp
// C#, orientativo
private readonly StringBuilder _buffer = new StringBuilder();

string LeerLinea(NetworkStream stream) {
    while (true) {
        int corte = _buffer.ToString().IndexOf('\n');
        if (corte >= 0) {
            string linea = _buffer.ToString(0, corte);
            _buffer.Remove(0, corte + 1);   // OJO: +1, para tragarse el \n
            return linea;
        }
        var trozo = new byte[4096];
        int leidos = stream.Read(trozo, 0, trozo.Length);
        if (leidos == 0) throw new IOException("el servidor cerro");
        _buffer.Append(Encoding.UTF8.GetString(trozo, 0, leidos));
    }
}
```

Hay tres tests que comprueban exactamente esto contra un socket de verdad, en
`tests/test_protocol.py::TestFragmentacionTCP`: un comando partido en cuatro envíos, tres comandos
pegados en uno, y cien peticiones seguidas sin que se pierda el emparejado.

### El modelo PULL: Unity pide, Python responde

Python **nunca** envía nada por su cuenta. No hay push, ni streaming, ni suscripciones. Y hay una
consecuencia que conviene entender antes de escribir el cliente:

> **`GET_STATE` avanza la simulación un paso.** El mundo se mueve porque alguien pregunta.

Es decir: el ritmo de la simulación lo marca el cliente. Pidiendo 10 veces por segundo
(`config.TICK_RATE`), el almacén corre a 10 ticks por segundo. Si el cliente se para, el almacén se
para; si pide el doble, va al doble de rápido.

Si **dos** clientes están conectados, cada `GET_STATE` de cada uno consume su propio tick, así que
verán pasos distintos. Es el comportamiento esperado, no un fallo: para mirar sin mover el mundo
todavía no hay comando (ver [Limitaciones](#7-limitaciones-conocidas-del-protocolo)).

---

## 2. Comandos

Cuatro, y ninguno distingue mayúsculas de minúsculas. Los espacios sobrantes se ignoran.

| Comando | Argumento | Qué hace | Avanza el tick |
|---|---|---|:--:|
| `GET_STATE` | — | Devuelve el estado completo de la simulación | **sí** |
| `RESET` | — | Reinicia la corrida: `step` vuelve a 1 y `run` sube | no |
| `PING` | — | Comprueba que el servidor vive | no |
| `SET_MODE` | `baseline` \| `qlearning` | Cambia de política **en caliente** y reinicia la corrida | no |

### `PING`

```
-> PING
<- {"ok":true}
```

### `GET_STATE`

Avanza un paso y devuelve el estado entero. El formato completo está en la
[sección 3](#3-el-snapshot).

```
-> GET_STATE
<- {"step":1,"agents":[...],"stats":{...},"mode":"qlearning"}
```

### `RESET`

Vuelve al paso cero de forma determinista: mismas rutas, mismas tareas, misma semilla. Lo único que
**no** se borra es `stats.deadlocks`, que cuenta los de la sesión entera.

```
-> RESET
<- {"ok":true}
```

Después de un `RESET`, el siguiente `GET_STATE` devuelve `step: 1` y `stats.run` una unidad más
alto. **Así se distingue un reinicio de un `step` estancado.**

### `SET_MODE`

El único comando con argumento. Cambia la política que decide qué hace cada AGV y **arranca una
corrida limpia con el mismo escenario**: media corrida con una política y media con otra no son
una corrida de ninguna de las dos.

```
-> SET_MODE baseline
<- {"ok":true,"mode":"baseline","run":2}
```

| Campo | Tipo | Qué es |
|---|---|---|
| `ok` | bool | Siempre `true` cuando el cambio salió bien |
| `mode` | str | La política que queda activa |
| `run` | int | El número de corrida nuevo |

Sus tres errores, y ninguno cierra la conexión:

```
-> SET_MODE turbo
<- {"error":"bad_mode","command":"SET_MODE","mode":"turbo","modes":["baseline","qlearning"]}

-> SET_MODE
<- {"error":"bad_mode","command":"SET_MODE","mode":"","modes":["baseline","qlearning"]}
```

| `error` | Cuándo pasa |
|---|---|
| `bad_mode` | El modo no existe, o no venía ninguno. `modes` trae los que sí valen |
| `set_mode_failed` | El modo existe pero no se pudo montar. Lleva `detail` con el motivo (lo típico: se pidió `qlearning` y no hay Q-table entrenada en el disco) |
| `mode_not_supported` | Esta simulación no sabe cambiar de política |

### Comando desconocido

No cierra la conexión: responde y sigue. El `command` que devuelve va **en mayúsculas** y sin los
argumentos.

```
-> BASURA algo
<- {"error":"unknown_command","command":"BASURA"}

->
<- {"error":"unknown_command","command":""}
```

---

## 3. El snapshot

Lo que devuelve `GET_STATE`. Aquí va **entero y de una corrida real**, partido en varias líneas
para que se lea; en el cable va en una sola:

```json
{
  "step": 1,
  "agents": [
    {"id": 1, "x": 1.0, "y": 0.0, "z": 0.0, "rotation": 90.0, "state": "moving",
     "node": "S1", "next_node": "S2",
     "path": ["S1", "S2", "S3", "G", "N4", "N5", "N6"],
     "task": 1, "wait_time": 0, "action": "advance", "blocked": false},
    {"id": 2, "x": 19.0, "y": 0.0, "z": 0.0, "rotation": 270.0, "state": "moving",
     "node": "S5", "next_node": "S4", "path": ["S5", "S4", "G"],
     "task": 2, "wait_time": 0, "action": "advance", "blocked": false},
    {"id": 3, "x": 1.0, "y": 0.0, "z": 8.0, "rotation": 90.0, "state": "moving",
     "node": "N1", "next_node": "N2", "path": ["N1", "N2", "N3", "G", "N4"],
     "task": 3, "wait_time": 0, "action": "advance", "blocked": false}
  ],
  "stats": {
    "run": 1, "policy": "qlearning", "conflicts": 0,
    "conflicts_by_type": {"vertex": 0, "edge": 0, "following": 0, "congestion": 0},
    "deadlocks": 0, "waiting": 0, "total_wait_time": 0, "finished_reason": null,
    "actions": {"advance": 3, "wait": 0, "reroute": 0}, "forced": 0, "penalties": 0
  },
  "mode": "qlearning"
}
```

### Nivel raíz

| Campo | Tipo | Desde | Qué es |
|---|---|---|---|
| `step` | int | fase 1 | Número de paso, empieza en 1 y **siempre crece** dentro de una corrida |
| `agents` | list | fase 1 | Un objeto por AGV, siempre todos y siempre en el mismo orden |
| `stats` | object | fase 5 | Los números de la corrida |
| `mode` | str | fase 8 | La política activa: `baseline` o `qlearning` |

### `agents[]`

| Campo | Tipo | Desde | Qué es |
|---|---|---|---|
| `id` | int | fase 1 | Identificador del AGV. Estable durante toda la sesión |
| `x`, `y`, `z` | float | fase 1 | Posición **ya en coordenadas de Unity** (ver sección 4) |
| `rotation` | float | fase 1 | Giro en grados sobre el eje vertical, 0-360 |
| `state` | str | fase 1 | `idle`, `moving`, `waiting` o `done` |
| `node` | str | fase 3 | Nodo en el que está, o del que acaba de salir |
| `next_node` | str \| null | fase 3 | Hacia dónde va; `null` si ya llegó |
| `path` | list[str] | fase 3 | La ruta entera, de origen a destino. Para pintarla |
| `task` | int \| null | fase 3 | Id de la tarea que lleva |
| `wait_time` | int | fase 5 | Ticks **acumulados** cediendo el paso en toda la corrida |
| `action` | str | fase 8 | Lo que **eligió** hacer: `advance`, `wait` o `reroute` |
| `blocked` | bool | fase 8 | Eligió `advance` y el motor **no le dejó** pasar |

**Los cuatro estados:**

| `state` | Qué significa | Qué hace el prefab |
|---|---|---|
| `idle` | Sin ruta que seguir (no hay camino a su destino) | Quieto |
| `moving` | Cruzando un tramo, o parado en un nodo a punto de salir | Se mueve hacia `x,y,z` |
| `waiting` | Cediendo el paso: no se movió en este tick | Quieto, y se le puede poner un icono |
| `done` | Llegó a su destino | Aparcado |

**`action` y `blocked` van juntos**, y es lo más útil para enseñar qué está pasando: `action` es lo
que el AGV **quiso** hacer y `blocked` es lo que el motor **le concedió**. Un AGV con
`action: "advance"` y `blocked: true` pidió pasar y no le dejaron, que es exactamente el momento
interesante en un cuello de botella. Un AGV a media travesía está ejecutando un avance
(`action: "advance"`); el que ya llegó sale como `wait`.

**La posición va interpolada.** Un AGV a mitad de camino entre `node` y `next_node` manda la
posición de la mitad del tramo, no la del nodo de destino. Así el prefab se puede mover a `x,y,z`
directamente, sin teletransportes ni interpolación por parte de Unity. Un tramo tarda entre 4 y 8
ticks en cruzarse, según su coste.

### `stats`

| Campo | Tipo | Desde | Qué es |
|---|---|---|---|
| `run` | int | fase 5 | Número de corrida. Sube en cada `RESET` y en cada `SET_MODE` |
| `policy` | str | fase 5 | La política activa (lo mismo que `mode`, en la raíz) |
| `conflicts` | int | fase 5 | Conflictos detectados en esta corrida |
| `conflicts_by_type` | object | fase 5 | Desglose: `vertex`, `edge`, `following`, `congestion` |
| `deadlocks` | int | fase 5 | Atascos de la **sesión**; no se borra al reiniciar |
| `waiting` | int | fase 5 | Cuántos AGVs están cediendo el paso ahora mismo |
| `total_wait_time` | int | fase 5 | Suma del `wait_time` de todos |
| `finished_reason` | str \| null | fase 5 | `"deadlock"` si la corrida murió atascada |
| `actions` | object | fase 8 | Decisiones de la corrida por tipo: `advance` / `wait` / `reroute` |
| `forced` | int | fase 8 | Veces que el motor tuvo que desatascar a la fuerza |
| `penalties` | int | fase 8 | Penalizaciones de ruta vivas ahora mismo |

**Los cuatro tipos de conflicto:**

| Tipo | Qué es |
|---|---|
| `vertex` | Dos o más AGVs quieren el mismo nodo. El que ya está encima cuenta como uno más |
| `edge` | Cruce de frente: A va de X a Y mientras B va de Y a X |
| `following` | A quiere entrar en el nodo que B está dejando. No se permite |
| `congestion` | Un AGV lleva demasiado esperando, o hay varios esperando en una zona |

---

## 4. Coordenadas y escala

La simulación piensa en un plano `(px, py)` sin altura. Unity usa **Y como eje vertical**, así que
el segundo eje del plano va a **Z**:

| Eje de Python | Eje de Unity | Cómo sale |
|---|---|---|
| `px` — el ancho del almacén | `x` | `px * UNITY_SCALE` |
| — | `y` — el vertical | **siempre `0.0`**: la altura la aplica Unity con el prefab |
| `py` — el fondo del almacén | **`z`** | `py * UNITY_SCALE` |

> La fila que importa es la última: **la Y de Python se convierte en la Z de Unity.**

`UNITY_SCALE` vale **`1.0`** y una unidad lógica es **un metro**, así que hoy los números coinciden.
Está en `python/config.py`, y cambiarlo cambia **todas** las coordenadas exportadas de golpe: las
del snapshot y las del mapa. La conversión vive en una sola función del proyecto
(`protocol.to_unity()`) y no hay ninguna copia ya convertida guardada en ningún sitio.

**Unity no tiene que convertir nada**: `x`, `y` y `z` llegan listos para asignar a un `Vector3`.

### El mapa

Para montar la escena hace falta el grafo. **No se puede pedir por el socket** (ver
[Limitaciones](#7-limitaciones-conocidas-del-protocolo)); hay dos formas de sacarlo:

**1. Verlo por consola**, con las coordenadas lógicas y las de Unity una al lado de la otra:

```bash
python3 python/main.py map --name warehouse
```

```
--- nodos: logicas (x, y) -> Unity (x, y, z) ---
G           (12, 4)  ->  (12, 0, 4)
N1           (0, 8)  ->  (0, 0, 8)
N2           (4, 8)  ->  (4, 0, 8)
--- aristas ---
G    -- N3    costo 5.7
```

**2. Exportarlo a JSON** con las coordenadas ya convertidas, que es lo que conviene para generar la
escena. `graph.to_unity_dict()` devuelve esta estructura:

```json
{
  "name": "warehouse",
  "directed": false,
  "scale": 1.0,
  "nodes": [{"id": "G", "x": 12.0, "y": 0.0, "z": 4.0},
            {"id": "N1", "x": 0.0, "y": 0.0, "z": 8.0}],
  "edges": [{"from": "G", "to": "N3", "cost": 5.7},
            {"from": "G", "to": "N4", "cost": 5.7}]
}
```

```bash
# volcarlo a un fichero para importarlo desde Unity
python3 -c "import sys,json; sys.path.insert(0,'python'); import graph; \
print(json.dumps(graph.warehouse_graph().to_unity_dict(), indent=2))" > warehouse_unity.json
```

Los ficheros de `python/maps/*.json` llevan las coordenadas **lógicas**, sin convertir: son la
fuente, no la exportación. Si `UNITY_SCALE` cambiara, las de Unity cambian y las del fichero no.

El mapa **no cambia durante una sesión**: se pide una vez al arrancar y se cachea.

---

## 5. Ciclo de vida del cliente

```
1. conectar a 127.0.0.1:5000
2. (opcional) PING para comprobar que responde
3. bucle a ~10 Hz:
      enviar GET_STATE
      leer una linea
      parsear el JSON
      mover los prefabs a (x, y, z) y girarlos a rotation
4. cerrar el socket
```

Un cliente de referencia en Python, con validación de la forma del contrato y medición de
latencias, está en `tests/fake_unity_client.py`:

```bash
python3 python/main.py serve --port 5055 &
python3 tests/fake_unity_client.py --port 5055 --seconds 30 --rate 10
```

### Reconexión y errores

- Si el servidor se cae, el `recv()` devuelve 0 bytes. Hay que reconectar; al reconectar se sigue
  la **misma** simulación, no una nueva.
- Un JSON que no parsea no debería ocurrir nunca. Si ocurre, lo más probable es que el cliente esté
  cortando mal por `\n` (ver sección 1).
- El servidor no cierra la conexión por un comando malo. Si se cierra, es que se cayó.
- Una línea sin `\n` de más de 64 KB sí cierra la conexión: es la protección contra un cliente que
  se quedó colgado a mitad de envío.

---

## 6. Compatibilidad

El formato está **congelado en cuanto a lo que ya existe**:

- Los campos que ya están **no cambian de nombre ni de tipo**, nunca.
- Las fases nuevas **solo añaden** campos.
- `JsonUtility` de Unity ignora lo que no conoce, así que un cliente escrito contra la fase 1
  sigue funcionando hoy sin tocarle una línea.

Lo que sí puede cambiar de una corrida a otra: el número de AGVs (`--agents`), el mapa (`--map`) y
la política (`--policy` o `SET_MODE`). Nada de eso cambia la **forma** del JSON.

---

## 7. Limitaciones conocidas del protocolo

Lo que hoy no se puede hacer, por si el cliente lo necesita:

- **No hay forma de mirar sin avanzar el mundo.** `GET_STATE` siempre consume un tick. Con dos
  clientes conectados, cada uno avanza la simulación por su cuenta y ven pasos distintos. Un
  comando `PEEK` que devolviera el estado sin tickear resolvería tanto esto como el modo pausa.
- **No hay pausa ni control de velocidad desde el socket.** El ritmo lo marca el cliente pidiendo
  más rápido o más despacio, que para lo que hace falta basta, pero no permite congelar la escena.
- **No se puede pedir el mapa por el socket.** Hay que sacarlo con `main.py map` y meterlo en la
  escena, o leer el JSON de `python/maps/`. Un comando `GET_MAP` sería la pieza que falta para que
  Unity se configure entero por red.
- **No se pueden crear tareas desde el cliente.** El reparto lo decide Python.
- **No hay autenticación ni cifrado.** Está pensado para `127.0.0.1`. No lo expongas a una red que
  no controles.
