# Protocolo HTTP del servidor de AGVs

Especificación completa del contrato entre el servidor Python y el cliente de Unity. **Este
fichero se basta solo**: no hace falta leer el código ni el README para escribir el cliente.

Transporte: **HTTP con JSON**. Todo el JSON de este documento está copiado tal cual de una corrida
real, no está escrito a mano.

---

## 1. Transporte

| | |
|---|---|
| Protocolo | **HTTP/1.1** |
| Dirección por defecto | `http://127.0.0.1:5000` (configurable con `--host` / `--port`) |
| Formato | JSON en la petición y en la respuesta (`Content-Type: application/json`) |
| Codificación | `utf-8` |
| Concurrencia | Un hilo por petición. Varios clientes a la vez están permitidos |

No hay que trocear nada ni buscar delimitadores: HTTP ya trae el `Content-Length` y `UnityWebRequest`
entrega el cuerpo entero.

### El modelo PULL: Unity pide, Python responde

Python **nunca** envía nada por su cuenta. No hay push, ni streaming, ni suscripciones.

> **`POST /step` avanza la simulación un paso. `GET /state` no.**

El ritmo lo marca el cliente: llamando a `POST /step` diez veces por segundo, el almacén corre a 10
ticks por segundo. Si el cliente se para, el almacén se para.

Que **mirar** y **avanzar** sean dos rutas distintas es a propósito: en HTTP un `GET` no puede tener
efectos, y así dos clientes conectados a la vez ya no se roban los ticks el uno al otro. Uno puede
llevar el reloj con `POST /step` y otro pintar con `GET /state` sin alterar nada.

---

## 2. Rutas

Cinco, y ninguna necesita cabeceras especiales.

| Ruta | Cuerpo | Qué hace | Avanza el tick |
|---|---|---|:--:|
| `GET /state` | — | El estado completo del almacén | no |
| `GET /health` | — | Comprueba que el servidor vive | no |
| `POST /step` | — (o `{}`) | Avanza un paso y devuelve el estado | **sí** |
| `POST /reset` | — (o `{}`) | Reinicia la corrida: `step` vuelve a 0 y `run` sube | no |
| `POST /mode` | `{"mode": "..."}` | Cambia de política **en caliente** y reinicia la corrida | no |

### Probarlo sin Unity

```bash
curl localhost:5000/health
curl localhost:5000/state
curl -X POST localhost:5000/step
curl -X POST -d '{"mode":"qlearning"}' localhost:5000/mode
curl -X POST localhost:5000/reset
```

### Desde Unity

```csharp
// C#, orientativo
IEnumerator Paso() {
    using var req = UnityWebRequest.Post("http://127.0.0.1:5000/step", "", "application/json");
    yield return req.SendWebRequest();
    if (req.result != UnityWebRequest.Result.Success) { Debug.LogError(req.error); yield break; }
    var snapshot = JsonUtility.FromJson<Snapshot>(req.downloadHandler.text);
    Pintar(snapshot);
}
```

### `POST /mode`

La única ruta con cuerpo. Arranca **una corrida limpia**: media corrida con una política y media con
otra no son una corrida de ninguna de las dos.

```
-> POST /mode  {"mode": "baseline"}
<- 200  {"ok":true,"mode":"baseline","run":2}
```

Sus errores, cada uno con su código HTTP:

| Código | `error` | Cuándo pasa |
|---|---|---|
| 400 | `bad_mode` | El modo no existe, o no venía ninguno. `modes` trae los que sí valen |
| 409 | `set_mode_failed` | El modo existe pero no se pudo montar. Lleva `detail` (lo típico: se pidió `qlearning` y no hay Q-table entrenada) |
| 501 | `mode_not_supported` | Esta simulación no sabe cambiar de política |

```
-> POST /mode  {"mode": "turbo"}
<- 400  {"error":"bad_mode","mode":"turbo","modes":["baseline","qlearning"]}
```

### Errores de forma

```
-> POST /mode  {no es json
<- 400  {"error":"bad_json","detail":"Expecting property name enclosed in double quotes: ..."}

-> GET /loquesea
<- 404  {"error":"unknown_route","path":"/loquesea","routes":["GET /health","GET /state","POST /mode","POST /reset","POST /step"]}
```

Una ruta desconocida **no** cierra nada: contesta 404 con la lista de las que hay y sigue.

---

## 3. El snapshot

Lo que devuelven `GET /state` y `POST /step`. Aquí va **entero y de una corrida real**, partido en varias líneas
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
| `boxes` | list | entregas | Una entrada por caja del almacen, con donde esta **ahora** |
| `mode` | str | fase 8 | La política activa: `baseline` o `qlearning` |

### `agents[]`

| Campo | Tipo | Desde | Qué es |
|---|---|---|---|
| `id` | int | fase 1 | Identificador del AGV. Estable durante toda la sesión |
| `x`, `y`, `z` | float | fase 1 | Posición **ya en coordenadas de Unity** (ver sección 4) |
| `rotation` | float | fase 1 | Giro en grados sobre el eje vertical, 0-360 |
| `state` | str | fase 1 | `idle`, `moving`, `waiting`, `picking`, `dropping` o `done` |
| `node` | str | fase 3 | Nodo en el que está, o del que acaba de salir |
| `next_node` | str \| null | fase 3 | Hacia dónde va; `null` si ya llegó |
| `path` | list[str] | fase 3 | La ruta entera, de origen a destino. Para pintarla |
| `task` | int \| null | fase 3 | Id de la tarea que lleva |
| `wait_time` | int | fase 5 | Ticks **acumulados** cediendo el paso en toda la corrida |
| `action` | str | fase 8 | Lo que **eligió** hacer: `advance`, `wait` o `reroute` |
| `blocked` | bool | fase 8 | Eligió `advance` y el motor **no le dejó** pasar |
| `leg` | str | entregas | `none`, `to_pick` (va a por la caja) o `to_drop` (la lleva al muelle) |
| `mission` | str \| null | entregas | Id de la mision que **gano pujando**, o `null` si va libre |
| `box` | str \| null | entregas | Id de la caja que tiene que recoger |
| `destination` | str \| null | entregas | Donde tiene que dejarla: un **rack** o un **muelle**, segun el flujo |
| `carrying` | str \| null | entregas | Id de la caja que lleva **encima ahora mismo** |
| `busy` | int | entregas | Ticks que le quedan de la recogida o de la entrega |

Hay **dos flujos**: `produccion -> rack` guarda lo que sale de la linea, y
`rack -> muelle` saca del almacen lo guardado. Y se **encadenan**: en cuanto una
caja queda guardada en un rack, el almacen le abre su mision de salida, asi que
`missions_total` sube durante la corrida.

Un AGV lleva **como mucho una mision a la vez**: `mission` pasa de `null` a un id cuando gana la
subasta, y vuelve a `null` en cuanto deja la caja en el muelle. Entonces vuelve a pujar.

**Los seis estados:**

| `state` | Qué significa | Qué hace el prefab |
|---|---|---|
| `idle` | Sin ruta que seguir (no hay camino a su destino) | Quieto |
| `moving` | Cruzando un tramo, o parado en un nodo a punto de salir | Se mueve hacia `x,y,z` |
| `waiting` | Cediendo el paso: no se movió en este tick | Quieto, y se le puede poner un icono |
| `picking` | Recogiendo la caja: ocupa su nodo y no se mueve | Animación de horquilla subiendo |
| `dropping` | Dejando la caja en el muelle | Animación de horquilla bajando |
| `done` | Llegó a su destino, o entregó su caja | Aparcado |

**Dibujar la caja que lleva un AGV:** con `carrying` basta — vale `null` mientras va a por ella y
trae el id en cuanto la recoge, así que el prefab la dibuja encima del AGV exactamente cuando
`carrying != null`. Para las demás cajas está `boxes[]`, que dice dónde está cada una y en qué
estado ([abajo](#boxes)).

Con `deliveries` desactivado —que es el modo por defecto— `leg` vale siempre `none`, los otros
campos de entrega valen `null`/`0`, `boxes[]` viene vacía y los estados `picking`, `dropping` y
`charging` no aparecen nunca.

**`action` y `blocked` van juntos**, y es lo más útil para enseñar qué está pasando: `action` es lo
que el AGV **quiso** hacer y `blocked` es lo que el motor **le concedió**. Un AGV con
`action: "advance"` y `blocked: true` pidió pasar y no le dejaron, que es exactamente el momento
interesante en un cuello de botella. Un AGV a media travesía está ejecutando un avance
(`action: "advance"`); el que ya llegó sale como `wait`.

**La posición va interpolada.** Un AGV a mitad de camino entre `node` y `next_node` manda la
posición de la mitad del tramo, no la del nodo de destino. Así el prefab se puede mover a `x,y,z`
directamente, sin teletransportes ni interpolación por parte de Unity. Un tramo tarda entre 4 y 8
ticks en cruzarse, según su coste.

### `boxes[]`

Las cajas se mueven durante la corrida: `graph.boxes` del mapa es solo el
inventario **inicial**, y esta lista dice donde esta cada una en este momento.

| Campo | Tipo | Qué es |
|---|---|---|
| `id` | str | Identificador de la caja, estable toda la corrida |
| `node` | str | Nodo donde esta **ahora**. Mientras la lleva un AGV, es el nodo del AGV |
| `level` | int | Nivel de la estanteria del que salio |
| `status` | str | Uno de los cinco de abajo |
| `mission` | str \| null | La mision que la esta moviendo, o `null` |

| `status` | Qué significa | Qué hace el prefab |
|---|---|---|
| `WAITING_PICKUP` | Recien fabricada, esperando en una linea de produccion | Caja en el suelo |
| `STORED` | Guardada en una estanteria | Caja en el rack, a su `level` |
| `RESERVED` | Un AGV gano su mision y viene a por ella | Caja marcada, aun sin mover |
| `IN_TRANSIT` | Va encima de un AGV | Caja sobre el AGV, sigue su posicion |
| `DELIVERED` | Ya salio por un muelle | Caja en el muelle |

### `stats`

| Campo | Tipo | Desde | Qué es |
|---|---|---|---|
| `run` | int | fase 5 | Número de corrida. Sube en cada `POST /reset` y en cada `POST /mode` |
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
| `deliveries` | bool | entregas | Si esta corrida va con entregas o solo con destinos |
| `picked` | int | entregas | Cajas recogidas en esta corrida |
| `delivered` | int | entregas | Cajas ya dejadas en un muelle |
| `missions_pending` | int | entregas | Misiones que siguen en la bolsa, sin dueño |
| `missions_total` | int | entregas | Misiones abiertas en total; **crece durante la corrida** |
| `boxes_delivered` | int | entregas | Cajas que ya salieron por un muelle |
| `messages` | int | entregas | Mensajes que lleva el bus de negociacion |

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
(`graph.to_unity()`) y no hay ninguna copia ya convertida guardada en ningún sitio.

**Unity no tiene que convertir nada**: `x`, `y` y `z` llegan listos para asignar a un `Vector3`.

### El mapa

Para montar la escena hace falta el grafo. **No se puede pedir por HTTP** (ver
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
1. (opcional) GET /health para comprobar que responde
2. bucle a ~10 Hz:
      POST /step
      parsear el JSON
      mover los prefabs a (x, y, z) y girarlos a rotation
```

No hay conexión que abrir ni que cerrar: cada petición se basta sola. Para probarlo a mano antes de
tener el cliente montado:

```bash
python3 python/main.py serve --port 5055 &
curl -X POST localhost:5055/step
```

### Reconexión y errores

- Si el servidor se cae, la petición falla con un error de red. Se reintenta y ya está: al volver se
  sigue la **misma** simulación, no una nueva.
- Un cuerpo mayor de 64 KB se rechaza con 413. Ninguna ruta de este contrato manda tanto.
- Los errores llevan **código HTTP y JSON**: 400 si el cuerpo viene mal, 404 si la ruta no existe,
  409 si la política no se pudo montar, 501 si la simulación no cambia de modo.

---

## 6. Compatibilidad

El formato está **congelado en cuanto a lo que ya existe**:

- Los campos que ya están **no cambian de nombre ni de tipo**, nunca.
- Las fases nuevas **solo añaden** campos.
- `JsonUtility` de Unity ignora lo que no conoce, así que un cliente escrito contra la fase 1
  sigue funcionando hoy sin tocarle una línea.

Lo que sí puede cambiar de una corrida a otra: el número de AGVs (`--agents`), el mapa (`--map`) y
la política (`--policy` o `POST /mode`). Nada de eso cambia la **forma** del JSON.

---

## 7. Limitaciones conocidas del protocolo

Lo que hoy no se puede hacer, por si el cliente lo necesita:

- **No hay pausa ni control de velocidad desde el servidor.** El ritmo lo marca el cliente llamando
  a `POST /step` más rápido o más despacio. Para congelar la escena basta con dejar de pedirlo, y
  `GET /state` sigue devolviendo el estado sin moverlo.
- **No se puede pedir el mapa por HTTP.** Hay que sacarlo con `main.py map` y meterlo en la escena,
  o leer el JSON de `python/maps/`. Una ruta `GET /map` sería la pieza que falta para que Unity se
  configure entero por red.
- **No se pueden crear tareas desde el cliente.** El reparto lo decide Python.
- **No hay autenticación ni cifrado.** Es HTTP plano pensado para `127.0.0.1`. No lo expongas a una
  red que no controles.
