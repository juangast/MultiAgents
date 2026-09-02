"""Fase 6: el entorno de Q-Learning. Estado, acciones y recompensa.

Aqui **no se entrena nada**. Se define el problema: que ve un AGV cuando le toca
decidir, que puede hacer, y cuanto vale cada cosa que pasa. La fase 7 pone el
bucle de aprendizaje encima de lo que hay en este modulo.

--- Q-Learning NO sustituye a A* ---

El pathfinding lo sigue resolviendo A*, igual que en la fase 3: quien dice por
donde se va de S1 a N6 es `astar.astar()`. Lo que se aprende aqui es mucho mas
chico: **que hacer AHORA** cuando la ruta que ya tengo me mete en un conflicto.
Avanzo, cedo el paso, o recalculo esquivando el atasco. Nada mas.

Si el estado fuera la ruta entera el espacio explotaria. En el mapa `warehouse`
hay 13 nodos, y solo las posiciones de 6 AGVs ya son 13^6 = 4.826.809 estados,
sin contar rutas, destinos ni progresos. Con el estado local de este modulo son
**72**. Esa es toda la diferencia, y es la razon de que el estado sea DISCRETO y
LOCAL: cinco preguntas sobre lo que este AGV tiene delante, nunca coordenadas
continuas ni el mapa completo.

--- Como se enchufa ---

`QLearningPolicy` cumple el mismo contrato que `conflicts.BaselinePolicy`
(`name` + `decide(agent, local_state)`), asi que entra en `Simulation` por el
constructor sin tocar una linea del motor:

    simulacion = Simulation(grafo, 6, policy=qlearning.QLearningPolicy())

Para verlo por el log:

    python3 python/qlearning.py
"""

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import astar
import config
import conflicts
from agent import STATE_MOVING, STATE_WAITING, Agent
from graph import WarehouseGraph
from logs import get_logger, setup_logging

log = get_logger("qlearning")


# --- El estado ---------------------------------------------------------------

# Los campos de la tupla de estado, en el orden en que salen de
# `get_local_state()`. Este orden es el contrato: es el de la clave de la
# Q-table y el del JSON en el que se guarda, asi que no se reordena sin
# invalidar las tablas ya entrenadas.
STATE_FIELDS: tuple[str, ...] = (
    "next_node_occupied",
    "edge_conflict",
    "queue_ahead",
    "distance_bucket",
    "has_priority",
)

# Cuantos valores distintos puede tomar cada campo. Mismo orden que arriba.
STATE_SIZES: tuple[int, ...] = (2, 2, 3, 3, 2)

# Una tupla de enteros, hasheable, del tamaño de STATE_FIELDS. Es la clave de Q.
State = tuple[int, ...]

# Tope de la cola: mas de dos AGVs esperando delante no cambian la decision, y
# saturar es lo que mantiene el campo en tres valores en vez de en N.
QUEUE_CAP: int = 2


class SimulationView(Protocol):
    """Lo que `get_local_state()` necesita mirar de la simulacion.

    Tres cosas, las tres publicas: el mapa, los agentes y la ocupacion. Se
    declara como Protocol por lo mismo que `conflicts.Policy`: asi el estado se
    puede construir en un test con un objeto de mentira, sin montar una
    `Simulation` entera.

    Que aparezca la simulacion completa no contradice que el estado sea local:
    lo local no es lo que se puede mirar, es lo que se acaba guardando en la
    tupla. De todo esto salen cinco enteros sobre el nodo de delante.
    """

    graph: WarehouseGraph
    agents: Sequence[Agent]
    occupancy: Mapping[str, int]


def get_local_state(agent: Agent, simulation: SimulationView) -> State:
    """El estado discreto y local de este AGV: siempre cinco enteros.

    | Campo                | Valores | Que pregunta                              |
    |----------------------|---------|-------------------------------------------|
    | `next_node_occupied` | 0/1     | ¿hay alguien en el nodo al que voy a entrar? |
    | `edge_conflict`      | 0/1     | ¿alguien viene de frente por mi tramo?    |
    | `queue_ahead`        | 0/1/2   | ¿cuantos esperan en mis 2 nodos siguientes? |
    | `distance_bucket`    | 0/1/2   | ¿cuanto me falta? cerca / medio / lejos   |
    | `has_priority`       | 0/1     | ¿soy el id menor de los que estamos en conflicto? |

    Devuelve **siempre** una tupla del mismo tamaño, del mismo tipo y con cada
    campo dentro de su rango: es la clave de la Q-table y dos situaciones
    iguales tienen que dar la misma clave. Un agente sin ruta, parado o ya
    llegado, no es un caso especial: da `(0, 0, 0, 0, 1)`, que es el estado de
    "no tengo nada delante".
    """
    siguiente = agent.next_node()
    return (
        _ocupado(agent, siguiente, simulation.occupancy),
        _viene_de_frente(agent, siguiente, simulation.agents),
        _cola_delante(agent, simulation.agents),
        distance_bucket(agent),
        _tengo_prioridad(agent, _rivales(agent, siguiente, simulation)),
    )


def state_from_local(agent: Agent, local_state: conflicts.LocalState) -> State:
    """El mismo estado, pero sacado de lo que el motor le pasa a la politica.

    `Simulation.decide()` entrega un `conflicts.LocalState`, que trae la
    ocupacion y los conflictos ya detectados de este tick, pero no la lista de
    agentes. Con eso cuatro campos salen **exactos** y `queue_ahead` sale
    **aproximado**: se cuentan los nodos ocupados de los dos siguientes de mi
    ruta, no los AGVs que esperan en ellos.

    Es el camino de respaldo de `QLearningPolicy` cuando no esta atada a una
    simulacion. Para entrenar hay que atarla con `bind()`, o el estado que se
    aprende no seria exactamente el que luego se ejecuta.
    """
    siguiente = agent.next_node()
    rivales = {
        otro
        for choque in local_state.conflicts
        for otro in choque.agents
        if otro != agent.id
    }
    rivales.update(local_state.blocked_by)
    de_frente = any(
        choque.type == conflicts.TYPE_EDGE for choque in local_state.conflicts
    )
    cola = sum(
        1
        for nodo in _proximos_nodos(agent)
        if local_state.occupancy.get(nodo) not in (None, agent.id)
    )
    return (
        _ocupado(agent, siguiente, local_state.occupancy),
        int(de_frente),
        min(cola, QUEUE_CAP),
        distance_bucket(agent),
        _tengo_prioridad(agent, rivales),
    )


def distance_bucket(agent: Agent) -> int:
    """0 cerca, 1 medio, 2 lejos, contando **nodos** que faltan.

    Los cortes son `config.DISTANCE_NEAR_NODES` y `config.DISTANCE_MID_NODES`, y
    lo que se cuenta son los nodos que quedan de `path`, no la distancia
    euclidiana al destino: en un almacen dos nodos pueden estar pegados y tener
    medio pasillo de por medio, asi que la geometria mentiria sobre lo que falta
    de verdad. Ademas los nodos que faltan es un numero que ya esta discretizado,
    que es lo que necesita la clave de la Q-table.
    """
    faltan = max(len(agent.path) - 1 - agent.path_index, 0)
    if faltan <= config.DISTANCE_NEAR_NODES:
        return 0
    if faltan <= config.DISTANCE_MID_NODES:
        return 1
    return 2


def state_space_size() -> int:
    """Cuantos estados distintos puede devolver `get_local_state()`."""
    return math.prod(STATE_SIZES)


def report_state_space() -> list[str]:
    """El espacio de estados linea a linea, listo para el log."""
    lineas = ["--- espacio de estados del Q-Learning (fase 6) ---"]
    for campo, tamano in zip(STATE_FIELDS, STATE_SIZES):
        valores = ", ".join(str(valor) for valor in range(tamano))
        lineas.append(f"{campo:<20} {tamano} valores  {{{valores}}}")
    lineas.append(
        f"{'TOTAL':<20} {state_space_size()} estados"
        f"  ({' x '.join(str(t) for t in STATE_SIZES)})"
    )
    lineas.append(
        f"{'acciones':<20} {len(ACTIONS)}  "
        f"({', '.join(accion.value for accion in ACTIONS)})"
    )
    lineas.append(
        f"{'celdas de Q(s,a)':<20} {state_space_size() * len(ACTIONS)}"
    )
    lineas.append(
        "cabe entero en un dict: con estos numeros la tabla se llena en pocos "
        "miles de ticks, no en millones"
    )
    return lineas


def _ocupado(agent: Agent, siguiente: str | None, occupancy: Mapping[str, int]) -> int:
    """1 si el nodo al que quiero entrar lo tiene otro AGV."""
    if siguiente is None:
        return 0
    ocupante = occupancy.get(siguiente)
    return int(ocupante is not None and ocupante != agent.id)


def _viene_de_frente(
    agent: Agent, siguiente: str | None, agents: Sequence[Agent]
) -> int:
    """1 si alguien esta en mi nodo siguiente y quiere entrar en el mio.

    Es el `edge conflict` de la fase 5 visto desde dentro del agente: el choque
    que la ocupacion sola no ve venir, porque cada uno esta parado en el nodo
    que el otro pide.
    """
    if siguiente is None:
        return 0
    for otro in agents:
        if otro.id == agent.id:
            continue
        if otro.current_node == siguiente and otro.next_node() == agent.current_node:
            return 1
    return 0


def _cola_delante(agent: Agent, agents: Sequence[Agent]) -> int:
    """Cuantos AGVs estan esperando en mis dos nodos siguientes. Saturado en 2."""
    proximos = set(_proximos_nodos(agent))
    if not proximos:
        return 0
    cuantos = sum(
        1
        for otro in agents
        if otro.id != agent.id
        and otro.state == STATE_WAITING
        and otro.current_node in proximos
    )
    return min(cuantos, QUEUE_CAP)


def _proximos_nodos(agent: Agent) -> tuple[str, ...]:
    """Los dos siguientes nodos de mi ruta, los que haya."""
    return tuple(agent.path[agent.path_index + 1 : agent.path_index + 3])


def _rivales(
    agent: Agent, siguiente: str | None, simulation: SimulationView
) -> set[int]:
    """Ids de los que se disputan el paso conmigo ahora mismo.

    Tres formas de estorbarme, que son las que la fase 5 llama `vertex`,
    `following` y `edge`: estar encima del nodo que pido, pedirlo a la vez que
    yo, o venir de frente por mi tramo.
    """
    if siguiente is None:
        return set()

    rivales: set[int] = set()
    ocupante = simulation.occupancy.get(siguiente)
    if ocupante is not None and ocupante != agent.id:
        rivales.add(ocupante)

    for otro in simulation.agents:
        if otro.id == agent.id:
            continue
        if otro.current_node == siguiente and otro.next_node() == agent.current_node:
            rivales.add(otro.id)
            continue
        # Solo los parados en un nodo estan pidiendo algo: el que va a media
        # travesia ya tiene su reserva concedida y no compite por ella.
        if (
            otro.progress <= 0.0
            and otro.state in (STATE_MOVING, STATE_WAITING)
            and otro.next_node() == siguiente
        ):
            rivales.add(otro.id)
    return rivales


def _tengo_prioridad(agent: Agent, rivales: Iterable[int]) -> int:
    """1 si soy el id menor de los que estamos en conflicto.

    Sin rivales tambien es 1: no hay a quien ceder el paso. Los otros campos ya
    dicen si hay conflicto o no, asi que el estado no pierde nada por juntar
    "no hay pelea" con "la pelea la gano yo".
    """
    ids = tuple(rivales)
    return int(not ids or agent.id < min(ids))


# --- Las acciones ------------------------------------------------------------


class Action(Enum):
    """Lo que un AGV puede decidir cuando le toca.

    | Accion    | Que hace                                                  |
    |-----------|-----------------------------------------------------------|
    | `ADVANCE` | Avanzar al siguiente nodo del path que trazo A*           |
    | `WAIT`    | Quedarse un tick donde esta                               |
    | `REROUTE` | Recalcular A* penalizando el nodo/tramo congestionado     |

    Fijate en lo que **no** hay: ninguna accion elige un nodo. La ruta la traza
    A*; el aprendizaje solo decide si se sigue, si se espera o si se pide otra.

    `REROUTE` **no mueve** al agente en el mismo tick. Cuando la politica decide,
    el motor ya fijo la intencion de este tick en la fase A, asi que la ruta
    nueva entra en vigor en el siguiente: recalcular cuesta un tick, y por eso
    hacia el motor un REROUTE se traduce a `wait`.
    """

    ADVANCE = "advance"
    WAIT = "wait"
    REROUTE = "reroute"


# El orden es el de la Q-table y el del desempate de `best_action()`: a igual
# valor gana ADVANCE, que es la accion por defecto de un almacen que funciona.
ACTIONS: tuple[Action, ...] = tuple(Action)


def enabled_actions(enable_reroute: bool | None = None) -> tuple[Action, ...]:
    """Las acciones que la politica puede elegir, segun `config.ENABLE_REROUTE`.

    Con el flag apagado quedan ADVANCE y WAIT, que es como la fase 7 puede
    empezar a entrenar si con dos acciones converge antes. El flag **no** cambia
    la Q-table: las filas siguen teniendo las tres acciones (ver `QTable`), asi
    que encenderlo mas tarde no obliga a migrar ningun fichero.
    """
    activo = config.ENABLE_REROUTE if enable_reroute is None else enable_reroute
    return ACTIONS if activo else (Action.ADVANCE, Action.WAIT)


def to_engine_action(action: Action) -> str:
    """Traduce una accion a lo que `Simulation` entiende: `go` o `wait`.

    ADVANCE es `go`; WAIT y REROUTE son `wait`, porque los dos terminan el tick
    sin moverse. Es la pieza que hace intercambiable esta politica con la
    baseline sin tocar el motor.
    """
    return conflicts.ACTION_GO if action is Action.ADVANCE else conflicts.ACTION_WAIT


# --- El reroute --------------------------------------------------------------


def reroute_penalties(
    agent: Agent, *, penalty: float | None = None
) -> dict[str | tuple[str, str], float]:
    """Encarece lo que este AGV tiene delante, para que A* lo esquive.

    Penaliza el nodo siguiente y el tramo hacia el, que es exactamente la
    congestion que el agente puede ver desde donde esta. Sale en el formato de
    `astar.Penalties`, el gancho que la fase 3 dejo preparado: no toca el mapa
    ni lo recarga, solo hace cara una casilla durante una busqueda.
    """
    siguiente = agent.next_node()
    if siguiente is None:
        return {}
    extra = config.REROUTE_PENALTY if penalty is None else float(penalty)
    return {siguiente: extra, (agent.current_node, siguiente): extra}


def reroute(
    agent: Agent,
    graph: WarehouseGraph,
    *,
    penalties: astar.Penalties | None = None,
) -> list[str] | None:
    """Recalcula la ruta desde donde esta el agente, esquivando lo de delante.

    Devuelve la ruta nueva ya puesta en el agente, o None si no hay ninguna (o
    si el agente esta a media travesia, que ahi no se puede cambiar de idea sin
    teletransportarlo).

    **No usa `Agent.assign_task()` a proposito**: esa reinicia `wait_time`, que
    es justo la medida con la que se comparan las politicas, y borrar el reloj
    de la espera en cada recalculo haria que el numero dejara de significar
    algo. Aqui solo se tocan `path`, `path_index` y `progress`; el destino, la
    tarea y el estado se quedan como estaban.
    """
    if agent.target_node is None or agent.progress > 0.0:
        return None

    if penalties is None:
        penalties = reroute_penalties(agent)

    nueva = astar.astar(graph, agent.current_node, agent.target_node, penalties)
    if nueva is None:
        log.debug("AGV %s: reroute sin salida desde %s", agent.id, agent.current_node)
        return None

    agent.path = list(nueva)
    agent.path_index = 0
    agent.progress = 0.0
    return agent.path


def is_useless_reroute(
    graph: WarehouseGraph,
    old_path: Sequence[str],
    new_path: Sequence[str],
    *,
    avoided_conflict: bool = False,
) -> bool:
    """Si el recalculo no sirvio para nada, que es lo que se penaliza.

    Un reroute es innecesario cuando **ni sale mas barato ni esquiva un
    conflicto real**. Las dos condiciones importan: cambiar a una ruta igual de
    larga esta bien pagado si con eso se sale de un cruce de frente, y esta mal
    pagado si solo fue por probar.

    Se compara por costo (`astar.path_cost`), no por numero de nodos: la ruta
    corta en saltos puede ser la cara. Una ruta que ni siquiera se puede recorrer
    cuenta como infinita, o sea que nunca es mejor.
    """
    if avoided_conflict:
        return False
    return _costo(graph, new_path) >= _costo(graph, old_path)


def _costo(graph: WarehouseGraph, path: Sequence[str]) -> float:
    """Costo de una ruta, o infinito si no se puede recorrer."""
    if len(path) < 2:
        return 0.0
    try:
        return astar.path_cost(graph, path)
    except KeyError:
        return math.inf


# --- La recompensa -----------------------------------------------------------


class Event(Enum):
    """Lo que le puede pasar a un AGV y tiene precio.

    `PROGRESS` es **`path_index` subio**, no que el AGV se haya movido por el
    mapa: un agente a media travesia se mueve en pantalla y no ha adelantado
    nada de su ruta, y pagarle por eso seria pagarle por ir despacio.
    """

    TASK_COMPLETE = "task_complete"
    PROGRESS = "progress"
    WAIT = "wait"
    CONFLICT = "conflict"
    DEADLOCK = "deadlock"
    USELESS_REROUTE = "useless_reroute"


# Cada evento con la constante de config.py de la que sale su valor. El valor se
# lee en cada llamada, no se copia aqui: asi se puede ajustar `config` y ver el
# efecto sin reimportar el modulo.
_CONFIG_KEY: dict[Event, str] = {
    Event.TASK_COMPLETE: "REWARD_TASK_COMPLETE",
    Event.PROGRESS: "REWARD_PROGRESS",
    Event.WAIT: "REWARD_WAIT",
    Event.CONFLICT: "REWARD_CONFLICT",
    Event.DEADLOCK: "REWARD_DEADLOCK",
    Event.USELESS_REROUTE: "REWARD_USELESS_REROUTE",
}


def reward(event: Event | str) -> float:
    """Lo que vale un evento. **Los numeros viven en `config.py`**, no aqui.

    | Evento            | Valor por defecto | Cuando                          |
    |-------------------|-------------------|---------------------------------|
    | `TASK_COMPLETE`   | +100              | el AGV llego a su destino       |
    | `PROGRESS`        | +2                | `path_index` subio en este tick |
    | `WAIT`            | -1                | se quedo un tick parado         |
    | `CONFLICT`        | -20               | intento entrar donde habia choque |
    | `DEADLOCK`        | -50               | la corrida murio atascada       |
    | `USELESS_REROUTE` | -3                | recalculo sin ganar nada        |

    Es una sola funcion y una sola tabla a proposito: ajustar el experimento
    tiene que ser editar seis constantes de `config.py`, no buscar numeros
    sueltos por el codigo.

    Acepta el `Event` o su cadena (`"wait"`), y lanza `ValueError` con la lista
    de los que hay si el evento no existe: un evento mal escrito que devolviera
    0.0 seria un premio invisible, y eso se busca durante dias.
    """
    try:
        clave = Event(event)
    except ValueError:
        conocidos = ", ".join(uno.value for uno in Event)
        raise ValueError(
            f"evento desconocido: {event!r}; los que hay son {conocidos}"
        ) from None
    return float(getattr(config, _CONFIG_KEY[clave]))


# --- La Q-table --------------------------------------------------------------

# Version del formato del JSON. Si el orden de STATE_FIELDS o los nombres de las
# acciones cambian, sube el numero: una tabla vieja leida con campos nuevos
# aprende sobre estados que no son los que dice.
FORMAT: str = "agv-qtable/1"

SEPARATOR: str = "|"


def encode_state(state: State) -> str:
    """La tupla de estado como clave de JSON: los campos en orden, con '|'.

    `(0, 1, 2, 1, 0)` se guarda como `"0|1|2|1|0"`. JSON no admite tuplas como
    clave, y esto se lee de un vistazo, que es mas de lo que se puede decir de
    un `str(tuple)` lleno de parentesis y espacios.
    """
    if len(state) != len(STATE_FIELDS):
        raise ValueError(
            f"el estado {state!r} tiene {len(state)} campos, no {len(STATE_FIELDS)}"
        )
    return SEPARATOR.join(str(int(valor)) for valor in state)


def decode_state(text: str) -> State:
    """Deshace `encode_state()`. Lanza ValueError si la clave no tiene la forma."""
    partes = text.split(SEPARATOR)
    if len(partes) != len(STATE_FIELDS):
        raise ValueError(
            f"la clave {text!r} tiene {len(partes)} campos, no {len(STATE_FIELDS)}"
        )
    try:
        return tuple(int(parte) for parte in partes)
    except ValueError:
        raise ValueError(f"la clave {text!r} no es una tupla de enteros") from None


def _fila_en_cero() -> dict[Action, float]:
    """Una fila nueva de la tabla: las tres acciones a cero."""
    return {accion: 0.0 for accion in ACTIONS}


class QTable:
    """Q(s, a) como `dict[tuple, dict[Action, float]]`, con `defaultdict`.

    Un estado que no se ha visto nunca nace con sus acciones a cero, asi que la
    fase 7 puede preguntar por cualquier estado sin comprobar antes si existe.
    Con 72 estados y 3 acciones la tabla entera son 216 numeros: cabe de sobra
    en memoria y en un JSON que se puede abrir y leer a mano.

    La fila lleva **siempre las tres acciones**, aunque `config.ENABLE_REROUTE`
    este apagado: el flag dice lo que la politica puede *elegir*, no lo que la
    tabla *guarda*. Asi una tabla entrenada solo con ADVANCE/WAIT se puede
    seguir entrenando con REROUTE encendido sin migrar el fichero.

    --- Formato del JSON ---

        {
          "format": "agv-qtable/1",
          "state_fields": ["next_node_occupied", "edge_conflict",
                           "queue_ahead", "distance_bucket", "has_priority"],
          "actions": ["advance", "wait", "reroute"],
          "q": {
            "0|1|2|1|0": {"advance": 1.5, "wait": -0.25, "reroute": 0.0}
          }
        }

    La clave de `q` es la tupla de estado con los campos en el orden de
    `state_fields`, separados por `|`. `state_fields` va escrito en el fichero a
    proposito: sin el, una tabla guardada hoy y leida despues de reordenar los
    campos seguiria cargando, y aprenderia sobre estados equivocados sin avisar.
    """

    def __init__(self, values: Mapping[State, Mapping[Action, float]] | None = None) -> None:
        self._q: defaultdict[State, dict[Action, float]] = defaultdict(_fila_en_cero)
        for estado, fila in (values or {}).items():
            self[estado].update(fila)

    def __repr__(self) -> str:
        return f"QTable(states={len(self)}/{state_space_size()})"

    def __len__(self) -> int:
        """Cuantos estados hay ya en la tabla."""
        return len(self._q)

    def __contains__(self, state: State) -> bool:
        """Si el estado ya se visito. No lo crea, al reves que `[]`."""
        return tuple(state) in self._q

    def __iter__(self):
        return iter(self._q)

    def __getitem__(self, state: State) -> dict[Action, float]:
        """La fila del estado, creandola a ceros si es la primera vez."""
        return self._q[tuple(state)]

    def value(self, state: State, action: Action) -> float:
        """Q(s, a)."""
        return self[state][action]

    def set_value(self, state: State, action: Action, value: float) -> None:
        """Escribe Q(s, a). Quien calcula el valor nuevo es la fase 7."""
        self[state][action] = float(value)

    def best_action(
        self, state: State, *, among: Sequence[Action] | None = None
    ) -> Action:
        """La accion de mas valor, con el empate resuelto por el orden de ACTIONS.

        El desempate es **determinista** para que dos corridas con la misma
        semilla decidan lo mismo. Con la tabla recien creada todo empata a cero
        y gana ADVANCE.
        """
        candidatas = tuple(among) if among else ACTIONS
        if not candidatas:
            raise ValueError("no hay ninguna accion entre la que elegir")
        fila = self[state]
        mejor = candidatas[0]
        for accion in candidatas[1:]:
            if fila[accion] > fila[mejor]:
                mejor = accion
        return mejor

    def best_value(self, state: State, *, among: Sequence[Action] | None = None) -> float:
        """max_a Q(s, a). Es el termino que la fase 7 mete en la actualizacion."""
        return self.value(state, self.best_action(state, among=among))

    def as_dict(self) -> dict[str, dict[str, float]]:
        """La tabla en claves de texto, tal y como se guarda. Ordenada."""
        return {
            encode_state(estado): {
                accion.value: float(self._q[estado][accion]) for accion in ACTIONS
            }
            for estado in sorted(self._q)
        }

    def save(self, path: str | Path) -> Path:
        """Escribe la tabla en JSON, creando el directorio si hace falta."""
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        contenido: dict[str, Any] = {
            "format": FORMAT,
            "state_fields": list(STATE_FIELDS),
            "actions": [accion.value for accion in ACTIONS],
            "q": self.as_dict(),
        }
        destino.write_text(
            json.dumps(contenido, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info("Q-table guardada en %s (%d estados)", destino, len(self))
        return destino

    @classmethod
    def load(cls, path: str | Path) -> "QTable":
        """Lee una tabla de disco, comprobando que el formato es el de ahora.

        Lanza ValueError si el fichero es de otra version o si los campos del
        estado no son los de `STATE_FIELDS`, en ese orden. Cargar a ciegas una
        tabla con otro orden de campos seria entrenar sobre estados que no son
        los que se creen, y eso no da error nunca: da resultados malos.
        """
        origen = Path(path)
        crudo = json.loads(origen.read_text(encoding="utf-8"))
        if not isinstance(crudo, dict):
            raise ValueError(f"{origen}: se esperaba un objeto JSON")

        formato = crudo.get("format")
        if formato != FORMAT:
            raise ValueError(f"{origen}: formato {formato!r}, se esperaba {FORMAT!r}")

        campos = crudo.get("state_fields")
        if campos is not None and list(campos) != list(STATE_FIELDS):
            raise ValueError(
                f"{origen}: los campos del estado son {campos}, "
                f"y ahora el estado es {list(STATE_FIELDS)}"
            )

        tabla = cls()
        for clave, fila in (crudo.get("q") or {}).items():
            estado = decode_state(str(clave))
            for nombre, valor in (fila or {}).items():
                try:
                    accion = Action(nombre)
                except ValueError:
                    raise ValueError(
                        f"{origen}: accion desconocida {nombre!r} en el estado {clave!r}"
                    ) from None
                tabla.set_value(estado, accion, float(valor))
        log.info("Q-table cargada de %s (%d estados)", origen, len(tabla))
        return tabla


# --- La politica -------------------------------------------------------------


class QLearningPolicy:
    """La politica de Q-Learning, con la misma interfaz que la baseline.

    En la fase 6 **no aprende nada**: elige la mejor accion de una Q-table que
    todavia esta a ceros, o sea que siempre avanza (con `epsilon > 0` prueba
    acciones al azar, con un generador sembrado para que la corrida siga siendo
    reproducible). Lo que se demuestra en esta fase no es lo que decide, es que
    entra en el hueco de `BaselinePolicy` sin tocar el motor:

        simulacion = Simulation(grafo, 6, policy=QLearningPolicy())

    Para que vea el estado completo hay que atarla a la simulacion, que existe
    despues que ella:

        politica = QLearningPolicy()
        simulacion = Simulation(grafo, 6, policy=politica)
        politica.bind(simulacion)

    Sin `bind()` sigue funcionando, pero saca el estado del `LocalState` que le
    pasa el motor, y ahi `queue_ahead` es una aproximacion
    (ver `state_from_local()`). Avisa una vez por el log; para entrenar, atala.

    La fase 7 solo tiene que anadir el `update()` de Bellman por encima:
    `last_decision(agent_id)` le devuelve el (estado, accion) con el que este
    AGV llego al tick, que es la mitad de la tupla que necesita.
    """

    name: str = "qlearning"

    def __init__(
        self,
        q_table: QTable | None = None,
        *,
        simulation: SimulationView | None = None,
        epsilon: float = 0.0,
        seed: int = config.RANDOM_SEED,
        enable_reroute: bool | None = None,
    ) -> None:
        self.q: QTable = q_table if q_table is not None else QTable()
        self.epsilon: float = float(epsilon)
        self.actions: tuple[Action, ...] = enabled_actions(enable_reroute)
        self._rng = random.Random(seed)
        self._simulation: SimulationView | None = simulation
        self._last: dict[int, tuple[State, Action]] = {}
        self._last_reroute: dict[int, tuple[list[str], list[str]]] = {}
        self._avisado: bool = False

    def __repr__(self) -> str:
        return (
            f"QLearningPolicy(states={len(self.q)}, epsilon={self.epsilon:g}, "
            f"actions={[accion.value for accion in self.actions]}, "
            f"bound={self._simulation is not None})"
        )

    def bind(self, simulation: SimulationView) -> None:
        """Le da la simulacion de la que sacar el estado completo."""
        self._simulation = simulation

    def reset(self) -> None:
        """Olvida las decisiones del episodio. La Q-table **no** se toca."""
        self._last.clear()
        self._last_reroute.clear()

    def decide(self, agent: Agent, local_state: conflicts.LocalState) -> str:
        """`go` o `wait`, que es lo unico que el motor entiende.

        Por dentro elige entre las tres acciones y guarda la decision para la
        fase 7. Un REROUTE recalcula la ruta aqui mismo y devuelve `wait`: la
        intencion de este tick ya estaba fijada cuando el motor pregunto, asi
        que la ruta nueva empieza a valer en el siguiente.
        """
        estado = self.observe(agent, local_state)
        accion = self.choose(estado)
        self._last[agent.id] = (estado, accion)

        if accion is Action.REROUTE:
            self._recalcula(agent)

        return to_engine_action(accion)

    def observe(
        self, agent: Agent, local_state: conflicts.LocalState | None = None
    ) -> State:
        """El estado de este agente: de la simulacion si esta atada, si no del motor."""
        if self._simulation is not None:
            return get_local_state(agent, self._simulation)
        if local_state is None:
            raise ValueError(
                "sin bind(simulation) hace falta el local_state del motor "
                "para poder construir el estado"
            )
        if not self._avisado:
            log.warning(
                "QLearningPolicy sin bind(): el estado sale del LocalState y "
                "queue_ahead va aproximado; para entrenar, atala a la simulacion"
            )
            self._avisado = True
        return state_from_local(agent, local_state)

    def choose(self, state: State) -> Action:
        """Epsilon-greedy sobre la Q-table. Con `epsilon = 0` es greedy puro.

        Con la tabla a ceros todas las acciones empatan y gana la primera,
        ADVANCE, que es la politica temeraria de la fase 6. El azar sale de un
        `random.Random` sembrado, asi que dos corridas con la misma semilla
        deciden exactamente lo mismo.
        """
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        return self.q.best_action(state, among=self.actions)

    def last_decision(self, agent_id: int) -> tuple[State, Action] | None:
        """El (estado, accion) con el que decidio este AGV la ultima vez."""
        return self._last.get(agent_id)

    def last_reroute(self, agent_id: int) -> tuple[list[str], list[str]] | None:
        """La (ruta vieja, ruta nueva) del ultimo REROUTE de este AGV.

        La fase 7 la necesita para saber si cobrar el `USELESS_REROUTE`:
        `is_useless_reroute(grafo, vieja, nueva, avoided_conflict=...)`.
        """
        return self._last_reroute.get(agent_id)

    def _recalcula(self, agent: Agent) -> None:
        """Aplica el REROUTE y se guarda las dos rutas para poder puntuarlo."""
        anterior = list(agent.path)
        nueva = reroute(agent, agent.graph)
        if nueva is None:
            return
        self._last_reroute[agent.id] = (anterior, list(nueva))
        log.debug(
            "AGV %s: reroute desde %s, %d nodos -> %d nodos",
            agent.id,
            agent.current_node,
            len(anterior),
            len(nueva),
        )


__all__ = [
    "ACTIONS",
    "FORMAT",
    "QUEUE_CAP",
    "STATE_FIELDS",
    "STATE_SIZES",
    "Action",
    "Event",
    "QLearningPolicy",
    "QTable",
    "SimulationView",
    "State",
    "decode_state",
    "distance_bucket",
    "enabled_actions",
    "encode_state",
    "get_local_state",
    "is_useless_reroute",
    "report_state_space",
    "reroute",
    "reroute_penalties",
    "reward",
    "state_from_local",
    "state_space_size",
    "to_engine_action",
]


if __name__ == "__main__":
    setup_logging()
    for _linea in report_state_space():
        log.info("%s", _linea)
