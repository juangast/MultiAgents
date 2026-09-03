"""Conflictos entre AGVs y la politica baseline contra la que comparar el resto.

Modulo puro: no mueve agentes, no guarda estado global y no sabe nada del reloj.
Recibe intenciones y devuelve conflictos; quien los aplica es
`simulation.Simulation.tick()`.

Deteccion y resolucion viven separadas a proposito. `detect_conflicts()` dice
QUE pasa y `resolve_baseline()` dice QUIEN gana, asi que la fase 8 puede cambiar
la segunda sin tocar la primera.

Modelo de ocupacion (**reserva doble**): el movimiento es continuo, un agente
tarda `cost(a, b)` ticks en cruzar un tramo y a media travesia su `current_node`
sigue siendo el nodo del que salio. Por eso un agente que cruza X -> Y **retiene
los dos nodos** y suelta X solo cuando llega a Y:

    tick 3:   AGV 1  ------>------   progreso 0.4
              X                 Y
       occupancy: X -> 1,  Y -> 1

`occupancy` sigue siendo `nodo -> un solo agent_id`, que es la invariante del
almacen. Lo que se relaja es lo contrario: un agente puede tener dos nodos, un
nodo nunca tiene dos agentes. La consecuencia es que el **following esta
prohibido**: nadie entra en el nodo que otro esta dejando hasta que lo suelta.
Cuesta throughput y provoca deadlocks, y eso es lo correcto para un baseline.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import astar
import config
from agent import STATE_WAITING, Agent
from graph import WarehouseGraph
from logs import get_logger

log = get_logger("conflicts")

# Los cuatro tipos de conflicto que el almacen sabe reconocer.
TYPE_VERTEX: str = "vertex"
TYPE_EDGE: str = "edge"
TYPE_FOLLOWING: str = "following"
TYPE_CONGESTION: str = "congestion"

TYPES: tuple[str, ...] = (TYPE_VERTEX, TYPE_EDGE, TYPE_FOLLOWING, TYPE_CONGESTION)

# Lo que una politica puede contestar. Son dos a proposito: la fase 8 aprende
# cuando ceder el paso, no a inventarse rutas.
ACTION_GO: str = "go"
ACTION_WAIT: str = "wait"
ACTIONS: tuple[str, ...] = (ACTION_GO, ACTION_WAIT)

# --- Las tres intenciones de la fase 8 ---------------------------------------
#
# Una INTENCION no es una garantia: dice lo que el AGV quiere hacer, no lo que va
# a pasar. Quien concede o niega es el motor, con el gate fisico. Por eso son
# vocabulario aparte del `go`/`wait` de arriba, que es lo que el motor entiende:
# `go` es "quiero pasar" y ADVANCE es "elijo avanzar", y entre las dos cosas esta
# la autoridad del motor.
INTENT_ADVANCE: str = "advance"
INTENT_WAIT: str = "wait"
INTENT_REROUTE: str = "reroute"
INTENTS: tuple[str, ...] = (INTENT_ADVANCE, INTENT_WAIT, INTENT_REROUTE)

# Como se lee una politica de las de antes. `go` es la forma vieja de decir
# ADVANCE, asi que `BaselinePolicy` y cualquier politica escrita para la fase 5
# siguen valiendo sin tocarles una linea.
_INTENT_ALIAS: dict[str, str] = {
    ACTION_GO: INTENT_ADVANCE,
    INTENT_ADVANCE: INTENT_ADVANCE,
    ACTION_WAIT: INTENT_WAIT,
    INTENT_REROUTE: INTENT_REROUTE,
}


def normalize_intent(value: str) -> str:
    """Traduce lo que devuelva una politica a una de las tres intenciones.

    Lanza `ValueError` con la lista de las que hay si el valor no es ninguna, por
    lo mismo que `qlearning.reward()` lanza con un evento mal escrito: una accion
    que se traga en silencio y acaba valiendo "wait" se busca durante dias.
    """
    intencion = _INTENT_ALIAS.get(str(value).strip().lower())
    if intencion is None:
        conocidas = ", ".join(sorted({*_INTENT_ALIAS}))
        raise ValueError(
            f"la politica devolvio {value!r}; las acciones que hay son {conocidas}"
        )
    return intencion

# Que agente quiere entrar en que nodo en este tick.
Intents = Mapping[int, str]
# Que agente ocupa cada nodo. Un nodo, un dueño.
Occupancy = Mapping[str, int]

_ORDEN: dict[str, int] = {tipo: numero for numero, tipo in enumerate(TYPES)}


@dataclass(frozen=True, slots=True)
class Conflict:
    """Un choque concreto, con su tipo, sus culpables y donde pasa.

    Es inmutable y hasheable a proposito: un conflicto es un hecho de un tick,
    no algo que se edite despues. `agents` va siempre como tupla ordenada por id
    para que dos detecciones del mismo choque sean el mismo objeto.

    `node` lo llevan `vertex`, `following` y `congestion`; `edge` lleva `edge`
    con el par ya ordenado. Nunca los dos a la vez.
    """

    type: str
    agents: tuple[int, ...]
    step: int
    node: str | None = None
    edge: tuple[str, str] | None = None

    def __post_init__(self) -> None:
        if self.type not in _ORDEN:
            raise ValueError(f"tipo de conflicto desconocido: {self.type!r}")

    def as_dict(self) -> dict[str, Any]:
        """El conflicto como JSON, para el registro de la corrida."""
        return {
            "step": self.step,
            "type": self.type,
            "agents": list(self.agents),
            "node": self.node,
            "edge": list(self.edge) if self.edge is not None else None,
        }


@dataclass(frozen=True, slots=True)
class Resolution:
    """Quien gana un conflicto y quien se queda esperando.

    En `congestion` no hay nada que arbitrar: no es una disputa por un nodo, es
    un sintoma. Sale `Resolution(None, ())`.
    """

    winner: int | None
    losers: tuple[int, ...]


class ConflictLog:
    """Los conflictos de una corrida, con el conteo ya hecho.

    Se vacia en cada `reset()`: el registro es *por corrida*, que es la unidad
    con la que se compara el baseline contra lo que venga despues.
    """

    def __init__(self) -> None:
        self._items: list[Conflict] = []

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __repr__(self) -> str:
        return f"ConflictLog(total={self.total}, by_type={self.by_type})"

    @property
    def total(self) -> int:
        """Cuantos conflictos lleva la corrida."""
        return len(self._items)

    @property
    def by_type(self) -> dict[str, int]:
        """Conteo por tipo, con los cuatro tipos siempre presentes.

        Los ceros van explicitos: un informe en el que falta `edge` no se
        distingue de uno en el que `edge` salio cero.
        """
        cuenta = dict.fromkeys(TYPES, 0)
        for conflicto in self._items:
            cuenta[conflicto.type] += 1
        return cuenta

    def append(self, conflict: Conflict) -> None:
        """Anota un conflicto."""
        self._items.append(conflict)

    def extend(self, conflicts: Iterable[Conflict]) -> None:
        """Anota varios de golpe."""
        self._items.extend(conflicts)

    def records(self) -> list[dict[str, Any]]:
        """La lista entera en JSON: paso, tipo y agentes de cada choque."""
        return [conflicto.as_dict() for conflicto in self._items]

    def clear(self) -> None:
        """Empieza una corrida nueva."""
        self._items.clear()


# --- Deteccion ---------------------------------------------------------------


def detect_conflicts(
    intents: Intents,
    graph: WarehouseGraph,
    *,
    occupancy: Occupancy | None = None,
    agents: Sequence[Agent] | None = None,
    step: int = 0,
    previous_zones: frozenset[str] = frozenset(),
) -> list[Conflict]:
    """Todos los conflictos que provocan estas intenciones, antes de mover nada.

    `intents` es `{agent_id: nodo al que quiere entrar en este tick}`. Solo
    declaran los agentes parados en un nodo: el que va a media travesia ya tiene
    su reserva concedida y no vuelve a pedirla.

    Los dos primeros argumentos bastan para la deteccion de vertices; `occupancy`
    y `agents` son opcionales porque el modulo tiene que poder probarse sin
    montar una simulacion entera. Sin ellos no hay forma de saber donde esta cada
    agente, asi que `edge`, `following` y `congestion` no se pueden ver.

    El orden de salida es estable (vertex, edge, following, congestion) y un
    mismo par de agentes sale una sola vez: si dos se cruzan de frente es un
    `edge`, no ademas dos `vertex`.
    """
    por_id: dict[int, Agent] = {agente.id: agente for agente in (agents or ())}
    ocupacion: Occupancy = occupancy if occupancy is not None else {}

    # El orden importa: `edge` y `following` explican el choque mejor que
    # `vertex`, asi que se detectan antes y se quedan con el par.
    de_arista = _edge(intents, por_id, graph, step)
    de_estela = _following(intents, por_id, ocupacion, step)

    explicados = {frozenset(conflicto.agents) for conflicto in de_arista + de_estela}
    de_vertice = _vertex(intents, ocupacion, step, explicados)

    de_atasco = detect_congestion(
        agents or (), graph, step, previous_zones=previous_zones
    )

    conflictos = de_vertice + de_arista + de_estela + de_atasco
    conflictos.sort(key=lambda c: (_ORDEN[c.type], c.node or "", c.edge or (), c.agents))
    return conflictos


def _vertex(
    intents: Intents,
    occupancy: Occupancy,
    step: int,
    explicados: set[frozenset[int]],
) -> list[Conflict]:
    """Dos o mas agentes queriendo el mismo nodo en el mismo tick.

    El que ya esta encima del nodo cuenta como uno mas que lo quiere: no lo esta
    pidiendo, lo esta usando, y eso es exactamente lo que impide entrar.
    """
    contendientes: dict[str, set[int]] = {}
    for agent_id, destino in intents.items():
        contendientes.setdefault(destino, set()).add(agent_id)
        ocupante = occupancy.get(destino)
        if ocupante is not None and ocupante != agent_id:
            contendientes[destino].add(ocupante)

    conflictos: list[Conflict] = []
    for destino, ids in contendientes.items():
        if len(ids) < 2 or frozenset(ids) in explicados:
            continue
        conflictos.append(
            Conflict(TYPE_VERTEX, tuple(sorted(ids)), step, node=destino)
        )
    return conflictos


def _edge(
    intents: Intents,
    por_id: Mapping[int, Agent],
    graph: WarehouseGraph,
    step: int,
) -> list[Conflict]:
    """El cruce de frente: A va de X a Y mientras B va de Y a X.

    Es el conflicto que la ocupacion sola no ve venir: los dos nodos estan
    "libres" desde el punto de vista del que sale, porque cada uno esta en el
    que el otro quiere. Sin esta comprobacion los dos AGVs se atraviesan.
    """
    conflictos: list[Conflict] = []
    ids = sorted(intents)
    for indice, a_id in enumerate(ids):
        for b_id in ids[indice + 1 :]:
            uno, otro = por_id.get(a_id), por_id.get(b_id)
            if uno is None or otro is None:
                continue
            if intents[a_id] != otro.current_node or intents[b_id] != uno.current_node:
                continue
            if not graph.has_edge(uno.current_node, otro.current_node):
                continue
            tramo = (uno.current_node, otro.current_node)
            conflictos.append(
                Conflict(
                    TYPE_EDGE,
                    (a_id, b_id),
                    step,
                    edge=tramo if tramo[0] <= tramo[1] else (tramo[1], tramo[0]),
                )
            )
    return conflictos


def _following(
    intents: Intents,
    por_id: Mapping[int, Agent],
    occupancy: Occupancy,
    step: int,
) -> list[Conflict]:
    """A quiere entrar en el nodo que B esta dejando en este mismo tick.

    **No se permite.** Con reserva doble, B no suelta su nodo de salida hasta
    que llega al otro extremo, asi que A se queda esperando. Se detecta y se
    cuenta igualmente: el agente pierde tiempo por culpa de otro, y eso es lo
    que la fase 8 tiene que aprender a evitar.
    """
    conflictos: list[Conflict] = []
    for agent_id in sorted(intents):
        destino = intents[agent_id]
        ocupante = occupancy.get(destino)
        if ocupante is None or ocupante == agent_id:
            continue
        otro = por_id.get(ocupante)
        # `progress > 0` y el nodo pedido es el suyo de salida: se esta yendo.
        if otro is None or otro.progress <= 0.0 or otro.current_node != destino:
            continue
        conflictos.append(
            Conflict(
                TYPE_FOLLOWING,
                tuple(sorted((agent_id, ocupante))),
                step,
                node=destino,
            )
        )
    return conflictos


def detect_congestion(
    agents: Sequence[Agent],
    graph: WarehouseGraph,
    step: int,
    *,
    threshold: int = config.CONFLICT_WAIT_THRESHOLD,
    zone_agents: int = config.CONGESTION_ZONE_AGENTS,
    previous_zones: frozenset[str] = frozenset(),
) -> list[Conflict]:
    """Atascos: un agente que lleva demasiado esperando, o una zona colapsada.

    Cada uno se emite **solo en el tick en que se cruza el umbral**, no en cada
    tick que dura. Si no, un atasco de cincuenta ticks contaria como cincuenta
    conflictos y el numero dejaria de significar nada.
    """
    conflictos: list[Conflict] = []

    for agente in sorted(agents, key=lambda a: a.id):
        if agente.wait_time == threshold:
            conflictos.append(
                Conflict(
                    TYPE_CONGESTION, (agente.id,), step, node=agente.current_node
                )
            )

    for zona in sorted(congested_zones(agents, graph, zone_agents=zone_agents)):
        if zona in previous_zones:
            continue
        conflictos.append(
            Conflict(TYPE_CONGESTION, _esperando_en(agents, graph, zona), step, node=zona)
        )
    return conflictos


def congested_zones(
    agents: Sequence[Agent],
    graph: WarehouseGraph,
    *,
    zone_agents: int = config.CONGESTION_ZONE_AGENTS,
) -> frozenset[str]:
    """Zonas con `zone_agents` o mas agentes esperando dentro.

    Una zona es un nodo mas sus vecinos directos: un atasco no cabe en un solo
    nodo, se forma alrededor de el. En el almacen la zona interesante es la de
    `graph.BOTTLENECK`, por donde pasan todas las rutas que lo cruzan.

    Va aparte de `detect_congestion()` porque quien recuerda las zonas de un tick
    al siguiente es el motor: aqui no se guarda estado.
    """
    return frozenset(
        nodo
        for nodo in graph.nodes()
        if len(_esperando_en(agents, graph, nodo)) >= zone_agents
    )


def _esperando_en(
    agents: Sequence[Agent], graph: WarehouseGraph, node: str
) -> tuple[int, ...]:
    """Ids de los agentes en `waiting` dentro de la zona de `node`."""
    zona = {node, *graph.neighbors(node)}
    return tuple(
        sorted(
            agente.id
            for agente in agents
            if agente.state == STATE_WAITING and agente.current_node in zona
        )
    )


# --- El reroute --------------------------------------------------------------
#
# Recalcular es mecanica del MOTOR, no del aprendizaje: la politica solo dice
# "reroute" y quien encarece el mapa y llama a A* es la simulacion. Por eso estas
# dos funciones viven aqui y no en `qlearning.py` (que las re-exporta para no
# romper a quien ya las importaba de alli).


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


# --- Resolucion --------------------------------------------------------------


def resolve_baseline(conflict: Conflict) -> Resolution:
    """La politica base: gana el agente con el id menor. Y ya.

    Simple **a proposito**. Es la referencia experimental contra la que medir el
    Q-Learning de la fase 8, no la solucion: no mira rutas, ni prioridades, ni
    quien lleva mas esperando, asi que el mismo AGV gana siempre y los de id
    alto se mueren de hambre en el cuello de botella. Eso es justo lo que tiene
    que verse en los numeros.

    Es **pura**: dice quien gana, no toca a nadie. Quien pone `state="waiting"` y
    suma el `wait_time` es el motor. Asi se prueba sin construir agentes y la
    politica de la fase 8 entra en el mismo hueco sin heredar mutaciones
    escondidas.
    """
    if conflict.type == TYPE_CONGESTION or not conflict.agents:
        return Resolution(None, ())

    ordenados = tuple(sorted(conflict.agents))
    return Resolution(ordenados[0], ordenados[1:])


# --- Politicas ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LocalState:
    """Lo unico que una politica ve del mundo cuando le toca decidir.

    Es deliberadamente **local**: el nodo en el que esta, a donde quiere ir, lo
    que lleva esperando y quien le ha ganado este tick. Ni el mapa entero ni las
    rutas de los demas. Si la fase 8 pudiera mirarlo todo aprenderia una politica
    centralizada, que es otro problema distinto del que plantea el proyecto.
    """

    step: int
    node: str
    intent: str | None
    wait_time: int
    blocked_by: tuple[int, ...]
    conflicts: tuple[Conflict, ...]
    occupancy: Occupancy
    neighbors: tuple[str, ...]


@runtime_checkable
class Policy(Protocol):
    """Lo que la simulacion necesita de una politica para poder usarla.

    Es el contrato de la inyeccion de dependencia, igual que
    `protocol.Simulation` lo es para el servidor: `Simulation` acepta cualquiera
    que lo cumpla, asi que el Q-Learning de la fase 8 se enchufa sin tocar el
    motor.
    """

    name: str

    def decide(self, agent: Agent, local_state: LocalState) -> str:
        """Devuelve `ACTION_GO` o `ACTION_WAIT` para este agente en este tick."""
        ...


class BaselinePolicy:
    """Cede el paso si alguien te gano el conflicto. Nada mas.

    Envuelve `resolve_baseline()`: el motor ya calculo quien gano y lo dejo en
    `local_state.blocked_by`. Sin aprendizaje, sin memoria y sin azar, asi que
    dos corridas con la misma semilla dan exactamente lo mismo.
    """

    name: str = "baseline"

    def __repr__(self) -> str:
        return "BaselinePolicy()"

    def decide(self, agent: Agent, local_state: LocalState) -> str:
        """`ACTION_WAIT` si le ganaron el paso, `ACTION_GO` si no."""
        return ACTION_WAIT if local_state.blocked_by else ACTION_GO


def read_only(occupancy: dict[str, int]) -> Occupancy:
    """Vista de solo lectura de la ocupacion, para pasarsela a una politica.

    Una politica no tiene por que poder reescribir el mapa de ocupacion, y menos
    una que en la fase 8 sera codigo de aprendizaje probando cosas.
    """
    return MappingProxyType(occupancy)


__all__ = [
    "ACTIONS",
    "ACTION_GO",
    "ACTION_WAIT",
    "INTENTS",
    "INTENT_ADVANCE",
    "INTENT_REROUTE",
    "INTENT_WAIT",
    "BaselinePolicy",
    "Conflict",
    "ConflictLog",
    "Intents",
    "LocalState",
    "Occupancy",
    "Policy",
    "Resolution",
    "TYPES",
    "TYPE_CONGESTION",
    "TYPE_EDGE",
    "TYPE_FOLLOWING",
    "TYPE_VERTEX",
    "congested_zones",
    "detect_conflicts",
    "detect_congestion",
    "normalize_intent",
    "read_only",
    "reroute",
    "reroute_penalties",
    "resolve_baseline",
]
