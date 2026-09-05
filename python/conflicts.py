"""Deteccion de conflictos y la politica base.

Cuatro tipos de choque (vertex, edge, following, congestion) y el desempate por
id menor. La politica es intercambiable: el Q-Learning entra por `Policy`.
"""

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from enum import Enum
from typing import Protocol

import config
from agent import Agent, State
from graph import Penalties, WarehouseGraph, astar
from config import get_logger

log = get_logger("conflicts")

CONFLICT_WAIT_THRESHOLD: int = 5
CONGESTION_ZONE_AGENTS: int = 3


class ConflictType(str, Enum):
    """Los cuatro tipos de choque que el almacen sabe reconocer."""

    __str__ = str.__str__

    VERTEX = "vertex"
    EDGE = "edge"
    FOLLOWING = "following"
    CONGESTION = "congestion"


class Intent(str, Enum):
    """Lo que una politica puede contestar.

    Son tres a proposito: se aprende cuando ceder el paso, no a inventar rutas.
    """

    __str__ = str.__str__

    ADVANCE = "advance"
    WAIT = "wait"
    REROUTE = "reroute"


_INTENT_ALIAS: dict[str, Intent] = {
    "go": Intent.ADVANCE,
    "advance": Intent.ADVANCE,
    "wait": Intent.WAIT,
    "reroute": Intent.REROUTE,
}


def normalize_intent(value: str) -> Intent:
    """Traduce lo que devuelva una politica a una de las tres intenciones."""
    intencion = _INTENT_ALIAS.get(str(value).strip().lower())
    if intencion is None:
        conocidas = ", ".join(sorted({*_INTENT_ALIAS}))
        raise ValueError(
            f"la politica devolvio {value!r}; las acciones que hay son {conocidas}"
        )
    return intencion

Intents = Mapping[int, str]
Occupancy = Mapping[str, int]

_ORDEN: dict[str, int] = {tipo: numero for numero, tipo in enumerate(ConflictType)}


class Conflict:
    """Un choque concreto, con su tipo, sus culpables y donde pasa."""

    def __init__(
        self,
        type: str,
        agents: tuple[int, ...],
        step: int,
        node: str | None = None,
        edge: tuple[str, str] | None = None,
    ) -> None:
        self.type = type
        self.agents = agents
        self.step = step
        self.node = node
        self.edge = edge

    def __post_init__(self) -> None:
        if self.type not in _ORDEN:
            raise ValueError(f"tipo de conflicto desconocido: {self.type!r}")


class Resolution:
    """Quien gana un conflicto y quien se queda esperando.

    En `congestion` no hay nada que arbitrar: no es una disputa por un nodo, es
    un sintoma. Sale `Resolution(None, ())`.
    """

    def __init__(self, winner: int | None, losers: tuple[int, ...]) -> None:
        self.winner = winner
        self.losers = losers


class ConflictLog:
    """Los conflictos de una corrida, con el conteo ya hecho.

    Se vacia en cada `reset()`: el registro es *por corrida*, que es la unidad
    con la que se compara el baseline contra lo que venga despues.
    """

    def __init__(self) -> None:
        self._items: list[Conflict] = []

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def total(self) -> int:
        """Cuantos conflictos lleva la corrida."""
        return len(self._items)

    def by_type(self) -> dict[str, int]:
        """Conteo por tipo, con los cuatro tipos siempre presentes.

        Los ceros van explicitos: un informe en el que falta `edge` no se
        distingue de uno en el que `edge` salio cero.
        """
        cuenta = dict.fromkeys(ConflictType, 0)
        for conflicto in self._items:
            cuenta[conflicto.type] += 1
        return cuenta

    def extend(self, conflicts: Iterable[Conflict]) -> None:
        """Anota varios de golpe."""
        self._items.extend(conflicts)

    def clear(self) -> None:
        """Empieza una corrida nueva."""
        self._items.clear()


def detect_conflicts(
    intents: Intents,
    graph: WarehouseGraph,
    *,
    occupancy: Occupancy | None = None,
    agents: Sequence[Agent] | None = None,
    step: int = 0,
    previous_zones: frozenset[str] = frozenset(),
) -> list[Conflict]:
    """Todos los conflictos que provocan estas intenciones, antes de mover nada."""
    por_id: dict[int, Agent] = {agente.id: agente for agente in (agents or ())}
    ocupacion: Occupancy = occupancy if occupancy is not None else {}

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
            Conflict(ConflictType.VERTEX, tuple(sorted(ids)), step, node=destino)
        )
    return conflictos


def _edge(
    intents: Intents,
    por_id: Mapping[int, Agent],
    graph: WarehouseGraph,
    step: int,
) -> list[Conflict]:
    """El cruce de frente: A va de X a Y mientras B va de Y a X."""
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
                    ConflictType.EDGE,
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
    """A quiere entrar en el nodo que B esta dejando en este mismo tick."""
    conflictos: list[Conflict] = []
    for agent_id in sorted(intents):
        destino = intents[agent_id]
        ocupante = occupancy.get(destino)
        if ocupante is None or ocupante == agent_id:
            continue
        otro = por_id.get(ocupante)
        if otro is None or otro.progress <= 0.0 or otro.current_node != destino:
            continue
        conflictos.append(
            Conflict(
                ConflictType.FOLLOWING,
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
    threshold: int = CONFLICT_WAIT_THRESHOLD,
    zone_agents: int = CONGESTION_ZONE_AGENTS,
    previous_zones: frozenset[str] = frozenset(),
) -> list[Conflict]:
    """Atascos: un agente que lleva demasiado esperando, o una zona colapsada."""
    conflictos: list[Conflict] = []

    for agente in sorted(agents, key=lambda a: a.id):
        if agente.wait_time == threshold:
            conflictos.append(
                Conflict(
                    ConflictType.CONGESTION, (agente.id,), step, node=agente.current_node
                )
            )

    for zona in sorted(congested_zones(agents, graph, zone_agents=zone_agents)):
        if zona in previous_zones:
            continue
        conflictos.append(
            Conflict(ConflictType.CONGESTION, _esperando_en(agents, graph, zona), step, node=zona)
        )
    return conflictos


def congested_zones(
    agents: Sequence[Agent],
    graph: WarehouseGraph,
    *,
    zone_agents: int = CONGESTION_ZONE_AGENTS,
) -> frozenset[str]:
    """Zonas con `zone_agents` o mas agentes esperando dentro."""
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
            if agente.state == State.WAITING and agente.current_node in zona
        )
    )


def reroute_penalties(
    agent: Agent, *, penalty: float | None = None
) -> dict[str | tuple[str, str], float]:
    """Encarece lo que este AGV tiene delante, para que A* lo esquive."""
    siguiente = agent.next_node()
    if siguiente is None:
        return {}
    extra = config.REROUTE_PENALTY if penalty is None else float(penalty)
    return {siguiente: extra, (agent.current_node, siguiente): extra}


def reroute(
    agent: Agent,
    graph: WarehouseGraph,
    *,
    penalties: Penalties | None = None,
) -> list[str] | None:
    """Recalcula la ruta desde donde esta el agente, esquivando lo de delante."""
    if agent.target_node is None or agent.progress > 0.0:
        return None

    if penalties is None:
        penalties = reroute_penalties(agent)

    nueva = astar(graph, agent.current_node, agent.target_node, penalties)
    if nueva is None:
        log.debug("AGV %s: reroute sin salida desde %s", agent.id, agent.current_node)
        return None

    agent.path = list(nueva)
    agent.path_index = 0
    agent.progress = 0.0
    return agent.path

def resolve_baseline(conflict: Conflict) -> Resolution:
    """La politica base: gana el agente con el id menor. Y ya.

    Gana el id menor. En congestion no hay a quien ceder, asi que no gana nadie.
    """
    if conflict.type == ConflictType.CONGESTION or not conflict.agents:
        return Resolution(None, ())

    ordenados = tuple(sorted(conflict.agents))
    return Resolution(ordenados[0], ordenados[1:])

class LocalState:
    """Lo unico que una politica ve del mundo cuando le toca decidir."""

    def __init__(
        self,
        step: int,
        node: str,
        intent: str | None,
        wait_time: int,
        blocked_by: tuple[int, ...],
        conflicts: tuple[Conflict, ...],
        occupancy: Occupancy,
        neighbors: tuple[str, ...],
    ) -> None:
        self.step = step
        self.node = node
        self.intent = intent
        self.wait_time = wait_time
        self.blocked_by = blocked_by
        self.conflicts = conflicts
        self.occupancy = occupancy
        self.neighbors = neighbors


class Policy(Protocol):
    """Lo que la simulacion necesita de una politica para poder usarla."""

    name: str

    def decide(self, agent: Agent, local_state: LocalState) -> str:
        """Devuelve `Intent.ADVANCE` o `Intent.WAIT` para este agente en este tick."""
        ...


class BaselinePolicy:
    """Cede el paso si alguien te gano el conflicto. Nada mas."""

    name: str = "baseline"

    def __repr__(self) -> str:
        return "BaselinePolicy()"

    def decide(self, agent: Agent, local_state: LocalState) -> str:
        """`Intent.WAIT` si le ganaron el paso, `Intent.ADVANCE` si no."""
        return Intent.WAIT if local_state.blocked_by else Intent.ADVANCE


def read_only(occupancy: dict[str, int]) -> Occupancy:
    """Vista de solo lectura de la ocupacion, para pasarsela a una politica.

    Una politica no tiene por que poder reescribir el mapa de ocupacion, y menos
    una que en la fase 8 sera codigo de aprendizaje probando cosas.
    """
    return MappingProxyType(occupancy)
