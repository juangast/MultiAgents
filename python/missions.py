"""Misiones, comunicacion y negociacion: el reparto de trabajo es una subasta.

Nadie asigna las misiones desde arriba. El `MissionManager` las publica al bus y
**no decide nada**: no calcula distancias, no compara AGVs y no elige ganador.
Cada AGV mira lo publicado, calcula su propia utilidad y puja; el que mas ofrece
se la lleva. Todo lo que se dicen queda registrado en el `MessageBus`.

Una mision es una entrega entera: recoger una caja de su estanteria y dejarla en
un muelle.
"""

import math
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

import config
from config import get_logger
from graph import ROLE_DOCK, ROLE_PRODUCTION, ROLE_STORAGE, Box, WarehouseGraph

log = get_logger("missions")


class MissionStatus(str, Enum):
    """El ciclo de vida de una mision."""

    __str__ = str.__str__

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


class BoxStatus(str, Enum):
    """Por donde va una caja. Son los cinco estados de M3."""

    __str__ = str.__str__

    WAITING_PICKUP = "WAITING_PICKUP"
    STORED = "STORED"
    RESERVED = "RESERVED"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"


class BoxState:
    """Una caja durante la corrida: donde esta ahora y como esta.

    Es lo unico que se mueve del almacen. `graph.boxes` es el inventario inicial
    y no cambia nunca; esto es su copia viva.
    """

    def __init__(
        self,
        id: str,
        node: str,
        level: int,
        status: BoxStatus,
        mission: str | None = None,
    ) -> None:
        self.id = id
        self.node = node
        self.level = level
        self.status = status
        self.mission = mission

    def as_dict(self) -> dict[str, Any]:
        """La caja tal y como la ve Unity."""
        return {
            "id": self.id,
            "node": self.node,
            "level": self.level,
            "status": self.status,
            "mission": self.mission,
        }


def build_inventory(graph: WarehouseGraph) -> dict[str, BoxState]:
    """Copia viva del inventario del mapa, con el estado que le toca a cada caja."""
    if not graph.boxes:
        raise ValueError(
            f"el mapa {graph.name!r} no tiene cajas: sin 'boxes' en el JSON no "
            f"hay nada que mover"
        )
    return {
        caja.id: BoxState(
            id=caja.id,
            node=caja.node,
            level=caja.level,
            status=(
                BoxStatus.WAITING_PICKUP
                if graph.role_of(caja.node) == ROLE_PRODUCTION
                else BoxStatus.STORED
            ),
        )
        for caja in sorted(graph.boxes, key=lambda c: c.id)
    }


class Flow(str, Enum):
    """Los dos flujos logisticos del almacen.

    `PRODUCTION_TO_RACK` guarda lo que sale de la linea de produccion;
    `RACK_TO_DOCK` saca del almacen lo que ya estaba guardado.
    """

    __str__ = str.__str__

    PRODUCTION_TO_RACK = "produccion -> rack"
    RACK_TO_DOCK = "rack -> muelle"


class MessageType(str, Enum):
    """Lo que se puede decir por el bus."""

    __str__ = str.__str__

    MISSION_PUBLISHED = "MISSION_PUBLISHED"
    BID = "BID"
    UNAVAILABLE = "UNAVAILABLE"
    ACCEPT = "ACCEPT"
    PICKED_UP = "PICKED_UP"
    COMPLETED = "COMPLETED"
    CHARGING = "CHARGING"


class Message:
    """Una linea del bus: quien lo dijo, a quien y que."""

    def __init__(
        self,
        t: int,
        emisor: str,
        receptor: str,
        tipo: MessageType,
        contenido: dict[str, Any] | None = None,
    ) -> None:
        self.t = t
        self.emisor = emisor
        self.receptor = receptor
        self.tipo = tipo
        self.contenido = contenido if contenido is not None else {}

    def __str__(self) -> str:
        detalle = " ".join(f"{k}={v}" for k, v in self.contenido.items())
        return f"[t={self.t:3d}] {self.emisor:>7} -> {self.receptor:<7} {self.tipo:<18} {detalle}"


class MessageBus:
    """El canal unico. Guarda el historial y un indice por paso."""

    def __init__(self) -> None:
        self.history: list[Message] = []
        self._por_t: dict[int, list[Message]] = {}

    def __len__(self) -> int:
        return len(self.history)

    def publish(self, message: Message) -> Message:
        """Mete un mensaje en el bus."""
        self.history.append(message)
        self._por_t.setdefault(message.t, []).append(message)
        return message

    def read(
        self, tipo: MessageType | None = None, t: int | None = None
    ) -> list[Message]:
        """Lee el bus, filtrando por tipo, por paso o por los dos."""
        fuente = self._por_t.get(t, []) if t is not None else self.history
        return [m for m in fuente if tipo is None or m.tipo is tipo]

    def clear(self) -> None:
        """Vacia el bus. Se llama en cada `reset()`."""
        self.history.clear()
        self._por_t.clear()


class Mission:
    """Recoger una caja de `node` y dejarla en `destination`."""

    def __init__(
        self,
        id: str,
        flow: Flow,
        box: str,
        node: str,
        level: int,
        destination: str,
        status: MissionStatus = MissionStatus.PENDING,
        agv_id: int | None = None,
        t_publicada: int | None = None,
        t_aceptada: int | None = None,
        t_completada: int | None = None,
    ) -> None:
        self.id = id
        self.flow = flow
        self.box = box
        self.node = node
        self.level = level
        self.destination = destination
        self.status = status
        self.agv_id = agv_id
        self.t_publicada = t_publicada
        self.t_aceptada = t_aceptada
        self.t_completada = t_completada


def flow_of(graph: WarehouseGraph, box: BoxState) -> Flow | None:
    """El flujo que pide una caja segun donde esta parada, o None si no pide nada.

    Una caja recien fabricada hay que guardarla; una guardada hay que sacarla;
    una que ya salio, ni una cosa ni la otra.
    """
    if box.status is BoxStatus.WAITING_PICKUP:
        return Flow.PRODUCTION_TO_RACK
    if box.status is BoxStatus.STORED:
        return Flow.RACK_TO_DOCK
    return None


class MissionManager:
    """Publica el trabajo que hay y apunta lo que pasa con el.

    Hace tres cosas y nada mas: publicar, registrar quien acepto y registrar
    quien termino. **No decide nada**: no calcula distancias, no compara AGVs y
    no elige ganador. Quien ejecuta cada mision lo deciden los AGVs pujando.

    Las misiones no estan escritas de antemano: salen del inventario en cada
    paso, y por eso una caja guardada en el rack genera despues su salida.
    """

    def __init__(self, bus: MessageBus, graph: WarehouseGraph) -> None:
        self.bus = bus
        self.graph = graph
        self.missions: dict[str, Mission] = {}
        self.completed: list[str] = []
        self._racks = graph.nodes_with_role(ROLE_STORAGE)
        self._muelles = graph.nodes_with_role(ROLE_DOCK)
        self._turno: dict[Flow, int] = {flujo: 0 for flujo in Flow}
        self._numero = 0

    def _destino(self, flow: Flow) -> str | None:
        """A donde va esta mision. Se reparte por turno para no amontonar."""
        pool = self._racks if flow is Flow.PRODUCTION_TO_RACK else self._muelles
        if not pool:
            return None
        destino = pool[self._turno[flow] % len(pool)]
        self._turno[flow] += 1
        return destino

    def open_work(self, inventory: dict[str, BoxState]) -> list[Mission]:
        """Abre una mision por cada caja que pide trabajo y no la tiene ya.

        Es lo que encadena los dos flujos: en cuanto una caja pasa a `STORED`,
        el paso siguiente le abre su mision de salida.
        """
        nuevas: list[Mission] = []
        for caja in inventory.values():
            if caja.mission is not None:
                continue
            flujo = flow_of(self.graph, caja)
            if flujo is None:
                continue
            destino = self._destino(flujo)
            if destino is None:
                continue

            self._numero += 1
            mision = Mission(
                id=f"M{self._numero:02d}",
                flow=flujo,
                box=caja.id,
                node=caja.node,
                level=caja.level,
                destination=destino,
            )
            self.missions[mision.id] = mision
            caja.mission = mision.id
            nuevas.append(mision)
        return nuevas

    def pool(self) -> list[Mission]:
        """Las misiones que siguen en la bolsa, sin dueño."""
        return [
            m for m in self.missions.values() if m.status is MissionStatus.PENDING
        ]

    def publish(self, t: int) -> list[Mission]:
        """Saca al bus todo lo que queda pendiente."""
        pendientes = self.pool()
        for mision in pendientes:
            if mision.t_publicada is None:
                mision.t_publicada = t
            self.bus.publish(Message(t, "MANAGER", "TODOS", MessageType.MISSION_PUBLISHED, {
                "mision": mision.id,
                "flujo": mision.flow,
                "caja": mision.box,
                "nodo": mision.node,
                "nivel": mision.level,
                "destino": mision.destination,
            }))
        return pendientes

    def accepted(self, t: int, mission: Mission, agv_id: int) -> None:
        """Apunta que un AGV se llevo la mision."""
        mission.status = MissionStatus.ACCEPTED
        mission.agv_id = agv_id
        mission.t_aceptada = t

    def picked_up(self, t: int, mission: Mission, agv_id: int) -> None:
        """Apunta que el AGV ya tiene la caja encima."""
        mission.status = MissionStatus.IN_PROGRESS
        self.bus.publish(Message(t, f"AGV-{agv_id}", "MANAGER", MessageType.PICKED_UP, {
            "mision": mission.id, "caja": mission.box,
        }))

    def finished(self, t: int, mission: Mission, agv_id: int) -> None:
        """Apunta que la caja ya esta en el muelle."""
        mission.status = MissionStatus.COMPLETED
        mission.t_completada = t
        self.completed.append(mission.id)
        self.bus.publish(Message(t, f"AGV-{agv_id}", "MANAGER", MessageType.COMPLETED, {
            "mision": mission.id, "caja": mission.box, "destino": mission.destination,
        }))

    def done(self) -> bool:
        """True cuando no queda ni una mision por servir."""
        return all(
            m.status is MissionStatus.COMPLETED for m in self.missions.values()
        )

    def by_agent(self, agv_id: int) -> Mission | None:
        """La mision que lleva ese AGV, si lleva alguna."""
        for mision in self.missions.values():
            if mision.agv_id == agv_id and mision.status in (
                MissionStatus.ACCEPTED, MissionStatus.IN_PROGRESS
            ):
                return mision
        return None


def resolve_auction(bids: Iterable[tuple[int, float]]) -> tuple[int, float] | None:
    """El ganador de una subasta: la utilidad mas alta, y a igualdad el id menor.

    El desempate por id no es un capricho: sin el, dos AGVs con la misma
    utilidad harian que la corrida dependiera del orden del diccionario.
    """
    pujas = sorted(bids, key=lambda p: (-p[1], p[0]))
    return pujas[0] if pujas else None


def resolve_auctions(
    bus: MessageBus,
    t: int,
    manager: MissionManager,
    agents: Sequence[Any],
) -> list[tuple[Any, Mission, float]]:
    """Resuelve todas las subastas del paso. Devuelve (agv, mision, utilidad).

    Un AGV solo puede ganar una mision por paso: en cuanto se lleva una sale de
    `libres` y sus pujas por las demas ya no cuentan.
    """
    por_id = {agv.id: agv for agv in agents}
    libres = {agv.id for agv in agents if agv.available()}
    del_paso = bus.read(MessageType.BID, t)
    ganadas: list[tuple[Any, Mission, float]] = []

    for mision in manager.pool():
        pujas = [
            (m.contenido["agv"], m.contenido["utilidad"])
            for m in del_paso
            if m.contenido.get("mision") == mision.id and m.contenido["agv"] in libres
        ]
        ganador = resolve_auction(pujas)
        if ganador is None:
            continue

        agv_id, utilidad = ganador
        bus.publish(Message(t, f"AGV-{agv_id}", "MANAGER", MessageType.ACCEPT, {
            "mision": mision.id, "utilidad": round(utilidad, 2),
        }))
        manager.accepted(t, mision, agv_id)
        libres.discard(agv_id)
        ganadas.append((por_id[agv_id], mision, utilidad))

    return ganadas


BID_W_DISTANCE: float = 2.0
BID_W_WORKLOAD: float = 20.0
BID_W_LEVEL: float = 4.0
BID_W_BATTERY: float = 0.5


def utility(distance: float, workload: float, level: int, battery: float) -> float:
    """Lo que vale una mision para un AGV. Cuanto mas alta, mas la quiere.

    Cuatro terminos: lo lejos que esta la caja, lo cargado de trabajo que va, lo
    alto que este la caja (subir la horquilla cuesta ticks) y la bateria que le
    queda. Gana el AGV libre, cercano, descansado y con las pilas llenas.
    """
    return (
        -BID_W_DISTANCE * distance
        - BID_W_WORKLOAD * workload
        - BID_W_LEVEL * max(0, level - 1)
        + BID_W_BATTERY * battery
    )


def estimated_cost(
    graph: WarehouseGraph, node: str, mission: Mission, chargers: Sequence[str]
) -> float:
    """Bateria que cuesta la mision entera: ida a la caja, transporte y salida al cargador.

    Las tres distancias van en linea recta, que es lo que sale barato de calcular
    al pujar; `BATTERY_DETOUR` corrige que la ruta de verdad rodea.
    """
    ida = _recta(graph, node, mission.node)
    carga = _recta(graph, mission.node, mission.destination)
    al_cargador = min(
        (_recta(graph, mission.destination, c) for c in chargers), default=0.0
    )
    return config.BATTERY_DRAIN * config.BATTERY_DETOUR * (ida + carga + al_cargador)


def battery_cost(distance: float) -> float:
    """Lo que cuesta en bateria recorrer esa distancia en linea recta."""
    return config.BATTERY_DRAIN * config.BATTERY_DETOUR * distance


def reaches(
    battery: float,
    graph: WarehouseGraph,
    node: str,
    mission: Mission,
    chargers: Sequence[str],
) -> bool:
    """Filtro de viabilidad: la bateria tiene que dar para terminar y quedar con reserva.

    No basta con estar por encima del umbral. Un AGV que acepta una mision que no
    puede terminar se queda tirado a medio pasillo.
    """
    gasto = estimated_cost(graph, node, mission, chargers)
    return battery - gasto >= config.BATTERY_RESERVE


def _recta(graph: WarehouseGraph, a: str, b: str) -> float:
    """Distancia en linea recta entre dos nodos. 0.0 si falta una posicion."""
    p, q = graph.positions.get(a), graph.positions.get(b)
    return 0.0 if p is None or q is None else math.dist(p, q)
