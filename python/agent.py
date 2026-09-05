"""El AGV: la ruta que sigue, en que punto va y que tarea o entrega lleva.

Cada agente es dueno de su ruta: `path` se copia, nunca se comparte. No sabe
nada del reloj, quien lo mueve es `simulation.Simulation.tick()`.
"""

import math
from enum import Enum

import config
import missions
from graph import Box, Penalties, WarehouseGraph, astar
from config import get_logger

log = get_logger("agent")

class State(str, Enum):
    """En que anda el AGV. El orden es el del ciclo de vida."""

    __str__ = str.__str__

    IDLE = "idle"
    MOVING = "moving"
    WAITING = "waiting"
    PICKING = "picking"
    DROPPING = "dropping"
    CHARGING = "charging"
    DONE = "done"


class Leg(str, Enum):
    """En que mitad de la entrega va el AGV."""

    __str__ = str.__str__

    NONE = "none"
    TO_PICK = "to_pick"
    TO_DROP = "to_drop"
    TO_CHARGER = "to_charger"


PICK_BASE_TICKS: int = 2
PICK_LEVEL_TICKS: int = 2
DROP_TICKS: int = 2


def pick_ticks(level: int) -> int:
    """Cuantos ticks cuesta recoger una caja de ese nivel.

    El nivel 1 cuesta `PICK_BASE_TICKS` y cada nivel por encima suma
    `PICK_LEVEL_TICKS`: subir la horquilla cuesta tiempo.
    """
    return PICK_BASE_TICKS + max(0, level - 1) * PICK_LEVEL_TICKS


class Agent:
    """Un AGV del almacen.

    `wait_time` acumula y solo vuelve a cero en `reset()`: es el tiempo que este
        AGV ha perdido cediendo el paso en toda la corrida.

        Una entrega son dos tramos, `Leg.TO_PICK` hasta la caja y `Leg.TO_DROP`
        hasta el muelle; `carrying` es lo que distingue a un AGV cargado de uno
        vacio.
    """

    def __init__(self, agent_id: int, graph: WarehouseGraph, start: str) -> None:
        self.id: int = agent_id
        self.graph: WarehouseGraph = graph
        self.start_node: str = start

        self.current_node: str = start
        self.target_node: str | None = None
        self.path: list[str] = []
        self.path_index: int = 0
        self.state: str = State.IDLE
        self.wait_time: int = 0
        self.task: int | None = None
        self.progress: float = 0.0

        self.leg: Leg = Leg.NONE
        self.box: str | None = None
        self.destination: str | None = None
        self.carrying: str | None = None
        self.busy: int = 0

        self.mission: str | None = None
        self.completed: int = 0
        self.battery: float = config.BATTERY_FULL
        self.charges: int = 0

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.id}, node={self.current_node!r}, "
            f"target={self.target_node!r}, state={self.state!r})"
        )

    def assign_task(
        self,
        origin: str,
        target: str,
        task: int | None = None,
        penalties: Penalties | None = None,
    ) -> bool:
        """Le da una tarea nueva y le calcula la ruta con A*."""
        trazada = self._traza(origin, target, task=task, penalties=penalties)
        self.leg = Leg.NONE
        self.box = None
        self.destination = None
        self.carrying = None
        self.busy = 0
        return trazada

    def _traza(
        self,
        origin: str,
        target: str,
        task: int | None = None,
        penalties: Penalties | None = None,
    ) -> bool:
        """Corre A* y deja al agente listo para recorrer la ruta.

        Es el trazado pelado, sin tocar nada de la entrega: lo que comparten
        `assign_task()`, `assign_delivery()` y `route_to_destination()`.
        """
        ruta = astar(self.graph, origin, target, penalties)

        self.target_node = target
        self.path_index = 0
        self.progress = 0.0
        self.wait_time = 0

        if ruta is None:
            self.path = []
            self.task = None
            self.state = State.IDLE
            log.warning(
                "AGV %s: no hay ruta de %s a %s, se queda en %s",
                self.id,
                origin,
                target,
                self.current_node,
            )
            return False

        self.current_node = origin
        self.path = list(ruta)
        self.task = task
        self.state = State.DONE if len(self.path) == 1 else State.MOVING
        return True

    def assign_delivery(
        self,
        origin: str,
        box: Box,
        destination: str,
        task: int | None = None,
        penalties: Penalties | None = None,
    ) -> bool:
        """Le da una entrega entera: ir a por `box` y llevarla a `destination`."""
        if not self._traza(origin, box.node, task=task, penalties=penalties):
            self.leg = Leg.NONE
            self.box = None
            self.destination = None
            self.carrying = None
            self.busy = 0
            return False

        self.leg = Leg.TO_PICK
        self.box = box.id
        self.destination = destination
        self.carrying = None
        self.busy = 0
        return True

    def route_to_destination(self, penalties: Penalties | None = None) -> bool:
        """Traza el segundo tramo: desde donde recogio hasta el muelle.

        La llama la simulacion cuando termina la recogida, porque hasta ese
        momento no se sabe en que nodo acaba el AGV ni que penalizaciones habra.
        """
        if self.destination is None:
            return False
        if not self._traza(
            self.current_node, self.destination, task=self.task, penalties=penalties
        ):
            self.leg = Leg.NONE
            return False
        self.leg = Leg.TO_DROP
        return True

    def start_pick(self, level: int) -> None:
        """Se pone a recoger: queda ocupado los ticks que pida el nivel."""
        self.state = State.PICKING
        self.busy = pick_ticks(level)
        self.progress = 0.0

    def start_drop(self) -> None:
        """Se pone a dejar la caja en el muelle."""
        self.state = State.DROPPING
        self.busy = DROP_TICKS
        self.progress = 0.0

    def work(self) -> bool:
        """Gasta un tick de la maniobra en curso. True si acaba de terminarla."""
        if self.busy <= 0:
            return True
        self.busy -= 1
        return self.busy == 0

    def finish_pick(self) -> None:
        """Cierra la recogida: la caja pasa a ir encima del AGV."""
        self.carrying = self.box
        self.leg = Leg.TO_DROP
        self.busy = 0

    def finish_drop(self) -> None:
        """Cierra la entrega: la caja se queda en el muelle y el AGV termina."""
        self.carrying = None
        self.leg = Leg.NONE
        self.busy = 0
        self.state = State.DONE

    def available(self) -> bool:
        """True si puede aceptar una mision: libre, parado y con bateria."""
        return (
            self.mission is None
            and self.state in (State.IDLE, State.DONE)
            and self.battery > config.BATTERY_THRESHOLD
        )

    def drain(self) -> float:
        """Gasta un tick de bateria. Devuelve lo que queda."""
        self.battery = max(0.0, self.battery - config.BATTERY_DRAIN)
        return self.battery

    def charge(self) -> bool:
        """Enchufado un tick. True cuando ya esta lleno."""
        self.battery = min(config.BATTERY_FULL, self.battery + config.BATTERY_CHARGE_RATE)
        return self.battery >= config.BATTERY_FULL

    def is_dead(self) -> bool:
        """Se quedo sin bateria en mitad del almacen."""
        return self.battery <= 0.0

    def workload(self) -> float:
        """Lo cargado de trabajo que va. Entra en su propia puja."""
        return (1.0 if self.mission else 0.0) + 0.25 * self.completed

    def viable(self, pool, chargers) -> list:
        """De lo publicado, lo que la bateria le da para terminar."""
        return [
            m for m in pool
            if missions.reaches(self.battery, self.graph, self.current_node, m, chargers)
        ]

    def needs_charge(self, pool, chargers) -> bool:
        """Se va a cargar por umbral, porque ya no llegaria, o porque no le alcanza.

        El umbral fijo no basta: cruzar un mapa grande puede costar mas de lo que
        queda por debajo del umbral, y el AGV se planta a cero en mitad de un
        pasillo. Por eso tambien se va **en cuanto dejaria de poder llegar a un
        cargador**, que es la condicion que no admite esperar.
        """
        if self.battery <= config.BATTERY_THRESHOLD:
            return True
        if not self.can_reach_charger(chargers):
            return True
        return bool(pool) and not self.viable(pool, chargers)

    def can_reach_charger(self, chargers) -> bool:
        """Le da la bateria para llegar al cargador mas cercano y quedar con reserva."""
        cerca = [d for c in chargers if (d := self.distance_to(c)) is not None]
        if not cerca:
            return True
        gasto = missions.battery_cost(min(cerca))
        return self.battery - gasto >= config.BATTERY_RESERVE

    def bid(self, bus, t: int, pool, chargers=()) -> list:
        """Mira lo publicado, calcula su utilidad y puja. O dice que no puede.

        Aqui es donde el AGV decide por si mismo: nadie le pregunta y nadie le
        asigna nada. Solo puja por lo que su bateria le da para **terminar**.
        """
        if not self.available():
            if pool:
                bus.publish(missions.Message(
                    t, f"AGV-{self.id}", "TODOS", missions.MessageType.UNAVAILABLE,
                    {"agv": self.id, "motivo": self._motivo()},
                ))
            return []

        viables = self.viable(pool, chargers)
        if pool and not viables:
            bus.publish(missions.Message(
                t, f"AGV-{self.id}", "TODOS", missions.MessageType.UNAVAILABLE,
                {"agv": self.id, "motivo": f"bateria insuficiente {self.battery:.0f}%"},
            ))
            return []

        pujas = []
        for mision in viables:
            distancia = self.distance_to(mision.node)
            if distancia is None:
                continue
            utilidad = missions.utility(
                distancia, self.workload(), mision.level, self.battery
            )
            pujas.append((mision.id, utilidad))
            bus.publish(missions.Message(
                t, f"AGV-{self.id}", "TODOS", missions.MessageType.BID,
                {"agv": self.id, "mision": mision.id, "utilidad": round(utilidad, 2)},
            ))
        return pujas

    def distance_to(self, node: str) -> float | None:
        """Distancia en linea recta hasta ese nodo. None si falta una posicion.

        Para pujar basta la geometria y sale barata; la ruta de verdad la traza
        A* despues, cuando ya se sabe que la mision es suya.
        """
        aqui = self.graph.positions.get(self.current_node)
        alli = self.graph.positions.get(node)
        if aqui is None or alli is None:
            return None
        return math.dist(aqui, alli)

    def _motivo(self) -> str:
        """Por que no puede pujar, para el log del bus."""
        if self.state not in (State.IDLE, State.DONE) or self.mission:
            return f"{self.state} {self.mission or ''}".strip()
        return f"bateria baja {self.battery:.0f}%"

    def next_node(self) -> str | None:
        """El nodo hacia el que se mueve ahora, o None si ya no queda camino."""
        siguiente = self.path_index + 1
        if 0 <= siguiente < len(self.path):
            return self.path[siguiente]
        return None

    def previous_node(self) -> str | None:
        """El nodo del que viene, o None si no ha salido del primero.

        Sirve para saber hacia donde mira un agente que ya llego: conserva el
        rumbo del ultimo tramo en vez de girar a cero de golpe.
        """
        anterior = self.path_index - 1
        if 0 <= anterior < len(self.path):
            return self.path[anterior]
        return None

    def has_arrived(self) -> bool:
        """True si `current_node` ya es el ultimo nodo de la ruta."""
        return bool(self.path) and self.path_index >= len(self.path) - 1

    def reset(self) -> None:
        """Lo deja como recien creado, en su nodo de partida y sin tarea."""
        self.current_node = self.start_node
        self.target_node = None
        self.path = []
        self.path_index = 0
        self.state = State.IDLE
        self.wait_time = 0
        self.task = None
        self.progress = 0.0
        self.leg = Leg.NONE
        self.box = None
        self.destination = None
        self.carrying = None
        self.busy = 0
