"""La simulacion de verdad: AGVs recorriendo el grafo del almacen.

Sustituye a la `FakeSimulation` de la fase 1 y cumple el mismo contrato,
`protocol.Simulation` (`get_snapshot()` + `reset()`), asi que el servidor la
acepta sin cambiar una linea de transporte.

El movimiento es continuo: un agente tarda `cost(a, b)` ticks en cruzar un tramo
y su `progress` avanza `1/cost` por tick, asi que el snapshot lleva la posicion
ya interpolada entre los dos nodos y Unity no ve teletransportes.
"""

import math
import random
import threading

import config
import protocol
from agent import STATE_DONE, STATE_IDLE, STATE_MOVING, STATE_WAITING, Agent
from graph import WarehouseGraph
from logs import get_logger

log = get_logger("simulation")

# Ruta de demostracion de cada mapa. En `warehouse` cruza el cuello de botella G
# a proposito: es el tramo por el que pasan todos los escenarios de congestion.
DEFAULT_ROUTES: dict[str, tuple[str, str]] = {
    "simple": ("A", "F"),
    "warehouse": ("S1", "N6"),
}


def default_route(graph: WarehouseGraph) -> tuple[str, str]:
    """Origen y destino por defecto del mapa.

    Si el mapa no esta en la tabla (o la pareja ya no existe en el), tira del
    primer y el ultimo nodo, que `nodes()` devuelve ordenados y por tanto son
    siempre los mismos.
    """
    ruta = DEFAULT_ROUTES.get(graph.name)
    if ruta is not None and all(nodo in graph.adjacency for nodo in ruta):
        return ruta

    nodos = graph.nodes()
    if not nodos:
        raise ValueError("el mapa no tiene ni un nodo")
    return nodos[0], nodos[-1]


class Simulation:
    """El almacen en marcha: un grafo, sus agentes y el contador de pasos.

    Es segura entre hilos porque el servidor comparte una sola instancia entre
    todos los clientes. El cerrojo es un `RLock` y no un `Lock` porque
    `get_snapshot()` re-entra en `tick()`.
    """

    def __init__(
        self,
        graph: WarehouseGraph,
        n_agents: int = 1,
        *,
        origin: str | None = None,
        target: str | None = None,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        if n_agents < 1:
            raise ValueError(f"hace falta al menos un agente, no {n_agents}")

        self.graph: WarehouseGraph = graph
        self.step: int = 0
        self.seed: int = seed
        self._lock = threading.RLock()

        por_defecto = default_route(graph)
        self._origen: str = origin if origin is not None else por_defecto[0]
        self._destino: str = target if target is not None else por_defecto[1]
        self._rutas: list[tuple[str, str]] = self._planea_rutas(n_agents)

        self.agents: list[Agent] = [
            Agent(numero, graph, origen)
            for numero, (origen, _) in enumerate(self._rutas, start=1)
        ]
        self.reset()

    def __repr__(self) -> str:
        return (
            f"Simulation(map={self.graph.name!r}, agents={len(self.agents)}, "
            f"step={self.step})"
        )

    @property
    def done(self) -> bool:
        """True cuando ya ningun agente tiene nada que hacer."""
        with self._lock:
            return all(
                agente.state in (STATE_DONE, STATE_IDLE) for agente in self.agents
            )

    def tick(self) -> int:
        """Avanza la simulacion un paso y devuelve el numero de paso.

        El contador sube siempre, aunque todos los agentes hayan llegado: el
        cliente de Unity comprueba que `step` crece de una peticion a la
        siguiente y no debe verlo estancarse nunca.
        """
        with self._lock:
            self.step += 1
            for agente in self.agents:
                self._mueve(agente)
            return self.step

    def get_snapshot(self) -> protocol.Snapshot:
        """Avanza un paso y devuelve el estado completo en coordenadas de Unity.

        Que la peticion sea la que mueve el mundo es el contrato de la fase 1:
        Python no empuja nada por su cuenta, y dos clientes a la vez ven pasos
        distintos porque cada `GET_STATE` consume su propio tick.
        """
        with self._lock:
            self.tick()
            return {
                "step": self.step,
                "agents": [self._describe(agente) for agente in self.agents],
            }

    def reset(self) -> None:
        """Vuelve al paso cero y reparte otra vez las mismas tareas.

        Determinista: con la misma semilla y el mismo mapa, dos simulaciones
        recien reiniciadas producen exactamente la misma secuencia de snapshots.
        """
        with self._lock:
            self.step = 0
            for agente, (origen, destino) in zip(self.agents, self._rutas):
                agente.reset()
                agente.assign_task(origen, destino, task=agente.id)
            log.info(
                "simulacion reiniciada: mapa %s, %d agente(s)",
                self.graph.name or "(sin nombre)",
                len(self.agents),
            )

    def _planea_rutas(self, n_agents: int) -> list[tuple[str, str]]:
        """Reparte origen y destino, siempre igual para la misma semilla.

        El primer agente hace la ruta fija del mapa, que es la que se demuestra.
        Los demas salen del generador sembrado con `seed`: la fase 3 corre con un
        solo agente, pero esto deja la fase 4 lista y reproducible.
        """
        rutas = [(self._origen, self._destino)]
        nodos = self.graph.nodes()
        if len(nodos) < 2:
            # Un mapa de un solo nodo: no hay de donde sacar pares distintos.
            return rutas * n_agents
        if n_agents == 1:
            return rutas

        rng = random.Random(self.seed)
        for _ in range(n_agents - 1):
            origen, destino = rng.sample(nodos, 2)
            rutas.append((origen, destino))
        return rutas

    def _mueve(self, agente: Agent) -> None:
        """Un tick de un agente. Se llama con el cerrojo ya tomado."""
        if agente.state == STATE_WAITING:
            # Todavia no lo usa nadie: con un agente no hay a quien cederle el
            # paso. Queda cableado para la gestion de colisiones de la fase 4.
            agente.wait_time = max(0, agente.wait_time - 1)
            if agente.wait_time == 0:
                agente.state = STATE_DONE if agente.has_arrived() else STATE_MOVING
            return

        if agente.state != STATE_MOVING:
            return

        siguiente = agente.next_node()
        if siguiente is None:
            agente.state = STATE_DONE
            agente.progress = 0.0
            return

        if not self.graph.has_edge(agente.current_node, siguiente):
            # A* nunca produce esto; solo puede pasar si alguien manipulo el
            # `path` a mano. Se para el agente en vez de reventar la simulacion.
            log.error(
                "AGV %s: su ruta pasa por %s -> %s, que no es una arista del mapa",
                agente.id,
                agente.current_node,
                siguiente,
            )
            agente.state = STATE_IDLE
            agente.progress = 0.0
            return

        costo = self.graph.cost(agente.current_node, siguiente)
        # Un tramo de costo cero se cruza en el mismo tick, sin dividir por cero.
        agente.progress = 1.0 if costo <= 0.0 else agente.progress + 1.0 / costo

        if agente.progress >= 1.0:
            agente.current_node = siguiente
            agente.path_index += 1
            agente.progress = 0.0
            if agente.has_arrived():
                agente.state = STATE_DONE

    def _describe(self, agente: Agent) -> dict[str, object]:
        """Un agente tal y como lo ve Unity."""
        px, py = self._posicion(agente)
        x, y, z = protocol.to_unity(px, py)
        return {
            # Los seis campos congelados en la fase 1: ni nombre ni tipo cambian.
            "id": agente.id,
            "x": x,
            "y": y,
            "z": z,
            "rotation": self._rotacion(agente),
            "state": agente.state,
            # Lo que agrega la fase 3, por encima del formato congelado.
            "node": agente.current_node,
            "next_node": agente.next_node(),
            "path": list(agente.path),
            "task": agente.task,
        }

    def _posicion(self, agente: Agent) -> tuple[float, float]:
        """Posicion logica, interpolada entre el nodo actual y el siguiente.

        Se interpola en coordenadas logicas y la conversion a Unity la hace
        `protocol.to_unity()` una sola vez, en `_describe()`: esa conversion no se
        duplica en ningun sitio del proyecto.
        """
        actual = self.graph.positions.get(agente.current_node)
        if actual is None:
            return 0.0, 0.0

        siguiente = agente.next_node()
        if siguiente is None or agente.progress <= 0.0:
            return actual

        destino = self.graph.positions.get(siguiente)
        if destino is None:
            return actual

        avance = min(agente.progress, 1.0)
        return (
            actual[0] + (destino[0] - actual[0]) * avance,
            actual[1] + (destino[1] - actual[1]) * avance,
        )

    def _rotacion(self, agente: Agent) -> float:
        """Rumbo del agente en grados sobre el eje vertical de Unity.

        En Unity 0 grados es mirar a +Z y se gira en sentido horario; como la `y`
        logica es la `z` de Unity, el angulo es `atan2(dx, dy)` sobre las
        coordenadas logicas. Un agente que ya llego mira hacia donde venia, en vez
        de girar a cero de golpe.
        """
        desde: str | None = agente.current_node
        hasta = agente.next_node()
        if hasta is None:
            desde, hasta = agente.previous_node(), agente.current_node
        if desde is None or hasta is None or desde == hasta:
            return 0.0

        origen = self.graph.positions.get(desde)
        destino = self.graph.positions.get(hasta)
        if origen is None or destino is None:
            return 0.0

        dx = destino[0] - origen[0]
        dy = destino[1] - origen[1]
        if dx == 0.0 and dy == 0.0:
            return 0.0
        return math.degrees(math.atan2(dx, dy)) % 360.0
