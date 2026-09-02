"""El AGV: la ruta que sigue, en que punto va y que tarea lleva.

Cada agente es dueño de su propia ruta. Nada de listas compartidas entre
agentes: `assign_task()` se queda con una copia de lo que devuelve A*, y `path`
se crea vacia en cada `__init__`, nunca como valor por defecto de un parametro.

No sabe nada del reloj: quien lo mueve es `simulation.Simulation.tick()`. Aqui
solo vive el estado y las preguntas que se le pueden hacer.
"""

import astar
from graph import WarehouseGraph
from logs import get_logger

log = get_logger("agent")

STATE_IDLE: str = "idle"
STATE_MOVING: str = "moving"
STATE_WAITING: str = "waiting"
STATE_DONE: str = "done"

# El orden es el del ciclo de vida: parado, moviendose, cediendo el paso, llegado.
STATES: tuple[str, ...] = (STATE_IDLE, STATE_MOVING, STATE_WAITING, STATE_DONE)


class Agent:
    """Un AGV del almacen.

    Campos publicos, que son los que salen en el snapshot:

    | Campo          | Que es                                                  |
    |----------------|---------------------------------------------------------|
    | `id`           | Identificador del AGV                                   |
    | `current_node` | Nodo en el que esta, o del que acaba de salir           |
    | `target_node`  | A donde va, o None si no tiene tarea                    |
    | `path`         | La ruta entera, de origen a destino                     |
    | `path_index`   | Que posicion de `path` es `current_node`                |
    | `state`        | Uno de STATES                                           |
    | `wait_time`    | Ticks acumulados cediendo el paso                       |
    | `task`         | Id de la tarea que lleva, o None                        |
    | `progress`     | 0..1 entre `current_node` y `next_node()`               |

    Ademas guarda `graph` y `start_node`, que no van al snapshot: el primero
    porque `assign_task()` tiene que correr A*, el segundo porque `reset()` tiene
    que saber a que nodo volver para que la simulacion sea reproducible.

    `wait_time` **acumula**, no descuenta: es el tiempo total que este AGV ha
    perdido cediendo el paso en toda la corrida, y por eso solo vuelve a cero en
    `reset()`. Quien lo sube es `simulation.Simulation`, un tick por cada vez que
    le toca esperar, y quien lo saca de `waiting` es tambien la simulacion cuando
    el nodo que pedia queda libre, no un contador que llega a cero.
    """

    def __init__(self, agent_id: int, graph: WarehouseGraph, start: str) -> None:
        self.id: int = agent_id
        self.graph: WarehouseGraph = graph
        self.start_node: str = start

        self.current_node: str = start
        self.target_node: str | None = None
        self.path: list[str] = []
        self.path_index: int = 0
        self.state: str = STATE_IDLE
        self.wait_time: int = 0
        self.task: int | None = None
        self.progress: float = 0.0

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
        penalties: astar.Penalties | None = None,
    ) -> bool:
        """Le da una tarea nueva y le calcula la ruta con A*.

        Devuelve True si encontro ruta. Si no la hay el agente se queda `idle`
        con la ruta vacia y **no lanza**: que dos zonas del almacen esten
        incomunicadas es un estado normal del mapa, no un error del programa.

        Llamarla otra vez con otro destino recalcula la ruta desde cero, que es
        lo que necesita el REROUTE de la fase 8.
        """
        ruta = astar.astar(self.graph, origin, target, penalties)

        self.target_node = target
        self.path_index = 0
        self.progress = 0.0
        self.wait_time = 0

        if ruta is None:
            # No se toca `current_node`: si `origin` no existe en el mapa,
            # apuntar ahi dejaria al agente en un sitio que no se puede dibujar.
            self.path = []
            self.task = None
            self.state = STATE_IDLE
            log.warning(
                "AGV %s: no hay ruta de %s a %s, se queda en %s",
                self.id,
                origin,
                target,
                self.current_node,
            )
            return False

        self.current_node = origin
        self.path = list(ruta)  # copia propia: nunca la misma lista que otro agente
        self.task = task
        # Una ruta de un solo nodo es que ya estaba en el destino.
        self.state = STATE_DONE if len(self.path) == 1 else STATE_MOVING
        return True

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
        self.state = STATE_IDLE
        self.wait_time = 0
        self.task = None
        self.progress = 0.0
