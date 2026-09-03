"""Las medidas de una corrida y la comparacion entre politicas (fase 9).

Modulo de **medicion**: no cambia el motor, lo observa. Ni una linea de
`simulation.py`, `conflicts.py` o `qlearning.py` se toca desde aqui, porque lo
que se compara son exactamente las fases 5-8 que ya tienen sus tests.

--- Por que el escenario se construye antes de saber la politica ---

El riesgo de esta fase no es el codigo, es el sesgo. Si el escenario del
baseline y el del Q-Learning no son el mismo, la comparacion no mide la
politica y no vale para nada. Por eso hay un `Scenario`, se construye **una vez
por semilla** y despues se corre con cada politica:

    escenario = build_scenario(grafo, 4, 16, seed=7)   # todavia no hay politica
    baseline  = run_once(grafo, escenario, "baseline")
    aprendida = run_once(grafo, escenario, "qlearning")

Las dos corridas comparten mapa, AGVs, origenes, destinos, la cola de tareas
entera y la semilla. Lo unico distinto es el nombre de la politica.

--- La cola de tareas ---

El motor de las fases 5-8 da UNA ruta por AGV y termina cuando llegan todos. La
cola de varias tareas vive **aqui**, en el runner, y se genera de golpe desde la
semilla, nunca segun hace falta. Es lo mismo que hacia `tasks.generate()` en la
fase 4 y por la misma razon: si los destinos se sortearan al asignarlos, el
orden de las extracciones dependeria del orden de llegada, el orden de llegada
depende de la politica, y ahi se acabo la comparacion pareada. Con la cola fija
la carga de trabajo es identica y lo unico que cambia es **quien** despacha que.
"""

import csv
import json
import random
import statistics
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
import conflicts
import qlearning
import simulation
from agent import STATE_DONE, Agent
from graph import WarehouseGraph
from logs import get_logger

log = get_logger("metrics")


@dataclass(frozen=True, slots=True)
class Scenario:
    """El trabajo de UNA semilla, listo antes de que exista ninguna politica.

    | Campo     | Que es                                                       |
    |-----------|--------------------------------------------------------------|
    | `seed`    | La semilla de la que salio todo                              |
    | `routes`  | Origen y destino inicial de cada AGV, sin repetir            |
    | `pending` | Los destinos que quedan en la cola, **en orden fijo**        |

    Es inmutable a proposito: un escenario que se pudiera editar entre una
    politica y la otra es exactamente el fallo que esta fase tiene que evitar.
    """

    seed: int
    routes: tuple[tuple[str, str], ...]
    pending: tuple[str, ...]

    @property
    def n_agents(self) -> int:
        """Cuantos AGVs corren este escenario."""
        return len(self.routes)

    @property
    def n_tasks(self) -> int:
        """Cuantas tareas hay que despachar en total, la inicial de cada AGV incluida."""
        return len(self.routes) + len(self.pending)


def build_scenario(
    graph: WarehouseGraph,
    n_agents: int,
    n_tasks: int | None = None,
    *,
    seed: int = config.RANDOM_SEED,
) -> Scenario:
    """Sortea el escenario entero de una semilla. Determinista y sin politica.

    Las rutas iniciales salen de `qlearning.random_routes()`, que ya garantiza
    origenes distintos, destinos distintos y que nadie arranque encima de su
    propio destino. La cola sale del **mismo** generador y a continuacion, asi
    que la semilla fija la lista completa: llamar dos veces con la misma semilla
    devuelve dos escenarios iguales campo a campo.
    """
    if n_agents < 1:
        raise ValueError(f"hace falta al menos un agente, no {n_agents}")

    total = n_agents * config.BENCHMARK_TASKS_PER_AGENT if n_tasks is None else int(n_tasks)
    if total < n_agents:
        raise ValueError(
            f"{total} tareas para {n_agents} agentes: cada AGV arranca con una, "
            f"asi que no pueden ser menos tareas que agentes"
        )

    rng = random.Random(seed)
    rutas = qlearning.random_routes(graph, n_agents, rng)
    nodos = graph.nodes()
    pendientes = tuple(rng.choice(nodos) for _ in range(total - n_agents))
    return Scenario(seed=int(seed), routes=tuple(rutas), pending=pendientes)


class RunMetrics:
    """Los numeros de UNA corrida, recogidos mientras pasa y no despues.

    Hay cosas que solo se pueden medir en marcha: la distancia recorrida no la
    guarda nadie, y el tick en que un AGV completo una tarea se pierde en cuanto
    se le da la siguiente. Por eso esto no es un informe final sino un
    recolector: `start()`, un `observe()` por tick y `close()` al terminar.

    | Metrica            | De donde sale                                       |
    |--------------------|-----------------------------------------------------|
    | `makespan`         | tick de la ultima tarea; si no se despacharon todas, |
    |                    | los ticks que duro la corrida                       |
    | `avg_task_time`    | media de (completada - asignada) por tarea          |
    | `total_wait_time`  | `agente.wait_time`, que acumula toda la corrida     |
    | `wait_by_agent`    | lo mismo, AGV por AGV                               |
    | `conflicts_by_type`| vertex / edge / following / congestion              |
    | `deadlock_count`   | 1 si la corrida murio atascada                      |
    | `total_distance`   | suma del costo de los tramos realmente pisados      |
    | `throughput`       | tareas completadas por 100 ticks                    |
    | `reroute_count`    | cuantas veces la politica pidio recalcular          |

    **Los totales crudos premian al que muere antes**: una corrida que se atasca
    en el tick 30 acumula menos conflictos y menos espera que una que corre 300
    y despacha el triple. Por eso van tambien `conflicts_per_tick` y
    `wait_per_tick`, y por eso `all_completed` sale en el CSV.
    """

    def __init__(
        self,
        *,
        policy: str,
        seed: int,
        map_name: str,
        n_agents: int,
        n_tasks: int,
    ) -> None:
        self.policy: str = policy
        self.seed: int = seed
        self.map_name: str = map_name
        self.n_agents: int = n_agents
        self.n_tasks: int = n_tasks

        self.ticks: int = 0
        self.completed_tasks: int = 0
        self.task_times: list[int] = []
        self.last_completion: int = 0
        self.total_distance: float = 0.0

        self.total_wait_time: int = 0
        self.wait_by_agent: dict[int, int] = {}
        self.conflicts_total: int = 0
        self.conflicts_by_type: dict[str, int] = dict.fromkeys(conflicts.TYPES, 0)
        self.deadlock_count: int = 0
        self.actions: dict[str, int] = dict.fromkeys(conflicts.INTENTS, 0)
        self.forced: int = 0
        self.finished_reason: str | None = None

        # Donde estaba cada AGV en el tick anterior, para la distancia.
        self._nodo: dict[int, str] = {}
        # AGV -> tick en que empezo la tarea que lleva ahora. Que este o no en el
        # dict es lo que distingue al que tiene tarea abierta del que ya no: sin
        # eso, un AGV que se queda `done` con la cola vacia sumaria una tarea
        # completada en cada tick de lo que quede de corrida.
        self._abierta: dict[int, int] = {}

    def __repr__(self) -> str:
        return (
            f"RunMetrics(policy={self.policy!r}, seed={self.seed}, "
            f"makespan={self.makespan}, conflicts={self.conflicts_total})"
        )

    @property
    def makespan(self) -> int:
        """Ticks hasta despachar todas las tareas, o los que duro si no se despacharon."""
        return self.last_completion if self.all_completed else self.ticks

    @property
    def all_completed(self) -> bool:
        """True si se despacho el trabajo entero. Lo que separa una corrida de una rota."""
        return self.completed_tasks >= self.n_tasks

    @property
    def avg_task_time(self) -> float:
        """Ticks que costo una tarea de media, sobre las que llegaron a terminar."""
        return statistics.fmean(self.task_times) if self.task_times else 0.0

    @property
    def throughput(self) -> float:
        """Tareas completadas por cada 100 ticks."""
        return 100.0 * self.completed_tasks / self.ticks if self.ticks else 0.0

    @property
    def reroute_count(self) -> int:
        """Cuantas veces la politica pidio recalcular la ruta."""
        return self.actions.get(conflicts.INTENT_REROUTE, 0)

    @property
    def conflicts_per_tick(self) -> float:
        """Conflictos por tick. Es lo comparable entre corridas de distinta duracion."""
        return self.conflicts_total / self.ticks if self.ticks else 0.0

    @property
    def wait_per_tick(self) -> float:
        """Ticks de espera por tick de corrida, entre todos los AGVs."""
        return self.total_wait_time / self.ticks if self.ticks else 0.0

    def start(self, sim: "simulation.Simulation") -> None:
        """Foto del paso cero: donde esta cada AGV y que tarea tiene abierta."""
        for agente in sim.agents:
            self._nodo[agente.id] = agente.current_node
            if agente.path:
                self._abierta[agente.id] = 0

    def observe(self, sim: "simulation.Simulation") -> None:
        """Un tick mas. Lo unico que hay que mirar en marcha es la distancia.

        Se mide por **cambio de `current_node`**, nunca por `path_index`: un
        REROUTE pone el indice a cero sin mover al AGV ni un metro, y contar por
        indice inflaria la cifra justo en la politica que mas recalcula.
        """
        self.ticks = sim.step
        for agente in sim.agents:
            anterior = self._nodo.get(agente.id)
            if anterior is None or anterior == agente.current_node:
                self._nodo[agente.id] = agente.current_node
                continue
            if sim.graph.has_edge(anterior, agente.current_node):
                self.total_distance += sim.graph.cost(anterior, agente.current_node)
            self._nodo[agente.id] = agente.current_node

    def start_task(self, agent_id: int, step: int) -> None:
        """Abre el reloj de una tarea nueva para este AGV."""
        self._abierta[agent_id] = step

    def complete_task(self, agent_id: int, step: int) -> bool:
        """Cierra la tarea abierta de este AGV. False si no tenia ninguna.

        Devolver False no es un error: es el AGV que ya entrego lo suyo y sigue
        `done` porque la cola esta vacia, y hay que poder distinguirlo del que
        acaba de llegar.
        """
        inicio = self._abierta.pop(agent_id, None)
        if inicio is None:
            return False
        self.completed_tasks += 1
        self.task_times.append(step - inicio)
        self.last_completion = step
        return True

    def close(self, sim: "simulation.Simulation") -> None:
        """Cierra la corrida y copia los contadores del motor."""
        numeros = sim.stats()
        self.ticks = sim.step
        self.total_wait_time = sum(agente.wait_time for agente in sim.agents)
        self.wait_by_agent = {agente.id: agente.wait_time for agente in sim.agents}
        self.conflicts_total = int(numeros["conflicts"])
        self.conflicts_by_type = dict(numeros["conflicts_by_type"])
        self.deadlock_count = int(sim.finished_reason == simulation.FINISHED_DEADLOCK)
        self.actions = dict(numeros["actions"])
        self.forced = int(numeros["forced"])
        self.finished_reason = numeros["finished_reason"]

    def to_dict(self) -> dict[str, Any]:
        """La corrida entera en JSON, con los diccionarios sin aplanar."""
        return {
            "policy": self.policy,
            "seed": self.seed,
            "map": self.map_name,
            "agents": self.n_agents,
            "tasks": self.n_tasks,
            "ticks": self.ticks,
            "makespan": self.makespan,
            "avg_task_time": round(self.avg_task_time, 3),
            "completed_tasks": self.completed_tasks,
            "all_completed": self.all_completed,
            "throughput": round(self.throughput, 3),
            "total_wait_time": self.total_wait_time,
            "wait_per_tick": round(self.wait_per_tick, 4),
            "wait_by_agent": dict(self.wait_by_agent),
            "total_distance": round(self.total_distance, 2),
            "conflicts": self.conflicts_total,
            "conflicts_per_tick": round(self.conflicts_per_tick, 4),
            "conflicts_by_type": dict(self.conflicts_by_type),
            "deadlocks": self.deadlock_count,
            "reroutes": self.reroute_count,
            "actions": dict(self.actions),
            "forced": self.forced,
            "finished_reason": self.finished_reason,
        }

    def to_row(self) -> dict[str, Any]:
        """La fila del CSV: todo plano, un valor por columna, sin anidar nada.

        Los booleanos salen como 0/1 y `finished_reason` vacio como cadena
        vacia, que es lo que Excel entiende sin preguntar.
        """
        fila: dict[str, Any] = {
            "policy": self.policy,
            "seed": self.seed,
            "map": self.map_name,
            "agents": self.n_agents,
            "tasks": self.n_tasks,
            "ticks": self.ticks,
            "makespan": self.makespan,
            "avg_task_time": round(self.avg_task_time, 3),
            "completed_tasks": self.completed_tasks,
            "all_completed": int(self.all_completed),
            "throughput": round(self.throughput, 3),
            "total_wait_time": self.total_wait_time,
            "wait_per_tick": round(self.wait_per_tick, 4),
            "total_distance": round(self.total_distance, 2),
            "conflicts": self.conflicts_total,
            "conflicts_per_tick": round(self.conflicts_per_tick, 4),
            "deadlocks": self.deadlock_count,
            "reroutes": self.reroute_count,
            "forced": self.forced,
            "finished_reason": self.finished_reason or "",
        }
        for tipo in conflicts.TYPES:
            fila[f"conflicts_{tipo}"] = self.conflicts_by_type.get(tipo, 0)
        for agent_id in sorted(self.wait_by_agent):
            fila[f"wait_agv_{agent_id}"] = self.wait_by_agent[agent_id]
        return fila


# --- El runner ---------------------------------------------------------------


def run_once(
    graph: WarehouseGraph,
    scenario: Scenario,
    policy: str,
    *,
    model: str | Path | None = None,
    max_steps: int = config.BENCHMARK_MAX_STEPS,
) -> RunMetrics:
    """Corre UN escenario con UNA politica y devuelve sus numeros.

    El escenario entra ya hecho: esta funcion no sortea nada, asi que dos
    llamadas con el mismo `Scenario` y distinta politica corren exactamente el
    mismo trabajo. Esa es toda la idea de la fase.

    La corrida termina cuando se despacha la cola, cuando el motor declara
    deadlock o cuando se acaban los ticks. Los tres finales son resultados
    validos y los tres se registran: una politica que se atasca no es un fallo
    del programa, es un dato.
    """
    simulacion = simulation.Simulation(
        graph,
        routes=list(scenario.routes),
        policy=policy,
        model=model,
        seed=scenario.seed,
    )
    medidas = RunMetrics(
        policy=simulacion.mode,
        seed=scenario.seed,
        map_name=graph.name or "(sin nombre)",
        n_agents=scenario.n_agents,
        n_tasks=scenario.n_tasks,
    )
    cola: deque[str] = deque(scenario.pending)

    medidas.start(simulacion)
    # Un AGV puede arrancar ya en su destino: cierra en el paso cero y coge la
    # siguiente de la cola antes del primer tick.
    _reparte(simulacion, cola, medidas)

    while simulacion.step < max_steps and not simulacion.done:
        simulacion.tick()
        # El orden importa: primero se anota lo que paso en el tick, y solo
        # despues se reparte. Al reves, la llegada del AGV quedaria tapada por la
        # tarea nueva y el tick de la entrega se perderia.
        medidas.observe(simulacion)
        _reparte(simulacion, cola, medidas)

    medidas.close(simulacion)
    return medidas


def _reparte(
    sim: "simulation.Simulation", cola: "deque[str]", medidas: RunMetrics
) -> None:
    """Al que llego le cierra la tarea y le da la siguiente de la cola.

    Se puede hacer desde fuera del motor sin romperle nada: un AGV en `done` esta
    parado en su nodo, `occupancy` tiene ese nodo y solo ese, y `assign_task()`
    con `origin == current_node` no lo mueve. Lo unico que reescribe son `path`,
    `path_index`, `progress` y `state`.
    """
    for agente in sim.agents:
        if agente.state != STATE_DONE:
            continue
        if not medidas.complete_task(agente.id, sim.step):
            # Ya habia entregado lo suyo y la cola estaba vacia: sigue `done`
            # tick tras tick y no hay nada nuevo que anotar.
            continue
        if not cola:
            continue
        _asigna(agente, cola.popleft(), sim.step, medidas)


def _asigna(agente: Agent, destino: str, step: int, medidas: RunMetrics) -> None:
    """Le da a este AGV la tarea siguiente, sin borrarle el reloj de la espera.

    `Agent.assign_task()` pone `wait_time` a 0, y `wait_time` es justo la medida
    con la que se comparan las politicas. Aqui se guarda y se restaura por la
    misma razon por la que `conflicts.reroute()` evita `assign_task()` a
    proposito: poner a cero el reloj de la espera en cada tarea nueva haria que
    el numero dejara de significar nada.
    """
    espera = agente.wait_time
    asignada = agente.assign_task(agente.current_node, destino, task=agente.id)
    agente.wait_time = espera

    if not asignada:
        log.warning(
            "AGV %s: no hay ruta de %s a %s, se queda sin tarea",
            agente.id,
            agente.current_node,
            destino,
        )
        return

    # Una ruta de un solo nodo es que el sorteo le toco el sitio donde ya estaba:
    # la tarea se cierra en el tick siguiente. Es legitimo, porque la cola es la
    # misma para las dos politicas; que a una le pille ahi y a la otra no es
    # exactamente la diferencia que se esta midiendo.
    medidas.start_task(agente.id, step)


def run_comparison(
    graph: WarehouseGraph,
    n_agents: int,
    n_tasks: int | None = None,
    seeds: Iterable[int] | None = None,
    policies: Sequence[str] = config.POLICIES,
    *,
    model: str | Path | None = None,
    max_steps: int = config.BENCHMARK_MAX_STEPS,
) -> dict[str, list[RunMetrics]]:
    """Enfrenta las politicas semilla a semilla, bajo condiciones identicas.

    Para CADA semilla se construye el escenario **una vez** y se corre con CADA
    politica. Que el escenario se construya fuera del bucle de politicas no es
    cosmetico: es lo que hace imposible que una politica vea un trabajo distinto
    del que vio la otra. Mismo mapa, mismos AGVs, mismos origenes, mismos
    destinos, misma cola y misma semilla; lo unico que cambia es el nombre.

    Devuelve `{politica: [RunMetrics por semilla]}`, con las listas en el mismo
    orden de semillas, que es lo que permite compararlas pareadas despues.
    """
    semillas = (
        list(range(1, config.BENCHMARK_RUNS + 1)) if seeds is None else [int(s) for s in seeds]
    )
    if not semillas:
        raise ValueError("hace falta al menos una semilla que correr")
    if not policies:
        raise ValueError("hace falta al menos una politica que comparar")

    resultados: dict[str, list[RunMetrics]] = {nombre: [] for nombre in policies}
    log.info(
        "--- benchmark: mapa %s, %d agente(s), %d tarea(s), %d semilla(s), politicas %s ---",
        graph.name or "(sin nombre)",
        n_agents,
        n_agents * config.BENCHMARK_TASKS_PER_AGENT if n_tasks is None else n_tasks,
        len(semillas),
        ", ".join(policies),
    )

    # Los logs de la simulacion son una linea de INFO por conflicto: con 20
    # semillas por dos politicas eso entierra el reporte, que es lo unico que hay
    # que leer. Los errores siguen saliendo.
    with qlearning._quiet("simulation", "agent", "conflicts"):
        for semilla in semillas:
            escenario = build_scenario(graph, n_agents, n_tasks, seed=semilla)
            for nombre in policies:
                resultados[nombre].append(
                    run_once(
                        graph, escenario, nombre, model=model, max_steps=max_steps
                    )
                )
            log.info(
                "semilla %3d | %s",
                semilla,
                " | ".join(
                    f"{nombre}: makespan {resultados[nombre][-1].makespan}, "
                    f"conflictos {resultados[nombre][-1].conflicts_total}"
                    for nombre in policies
                ),
            )
    return resultados


# --- Agregacion --------------------------------------------------------------

# Las metricas que se resumen y se comparan, en el orden en que se leen.
METRIC_FIELDS: tuple[str, ...] = (
    "makespan",
    "ticks",
    "avg_task_time",
    "completed_tasks",
    "throughput",
    "total_wait_time",
    "wait_per_tick",
    "total_distance",
    "conflicts",
    "conflicts_per_tick",
    "conflicts_vertex",
    "conflicts_edge",
    "conflicts_following",
    "conflicts_congestion",
    "deadlocks",
    "reroutes",
    "forced",
)

# En que direccion esta lo bueno. Sin esto, "diferencia %" es un numero sin
# signo util: bajar el makespan es ganar y bajar el throughput es perder.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "makespan",
        "ticks",
        "avg_task_time",
        "total_wait_time",
        "wait_per_tick",
        "total_distance",
        "conflicts",
        "conflicts_per_tick",
        "conflicts_vertex",
        "conflicts_edge",
        "conflicts_following",
        "conflicts_congestion",
        "deadlocks",
        "reroutes",
        "forced",
    }
)


def summarize(runs: Sequence[RunMetrics]) -> dict[str, Any]:
    """Media, desviacion tipica y mediana de cada metrica, mas las tasas de exito.

    La desviacion es la poblacional (`pstdev`): estas N corridas son el
    experimento entero, no una muestra de algo mayor, y ademas con una sola
    semilla da 0.0 en vez de reventar.

    Las tres tasas de arriba van aparte de las medias a proposito. Una politica
    que se atasca acumula menos de todo, asi que sus medias salen enganosamente
    buenas; `completion_rate` es lo que dice si esas medias se pueden leer.
    """
    if not runs:
        return {"runs": 0, "completion_rate": 0.0, "deadlock_free_rate": 0.0,
                "task_rate": 0.0, "metrics": {}}

    filas = [corrida.to_row() for corrida in runs]
    resumen: dict[str, dict[str, float]] = {}
    for campo in METRIC_FIELDS:
        valores = [float(fila[campo]) for fila in filas]
        resumen[campo] = {
            "mean": round(statistics.fmean(valores), 4),
            "stdev": round(statistics.pstdev(valores), 4),
            "median": round(statistics.median(valores), 4),
            "min": round(min(valores), 4),
            "max": round(max(valores), 4),
        }

    despachadas = sum(corrida.completed_tasks for corrida in runs)
    pedidas = sum(corrida.n_tasks for corrida in runs)
    return {
        "runs": len(runs),
        "completion_rate": round(
            100.0 * sum(1 for c in runs if c.all_completed) / len(runs), 2
        ),
        "deadlock_free_rate": round(
            100.0 * sum(1 for c in runs if c.deadlock_count == 0) / len(runs), 2
        ),
        "task_rate": round(100.0 * despachadas / pedidas, 2) if pedidas else 0.0,
        "metrics": resumen,
    }


def paired_wins(
    left: Sequence[RunMetrics], right: Sequence[RunMetrics], field: str = "makespan"
) -> tuple[int, int, int]:
    """Semilla a semilla, cuantas gana cada uno. Devuelve (izquierda, derecha, empates).

    Es lo que la media sola no dice: dos politicas pueden tener el mismo
    makespan medio porque una gana siempre por poco y pierde una vez por mucho.
    Las dos listas tienen que venir en el mismo orden de semillas, que es como
    las devuelve `run_comparison()`.
    """
    izquierda = derecha = empates = 0
    for una, otra in zip(left, right):
        if una.seed != otra.seed:
            raise ValueError(
                f"las corridas no estan pareadas: semilla {una.seed} contra {otra.seed}"
            )
        valor_una = una.to_row()[field]
        valor_otra = otra.to_row()[field]
        mejor_menor = field in LOWER_IS_BETTER
        if valor_una == valor_otra:
            empates += 1
        elif (valor_una < valor_otra) == mejor_menor:
            izquierda += 1
        else:
            derecha += 1
    return izquierda, derecha, empates


# --- Los ficheros de results/ -------------------------------------------------


def run_columns(runs: Sequence[RunMetrics]) -> list[str]:
    """Las columnas del CSV, en orden fijo. Es el contrato del fichero."""
    if not runs:
        return []
    fila = runs[0].to_row()
    fijas = [clave for clave in fila if not clave.startswith("wait_agv_")]
    por_agente = sorted(
        (clave for clave in fila if clave.startswith("wait_agv_")),
        key=lambda clave: int(clave.rsplit("_", 1)[1]),
    )
    return fijas + por_agente


def write_runs_csv(runs: Sequence[RunMetrics], path: str | Path) -> Path:
    """Escribe una fila por corrida, con cabecera. Se abre en Excel tal cual.

    Coma y punto decimal, sin dialecto raro: es lo que abre Excel y ademas lo
    que lee `csv.DictReader`, pandas o cualquier otra cosa sin configurar nada.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    columnas = run_columns(runs)
    with destino.open("w", encoding=config.ENCODING, newline="") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=columnas)
        escritor.writeheader()
        for corrida in runs:
            escritor.writerow(corrida.to_row())
    log.info("%s: %d corrida(s), %d columnas", destino, len(runs), len(columnas))
    return destino


def read_runs_csv(path: str | Path) -> list[dict[str, str]]:
    """Lee el CSV tal cual, en texto. Para los tests y para mirarlo a mano."""
    with Path(path).open("r", encoding=config.ENCODING, newline="") as fichero:
        return list(csv.DictReader(fichero))


def write_comparison_json(
    results: dict[str, list[RunMetrics]],
    path: str | Path = config.COMPARISON_JSON,
    *,
    header: dict[str, Any] | None = None,
) -> Path:
    """Escribe `results/comparison.json`: el resumen agregado de cada politica.

    Lleva tambien la cabecera del experimento (mapa, agentes, tareas, semillas,
    modelo). Un resumen sin decir sobre que se midio no se puede volver a leer
    dentro de un mes, y menos comparar contra otro.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    contenido: dict[str, Any] = {
        "experiment": dict(header or {}),
        "policies": {
            nombre: summarize(corridas) for nombre, corridas in results.items()
        },
    }
    destino.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False) + "\n",
        encoding=config.ENCODING,
    )
    log.info("resumen agregado en %s", destino)
    return destino


# --- El reporte ---------------------------------------------------------------

# Las filas de la tabla comparativa: (campo, como se llama en el informe).
REPORT_ROWS: tuple[tuple[str, str], ...] = (
    ("makespan", "makespan (ticks)"),
    ("avg_task_time", "tiempo por tarea"),
    ("completed_tasks", "tareas completadas"),
    ("throughput", "throughput /100t"),
    ("total_wait_time", "espera total"),
    ("wait_per_tick", "espera por tick"),
    ("total_distance", "distancia recorrida"),
    ("conflicts", "conflictos"),
    ("conflicts_per_tick", "conflictos por tick"),
    ("conflicts_vertex", "  de nodo"),
    ("conflicts_edge", "  de arista"),
    ("conflicts_following", "  de seguimiento"),
    ("conflicts_congestion", "  de congestion"),
    ("deadlocks", "deadlocks"),
    ("reroutes", "reroutes"),
)

# La metrica con la que se decide quien gana. Es la que pedia la fase: en
# cuantos ticks se despacha el mismo trabajo.
MAIN_FIELD: str = "makespan"


def _diferencia(base: float, otro: float) -> str:
    """El cambio porcentual de `otro` respecto a `base`, o `n/a` si base es cero."""
    if base == 0.0:
        return "n/a" if otro == 0.0 else "+inf"
    return f"{100.0 * (otro - base) / abs(base):+.1f}%"


def _mejora(campo: str, base: float, otro: float) -> str:
    """Si `otro` es mejor que `base` en este campo. Nada de interpretaciones."""
    if otro == base:
        return "igual"
    mejor = otro < base if campo in LOWER_IS_BETTER else otro > base
    return "si" if mejor else "NO"


def comparison_lines(
    results: dict[str, list[RunMetrics]],
    *,
    baseline: str = config.POLICY_BASELINE,
    learned: str = config.POLICY_QLEARNING,
) -> list[str]:
    """La tabla comparativa, lista para el log. Sin maquillar nada.

    Dos avisos van dentro del propio informe y no en la documentacion, porque es
    donde se leen los numeros:

    - **Los totales crudos premian al que muere antes.** Una corrida que se
      atasca acumula menos conflictos y menos espera que una que corre el triple
      y despacha el triple. Por eso van las tasas por tick y las de completitud.
    - **La media puede esconder el reparto.** Por eso va tambien en cuantas de
      las N semillas pareadas gana cada politica.

    Y si el Q-Learning no mejora, lo dice con esas palabras y con el numero
    delante. Un resultado real vale mas que uno maquillado.
    """
    izquierda = results.get(baseline, [])
    derecha = results.get(learned, [])
    if not izquierda or not derecha:
        return ["(no hay dos politicas con las que comparar)"]

    resumen_base = summarize(izquierda)
    resumen_otro = summarize(derecha)

    cabecera = (
        f"{'Metrica':<22}{'Baseline':>12}{'Q-Learning':>13}"
        f"{'Diferencia %':>15}{'¿Mejora?':>11}"
    )
    lineas = [
        f"--- comparacion: {len(izquierda)} semillas pareadas, mismo mapa, "
        f"mismos AGVs, mismas tareas ---",
        cabecera,
        "-" * len(cabecera),
    ]

    for campo, etiqueta in REPORT_ROWS:
        base = resumen_base["metrics"][campo]["mean"]
        otro = resumen_otro["metrics"][campo]["mean"]
        lineas.append(
            f"{etiqueta:<22}{base:>12.2f}{otro:>13.2f}"
            f"{_diferencia(base, otro):>15}{_mejora(campo, base, otro):>11}"
        )

    lineas.append("-" * len(cabecera))
    for clave, etiqueta in (
        ("completion_rate", "corridas completas %"),
        ("task_rate", "tareas despachadas %"),
        ("deadlock_free_rate", "sin deadlock %"),
    ):
        base, otro = resumen_base[clave], resumen_otro[clave]
        # Estas tres suben mejor, y no estan en LOWER_IS_BETTER porque no son
        # metricas de corrida sino tasas del conjunto.
        veredicto = "igual" if otro == base else ("si" if otro > base else "NO")
        lineas.append(
            f"{etiqueta:<22}{base:>12.2f}{otro:>13.2f}"
            f"{_diferencia(base, otro):>15}{veredicto:>11}"
        )

    gana_base, gana_otro, empates = paired_wins(izquierda, derecha, MAIN_FIELD)
    lineas.append("-" * len(cabecera))
    lineas.append(
        f"{'semillas ganadas':<22}{gana_base:>12}{gana_otro:>13}"
        f"{f'{empates} empates':>15}{'':>11}"
    )

    lineas.append("")
    lineas.extend(_veredicto(resumen_base, resumen_otro, gana_base, gana_otro, empates))
    lineas.append(
        "nota: los totales crudos (conflictos, espera, distancia) premian a la "
        "politica que muere antes;"
    )
    lineas.append(
        "      leelos junto a 'tareas despachadas %' y a las tasas por tick."
    )
    return lineas


def _veredicto(
    base: dict[str, Any],
    otro: dict[str, Any],
    gana_base: int,
    gana_otro: int,
    empates: int,
) -> list[str]:
    """Las lineas que dicen si el Q-Learning mejoro. Sin adornos y sin escoger dato.

    Se miran **las dos cosas a la vez**, la media y el reparto, porque pueden
    discrepar y ese desacuerdo es informacion, no ruido: una politica que gana
    en 11 semillas de 20 y aun asi tiene peor makespan medio es una politica que
    normalmente va mejor y de vez en cuando se cuelga del todo. Quedarse con el
    numero que mas conviene de los dos seria maquillar el resultado.
    """
    makespan_base = base["metrics"][MAIN_FIELD]["mean"]
    makespan_otro = otro["metrics"][MAIN_FIELD]["mean"]
    total = gana_base + gana_otro + empates
    media_mejor = makespan_otro < makespan_base
    reparto_mejor = gana_otro > gana_base
    cambio = _diferencia(makespan_base, makespan_otro)

    if media_mejor and reparto_mejor:
        lineas = [
            f"VEREDICTO: el Q-Learning MEJORA. Makespan medio {makespan_otro:.1f} "
            f"contra {makespan_base:.1f} ticks ({cambio}), y gana en "
            f"{gana_otro} de {total} semillas."
        ]
    elif not media_mejor and not reparto_mejor:
        lineas = [
            f"VEREDICTO: el Q-Learning NO mejora. Makespan medio {makespan_otro:.1f} "
            f"contra {makespan_base:.1f} ticks del baseline ({cambio}), y solo gana "
            f"en {gana_otro} de {total} semillas."
        ]
    elif reparto_mejor:
        lineas = [
            f"VEREDICTO: mixto, y la media sola engana. El Q-Learning gana en "
            f"{gana_otro} de {total} semillas, mas que el baseline ({gana_base}), "
            f"pero su makespan MEDIO es peor: {makespan_otro:.1f} contra "
            f"{makespan_base:.1f} ticks ({cambio}).",
            "           Eso es que suele ir mejor y de vez en cuando se cuelga del "
            "todo: son las corridas malas las que se llevan la media, no un peor "
            "rendimiento tipico.",
        ]
    else:
        lineas = [
            f"VEREDICTO: mixto. El Q-Learning tiene mejor makespan medio "
            f"({makespan_otro:.1f} contra {makespan_base:.1f} ticks, {cambio}) pero "
            f"gana en menos semillas que el baseline ({gana_otro} contra {gana_base} "
            f"de {total}).",
            "           La media se la lleva unas pocas corridas muy buenas, asi "
            "que no es una mejora que se pueda dar por sistematica.",
        ]

    if otro["completion_rate"] < base["completion_rate"]:
        lineas.append(
            f"           Y termina menos corridas: {otro['completion_rate']:.0f}% "
            f"contra {base['completion_rate']:.0f}% del baseline. La que se queda a "
            f"medias cuenta como makespan = tope de ticks, que es de donde sale "
            f"buena parte de esa media."
        )
    if otro["task_rate"] < base["task_rate"]:
        lineas.append(
            f"           Despacha menos trabajo: {otro['task_rate']:.1f}% de las "
            f"tareas contra {base['task_rate']:.1f}%, asi que sus totales de "
            f"conflictos y espera salen bajos por no llegar, no por ir mejor."
        )
    return lineas


# --- Las graficas (opcionales) ------------------------------------------------

# Que se dibuja, en el orden de los paneles.
PLOT_PANELS: tuple[tuple[str, str], ...] = (
    ("makespan", "makespan (ticks)"),
    ("conflicts", "conflictos"),
    ("total_wait_time", "espera total (ticks)"),
)


def save_comparison_plot(
    results: dict[str, list[RunMetrics]], path: str | Path = config.COMPARISON_PLOT
) -> Path | None:
    """Barras comparativas con barras de error. Devuelve None si no hay matplotlib.

    matplotlib es opcional a proposito: el proyecto no tiene dependencias y los
    CSV ya estan escritos antes de llegar aqui. Sin la libreria se avisa por el
    log y se sigue; el benchmark no se pierde por no poder dibujarlo.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # sin ventana: esto corre en una terminal
        from matplotlib import pyplot as plt
    except ImportError:
        log.warning(
            "matplotlib no esta instalado, me salto %s "
            "(los numeros estan en los CSV igualmente)",
            path,
        )
        return None

    politicas = [nombre for nombre, corridas in results.items() if corridas]
    if not politicas:
        log.warning("no hay corridas que dibujar")
        return None

    resumenes = {nombre: summarize(results[nombre]) for nombre in politicas}
    colores = ["tab:blue", "tab:orange", "tab:green", "tab:red"][: len(politicas)]

    figura, ejes = plt.subplots(1, len(PLOT_PANELS), figsize=(4.2 * len(PLOT_PANELS), 4.6))
    ejes = ejes if len(PLOT_PANELS) > 1 else [ejes]
    for eje, (campo, titulo) in zip(ejes, PLOT_PANELS):
        medias = [resumenes[n]["metrics"][campo]["mean"] for n in politicas]
        errores = [resumenes[n]["metrics"][campo]["stdev"] for n in politicas]
        barras = eje.bar(
            politicas, medias, yerr=errores, capsize=6, color=colores, alpha=0.85
        )
        eje.bar_label(barras, fmt="%.0f", padding=3, fontsize="small")
        eje.set_title(titulo)
        eje.grid(axis="y", alpha=0.25)
        eje.set_axisbelow(True)

    corridas = len(results[politicas[0]])
    figura.suptitle(
        f"AGVs: baseline contra Q-Learning ({corridas} semillas pareadas, "
        f"barras de error = desviacion tipica)"
    )
    figura.tight_layout()

    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=120)
    plt.close(figura)
    log.info("graficas comparativas en %s", destino)
    return destino
