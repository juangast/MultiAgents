"""Q-Learning: el estado local, la recompensa, la Q-table y el entrenamiento.

No sustituye a A*: A* traza la ruta y aqui solo se aprende que hacer ahora ante
un riesgo de conflicto (avanzar, esperar o recalcular).
"""

import csv
import json
import logging
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import config
import conflicts
import simulation
from agent import Agent, State as AgentState
from graph import WarehouseGraph, path_cost
from config import get_logger, setup_logging

log = get_logger("qlearning")


STATE_FIELDS: tuple[str, ...] = (
    "next_node_occupied",
    "edge_conflict",
    "queue_ahead",
    "distance_bucket",
    "has_priority",
    "carrying",
)

STATE_SIZES: tuple[int, ...] = (2, 2, 3, 3, 2, 2)

State = tuple[int, ...]

QUEUE_CAP: int = 2


class SimulationView(Protocol):
    """Lo que `get_local_state()` necesita mirar de la simulacion."""

    graph: WarehouseGraph
    agents: Sequence[Agent]
    occupancy: Mapping[str, int]


def get_local_state(agent: Agent, simulation: SimulationView) -> State:
    """El estado discreto y local de este AGV: siempre seis enteros."""
    siguiente = agent.next_node()
    return (
        _ocupado(agent, siguiente, simulation.occupancy),
        _viene_de_frente(agent, siguiente, simulation.agents),
        _cola_delante(agent, simulation.agents),
        distance_bucket(agent),
        _tengo_prioridad(agent, _rivales(agent, siguiente, simulation)),
        int(agent.carrying is not None),
    )


def state_from_local(agent: Agent, local_state: conflicts.LocalState) -> State:
    """El mismo estado, pero sacado de lo que el motor le pasa a la politica."""
    siguiente = agent.next_node()
    rivales = {
        otro
        for choque in local_state.conflicts
        for otro in choque.agents
        if otro != agent.id
    }
    rivales.update(local_state.blocked_by)
    de_frente = any(
        choque.type == conflicts.ConflictType.EDGE for choque in local_state.conflicts
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
        int(agent.carrying is not None),
    )


DISTANCE_NEAR_NODES: int = 3
DISTANCE_MID_NODES: int = 8
REPORT_EVERY: int = 100


def distance_bucket(agent: Agent) -> int:
    """0 cerca, 1 medio, 2 lejos, contando **nodos** que faltan."""
    faltan = max(len(agent.path) - 1 - agent.path_index, 0)
    if faltan <= DISTANCE_NEAR_NODES:
        return 0
    if faltan <= DISTANCE_MID_NODES:
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
    """1 si alguien esta en mi nodo siguiente y quiere entrar en el mio."""
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
        and otro.state == AgentState.WAITING
        and otro.current_node in proximos
    )
    return min(cuantos, QUEUE_CAP)


def _proximos_nodos(agent: Agent) -> tuple[str, ...]:
    """Los dos siguientes nodos de mi ruta, los que haya."""
    return tuple(agent.path[agent.path_index + 1 : agent.path_index + 3])


def _rivales(
    agent: Agent, siguiente: str | None, simulation: SimulationView
) -> set[int]:
    """Ids de los que se disputan el paso conmigo ahora mismo."""
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
        if (
            otro.progress <= 0.0
            and otro.state in (AgentState.MOVING, AgentState.WAITING)
            and otro.next_node() == siguiente
        ):
            rivales.add(otro.id)
    return rivales


def _tengo_prioridad(agent: Agent, rivales: Iterable[int]) -> int:
    """1 si soy el id menor de los que estamos en conflicto."""
    ids = tuple(rivales)
    return int(not ids or agent.id < min(ids))


class Action(Enum):
    """Lo que un AGV puede decidir cuando le toca."""

    ADVANCE = "advance"
    WAIT = "wait"
    REROUTE = "reroute"


ACTIONS: tuple[Action, ...] = tuple(Action)


def enabled_actions(enable_reroute: bool | None = None) -> tuple[Action, ...]:
    """Las acciones que la politica puede elegir, segun `config.ENABLE_REROUTE`."""
    activo = config.ENABLE_REROUTE if enable_reroute is None else enable_reroute
    return ACTIONS if activo else (Action.ADVANCE, Action.WAIT)


def is_useless_reroute(
    graph: WarehouseGraph,
    old_path: Sequence[str],
    new_path: Sequence[str],
    *,
    avoided_conflict: bool = False,
) -> bool:
    """Si el recalculo no sirvio para nada, que es lo que se penaliza."""
    nuevo, viejo = _costo(graph, new_path), _costo(graph, old_path)
    if nuevo > viejo:
        return True
    if avoided_conflict:
        return False
    return nuevo >= viejo


def _costo(graph: WarehouseGraph, path: Sequence[str]) -> float:
    """Costo de una ruta, o infinito si no se puede recorrer."""
    if len(path) < 2:
        return 0.0
    try:
        return path_cost(graph, path)
    except KeyError:
        return math.inf


class Event(Enum):
    """Lo que le puede pasar a un AGV y tiene precio."""

    TASK_COMPLETE = "task_complete"
    PICKED = "picked"
    PROGRESS = "progress"
    WAIT = "wait"
    CONFLICT = "conflict"
    DEADLOCK = "deadlock"
    USELESS_REROUTE = "useless_reroute"


_CONFIG_KEY: dict[Event, str] = {
    Event.TASK_COMPLETE: "REWARD_TASK_COMPLETE",
    Event.PICKED: "REWARD_PICKED",
    Event.PROGRESS: "REWARD_PROGRESS",
    Event.WAIT: "REWARD_WAIT",
    Event.CONFLICT: "REWARD_CONFLICT",
    Event.DEADLOCK: "REWARD_DEADLOCK",
    Event.USELESS_REROUTE: "REWARD_USELESS_REROUTE",
}


def reward(event: Event | str) -> float:
    """Lo que vale un evento. **Los numeros viven en `config.py`**, no aqui."""
    try:
        clave = Event(event)
    except ValueError:
        conocidos = ", ".join(uno.value for uno in Event)
        raise ValueError(
            f"evento desconocido: {event!r}; los que hay son {conocidos}"
        ) from None
    return float(getattr(config, _CONFIG_KEY[clave]))


FORMAT: str = "agv-qtable/1"

SEPARATOR: str = "|"


def encode_state(state: State) -> str:
    """La tupla de estado como clave de JSON: los campos en orden, con '|'."""
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

    Cada fila lleva SIEMPRE las tres acciones aunque el action set las deje
        fuera, para que una tabla entrenada sin REROUTE se pueda leer igual.
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

    def _fila(self, state: State) -> dict[Action, float]:
        """La fila del estado **sin crearla**: si no esta, una de ceros de usar y tirar."""
        return self._q.get(tuple(state)) or _fila_en_cero()

    def value(self, state: State, action: Action) -> float:
        """Q(s, a). No crea la fila: leer no escribe."""
        return self._fila(state)[action]

    def set_value(self, state: State, action: Action, value: float) -> None:
        """Escribe Q(s, a). Quien calcula el valor nuevo es la fase 7."""
        self[state][action] = float(value)

    def best_action(
        self, state: State, *, among: Sequence[Action] | None = None
    ) -> Action:
        """La accion de mas valor, con el empate resuelto por el orden de ACTIONS."""
        candidatas = tuple(among) if among else ACTIONS
        if not candidatas:
            raise ValueError("no hay ninguna accion entre la que elegir")
        fila = self._fila(state)
        mejor = candidatas[0]
        for accion in candidatas[1:]:
            if fila[accion] > fila[mejor]:
                mejor = accion
        return mejor

    def best_value(self, state: State, *, among: Sequence[Action] | None = None) -> float:
        """max_a Q(s, a). Es el termino que la fase 7 mete en la actualizacion."""
        return self.value(state, self.best_action(state, among=among))

    def update(
        self,
        state: State,
        action: Action,
        reward_value: float,
        next_state: State,
        *,
        alpha: float,
        gamma: float,
        terminal: bool = False,
        among: Sequence[Action] | None = None,
    ) -> float:
        """La actualizacion de Bellman. Devuelve el Q(s, a) nuevo."""
        actual = self.value(state, action)
        futuro = 0.0 if terminal else self.best_value(next_state, among=among)
        nuevo = actual + alpha * (reward_value + gamma * futuro - actual)
        self.set_value(state, action, nuevo)
        return nuevo

    def as_dict(self) -> dict[str, dict[str, float]]:
        """La tabla en claves de texto, tal y como se guarda. Ordenada."""
        return {
            encode_state(estado): {
                accion.value: float(self._q[estado][accion]) for accion in ACTIONS
            }
            for estado in sorted(self._q)
        }

    def save(
        self, path: str | Path, *, metadata: Mapping[str, Any] | None = None
    ) -> Path:
        """Escribe la tabla en JSON, creando el directorio si hace falta."""
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        contenido: dict[str, Any] = {
            "format": FORMAT,
            "state_fields": list(STATE_FIELDS),
            "actions": [accion.value for accion in ACTIONS],
            "metadata": dict(metadata) if metadata is not None else {},
            "q": self.as_dict(),
        }
        destino.write_text(
            json.dumps(contenido, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info("Q-table guardada en %s (%d estados)", destino, len(self))
        return destino


def load_qtable(path: str | Path) -> QTable:
    """Lee una tabla de disco, comprobando que el formato es el de ahora.

    Lanza ValueError si el fichero es de otra version o si los campos del estado
    no son los de `STATE_FIELDS`, en ese orden: cargar a ciegas una tabla con
    otro orden seria entrenar sobre estados que no son los que se creen, y eso
    no da error nunca, da resultados malos.
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

    tabla = QTable()
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


def trained_enable_reroute(path: str | Path) -> bool | None:
    """Si este modelo se entreno con REROUTE. None si el fichero no lo dice.

    Servir un modelo fuera de su action set es un fallo silencioso: la columna
    que nunca se entreno esta a ceros, y como lo aprendido es casi todo
    negativo, el cero gana y la politica elige justo lo que no probo.
    """
    hiper = load_metadata(path).get("hyperparameters") or {}
    nombres = hiper.get("actions")
    if not nombres:
        return None
    return Action.REROUTE.value in {str(nombre) for nombre in nombres}


def load_action_visits(path: str | Path) -> dict[State, dict[Action, int]]:
    """Cuantas veces se actualizo cada celda (estado, accion) al entrenar.

    Vacio si el modelo es anterior a que esto se guardara: entonces no hay con
    que filtrar y la politica se comporta como siempre.
    """
    datos = load_metadata(path).get("action_visits")
    if not isinstance(datos, dict):
        return {}

    visitas: dict[State, dict[Action, int]] = {}
    for clave, fila in datos.items():
        if not isinstance(fila, dict):
            continue
        try:
            estado = decode_state(str(clave))
        except ValueError:
            continue
        cuenta: dict[Action, int] = {}
        for nombre, cuantas in fila.items():
            try:
                cuenta[Action(str(nombre))] = int(cuantas)
            except (ValueError, TypeError):
                continue
        visitas[estado] = cuenta
    return visitas


def load_metadata(path: str | Path) -> dict[str, Any]:
    """La `metadata` con la que se guardo una Q-table, o `{}` si no lleva.

    Se lee aparte de `load_qtable()`: una tabla vieja sin metadata tiene que
    poder seguir cargando.
    """
    crudo = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(crudo, dict):
        return {}
    datos = crudo.get("metadata")
    return dict(datos) if isinstance(datos, dict) else {}


class Decision:
    """Lo que un AGV decidio en un tick, con el paso en que lo decidio.

    Lleva el `step` porque se guarda **la ultima** decision de cada AGV, no
    todas: sin la marca de paso, el entrenamiento le atribuiria a la decision de
    ahora lo que paso hace cuatro ticks.
    """

    def __init__(self, state: State, action: Action, step: int) -> None:
        self.state = state
        self.action = action
        self.step = step


class QLearningPolicy:
    """La politica de Q-Learning, con la misma interfaz que la baseline.

    Para que vea el estado completo hay que atarla con `bind()`. Sin atar sigue
    funcionando, pero saca el estado del `LocalState` que le pasa el motor, y
    ahi `queue_ahead` es una aproximacion.
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
        visits: Mapping[State, Mapping[Action, int]] | None = None,
        min_visits: int = 0,
    ) -> None:
        self.q: QTable = q_table if q_table is not None else QTable()
        self.epsilon: float = float(epsilon)
        self.actions: tuple[Action, ...] = enabled_actions(enable_reroute)
        self._rng = random.Random(seed)
        self._simulation: SimulationView | None = simulation
        self._last: dict[int, Decision] = {}
        self._avisado: bool = False
        self._visits: Mapping[State, Mapping[Action, int]] = visits or {}
        self.min_visits: int = int(min_visits)

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

    def decide(self, agent: Agent, local_state: conflicts.LocalState) -> str:
        """La accion elegida: `"advance"`, `"wait"` o `"reroute"`."""
        estado = self.observe(agent, local_state)
        accion = self.choose(estado)
        self._last[agent.id] = Decision(estado, accion, local_state.step)
        return accion.value

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
        """Epsilon-greedy sobre la Q-table. Con `epsilon = 0` es greedy puro."""
        if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
            return self._rng.choice(self.actions)
        return self.q.best_action(state, among=self._respaldadas(state))

    def _respaldadas(self, state: State) -> tuple[Action, ...]:
        """Las acciones que esta tabla probo lo bastante en este estado."""
        if self.min_visits <= 0 or not self._visits:
            return self.actions
        fila = self._visits.get(tuple(state), {})
        respaldadas = tuple(
            accion for accion in self.actions if fila.get(accion, 0) >= self.min_visits
        )
        return respaldadas or self.actions

    def last_decision(self, agent_id: int) -> tuple[State, Action] | None:
        """El (estado, accion) con el que decidio este AGV la ultima vez."""
        decision = self._last.get(agent_id)
        return None if decision is None else (decision.state, decision.action)

    def decision(self, agent_id: int) -> Decision | None:
        """La ultima decision entera de este AGV: estado, accion, paso y reroute."""
        return self._last.get(agent_id)


class TrainingConfig:
    """Los numeros de una corrida de entrenamiento. Por defecto, los de `config.py`."""

    def __init__(
        self,
        map_name: str = config.DEFAULT_MAP,
        agents: int = config.TRAIN_AGENTS,
        episodes: int = config.EPISODES,
        seed: int = config.RANDOM_SEED,
        alpha: float = config.ALPHA,
        gamma: float = config.GAMMA,
        epsilon_start: float = config.EPSILON_START,
        epsilon_end: float = config.EPSILON_END,
        epsilon_decay: float = config.EPSILON_DECAY,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
        enable_reroute: bool | None = None,
        report_every: int = REPORT_EVERY,
        scenario: str = '',
        deliveries: bool = False,
    ) -> None:
        self.map_name = map_name
        self.agents = agents
        self.episodes = episodes
        self.seed = seed
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_steps = max_steps
        self.enable_reroute = enable_reroute
        self.report_every = report_every
        self.scenario = scenario
        self.deliveries = deliveries

    def as_dict(self) -> dict[str, Any]:
        """Los hiperparametros en JSON, para la metadata del modelo."""
        return {
            "map": self.map_name,
            "scenario": self.scenario,
            "deliveries": self.deliveries,
            "agents": self.agents,
            "episodes": self.episodes,
            "seed": self.seed,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "max_steps_per_episode": self.max_steps,
            "actions": [
                accion.value for accion in enabled_actions(self.enable_reroute)
            ],
        }


class Transition:
    """Una (s, a, r, s') lista para meter en Bellman."""

    def __init__(
        self,
        agent_id: int,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        terminal: bool,
    ) -> None:
        self.agent_id = agent_id
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.terminal = terminal


class EpisodeStats:
    """Una fila de `results/training_log.csv`: como fue un episodio."""

    def __init__(
        self,
        episode: int,
        epsilon: float,
        total_reward: float,
        avg_reward: float,
        conflicts: int,
        deadlocks: int,
        completed_tasks: int,
        makespan: int,
        total_wait: int,
        states_visited: int,
    ) -> None:
        self.episode = episode
        self.epsilon = epsilon
        self.total_reward = total_reward
        self.avg_reward = avg_reward
        self.conflicts = conflicts
        self.deadlocks = deadlocks
        self.completed_tasks = completed_tasks
        self.makespan = makespan
        self.total_wait = total_wait
        self.states_visited = states_visited

    def con_recompensa(self, total: float, avg: float) -> "EpisodeStats":
        """Una copia con la recompensa del episodio ya contada."""
        return EpisodeStats(
            episode=self.episode,
            epsilon=self.epsilon,
            total_reward=total,
            avg_reward=avg,
            conflicts=self.conflicts,
            deadlocks=self.deadlocks,
            completed_tasks=self.completed_tasks,
            makespan=self.makespan,
            total_wait=self.total_wait,
            states_visited=self.states_visited,
        )

    def as_row(self) -> dict[str, Any]:
        """La fila del CSV, con los flotantes ya redondeados."""
        return {
            "episode": self.episode,
            "epsilon": round(self.epsilon, 6),
            "total_reward": round(self.total_reward, 3),
            "avg_reward": round(self.avg_reward, 4),
            "conflicts": self.conflicts,
            "deadlocks": self.deadlocks,
            "completed_tasks": self.completed_tasks,
            "makespan": self.makespan,
            "total_wait": self.total_wait,
            "states_visited": self.states_visited,
        }


LOG_COLUMNS: tuple[str, ...] = (
    "episode",
    "epsilon",
    "total_reward",
    "avg_reward",
    "conflicts",
    "deadlocks",
    "completed_tasks",
    "makespan",
    "total_wait",
    "states_visited",
)


class TrainablePolicy(Protocol):
    """Lo que `TrainingEnv` necesita de una politica para poder puntuarla."""

    name: str

    def decide(self, agent: Agent, local_state: conflicts.LocalState) -> str: ...

    def bind(self, simulation: SimulationView) -> None: ...

    def reset(self) -> None: ...

    def decision(self, agent_id: int) -> Decision | None: ...


class BaselineAdapter:
    """La politica de la fase 5 con el cuaderno de la fase 7."""

    name: str = "baseline"

    def __init__(self, *, simulation: SimulationView | None = None) -> None:
        self._inner = conflicts.BaselinePolicy()
        self._simulation: SimulationView | None = simulation
        self._last: dict[int, Decision] = {}

    def __repr__(self) -> str:
        return f"BaselineAdapter(bound={self._simulation is not None})"

    def bind(self, simulation: SimulationView) -> None:
        self._simulation = simulation

    def reset(self) -> None:
        self._last.clear()

    def decide(self, agent: Agent, local_state: conflicts.LocalState) -> str:
        estado = (
            get_local_state(agent, self._simulation)
            if self._simulation is not None
            else state_from_local(agent, local_state)
        )
        motor = self._inner.decide(agent, local_state)
        accion = Action.ADVANCE if motor == conflicts.Intent.ADVANCE else Action.WAIT
        self._last[agent.id] = Decision(estado, accion, local_state.step)
        return motor

    def decision(self, agent_id: int) -> Decision | None:
        return self._last.get(agent_id)


def random_routes(
    graph: WarehouseGraph, n_agents: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Un escenario nuevo: origenes distintos, destinos distintos, y nadie ya en casa."""
    nodos = graph.nodes()
    if n_agents > len(nodos):
        raise ValueError(
            f"no caben {n_agents} agentes en un mapa de {len(nodos)} nodo(s)"
        )

    origenes = rng.sample(nodos, n_agents)
    destinos = rng.sample(nodos, n_agents)
    for _ in range(20):
        if all(origen != destino for origen, destino in zip(origenes, destinos)):
            break
        destinos = rng.sample(nodos, n_agents)
    return list(zip(origenes, destinos))


class _Abierta:
    """Una decision a la que todavia se le esta sumando la recompensa."""

    def __init__(self, decision: Decision, reward: float = 0.0) -> None:
        self.decision = decision
        self.reward = reward

    def close(self, agent_id: int, next_state: State, *, terminal: bool) -> Transition:
        return Transition(
            agent_id=agent_id,
            state=self.decision.state,
            action=self.decision.action,
            reward=self.reward,
            next_state=next_state,
            terminal=terminal,
        )


class _Foto:
    """Como estaba un AGV al empezar el tick. Con esto se ve que le paso."""

    def __init__(
        self,
        path_index: int,
        state: str,
        wait_time: int,
        carrying: str | None = None,
    ) -> None:
        self.path_index = path_index
        self.state = state
        self.wait_time = wait_time
        self.carrying = carrying


class TrainingEnv:
    """Un episodio: monta la simulacion, la tickea y reparte la recompensa."""

    def __init__(
        self,
        graph: WarehouseGraph,
        n_agents: int,
        policy: TrainablePolicy,
        *,
        seed: int = config.RANDOM_SEED,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
        routes_factory: Callable[[random.Random], Sequence[tuple[str, str]]] | None = None,
        deliveries: bool = False,
    ) -> None:
        self.graph = graph
        self.n_agents = int(n_agents)
        self.policy = policy
        self.max_steps = int(max_steps)
        self.deliveries = bool(deliveries)
        self._rng = random.Random(seed)
        self._routes_factory = routes_factory

        self.sim: simulation.Simulation | None = None
        self._abiertas: dict[int, _Abierta] = {}
        self._llegadas: dict[int, int] = {}
        self._deadlock: bool = False

    def __repr__(self) -> str:
        return (
            f"TrainingEnv(map={self.graph.name!r}, agents={self.n_agents}, "
            f"policy={self.policy.name!r}, max_steps={self.max_steps})"
        )

    def done(self) -> bool:
        """True si llegaron todos, si hubo deadlock o si se acabaron los ticks."""
        return self.sim.done() or self.sim.step >= self.max_steps

    def reset(self) -> None:
        """Arranca un episodio nuevo, con un reparto de tareas nuevo."""
        rutas = (
            list(self._routes_factory(self._rng))
            if self._routes_factory is not None
            else random_routes(self.graph, self.n_agents, self._rng)
        )
        self.sim = simulation.Simulation(
            self.graph,
            self.n_agents,
            routes=rutas,
            policy=self.policy,
            deliveries=self.deliveries,
        )
        self.policy.reset()
        self.policy.bind(self.sim)
        self._abiertas.clear()
        self._llegadas.clear()
        self._deadlock = False

    def step(self) -> list[Transition]:
        """Un tick de la simulacion. Devuelve las transiciones que se cerraron."""
        paso = self.sim.step + 1
        antes = {
            agente.id: _Foto(
                agente.path_index, agente.state, agente.wait_time, agente.carrying
            )
            for agente in self.sim.agents
        }
        ya_habia = len(self.sim.conflicts)

        self.sim.tick()

        murio_ahora = (
            self.sim.finished_reason == simulation.FINISHED_DEADLOCK
            and not self._deadlock
        )
        self._deadlock = self._deadlock or murio_ahora
        en_conflicto = {
            agent_id
            for choque in islice(self.sim.conflicts, ya_habia, None)
            for agent_id in choque.agents
        }

        cerradas: list[Transition] = []

        for agente in self.sim.agents:
            decision = self.policy.decision(agente.id)
            if decision is None or decision.step != paso:
                continue
            abierta = self._abiertas.get(agente.id)
            if abierta is not None:
                cerradas.append(
                    abierta.close(agente.id, decision.state, terminal=False)
                )
            self._abiertas[agente.id] = _Abierta(decision)

        for agente in self.sim.agents:
            if agente.state == AgentState.DONE and antes[agente.id].state != AgentState.DONE:
                self._llegadas[agente.id] = paso

            abierta = self._abiertas.get(agente.id)
            if abierta is None:
                continue

            eventos = self._eventos(
                agente,
                antes[agente.id],
                abierta.decision,
                paso=paso,
                en_conflicto=agente.id in en_conflicto,
                deadlock=murio_ahora,
            )
            abierta.reward += sum(reward(evento) for evento in eventos)

            if agente.state == AgentState.DONE:
                del self._abiertas[agente.id]
                cerradas.append(
                    abierta.close(
                        agente.id, get_local_state(agente, self.sim), terminal=True
                    )
                )

        return cerradas

    def close_pending(self) -> list[Transition]:
        """Cierra las transiciones que quedaron abiertas al acabar el episodio."""
        cerradas: list[Transition] = []
        for agente in self.sim.agents:
            abierta = self._abiertas.pop(agente.id, None)
            if abierta is None:
                continue
            cerradas.append(
                abierta.close(
                    agente.id,
                    get_local_state(agente, self.sim),
                    terminal=self._deadlock or agente.state == AgentState.DONE,
                )
            )
        return cerradas

    def stats(self, episode: int, epsilon: float, *, states_visited: int) -> EpisodeStats:
        """Los numeros del episodio que acaba de terminar."""
        completadas = sum(
            1 for agente in self.sim.agents if agente.state == AgentState.DONE
        )
        llegaron_todos = len(self._llegadas) == len(self.sim.agents)
        return EpisodeStats(
            episode=episode,
            epsilon=epsilon,
            total_reward=0.0,
            avg_reward=0.0,
            conflicts=self.sim.conflicts.total(),
            deadlocks=int(self._deadlock),
            completed_tasks=completadas,
            makespan=max(self._llegadas.values()) if llegaron_todos else self.sim.step,
            total_wait=sum(agente.wait_time for agente in self.sim.agents),
            states_visited=states_visited,
        )

    def _eventos(
        self,
        agent: Agent,
        antes: _Foto,
        decision: Decision,
        *,
        paso: int,
        en_conflicto: bool,
        deadlock: bool,
    ) -> list[Event]:
        """Que le paso a este AGV en este tick, en eventos con precio.

        El CONFLICT no se le cobra al que gano el desempate: el estado local no
            distingue "camino libre" de "camino libre y gano la disputa", asi que
            cobrarselo envenenaria la celda que sostiene toda la politica. Al perdedor
            si, que es el que puede aprender algo.
        """
        eventos: list[Event] = []
        registro = self.sim.action_record(agent.id)
        fresco = registro is not None and registro.step == paso

        if agent.path_index > antes.path_index:
            eventos.append(Event.PROGRESS)
        if agent.state == AgentState.DONE and antes.state != AgentState.DONE:
            eventos.append(Event.TASK_COMPLETE)
        if agent.carrying is not None and antes.carrying is None:
            eventos.append(Event.PICKED)

        if agent.wait_time > antes.wait_time:
            eventos.append(
                Event.CONFLICT if fresco and registro.blocked else Event.WAIT
            )

        if fresco and registro.forced and Event.CONFLICT not in eventos:
            eventos.append(Event.CONFLICT)

        if fresco and registro.reroute is not None:
            vieja, nueva = registro.reroute
            esquiva = (
                en_conflicto
                and len(vieja) > 1
                and len(nueva) > 1
                and nueva[1] != vieja[1]
            )
            if is_useless_reroute(
                self.graph, vieja, nueva, avoided_conflict=esquiva
            ):
                eventos.append(Event.USELESS_REROUTE)

        if deadlock and agent.state != AgentState.DONE:
            eventos.append(Event.DEADLOCK)

        return eventos


class Trainer:
    """El bucle de Q-Learning: episodios, epsilon-greedy y Bellman."""

    def __init__(
        self,
        graph: WarehouseGraph,
        cfg: TrainingConfig,
        *,
        q_table: QTable | None = None,
        learn: bool = True,
        routes_factory: Callable[[random.Random], Sequence[tuple[str, str]]] | None = None,
    ) -> None:
        self.graph = graph
        self.cfg = cfg
        self.learn = learn
        self.q: QTable = q_table if q_table is not None else QTable()
        self.actions = enabled_actions(cfg.enable_reroute)
        self.epsilon: float = cfg.epsilon_start if learn else 0.0
        self.history: list[EpisodeStats] = []
        self.visits: Counter[State] = Counter()
        self.action_visits: Counter[tuple[State, Action]] = Counter()

        maestro = random.Random(cfg.seed)
        semilla_escenarios = maestro.randrange(2**31)
        semilla_politica = maestro.randrange(2**31)

        self.policy = QLearningPolicy(
            self.q,
            epsilon=self.epsilon,
            seed=semilla_politica,
            enable_reroute=cfg.enable_reroute,
        )
        self.env = TrainingEnv(
            graph,
            cfg.agents,
            self.policy,
            seed=semilla_escenarios,
            max_steps=cfg.max_steps,
            routes_factory=routes_factory,
            deliveries=cfg.deliveries,
        )

    def __repr__(self) -> str:
        return (
            f"Trainer(mode={'train' if self.learn else 'evaluate'}, "
            f"map={self.graph.name!r}, agents={self.cfg.agents}, "
            f"epsilon={self.epsilon:g}, states={len(self.q)})"
        )

    def decay_epsilon(self) -> float:
        """Decaimiento exponencial con suelo: `eps <- max(END, eps * DECAY)`."""
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
        return self.epsilon

    def run_episode(self, episode: int) -> EpisodeStats:
        """Un episodio entero: reset, ticks hasta el final y las Q que toquen."""
        self.policy.epsilon = self.epsilon
        self.env.reset()

        total = 0.0
        decisiones = 0
        while not self.env.done():
            for transicion in self.env.step():
                total += transicion.reward
                decisiones += 1
                self._aprende(transicion)

        for transicion in self.env.close_pending():
            total += transicion.reward
            decisiones += 1
            self._aprende(transicion)

        numeros = self.env.stats(episode, self.epsilon, states_visited=len(self.q))
        return numeros.con_recompensa(
            total, total / decisiones if decisiones else 0.0
        )

    def run(self, episodes: int | None = None) -> list[EpisodeStats]:
        """El bucle: `EPISODES` episodios y un `decay_epsilon()` detras de cada uno."""
        cuantos = self.cfg.episodes if episodes is None else int(episodes)
        modo = "TRAIN" if self.learn else "EVALUATE"
        log.info(
            "--- %s: mapa %s, %d AGVs, %d episodios, semilla %d ---",
            modo,
            self.graph.name or "(sin nombre)",
            self.cfg.agents,
            cuantos,
            self.cfg.seed,
        )
        if self.learn:
            log.info(
                "alpha %g | gamma %g | epsilon %g -> %g (x%g por episodio) | "
                "tope %d ticks | acciones %s",
                self.cfg.alpha,
                self.cfg.gamma,
                self.cfg.epsilon_start,
                self.cfg.epsilon_end,
                self.cfg.epsilon_decay,
                self.cfg.max_steps,
                ", ".join(accion.value for accion in self.actions),
            )

        with _quiet("simulation", "agent"):
            for episode in range(1, cuantos + 1):
                self.history.append(self.run_episode(episode))
                if self.learn:
                    self.decay_epsilon()
        return self.history

    def save(self, path: str | Path = config.Q_TABLE_FILE) -> Path:
        """Guarda la Q-table con la metadata de esta corrida."""
        return self.q.save(path, metadata=self.metadata())

    def metadata(self) -> dict[str, Any]:
        """Con que se entreno esto: mapa, agentes, hiperparametros, semilla y fecha."""
        ultimo = self.history[-1] if self.history else None
        return {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "map": self.graph.name or "(sin nombre)",
            "map_nodes": len(self.graph.nodes()),
            "agents": self.cfg.agents,
            "seed": self.cfg.seed,
            "episodes_run": len(self.history),
            "hyperparameters": self.cfg.as_dict(),
            "final_epsilon": round(self.epsilon, 6),
            "states_visited": len(self.q),
            "state_space_size": state_space_size(),
            "shared_q_table": True,
            "visits": {
                encode_state(estado): cuantas
                for estado, cuantas in sorted(self.visits.items())
            },
            "action_visits": {
                encode_state(estado): {
                    accion.value: self.action_visits[(estado, accion)]
                    for accion in ACTIONS
                }
                for estado in sorted(self.visits)
            },
            "last_episode": ultimo.as_row() if ultimo is not None else None,
        }

    def _aprende(self, transicion: Transition) -> None:
        """Bellman, pero solo en TRAIN. En EVALUATE esto no hace nada."""
        if not self.learn:
            return
        self.visits[transicion.state] += 1
        self.action_visits[(transicion.state, transicion.action)] += 1
        self.q.update(
            transicion.state,
            transicion.action,
            transicion.reward,
            transicion.next_state,
            alpha=self.cfg.alpha,
            gamma=self.cfg.gamma,
            terminal=transicion.terminal,
            among=self.actions,
        )


def train(
    graph: WarehouseGraph,
    cfg: TrainingConfig | None = None,
    *,
    model_path: str | Path = config.Q_TABLE_FILE,
    log_path: str | Path | None = config.TRAINING_LOG_FILE,
) -> Trainer:
    """Modo TRAIN: entrena, guarda el modelo y el CSV. Sin servidor.

    Devuelve el `Trainer` con `history` lleno, por si quien llama quiere seguir
    mirando los numeros.
    """
    entrenador = Trainer(graph, cfg if cfg is not None else TrainingConfig())
    entrenador.run()

    entrenador.save(model_path)
    if log_path is not None:
        write_training_log(entrenador.history, log_path)
    for linea in summary_lines(entrenador.history, entrenador.cfg.report_every):
        log.info("%s", linea)
    return entrenador


def evaluate(
    graph: WarehouseGraph,
    cfg: TrainingConfig | None = None,
    *,
    model_path: str | Path = config.Q_TABLE_FILE,
    episodes: int = 100,
) -> tuple[Trainer, list[EpisodeStats]]:
    """Modo EVALUATE: carga la Q-table del disco y juega greedy puro."""
    ajustes = cfg if cfg is not None else TrainingConfig()
    tabla = load_qtable(model_path)

    aprendida = Trainer(graph, ajustes, q_table=tabla, learn=False)
    aprendida.run(episodes)

    referencia = _run_baseline(graph, ajustes, episodes)
    return aprendida, referencia


def _run_baseline(
    graph: WarehouseGraph, cfg: TrainingConfig, episodes: int
) -> list[EpisodeStats]:
    """La baseline de la fase 5 sobre los mismos escenarios, con la misma vara."""
    maestro = random.Random(cfg.seed)
    semilla_escenarios = maestro.randrange(2**31)

    politica = BaselineAdapter()
    entorno = TrainingEnv(
        graph,
        cfg.agents,
        politica,
        seed=semilla_escenarios,
        max_steps=cfg.max_steps,
    )

    historia: list[EpisodeStats] = []
    with _quiet("simulation", "agent"):
        for episode in range(1, episodes + 1):
            entorno.reset()
            total = 0.0
            decisiones = 0
            while not entorno.done():
                for transicion in entorno.step():
                    total += transicion.reward
                    decisiones += 1
            for transicion in entorno.close_pending():
                total += transicion.reward
                decisiones += 1
            numeros = entorno.stats(episode, 0.0, states_visited=0)
            historia.append(
                numeros.con_recompensa(
                    total, total / decisiones if decisiones else 0.0
                )
            )
    return historia


def write_training_log(
    stats: Sequence[EpisodeStats], path: str | Path = config.TRAINING_LOG_FILE
) -> Path:
    """Escribe `results/training_log.csv`: una fila por episodio, cabecera incluida."""
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding=config.ENCODING, newline="") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=list(LOG_COLUMNS))
        escritor.writeheader()
        for fila in stats:
            escritor.writerow(fila.as_row())
    log.info("log de entrenamiento en %s (%d episodios)", destino, len(stats))
    return destino


def summary_lines(
    stats: Sequence[EpisodeStats], every: int = REPORT_EVERY
) -> list[str]:
    """La tabla resumen por bloques de `every` episodios, lista para el log."""
    if not stats:
        return ["(sin episodios)"]

    cabecera = (
        f"{'episodios':>12} {'epsilon':>8} {'recompensa':>11} {'r/decision':>11} "
        f"{'conflictos':>11} {'deadlocks':>10} {'completadas':>12} "
        f"{'makespan':>9} {'espera':>8} {'estados':>8}"
    )
    lineas = [f"--- resumen cada {every} episodios ---", cabecera, "-" * len(cabecera)]

    agentes = max((fila.completed_tasks for fila in stats), default=0)
    for arranque in range(0, len(stats), every):
        bloque = stats[arranque : arranque + every]
        lineas.append(
            f"{bloque[0].episode:>5}-{bloque[-1].episode:<6} "
            f"{statistics.fmean(f.epsilon for f in bloque):>8.3f} "
            f"{statistics.fmean(f.total_reward for f in bloque):>11.1f} "
            f"{statistics.fmean(f.avg_reward for f in bloque):>11.2f} "
            f"{statistics.fmean(f.conflicts for f in bloque):>11.1f} "
            f"{statistics.fmean(f.deadlocks for f in bloque):>10.2f} "
            f"{statistics.fmean(f.completed_tasks for f in bloque):>12.2f} "
            f"{statistics.fmean(f.makespan for f in bloque):>9.1f} "
            f"{statistics.fmean(f.total_wait for f in bloque):>8.1f} "
            f"{bloque[-1].states_visited:>8}"
        )
    if agentes:
        lineas.append(f"(completadas es sobre {agentes} AGVs por episodio)")
    return lineas


def compare_lines(
    aprendida: Sequence[EpisodeStats], baseline: Sequence[EpisodeStats]
) -> list[str]:
    """Q-Learning contra baseline, promedio a promedio, listo para el log."""
    if not aprendida or not baseline:
        return ["(no hay con que comparar)"]

    def medias(filas: Sequence[EpisodeStats]) -> dict[str, float]:
        ticks = max(statistics.fmean(f.makespan for f in filas), 1.0)
        return {
            "recompensa": statistics.fmean(f.total_reward for f in filas),
            "completadas": statistics.fmean(f.completed_tasks for f in filas),
            "deadlocks": statistics.fmean(f.deadlocks for f in filas),
            "makespan": ticks,
            "conflictos": statistics.fmean(f.conflicts for f in filas),
            "conflictos/tick": statistics.fmean(f.conflicts for f in filas) / ticks,
            "espera": statistics.fmean(f.total_wait for f in filas),
            "espera/tick": statistics.fmean(f.total_wait for f in filas) / ticks,
        }

    izquierda, derecha = medias(aprendida), medias(baseline)
    lineas = [
        f"--- {len(aprendida)} episodios, los mismos escenarios, medias ---",
        f"{'metrica':<17}{'q-learning':>12}{'baseline':>12}{'diferencia':>13}",
        "-" * 54,
    ]
    for nombre in izquierda:
        uno, otro = izquierda[nombre], derecha[nombre]
        lineas.append(f"{nombre:<17}{uno:>12.2f}{otro:>12.2f}{uno - otro:>+13.2f}")
    lineas.append(
        "los totales crudos (conflictos, espera) premian al que muere antes: "
        "miralos por tick"
    )
    return lineas


class _quiet:
    """Baja el nivel de unos loggers mientras dura el bloque, y lo devuelve luego."""

    def __init__(self, *names: str, level: int = logging.ERROR) -> None:
        self.names = names
        self.level = level
        self.anteriores: list[tuple[logging.Logger, int]] = []

    def __enter__(self) -> None:
        self.anteriores = [
            (logging.getLogger(nombre), logging.getLogger(nombre).level)
            for nombre in self.names
        ]
        for logger, _ in self.anteriores:
            logger.setLevel(self.level)

    def __exit__(self, *_excepcion: object) -> None:
        for logger, nivel in self.anteriores:
            logger.setLevel(nivel)


if __name__ == "__main__":
    setup_logging()
    for _linea in report_state_space():
        log.info("%s", _linea)
