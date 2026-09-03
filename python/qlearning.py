"""Fases 6 y 7: el entorno de Q-Learning y el bucle que aprende encima de el.

El modulo va en dos mitades, y se leen en ese orden:

**Fase 6, el entorno.** Que ve un AGV cuando le toca decidir (`get_local_state`),
que puede hacer (`Action`), cuanto vale cada cosa que pasa (`reward`) y donde se
guarda lo aprendido (`QTable`). Aqui no se entrena nada: se define el problema.

**Fase 7, el entrenamiento.** `TrainingEnv` corre un episodio y reparte la
recompensa, `Trainer` pone encima la actualizacion de Bellman

    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

y `train()` / `evaluate()` son los dos modos, bien separados: TRAIN explora con
epsilon-greedy y escribe en la tabla, EVALUATE es greedy puro, carga la tabla del
disco y no toca nada. **Ni el uno ni el otro levantan el servidor ni hablan con
Unity**: entrenar contra un socket solo lo haria mas lento y no le da al
algoritmo ni un dato mas.

--- Una sola Q-table para todos los AGVs (politica homogenea) ---

Los N agentes comparten la **misma** tabla: todos leen de ella y todos escriben
en ella. No es un atajo, es la decision de diseño de la fase:

- Un AGV es intercambiable con otro. El estado es local y no lleva el id dentro
  (`has_priority` dice si soy el menor, no quien soy), asi que lo que aprende el
  AGV 3 sobre "hay alguien delante y no tengo prioridad" vale igual para el 1.
- Cada episodio produce N veces mas experiencia. Con 4 agentes la tabla ve
  ~4x transiciones por episodio que con politicas separadas, y son 72 estados:
  se llenan en decenas de episodios en vez de en miles.
- Cambiar el numero de AGVs no invalida el modelo. Se entrena con 4 y se evalua
  con 6 sin reentrenar, porque la tabla no esta indexada por agente.

Lo que se pierde es la especializacion (no puede haber un AGV "agresivo" y otro
"cauto"), y en un almacen de AGVs identicos eso no es una perdida.

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

    python3 python/qlearning.py                                  # el entorno
    python3 python/main.py train --map warehouse --agents 4       # entrenar
    python3 python/main.py evaluate --map warehouse --agents 4     # evaluar
"""

import contextlib
import csv
import json
import logging
import math
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any, Protocol

import astar
import config
import conflicts
import simulation
from agent import STATE_DONE, STATE_MOVING, STATE_WAITING, Agent
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


# Recalcular es mecanica del MOTOR, no del aprendizaje, asi que estas dos viven
# en `conflicts` desde la fase 8. Se re-exportan aqui porque la fase 6 las
# publico en este modulo y lo que se importa de un sitio no se muda sin avisar.
reroute_penalties = conflicts.reroute_penalties
reroute = conflicts.reroute


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

    Esquivar un conflicto justifica una ruta **igual** de cara, no una mas cara.
    Sin ese matiz el castigo no salta nunca (bloqueado siempre hay un conflicto)
    y la politica aprende a recalcular sin parar: como `PROGRESS` paga por cada
    nodo cruzado, dar la vuelta al almacen sale rentable, y dos AGVs sentados en
    el destino del otro se pasan la corrida dando vueltas.

    Se compara por costo (`astar.path_cost`), no por numero de nodos: la ruta
    corta en saltos puede ser la cara. Una ruta que ni siquiera se puede recorrer
    cuenta como infinita, o sea que nunca es mejor.
    """
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

    def _fila(self, state: State) -> dict[Action, float]:
        """La fila del estado **sin crearla**: si no esta, una de ceros de usar y tirar.

        Leer no puede escribir. Con el `defaultdict` pelado, preguntar por un
        estado que no se ha visto lo mete en la tabla, y entonces un `evaluate`
        (que no aprende nada) acabaria con mas estados de los que tenia el modelo
        que cargo. `states_visited` dejaria de contar lo que dice contar.
        """
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
        """La accion de mas valor, con el empate resuelto por el orden de ACTIONS.

        El desempate es **determinista** para que dos corridas con la misma
        semilla decidan lo mismo. Con la tabla recien creada todo empata a cero
        y gana ADVANCE.
        """
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
        """La actualizacion de Bellman. Devuelve el Q(s, a) nuevo.

            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        `terminal=True` pone a cero el termino del futuro, que es lo correcto
        cuando `s'` es el final del episodio: el AGV ya llego y detras no hay
        nada que valorar. Sin eso la tabla arrastraria el valor de un estado que
        no se va a vivir, y el +100 de la llegada se contaria dos veces.

        `among` limita el max a las acciones que la politica puede elegir de
        verdad (`enabled_actions()`): si REROUTE esta apagado, su celda no debe
        entrar en el maximo o se aprenderia sobre una accion que nadie va a
        tomar.
        """
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
        """Escribe la tabla en JSON, creando el directorio si hace falta.

        `metadata` va en el mismo fichero, bajo la clave `metadata`: con que
        mapa, cuantos agentes, que hiperparametros y que semilla se entreno esto,
        y cuando. **Una Q-table sin saber con que se entreno no sirve**: 216
        numeros sueltos no dicen si salieron de `warehouse` con 4 AGVs o de
        `simple` con 2, y evaluarla en el mapa equivocado no da error, da
        resultados malos. Lo escribe `Trainer.save()`.
        """
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


def load_metadata(path: str | Path) -> dict[str, Any]:
    """La `metadata` con la que se guardo una Q-table, o `{}` si no lleva.

    Se lee aparte de `QTable.load()` a proposito: la tabla se carga para
    *usarla* y la metadata para *contarla*, y una tabla vieja sin metadata tiene
    que poder seguir cargando.
    """
    crudo = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(crudo, dict):
        return {}
    datos = crudo.get("metadata")
    return dict(datos) if isinstance(datos, dict) else {}


# --- La politica -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Decision:
    """Lo que un AGV decidio en un tick concreto, con todo lo que hace falta
    para poder pagarselo despues.

    Lleva el `step` porque `QLearningPolicy` guarda **la ultima** decision de
    cada AGV y no todas: un agente a media travesia no decide nada, asi que su
    entrada se queda de un tick anterior. Sin la marca de paso, el entrenamiento
    le atribuiria a la decision de ahora lo que paso hace cuatro ticks.

    Lo que **no** lleva es que paso despues: si el motor concedio el paso, si lo
    nego o si el recalculo cambio algo. Eso es cosa del motor y esta en
    `simulation.ActionRecord`. Aqui solo vive lo que la politica eligio.
    """

    state: State
    action: Action
    step: int


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
        self._last: dict[int, Decision] = {}
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

    def decide(self, agent: Agent, local_state: conflicts.LocalState) -> str:
        """La accion elegida: `"advance"`, `"wait"` o `"reroute"`.

        Y **nada mas**: aqui no se mueve a nadie ni se le reescribe la ruta. La
        politica propone y el motor dispone, que es lo que la fase 8 vino a
        dejar claro. Quien ejecuta el REROUTE (encarecer el mapa y volver a
        llamar a A*) es `simulation.Simulation._recalcula()`, con la tabla de
        penalizaciones del almacen, no con un dict de usar y tirar.

        Antes esta funcion recalculaba la ruta ella misma, y eso dejaba al motor
        con una intencion que apuntaba a la ruta vieja: una fuente de errores
        silenciosos que la fase 8 se lleva por delante.
        """
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
        decision = self._last.get(agent_id)
        return None if decision is None else (decision.state, decision.action)

    def decision(self, agent_id: int) -> Decision | None:
        """La ultima decision entera de este AGV: estado, accion, paso y reroute.

        Es lo que usa `TrainingEnv` para saber **en que tick** decidio cada uno:
        `last_decision()` da la pareja de siempre, y esto da ademas la marca de
        paso, que es lo que distingue una decision de este tick de la que quedo
        de hace cuatro.
        """
        return self._last.get(agent_id)



# --- Fase 7: el entrenamiento -------------------------------------------------
#
# Todo lo que sigue corre SIN servidor y SIN Unity. Un episodio son unos cientos
# de ticks y se entrenan mil episodios: meter un socket en medio multiplicaria el
# tiempo por el ping y no le daria al algoritmo ni un dato mas de los que ya
# tiene. Unity se conecta a `main.py serve` cuando hay algo que enseñar.


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Los numeros de una corrida de entrenamiento. Por defecto, los de `config.py`.

    Es inmutable y se guarda tal cual en la metadata de la Q-table: un modelo y
    los hiperparametros con los que salio viajan juntos o no sirven.
    """

    map_name: str = config.DEFAULT_MAP
    agents: int = config.TRAIN_AGENTS
    episodes: int = config.EPISODES
    seed: int = config.RANDOM_SEED
    alpha: float = config.ALPHA
    gamma: float = config.GAMMA
    epsilon_start: float = config.EPSILON_START
    epsilon_end: float = config.EPSILON_END
    epsilon_decay: float = config.EPSILON_DECAY
    max_steps: int = config.MAX_STEPS_PER_EPISODE
    enable_reroute: bool | None = None
    report_every: int = config.REPORT_EVERY

    def as_dict(self) -> dict[str, Any]:
        """Los hiperparametros en JSON, para la metadata del modelo."""
        return {
            "map": self.map_name,
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


@dataclass(frozen=True, slots=True)
class Transition:
    """Una (s, a, r, s') lista para meter en Bellman.

    `terminal` dice si `next_state` es el final: cuando un AGV llega a su
    destino no hay futuro que descontar, y `QTable.update()` pone el termino de
    `gamma` a cero. Un episodio cortado por el tope de pasos **no** es terminal:
    el mundo seguia, el que se acabo fue el reloj del experimento.
    """

    agent_id: int
    state: State
    action: Action
    reward: float
    next_state: State
    terminal: bool


@dataclass(frozen=True, slots=True)
class EpisodeStats:
    """Una fila de `results/training_log.csv`: como fue un episodio.

    | Columna           | Que es                                               |
    |-------------------|------------------------------------------------------|
    | `episode`         | Numero de episodio, desde 1                          |
    | `epsilon`         | El epsilon con el que se jugo (0 en EVALUATE)        |
    | `total_reward`    | Suma de la recompensa de **todos** los AGVs          |
    | `avg_reward`      | `total_reward` por decision tomada                   |
    | `conflicts`       | Conflictos detectados en el episodio                 |
    | `deadlocks`       | 1 si la corrida murio atascada, 0 si no              |
    | `completed_tasks` | Cuantos AGVs llegaron a su destino                   |
    | `makespan`        | Tick en que llego el ultimo; si no llegaron todos,   |
    |                   | los ticks que duro el episodio                       |
    | `total_wait`      | Ticks perdidos cediendo el paso, entre todos         |
    | `states_visited`  | Estados distintos en la Q-table (acumulado, tope 72) |

    `avg_reward` va **por decision** y no por agente: el episodio del principio
    dura 200 ticks y el del final 60, asi que dividir por el numero de AGVs solo
    reescalaria `total_reward` y no diria nada nuevo. Por decision si dice algo
    distinto: cuanto saca el AGV cada vez que le toca elegir.
    """

    episode: int
    epsilon: float
    total_reward: float
    avg_reward: float
    conflicts: int
    deadlocks: int
    completed_tasks: int
    makespan: int
    total_wait: int
    states_visited: int

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


# Las columnas del CSV, en este orden. Es el contrato del fichero.
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
    """Lo que `TrainingEnv` necesita de una politica para poder puntuarla.

    Es `conflicts.Policy` mas tres cosas: `bind()` para que vea el estado
    completo, `reset()` entre episodios, y `decision()` para saber **que** eligio
    cada AGV y **en que tick**, que es lo unico con lo que se puede repartir la
    recompensa. Lo cumplen `QLearningPolicy` y `BaselineAdapter`.
    """

    name: str

    def decide(self, agent: Agent, local_state: conflicts.LocalState) -> str: ...

    def bind(self, simulation: SimulationView) -> None: ...

    def reset(self) -> None: ...

    def decision(self, agent_id: int) -> Decision | None: ...


class BaselineAdapter:
    """La politica de la fase 5 con el cuaderno de la fase 7.

    Decide **exactamente** lo mismo que `conflicts.BaselinePolicy` (cede el paso
    si te ganaron el conflicto), pero ademas apunta el estado y la accion en el
    formato de `Decision`. Sirve para medir a la baseline con la MISMA vara y en
    los MISMOS escenarios que al Q-Learning: sin esto la comparacion del
    `evaluate` seria entre dos numeros calculados de forma distinta.

    No aprende ni guarda nada: es la referencia contra la que se compara.
    """

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
        accion = Action.ADVANCE if motor == conflicts.ACTION_GO else Action.WAIT
        self._last[agent.id] = Decision(estado, accion, local_state.step)
        return motor

    def decision(self, agent_id: int) -> Decision | None:
        return self._last.get(agent_id)


def random_routes(
    graph: WarehouseGraph, n_agents: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Un escenario nuevo: origenes distintos, destinos distintos, y nadie ya en casa.

    Cada episodio se juega en un reparto de tareas **distinto**, sacado de un
    generador sembrado. Con el reparto fijo de `Simulation._planea_rutas()` los
    mil episodios serian el mismo, y la tabla aprenderia ese escenario de
    memoria en vez de aprender a ceder el paso; ademas la curva de aprendizaje no
    diria nada, porque el unico ruido seria el de la exploracion.

    Origenes y destinos van sin repetir por lo mismo que en la fase 5: dos AGVs
    en el mismo nodo rompen la invariante antes de mover nada, y un AGV aparcado
    encima del destino de otro lo bloquea para siempre.
    """
    nodos = graph.nodes()
    if n_agents > len(nodos):
        raise ValueError(
            f"no caben {n_agents} agentes en un mapa de {len(nodos)} nodo(s)"
        )

    origenes = rng.sample(nodos, n_agents)
    destinos = rng.sample(nodos, n_agents)
    # Rechazo simple: un AGV que sale de su propio destino tiene una ruta de un
    # nodo, llega en el tick cero y no aporta ni una transicion al episodio.
    for _ in range(20):
        if all(origen != destino for origen, destino in zip(origenes, destinos)):
            break
        destinos = rng.sample(nodos, n_agents)
    return list(zip(origenes, destinos))


@dataclass(slots=True)
class _Abierta:
    """Una decision a la que todavia se le esta sumando la recompensa.

    Un AGV decide **solo cuando esta parado en un nodo**: si elige ADVANCE, cruza
    un tramo que cuesta entre 4 y 8 ticks y durante esos ticks no vuelve a
    decidir nada. Todo lo que pase mientras cruza (el `+2` de llegar al nodo, el
    `+100` de terminar) es consecuencia de aquella decision, asi que se le suma a
    ella y la transicion se cierra cuando el AGV vuelve a decidir o cuando
    termina. Cerrar la transicion en el mismo tick daria recompensa 0 a todos los
    ADVANCE, y con eso no se aprende nada.
    """

    decision: Decision
    reward: float = 0.0

    def close(self, agent_id: int, next_state: State, *, terminal: bool) -> Transition:
        return Transition(
            agent_id=agent_id,
            state=self.decision.state,
            action=self.decision.action,
            reward=self.reward,
            next_state=next_state,
            terminal=terminal,
        )


@dataclass(frozen=True, slots=True)
class _Foto:
    """Como estaba un AGV al empezar el tick. Con esto se ve que le paso."""

    path_index: int
    state: str
    wait_time: int


class TrainingEnv:
    """Un episodio: monta la simulacion, la tickea y reparte la recompensa.

    No aprende: eso es `Trainer`. Lo que hace es traducir lo que pasa en la
    `Simulation` a transiciones `(s, a, r, s')`, que es lo unico que Bellman
    necesita. La separacion importa porque asi el mismo entorno puntua al
    Q-Learning y a la baseline exactamente igual.

        entorno = TrainingEnv(grafo, 4, politica, seed=42, max_steps=200)
        entorno.reset()
        while not entorno.done:
            for transicion in entorno.step():
                ...
        for transicion in entorno.close_pending():
            ...

    **Cada `reset()` es un escenario nuevo** (ver `random_routes`), sacado del
    generador sembrado en el constructor: la secuencia de episodios de una
    semilla es siempre la misma.
    """

    def __init__(
        self,
        graph: WarehouseGraph,
        n_agents: int,
        policy: TrainablePolicy,
        *,
        seed: int = config.RANDOM_SEED,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
    ) -> None:
        self.graph = graph
        self.n_agents = int(n_agents)
        self.policy = policy
        self.max_steps = int(max_steps)
        self._rng = random.Random(seed)

        self.simulation: simulation.Simulation | None = None
        self._abiertas: dict[int, _Abierta] = {}
        self._llegadas: dict[int, int] = {}
        self._deadlock: bool = False

    def __repr__(self) -> str:
        return (
            f"TrainingEnv(map={self.graph.name!r}, agents={self.n_agents}, "
            f"policy={self.policy.name!r}, max_steps={self.max_steps})"
        )

    @property
    def sim(self) -> simulation.Simulation:
        """La simulacion del episodio en curso. Revienta si no hubo `reset()`."""
        if self.simulation is None:
            raise RuntimeError("hace falta un reset() antes de empezar el episodio")
        return self.simulation

    @property
    def done(self) -> bool:
        """True si llegaron todos, si hubo deadlock o si se acabaron los ticks."""
        return self.sim.done or self.sim.step >= self.max_steps

    def reset(self) -> None:
        """Arranca un episodio nuevo, con un reparto de tareas nuevo."""
        rutas = random_routes(self.graph, self.n_agents, self._rng)
        self.simulation = simulation.Simulation(
            self.graph, self.n_agents, routes=rutas, policy=self.policy
        )
        self.policy.reset()
        self.policy.bind(self.simulation)
        self._abiertas.clear()
        self._llegadas.clear()
        self._deadlock = False

    def step(self) -> list[Transition]:
        """Un tick de la simulacion. Devuelve las transiciones que se cerraron.

        El reparto de la recompensa va en dos pasadas, y el orden no es un
        detalle:

        1. El AGV que **ha vuelto a decidir** en este tick cierra su transicion
           anterior: el estado que acaba de observar es el `s'` de aquella.
        2. Los eventos del tick se le suman a la transicion abierta de cada AGV,
           que para el que decidio ahora es la nueva, y para el que sigue
           cruzando un tramo es la de hace varios ticks.
        """
        paso = self.sim.step + 1
        antes = {
            agente.id: _Foto(agente.path_index, agente.state, agente.wait_time)
            for agente in self.sim.agents
        }
        # El registro de conflictos de la corrida solo crece, asi que lo que se
        # anada de aqui en adelante es exactamente lo de este tick. Se marca la
        # posicion en vez de recorrer el log entero filtrando por paso: sobre un
        # episodio de 200 ticks eso seria cuadratico, y no hace falta.
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
            if agente.state == STATE_DONE and antes[agente.id].state != STATE_DONE:
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

            if agente.state == STATE_DONE:
                del self._abiertas[agente.id]
                cerradas.append(
                    abierta.close(
                        agente.id, get_local_state(agente, self.sim), terminal=True
                    )
                )

        return cerradas

    def close_pending(self) -> list[Transition]:
        """Cierra las transiciones que quedaron abiertas al acabar el episodio.

        Un episodio cortado por el tope de pasos **no** es terminal: el AGV
        seguia teniendo camino por delante y su `s'` vale lo que valga, asi que
        se descuenta con gamma como cualquier otro. Uno que murio en deadlock si
        lo es: detras de un atasco no hay futuro que valorar.
        """
        cerradas: list[Transition] = []
        for agente in self.sim.agents:
            abierta = self._abiertas.pop(agente.id, None)
            if abierta is None:
                continue
            cerradas.append(
                abierta.close(
                    agente.id,
                    get_local_state(agente, self.sim),
                    terminal=self._deadlock or agente.state == STATE_DONE,
                )
            )
        return cerradas

    def stats(self, episode: int, epsilon: float, *, states_visited: int) -> EpisodeStats:
        """Los numeros del episodio que acaba de terminar."""
        completadas = sum(
            1 for agente in self.sim.agents if agente.state == STATE_DONE
        )
        llegaron_todos = len(self._llegadas) == len(self.sim.agents)
        return EpisodeStats(
            episode=episode,
            epsilon=epsilon,
            total_reward=0.0,  # lo rellena el Trainer, que es quien las suma
            avg_reward=0.0,
            conflicts=self.sim.conflicts.total,
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

        Cada evento salta **como mucho una vez**, y el precio de todos sale de
        `config.py` por `reward()`. Las reglas:

        - `PROGRESS`: `path_index` subio, o sea que cruzo un tramo entero. Que se
          haya movido en pantalla no cuenta: pagar por ir a media travesia seria
          pagar por ir despacio.
        - `TASK_COMPLETE`: entro en `done` en este tick.
        - `CONFLICT` frente a `WAIT`: los dos son "no se movio", y lo que los
          separa es **haberlo intentado**. Quien lo dice ya no es la decision,
          es el motor: `ActionRecord.blocked` es "elegi ADVANCE y no me dejaron
          pasar", que es exactamente intentar entrar donde no cabia. -20. El que
          cedio el paso, -1.

          El castigo **no** cae sobre el que gano el desempate y si paso. Y no es
          un descuido: el estado local no distingue "camino libre sin rivales" de
          "camino libre y gano la disputa" (en los dos `next_node_occupied = 0` y
          `has_priority = 1`), asi que cobrarselo al ganador envenenaria la celda
          del camino libre, que es la que sostiene toda la politica, y el almacen
          se pararia entero. Al perdedor si se le cobra, y es el que puede
          aprender algo: su `has_priority` vale 0.
        - `CONFLICT` tambien al que el desatasco tuvo que forzar
          (`ActionRecord.forced`). Es la leccion de la fase 8: quedarse todos
          parados sale caro, y quien paga es la accion que cada uno eligio.
        - `USELESS_REROUTE`: solo el tick en que se recalculo, y solo si la ruta
          nueva ni salia mas barata ni esquivaba un conflicto de verdad.
        - `DEADLOCK`: a todo el que seguia en marcha cuando la corrida murio.
        """
        eventos: list[Event] = []
        # Lo que el MOTOR hizo con la intencion de este AGV. La decision dice lo
        # que quiso; esto dice lo que le dejaron.
        registro = self.sim.action_record(agent.id)
        fresco = registro is not None and registro.step == paso

        if agent.path_index > antes.path_index:
            eventos.append(Event.PROGRESS)
        if agent.state == STATE_DONE and antes.state != STATE_DONE:
            eventos.append(Event.TASK_COMPLETE)

        if agent.wait_time > antes.wait_time:
            eventos.append(
                Event.CONFLICT if fresco and registro.blocked else Event.WAIT
            )

        if fresco and registro.forced and Event.CONFLICT not in eventos:
            eventos.append(Event.CONFLICT)

        if fresco and registro.reroute is not None:
            vieja, nueva = registro.reroute
            # "Esquivar un conflicto" es irse a otro sitio, no que hubiera un
            # conflicto. Con `en_conflicto` a secas, cualquier recalculo hecho
            # estando bloqueado (o sea, casi todos) contaba como util y el
            # castigo no saltaba nunca: la politica aprendia a recalcular sin
            # parar, y dos AGVs sentados en el destino del otro se pasaban la
            # corrida dando vueltas al almacen.
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

        if deadlock and agent.state != STATE_DONE:
            eventos.append(Event.DEADLOCK)

        return eventos


class Trainer:
    """El bucle de Q-Learning: episodios, epsilon-greedy y Bellman.

    Los dos modos van bien separados y el que manda es `learn`:

    | Modo       | `learn` | epsilon                 | Q-table              |
    |------------|---------|-------------------------|----------------------|
    | TRAIN      | `True`  | de EPSILON_START a END  | se actualiza         |
    | EVALUATE   | `False` | **0**, greedy puro      | **no se toca**       |

    En EVALUATE no hay ni azar ni escritura: `epsilon = 0` hace que
    `QLearningPolicy.choose()` sea siempre `best_action()`, y `_aprende()` no se
    llama. Es lo que hace que evaluar dos veces el mismo modelo devuelva
    exactamente los mismos numeros.

    Los N agentes comparten **una sola** Q-table (politica homogenea): es la
    decision de diseño explicada arriba en el docstring del modulo.
    """

    def __init__(
        self,
        graph: WarehouseGraph,
        cfg: TrainingConfig,
        *,
        q_table: QTable | None = None,
        learn: bool = True,
    ) -> None:
        self.graph = graph
        self.cfg = cfg
        self.learn = learn
        self.q: QTable = q_table if q_table is not None else QTable()
        self.actions = enabled_actions(cfg.enable_reroute)
        self.epsilon: float = cfg.epsilon_start if learn else 0.0
        self.history: list[EpisodeStats] = []
        # Cuantas veces se actualizo cada estado. Sin esto, una celda que se
        # visito 4.000 veces y otra que se visito 3 se leen igual en el JSON, y
        # la segunda no es una politica aprendida: es ruido con forma de numero.
        self.visits: Counter[State] = Counter()

        # Dos generadores distintos, los dos hijos de la misma semilla: si el
        # escenario y la exploracion salieran del mismo, cambiar el numero de
        # acciones exploradas moveria tambien el reparto de tareas de cada
        # episodio, y dos corridas dejarian de ser comparables.
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
        )

    def __repr__(self) -> str:
        return (
            f"Trainer(mode={'train' if self.learn else 'evaluate'}, "
            f"map={self.graph.name!r}, agents={self.cfg.agents}, "
            f"epsilon={self.epsilon:g}, states={len(self.q)})"
        )

    def decay_epsilon(self) -> float:
        """Decaimiento exponencial con suelo: `eps <- max(END, eps * DECAY)`.

        Exponencial y no lineal porque lo que hace falta es explorar mucho al
        principio, cuando la tabla esta a ceros y cualquier cosa es informacion,
        y cada vez menos despues. El suelo `EPSILON_END` no se quita nunca: una
        politica que deja de explorar del todo no vuelve a corregir un estado
        que aprendio mal.
        """
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
        return self.epsilon

    def run_episode(self, episode: int) -> EpisodeStats:
        """Un episodio entero: reset, ticks hasta el final y las Q que toquen."""
        self.policy.epsilon = self.epsilon
        self.env.reset()

        total = 0.0
        decisiones = 0
        while not self.env.done:
            for transicion in self.env.step():
                total += transicion.reward
                decisiones += 1
                self._aprende(transicion)

        for transicion in self.env.close_pending():
            total += transicion.reward
            decisiones += 1
            self._aprende(transicion)

        numeros = self.env.stats(episode, self.epsilon, states_visited=len(self.q))
        return replace(
            numeros,
            total_reward=total,
            avg_reward=total / decisiones if decisiones else 0.0,
        )

    def run(self, episodes: int | None = None) -> list[EpisodeStats]:
        """El bucle: `EPISODES` episodios y un `decay_epsilon()` detras de cada uno.

            for episode in range(EPISODES):
                env.reset()
                while not env.done:
                    ... s, a, r, s' de cada agente, y Q(s,a) actualizada ...
                decay_epsilon()

        El log de la simulacion se calla durante la corrida: mil episodios con
        una linea por conflicto son cientos de miles de lineas por consola, y lo
        que se quiere leer es el resumen de cada 100.
        """
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
            "last_episode": ultimo.as_row() if ultimo is not None else None,
        }

    def _aprende(self, transicion: Transition) -> None:
        """Bellman, pero solo en TRAIN. En EVALUATE esto no hace nada."""
        if not self.learn:
            return
        self.visits[transicion.state] += 1
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


# --- Los dos modos -----------------------------------------------------------


def train(
    graph: WarehouseGraph,
    cfg: TrainingConfig | None = None,
    *,
    model_path: str | Path = config.Q_TABLE_FILE,
    log_path: str | Path | None = config.TRAINING_LOG_FILE,
    curve_path: str | Path | None = config.LEARNING_CURVE_FILE,
) -> Trainer:
    """Modo TRAIN: entrena, guarda el modelo, el CSV y la curva. Sin servidor.

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
    if curve_path is not None:
        save_learning_curve(entrenador.history, curve_path)
    return entrenador


def evaluate(
    graph: WarehouseGraph,
    cfg: TrainingConfig | None = None,
    *,
    model_path: str | Path = config.Q_TABLE_FILE,
    episodes: int = 100,
) -> tuple[Trainer, list[EpisodeStats]]:
    """Modo EVALUATE: carga la Q-table del disco y juega greedy puro.

    `epsilon = 0`, `learn = False`: no explora, no escribe en la tabla y no
    guarda nada. Devuelve `(trainer, baseline)`, con la baseline corrida sobre
    **los mismos escenarios** para que la comparacion signifique algo: los dos
    `TrainingEnv` salen de la misma semilla, asi que el episodio 7 de uno es el
    episodio 7 del otro, con las mismas tareas y los mismos AGVs.
    """
    ajustes = cfg if cfg is not None else TrainingConfig()
    tabla = QTable.load(model_path)

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
            while not entorno.done:
                for transicion in entorno.step():
                    total += transicion.reward
                    decisiones += 1
            for transicion in entorno.close_pending():
                total += transicion.reward
                decisiones += 1
            numeros = entorno.stats(episode, 0.0, states_visited=0)
            historia.append(
                replace(
                    numeros,
                    total_reward=total,
                    avg_reward=total / decisiones if decisiones else 0.0,
                )
            )
    return historia


# --- El registro de la corrida -----------------------------------------------


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


def read_training_log(path: str | Path = config.TRAINING_LOG_FILE) -> list[dict[str, str]]:
    """Lee el CSV tal cual, en texto. Para los tests y para mirarlo a mano."""
    with Path(path).open("r", encoding=config.ENCODING, newline="") as fichero:
        return list(csv.DictReader(fichero))


def summary_lines(
    stats: Sequence[EpisodeStats], every: int = config.REPORT_EVERY
) -> list[str]:
    """La tabla resumen por bloques de `every` episodios, lista para el log.

    Se promedia por bloques y no episodio a episodio porque cada episodio tiene
    un reparto de tareas distinto: dos AGVs que salen pegados chocan y dos que
    salen en esquinas opuestas no, y esa varianza tapa la tendencia. La media de
    100 episodios sortea el mismo tipo de escenarios en todos los bloques, asi
    que lo que cambie de un bloque a otro es la politica.
    """
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


def moving_average(values: Sequence[float], window: int) -> list[float]:
    """Media movil centrada-a-la-izquierda, del mismo largo que la entrada."""
    if window <= 1:
        return [float(valor) for valor in values]
    salida: list[float] = []
    acumulado = 0.0
    for indice, valor in enumerate(values):
        acumulado += float(valor)
        if indice >= window:
            acumulado -= float(values[indice - window])
        salida.append(acumulado / min(indice + 1, window))
    return salida


def save_learning_curve(
    stats: Sequence[EpisodeStats],
    path: str | Path = config.LEARNING_CURVE_FILE,
    *,
    window: int | None = None,
) -> Path | None:
    """Guarda la curva de aprendizaje en PNG. Devuelve None si no hay matplotlib.

    matplotlib es opcional a proposito: el proyecto no tiene dependencias, y el
    CSV con los mil episodios ya esta escrito antes de llegar aqui. Sin la
    libreria se avisa por el log y se sigue; el entrenamiento no se pierde por
    no poder dibujarlo.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # sin ventana: esto corre en una terminal
        from matplotlib import pyplot as plt
    except ImportError:
        log.warning(
            "matplotlib no esta instalado, me salto %s "
            "(los mil episodios estan en el CSV igualmente)",
            path,
        )
        return None

    if not stats:
        log.warning("no hay episodios que dibujar")
        return None

    ventana = window if window is not None else max(1, len(stats) // 50)
    episodios = [fila.episode for fila in stats]
    paneles = (
        ("recompensa total", [fila.total_reward for fila in stats], "tab:blue"),
        ("conflictos", [float(fila.conflicts) for fila in stats], "tab:red"),
        ("tareas completadas", [float(fila.completed_tasks) for fila in stats], "tab:green"),
        ("makespan (ticks)", [float(fila.makespan) for fila in stats], "tab:orange"),
    )

    figura, ejes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for eje, (titulo, valores, color) in zip(ejes.flat, paneles):
        eje.plot(episodios, valores, color=color, alpha=0.22, linewidth=0.8)
        eje.plot(
            episodios,
            moving_average(valores, ventana),
            color=color,
            linewidth=1.8,
            label=f"media movil ({ventana})",
        )
        eje.set_title(titulo)
        eje.grid(alpha=0.25)
        eje.legend(loc="best", fontsize="small")

    for eje in ejes[-1]:
        eje.set_xlabel("episodio")

    gemelo = ejes.flat[0].twinx()
    gemelo.plot(episodios, [fila.epsilon for fila in stats], color="grey", linestyle="--", linewidth=1.0)
    gemelo.set_ylabel("epsilon", color="grey")

    figura.suptitle("Q-Learning de AGVs: curva de aprendizaje")
    figura.tight_layout()

    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=120)
    plt.close(figura)
    log.info("curva de aprendizaje en %s", destino)
    return destino


def compare_lines(
    aprendida: Sequence[EpisodeStats], baseline: Sequence[EpisodeStats]
) -> list[str]:
    """Q-Learning contra baseline, promedio a promedio, listo para el log.

    Ojo con leer los totales crudos: **un episodio que muere antes acumula menos
    de todo**. Una politica que se atasca en el tick 30 sale con menos
    conflictos y menos espera que una que corre 90 ticks y entrega el triple de
    tareas, y eso no la hace mejor. Por eso van tambien las dos tasas por tick,
    que es lo que se puede comparar entre corridas de distinta duracion, y por
    eso las que mandan son `completadas` y `deadlocks`.
    """
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


@contextlib.contextmanager
def _quiet(*names: str, level: int = logging.ERROR) -> Iterator[None]:
    """Baja el nivel de unos loggers mientras dura el bloque, y lo devuelve luego.

    Mil episodios de `Simulation` son una linea de INFO por conflicto y otra por
    reset: cientos de miles de lineas que tapan lo unico que hay que leer, que es
    el resumen. Los errores siguen saliendo.
    """
    anteriores = [(logging.getLogger(nombre), logging.getLogger(nombre).level) for nombre in names]
    for logger, _ in anteriores:
        logger.setLevel(level)
    try:
        yield
    finally:
        for logger, nivel in anteriores:
            logger.setLevel(nivel)



__all__ = [
    "ACTIONS",
    "FORMAT",
    "LOG_COLUMNS",
    "QUEUE_CAP",
    "STATE_FIELDS",
    "STATE_SIZES",
    "Action",
    "BaselineAdapter",
    "Decision",
    "EpisodeStats",
    "Event",
    "QLearningPolicy",
    "QTable",
    "SimulationView",
    "State",
    "TrainablePolicy",
    "Trainer",
    "TrainingConfig",
    "TrainingEnv",
    "Transition",
    "compare_lines",
    "decode_state",
    "distance_bucket",
    "enabled_actions",
    "encode_state",
    "evaluate",
    "get_local_state",
    "is_useless_reroute",
    "load_metadata",
    "moving_average",
    "random_routes",
    "read_training_log",
    "report_state_space",
    "reroute",
    "reroute_penalties",
    "reward",
    "save_learning_curve",
    "state_from_local",
    "state_space_size",
    "summary_lines",
    "to_engine_action",
    "train",
    "write_training_log",
]


if __name__ == "__main__":
    setup_logging()
    for _linea in report_state_space():
        log.info("%s", _linea)
