"""El almacen en marcha: ticks, conflictos, desatasco y snapshot.

Cada tick va en dos fases dentro del mismo paso: primero cada AGV parado declara
a que nodo quiere entrar, y despues se detectan los conflictos, la politica
decide quien cede y solo entonces se mueve a alguien.

A* dice POR DONDE y la politica dice QUE HACER AHORA: ninguna accion elige un
nodo. `policy` es la unica variable experimental.
"""

import math
import random
import threading
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import config
import conflicts
import missions
import server
from agent import Agent, Leg, State
from graph import (
    ROLE_CHARGING,
    ROLE_DOCK,
    Penalties,
    TemporaryPenalties,
    WarehouseGraph,
    astar,
    path_cost,
    to_unity,
)
from config import get_logger

log = get_logger("simulation")

FINISHED_DEADLOCK: str = "deadlock"

DEFAULT_ROUTES: dict[str, tuple[str, str]] = {
    "simple": ("A", "F"),
    "warehouse": ("S1", "N6"),
}


DEADLOCK_FORCE_TICKS: int = 8
YIELD_TICKS: int = 10
STARVED_TICKS: int = 45
REROUTE_COOLDOWN: int = 8
SERVE_EPSILON: float = 0.0
SERVE_MIN_VISITS: int = 30


def _recta(graph: WarehouseGraph, a: str, b: str) -> float:
    """Distancia en linea recta entre dos nodos. 0.0 si falta una posicion."""
    p, q = graph.positions.get(a), graph.positions.get(b)
    return 0.0 if p is None or q is None else math.dist(p, q)


def default_route(graph: WarehouseGraph) -> tuple[str, str]:
    """Origen y destino por defecto del mapa."""
    ruta = DEFAULT_ROUTES.get(graph.name)
    if ruta is not None and all(nodo in graph.adjacency for nodo in ruta):
        return ruta

    nodos = graph.nodes()
    if not nodos:
        raise ValueError("el mapa no tiene ni un nodo")
    return nodos[0], nodos[-1]


class ActionRecord:
    """Que decidio un AGV en un tick, y que le concedio el motor."""

    def __init__(
        self,
        action: str,
        step: int,
        blocked: bool = False,
        forced: bool = False,
        reroute: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
    ) -> None:
        self.action = action
        self.step = step
        self.blocked = blocked
        self.forced = forced
        self.reroute = reroute


def make_policy(
    name: str,
    *,
    model: str | Path | None = None,
    seed: int = config.RANDOM_SEED,
) -> conflicts.Policy:
    """Monta una politica por su nombre: `"baseline"` o `"qlearning"`."""
    nombre = str(name).strip().lower()

    if nombre == config.POLICY_BASELINE:
        return conflicts.BaselinePolicy()

    if nombre == config.POLICY_QLEARNING:
        import qlearning

        ruta = Path(model) if model is not None else config.Q_TABLE_FILE
        if not ruta.is_file():
            raise ValueError(
                f"no existe la Q-table {ruta}; entrenala antes con: "
                f"python3 python/main.py train --map {config.DEFAULT_MAP}"
            )
        entrenada_con_reroute = qlearning.trained_enable_reroute(ruta)
        if entrenada_con_reroute is False and config.ENABLE_REROUTE:
            log.info(
                "%s se entreno sin REROUTE: lo dejo fuera tambien al servir",
                ruta,
            )

        visitas = qlearning.load_action_visits(ruta)
        if not visitas and SERVE_MIN_VISITS > 0:
            log.warning(
                "%s no guarda las visitas por celda: sin ellas no se puede "
                "filtrar lo que la tabla no llego a aprender (reentrenala)",
                ruta,
            )

        return qlearning.QLearningPolicy(
            qlearning.load_qtable(ruta),
            epsilon=SERVE_EPSILON,
            seed=seed,
            enable_reroute=entrenada_con_reroute,
            visits=visitas,
            min_visits=SERVE_MIN_VISITS,
        )

    raise ValueError(
        f"no conozco la politica {name!r}; las que hay son "
        + ", ".join(config.POLICIES)
    )


class Simulation:
    """El almacen en marcha: un grafo, sus agentes y el contador de pasos."""

    def __init__(
        self,
        graph: WarehouseGraph,
        n_agents: int = 1,
        *,
        origin: str | None = None,
        target: str | None = None,
        seed: int = config.RANDOM_SEED,
        policy: conflicts.Policy | str | None = None,
        model: str | Path | None = None,
        routes: Sequence[tuple[str, str]] | None = None,
        deliveries: bool = False,
    ) -> None:
        """`policy` acepta el nombre del modo (`"baseline"` / `"qlearning"`) o un"""
        if routes is not None:
            n_agents = len(routes)
        if n_agents < 1:
            raise ValueError(f"hace falta al menos un agente, no {n_agents}")

        nodos = graph.nodes()
        if n_agents > len(nodos):
            raise ValueError(
                f"no caben {n_agents} agentes en un mapa de {len(nodos)} nodo(s): "
                f"cada AGV necesita un nodo de salida para el solo"
            )

        self.graph: WarehouseGraph = graph
        self.step: int = 0
        self.seed: int = seed
        self._model: str | Path | None = model
        self.policy: conflicts.Policy = conflicts.BaselinePolicy()
        self.mode: str = self.policy.name
        self._lock = threading.RLock()

        por_defecto = default_route(graph)
        self._origen: str = origin if origin is not None else por_defecto[0]
        self._destino: str = target if target is not None else por_defecto[1]
        self._rutas: list[tuple[str, str]] = (
            self._comprueba_rutas(routes)
            if routes is not None
            else self._planea_rutas(n_agents)
        )

        self.deliveries: bool = bool(deliveries)
        self.bus: missions.MessageBus = missions.MessageBus()
        self.manager: missions.MissionManager | None = (
            missions.MissionManager(self.bus, graph) if self.deliveries else None
        )
        self.inventory: dict[str, missions.BoxState] = (
            missions.build_inventory(graph) if self.deliveries else {}
        )
        self.delivered: int = 0
        self.picked: int = 0

        self.agents: list[Agent] = [
            Agent(numero, graph, origen)
            for numero, (origen, _) in enumerate(self._rutas, start=1)
        ]

        self.occupancy: dict[str, int] = {}
        self.conflicts: conflicts.ConflictLog = conflicts.ConflictLog()
        self.run: int = 0
        self.deadlocks: int = 0
        self.finished_reason: str | None = None
        self._ticks_sin_avance: int = 0
        self._zonas: frozenset[str] = frozenset()

        self.penalties: TemporaryPenalties = TemporaryPenalties()
        self._acciones: dict[int, ActionRecord] = {}
        self._reservas: dict[str, tuple[int, int]] = {}
        self._parado: dict[int, int] = {}
        self._proximo_reroute: dict[int, int] = {}
        self._recuento: dict[str, int] = dict.fromkeys(conflicts.Intent, 0)
        self._forzados: int = 0
        self._por_id: dict[int, Agent] = {a.id: a for a in self.agents}

        if policy is not None:
            self._monta_politica(policy)

        self.reset()

    def _monta_politica(self, policy: conflicts.Policy | str) -> None:
        """Deja lista la politica, venga por nombre o ya construida."""
        if isinstance(policy, str):
            self.policy = make_policy(policy, model=self._model, seed=self.seed)
            atar = getattr(self.policy, "bind", None)
            if atar is not None:
                atar(self)
        else:
            self.policy = policy
        self.mode = self.policy.name

    def __repr__(self) -> str:
        return (
            f"Simulation(map={self.graph.name!r}, agents={len(self.agents)}, "
            f"step={self.step}, mode={self.mode!r})"
        )

    def done(self) -> bool:
        """True cuando ya ningun agente tiene nada que hacer, o hay deadlock."""
        with self._lock:
            if self.finished_reason is not None:
                return True
            parados = all(
                agente.state in (State.DONE, State.IDLE, State.CHARGING)
                for agente in self.agents
            )
            if self.manager is None:
                return parados
            entregadas = all(
                caja.status is missions.BoxStatus.DELIVERED
                for caja in self.inventory.values()
            )
            return parados and entregadas

    def tick(self) -> int:
        """Avanza la simulacion un paso y devuelve el numero de paso."""
        with self._lock:
            self.step += 1
            self._fase_bateria()
            self._fase_subasta()
            self._cierra_los_que_llegaron()
            self._fase_maniobras()

            huella = self._huella()
            por_agente = self._huella_por_agente()
            intenciones = self._fase_a_intenciones()
            self._fase_b_resuelve_y_aplica(intenciones)
            self._cuenta_los_parados(por_agente)
            self._vigila_el_deadlock(huella)

            return self.step

    def snapshot(self) -> server.Snapshot:
        """El estado ahora mismo, en coordenadas de Unity. **No avanza el reloj.**"""
        with self._lock:
            return {
                "step": self.step,
                "agents": [self._describe(agente) for agente in self.agents],
                "stats": self.stats(),
                "boxes": [caja.as_dict() for caja in self.inventory.values()],
                "mode": self.mode,
            }

    def get_snapshot(self) -> server.Snapshot:
        """Avanza un paso y devuelve el estado. Es lo que pide `POST /step`."""
        with self._lock:
            if self.finished_reason == FINISHED_DEADLOCK:
                log.warning(
                    "la corrida %d acabo en deadlock, arranco la %d",
                    self.run,
                    self.run + 1,
                )
                self.reset()

            self.tick()
            return self.snapshot()

    def stats(self) -> dict[str, Any]:
        """Los numeros de la corrida, que son con los que se compara el baseline.

        Todo aqui dentro es determinista y serializable: dos simulaciones con la
        misma semilla tienen que producir snapshots identicos, `stats` incluido.
        """
        with self._lock:
            return {
                "run": self.run,
                "policy": self.policy.name,
                "conflicts": self.conflicts.total(),
                "conflicts_by_type": self.conflicts.by_type(),
                "deadlocks": self.deadlocks,
                "waiting": sum(
                    1 for agente in self.agents if agente.state == State.WAITING
                ),
                "total_wait_time": sum(agente.wait_time for agente in self.agents),
                "finished_reason": self.finished_reason,
                "actions": dict(self._recuento),
                "forced": self._forzados,
                "penalties": len(self.penalties),
                "deliveries": self.deliveries,
                "picked": self.picked,
                "delivered": self.delivered,
                "missions_pending": (
                    len(self.manager.pool()) if self.manager is not None else 0
                ),
                "missions_total": (
                    len(self.manager.missions) if self.manager is not None else 0
                ),
                "boxes_delivered": sum(
                    1 for c in self.inventory.values()
                    if c.status is missions.BoxStatus.DELIVERED
                ),
                "messages": len(self.bus),
                "battery": [round(a.battery, 1) for a in self.agents],
                "charges": sum(a.charges for a in self.agents),
            }

    def set_mode(self, mode: str, *, model: str | Path | None = None) -> str:
        """Cambia de politica **en caliente** y arranca una corrida limpia."""
        with self._lock:
            if model is not None:
                self._model = model
            self._monta_politica(str(mode).strip().lower())
            self.penalties.clear()
            self._reservas.clear()
            self._ticks_sin_avance = 0
            self.reset()
            log.info("modo %s en caliente, arranca la corrida %d", self.mode, self.run)
            return self.mode

    def action_record(self, agent_id: int) -> ActionRecord | None:
        """Lo que decidio este AGV en el ultimo tick y lo que el motor le concedio.

        Es por donde el entrenamiento se entera de que un ADVANCE quedo
        `blocked` (y cobra el -20) o de que el desatasco forzo al agente.
        """
        with self._lock:
            return self._acciones.get(agent_id)

    def reset(self) -> None:
        """Vuelve al paso cero y reparte otra vez las mismas tareas."""
        with self._lock:
            self.step = 0
            self.run += 1
            self.finished_reason = None
            self._ticks_sin_avance = 0
            self._zonas = frozenset()
            self.conflicts.clear()
            self.occupancy = {}

            self.penalties.clear()
            self._acciones.clear()
            self._reservas.clear()
            self._parado.clear()
            self._proximo_reroute.clear()
            self._recuento = dict.fromkeys(conflicts.Intent, 0)
            self._forzados = 0

            self.delivered = 0
            self.picked = 0
            self.bus.clear()
            if self.manager is not None:
                self.manager = missions.MissionManager(self.bus, self.graph)
                self.inventory = missions.build_inventory(self.graph)

            for agente, (origen, destino) in zip(self.agents, self._rutas):
                agente.reset()
                if self.deliveries:
                    agente.current_node = origen
                    agente.state = State.IDLE
                else:
                    agente.assign_task(origen, destino, task=agente.id)
                self.occupancy[agente.current_node] = agente.id

            log.info(
                "simulacion reiniciada: mapa %s, %d agente(s), modo %s, corrida %d",
                self.graph.name or "(sin nombre)",
                len(self.agents),
                self.mode,
                self.run,
            )

    def _comprueba_rutas(
        self, routes: Sequence[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Acepta un reparto de tareas escrito a mano, si es que se sostiene."""
        rutas = [(str(origen), str(destino)) for origen, destino in routes]

        desconocidos = sorted(
            {nodo for ruta in rutas for nodo in ruta} - set(self.graph.adjacency)
        )
        if desconocidos:
            raise ValueError(
                f"estos nodos no estan en el mapa {self.graph.name!r}: "
                + ", ".join(repr(nodo) for nodo in desconocidos)
            )

        origenes = [origen for origen, _ in rutas]
        if len(set(origenes)) != len(origenes):
            raise ValueError(
                f"dos AGVs no pueden salir del mismo nodo: {origenes}"
            )
        return rutas

    def _planea_rutas(self, n_agents: int) -> list[tuple[str, str]]:
        """Reparte origen y destino, siempre igual para la misma semilla."""
        rutas = [(self._origen, self._destino)]
        nodos = self.graph.nodes()
        if len(nodos) < 2:
            return rutas * n_agents
        if n_agents == 1:
            return rutas

        rng = random.Random(self.seed)
        origenes = rng.sample([n for n in nodos if n != self._origen], n_agents - 1)
        destinos = rng.sample([n for n in nodos if n != self._destino], n_agents - 1)
        rutas.extend(zip(origenes, destinos))
        return rutas


    def _cierra_los_que_llegaron(self) -> None:
        """Pasa a `done` al que ya no tiene siguiente nodo. Antes de declarar nada."""
        for agente in self.agents:
            if agente.state == State.MOVING and agente.next_node() is None:
                agente.progress = 0.0
                if not self._empieza_maniobra(agente):
                    agente.state = State.DONE

    def _suelta_mision(self, agente: Agent) -> None:
        """Devuelve a la bolsa la mision de un AGV que no la puede terminar.

        Sin esto, un AGV varado se queda la mision cogida para siempre: la caja
        no vuelve a publicarse, nadie mas puede ir a por ella y la corrida no
        termina nunca. La caja se queda donde este, y desde ahi se vuelve a
        pedir su trabajo.
        """
        if self.manager is None or agente.mission is None:
            return

        mision = self.manager.missions.pop(agente.mission, None)
        caja = self.inventory.get(mision.box) if mision is not None else None
        if caja is not None:
            caja.mission = None
            caja.node = agente.current_node if agente.carrying else caja.node
            caja.status = (
                missions.BoxStatus.DELIVERED
                if self.graph.role_of(caja.node) == ROLE_DOCK
                else missions.BoxStatus.STORED
            )
        agente.mission = None
        agente.carrying = None
        agente.leg = Leg.NONE

    def _aterriza_caja(self, mission, nodo: str) -> None:
        """Deja la caja en el nodo donde el AGV la solto y le pone su estado.

        En un muelle la caja termina su viaje; en una estanteria queda guardada,
        y al soltar su mision el manager le abrira la de salida en el paso
        siguiente. Ese es el encadenado de los dos flujos.
        """
        caja = self.inventory.get(mission.box)
        if caja is None:
            return
        caja.node = nodo
        caja.mission = None
        if self.graph.role_of(nodo) == ROLE_DOCK:
            caja.status = missions.BoxStatus.DELIVERED
        else:
            caja.status = missions.BoxStatus.STORED

    def _fase_bateria(self) -> None:
        """Los enchufados cargan; el que se quedo a cero se queda tirado."""
        for agente in self.agents:
            if agente.state == State.CHARGING:
                if agente.charge():
                    agente.charges += 1
                    agente.state = State.IDLE
                    log.debug("paso %3d | AGV %s | cargado al 100%%, vuelve a pujar",
                              self.step, agente.id)
            elif (
                agente.mission is not None
                and agente.leg is not Leg.TO_CHARGER
                and not agente.can_reach_charger(
                    self.graph.nodes_with_role(ROLE_CHARGING)
                )
            ):
                log.warning(
                    "AGV %s abandona %s al %.0f%%: no llegaria a un cargador",
                    agente.id, agente.mission, agente.battery,
                )
                self._suelta_mision(agente)
                self._al_cargador(agente)

            elif agente.is_dead():
                if agente.mission is not None:
                    log.warning("AGV %s se quedo sin bateria en %s con %s a medias",
                                agente.id, agente.current_node, agente.mission)
                    self._suelta_mision(agente)
                agente.state = State.IDLE

    def _al_cargador(self, agente: Agent) -> bool:
        """Le traza ruta al cargador mas cercano. False si no hay ninguno."""
        cargadores = self.graph.nodes_with_role(ROLE_CHARGING)
        if not cargadores:
            return False

        candidatos = sorted(
            (d, nodo)
            for nodo in cargadores
            if (d := agente.distance_to(nodo)) is not None
        )
        for _, nodo in candidatos:
            if agente.assign_task(
                agente.current_node, nodo, task=agente.task, penalties=self.penalties
            ):
                agente.leg = Leg.TO_CHARGER
                if len(agente.path) == 1:
                    agente.leg = Leg.NONE
                    agente.state = State.CHARGING
                self.bus.publish(missions.Message(
                    self.step, f"AGV-{agente.id}", "MANAGER",
                    missions.MessageType.CHARGING,
                    {"agv": agente.id, "estacion": nodo,
                     "bateria": round(agente.battery, 1)},
                ))
                log.debug("paso %3d | AGV %s | al %.0f%% se va a cargar a %s",
                          self.step, agente.id, agente.battery, nodo)
                return True
        return False

    def _fase_subasta(self) -> None:
        """El manager publica, cada AGV puja por su cuenta y se reparte lo ganado.

        Es todo el reparto de trabajo del almacen: aqui nadie asigna nada, el
        motor solo traslada al AGV lo que su propia puja le gano.
        """
        if self.manager is None:
            return

        nuevas = self.manager.open_work(self.inventory)
        for mision in nuevas:
            log.debug("paso %3d | MANAGER | abre %s (%s) caja %s en %s -> %s",
                      self.step, mision.id, mision.flow, mision.box,
                      mision.node, mision.destination)

        pendientes = self.manager.publish(self.step)
        if not pendientes:
            return

        cargadores = self.graph.nodes_with_role(ROLE_CHARGING)
        for agente in self.agents:
            agente.bid(self.bus, self.step, pendientes, cargadores)

        for agente, mision, utilidad in missions.resolve_auctions(
            self.bus, self.step, self.manager, self.agents
        ):
            caja = self.inventory.get(mision.box)
            if caja is None:
                log.error("la caja %r de %s ya no esta en el mapa", mision.box, mision.id)
                continue
            if agente.assign_delivery(
                agente.current_node, caja, mision.destination,
                task=agente.id, penalties=self.penalties,
            ):
                agente.mission = mision.id
                caja_viva = self.inventory.get(mision.box)
                if caja_viva is not None:
                    caja_viva.status = missions.BoxStatus.RESERVED
                if agente.has_arrived() and not self._empieza_maniobra(agente):
                    agente.state = State.DONE
                log.debug(
                    "paso %3d | AGV %s | GANA %s (caja %s en %s nivel %d -> %s) con %.1f",
                    self.step, agente.id, mision.id, caja.id, caja.node,
                    caja.level, mision.destination, utilidad,
                )
            else:
                mision.status = missions.MissionStatus.PENDING
                mision.agv_id = None
                log.warning(
                    "AGV %s gano %s pero no hay ruta a %s; vuelve a la bolsa",
                    agente.id, mision.id, caja.node,
                )

        for agente in self.agents:
            if (
                agente.mission is None
                and agente.state in (State.IDLE, State.DONE)
                and agente.needs_charge(pendientes, cargadores)
            ):
                self._al_cargador(agente)

    def _empieza_maniobra(self, agente: Agent) -> bool:
        """Pone a recoger o a dejar al que acaba de llegar. False si no tocaba."""
        if agente.leg == Leg.TO_PICK:
            caja = self.inventory.get(agente.box) if agente.box else None
            if caja is None:
                log.error(
                    "AGV %s: llego a por la caja %r, que ya no esta en el mapa",
                    agente.id,
                    agente.box,
                )
                return False
            agente.start_pick(caja.level)
            log.debug(
                "paso %3d | AGV %s | PICK %s en %s nivel %d | %d tick(s)",
                self.step,
                agente.id,
                caja.id,
                caja.node,
                caja.level,
                agente.busy,
            )
            return True

        if agente.leg == Leg.TO_CHARGER:
            agente.leg = Leg.NONE
            agente.state = State.CHARGING
            log.debug("paso %3d | AGV %s | enchufado en %s al %.0f%%",
                      self.step, agente.id, agente.current_node, agente.battery)
            return True

        if agente.leg == Leg.TO_DROP:
            agente.start_drop()
            log.debug(
                "paso %3d | AGV %s | DROP %s en %s | %d tick(s)",
                self.step,
                agente.id,
                agente.carrying,
                agente.current_node,
                agente.busy,
            )
            return True

        return False

    def _fase_maniobras(self) -> None:
        """Gasta un tick de cada recogida y de cada entrega en curso."""
        for agente in self.agents:
            if agente.state == State.PICKING:
                if agente.work():
                    agente.finish_pick()
                    self.picked += 1
                    if self.manager is not None and agente.mission:
                        mision = self.manager.missions.get(agente.mission)
                        if mision is not None:
                            self.manager.picked_up(self.step, mision, agente.id)
                            caja_viva = self.inventory.get(mision.box)
                            if caja_viva is not None:
                                caja_viva.status = missions.BoxStatus.IN_TRANSIT
                                caja_viva.node = agente.current_node
                    if agente.route_to_destination(self.penalties):
                        if len(agente.path) == 1:
                            agente.start_drop()
                        else:
                            agente.state = State.MOVING
                        log.debug(
                            "paso %3d | AGV %s | recogio %s, al muelle %s por %s",
                            self.step,
                            agente.id,
                            agente.carrying,
                            agente.destination,
                            " -> ".join(agente.path),
                        )
                    else:
                        agente.state = State.IDLE
                        log.warning(
                            "AGV %s: recogio %s pero no hay ruta al muelle %s",
                            agente.id,
                            agente.carrying,
                            agente.destination,
                        )

            elif agente.state == State.DROPPING and agente.work():
                entregada = agente.carrying
                agente.finish_drop()
                self.delivered += 1
                if self.manager is not None and agente.mission:
                    mision = self.manager.missions.get(agente.mission)
                    if mision is not None:
                        self.manager.finished(self.step, mision, agente.id)
                        self._aterriza_caja(mision, agente.current_node)
                    agente.completed += 1
                    agente.mission = None
                log.debug(
                    "paso %3d | AGV %s | entrego %s en %s",
                    self.step,
                    agente.id,
                    entregada,
                    agente.current_node,
                )

    def _fase_a_intenciones(self) -> dict[int, str]:
        """FASE A: cada agente parado dice a que nodo quiere entrar este tick."""
        intenciones: dict[int, str] = {}
        for agente in self.agents:
            if agente.state not in (State.MOVING, State.WAITING):
                continue
            if agente.progress > 0.0:
                continue

            siguiente = agente.next_node()
            if siguiente is None:
                continue

            if not self.graph.has_edge(agente.current_node, siguiente):
                log.error(
                    "AGV %s: su ruta pasa por %s -> %s, que no es una arista del mapa",
                    agente.id,
                    agente.current_node,
                    siguiente,
                )
                agente.state = State.IDLE
                agente.progress = 0.0
                continue

            intenciones[agente.id] = siguiente
        return intenciones

    def _fase_b_resuelve_y_aplica(self, intenciones: dict[int, str]) -> None:
        """FASE B: detectar -> decidir -> desatascar -> aplicar. En ese orden."""
        detectados = conflicts.detect_conflicts(
            intenciones,
            self.graph,
            occupancy=self.occupancy,
            agents=self.agents,
            step=self.step,
            previous_zones=self._zonas,
        )
        self._registra(detectados)
        bloqueado_por, suyos = self._arbitra(detectados)

        en_travesia = [agente for agente in self.agents if agente.progress > 0.0]

        acciones: dict[int, str] = {}
        for agente in self.agents:
            destino = intenciones.get(agente.id)
            if destino is None:
                continue
            acciones[agente.id] = conflicts.normalize_intent(
                self.policy.decide(
                    agente, self._estado_local(agente, destino, bloqueado_por, suyos)
                )
            )

        forzados = self._desatasca(intenciones, acciones)

        for agente in self.agents:
            destino = intenciones.get(agente.id)
            if destino is None:
                continue

            accion = acciones[agente.id]
            self._recuento[accion] += 1

            recalculo = (
                self._recalcula(agente)
                if accion == conflicts.Intent.REROUTE
                else None
            )

            quiere_pasar = accion == conflicts.Intent.ADVANCE
            pasa = quiere_pasar and self._puede_entrar(agente, destino)
            if pasa:
                self._empieza_travesia(agente, destino)
            else:
                self._cede_el_paso(agente)

            self._anota(
                agente,
                accion,
                blocked=quiere_pasar and not pasa,
                forced=agente.id in forzados,
                reroute=recalculo,
            )

        for agente in en_travesia:
            self._avanza(agente)

        cruzando = {agente.id for agente in en_travesia}
        for agente in self.agents:
            registro = self._acciones.get(agente.id)
            if registro is not None and registro.step == self.step:
                continue
            self._anota(
                agente,
                conflicts.Intent.ADVANCE
                if agente.id in cruzando or agente.progress > 0.0
                else conflicts.Intent.WAIT,
            )

        self.penalties.expire(self.step)
        self._reservas = {
            nodo: reserva
            for nodo, reserva in self._reservas.items()
            if self.step < reserva[1]
        }
        self._zonas = conflicts.congested_zones(self.agents, self.graph)

    def _arbitra(
        self, detectados: list[conflicts.Conflict]
    ) -> tuple[dict[int, list[int]], dict[int, list[conflicts.Conflict]]]:
        """Quien le ha ganado el paso a quien, y que choques lleva cada uno encima."""
        bloqueado_por: dict[int, list[int]] = {}
        suyos: dict[int, list[conflicts.Conflict]] = {}
        for conflicto in detectados:
            for agent_id in conflicto.agents:
                suyos.setdefault(agent_id, []).append(conflicto)
            resolucion = conflicts.resolve_baseline(conflicto)
            if resolucion.winner is None:
                continue
            for perdedor in resolucion.losers:
                bloqueado_por.setdefault(perdedor, []).append(resolucion.winner)
        return bloqueado_por, suyos

    def _anota(
        self,
        agente: Agent,
        accion: str,
        *,
        blocked: bool = False,
        forced: bool = False,
        reroute: tuple[tuple[str, ...], tuple[str, ...]] | None = None,
    ) -> None:
        """Deja constancia de lo que este AGV quiso y de lo que el motor le dio."""
        self._acciones[agente.id] = ActionRecord(
            action=accion,
            step=self.step,
            blocked=blocked,
            forced=forced,
            reroute=reroute,
        )

    def _estado_local(
        self,
        agente: Agent,
        destino: str,
        bloqueado_por: dict[int, list[int]],
        suyos: dict[int, list[conflicts.Conflict]],
    ) -> conflicts.LocalState:
        """Lo que la politica ve de este agente. Solo lo local, a proposito."""
        return conflicts.LocalState(
            step=self.step,
            node=agente.current_node,
            intent=destino,
            wait_time=agente.wait_time,
            blocked_by=tuple(sorted(bloqueado_por.get(agente.id, ()))),
            conflicts=tuple(suyos.get(agente.id, ())),
            occupancy=conflicts.read_only(self.occupancy),
            neighbors=tuple(self.graph.neighbors(agente.current_node)),
        )

    def _registra(self, detectados: list[conflicts.Conflict]) -> None:
        """Anota los conflictos de este tick y los cuenta por el log."""
        self.conflicts.extend(detectados)
        for conflicto in detectados:
            donde = (
                conflicto.node
                if conflicto.edge is None
                else " <-> ".join(conflicto.edge)
            )
            log.info(
                "paso %3d | CONFLICTO %-10s | AGV %s | %s",
                self.step,
                conflicto.type,
                ", ".join(str(agent_id) for agent_id in conflicto.agents),
                donde,
            )


    def _puede_entrar(self, agente: Agent, destino: str) -> bool:
        """Dice si el nodo esta libre **para este agente**. Sin excepciones."""
        ocupante = self.occupancy.get(destino)
        if ocupante is not None and ocupante != agente.id:
            return False

        reserva = self._reservas.get(destino)
        return reserva is None or reserva[0] == agente.id or self.step >= reserva[1]

    def _empieza_travesia(self, agente: Agent, destino: str) -> None:
        """Le concede el paso: reserva el destino y da el primer trozo de tramo.

        Reserva doble: se queda tambien con el nodo del que sale, y lo suelta
        solo al llegar. Un tramo que se cruza en un tick llega aqui mismo.
        """
        costo = self.graph.cost(agente.current_node, destino)
        self.occupancy[destino] = agente.id
        agente.state = State.MOVING
        agente.progress = 1.0 if costo <= 0.0 else 1.0 / costo
        if self.deliveries:
            agente.drain()
        self._llega_si_toca(agente, destino)

    def _avanza(self, agente: Agent) -> None:
        """Un tick mas de travesia para el que ya iba por el tramo."""
        destino = agente.next_node()
        if destino is None:
            return
        costo = self.graph.cost(agente.current_node, destino)
        agente.progress = 1.0 if costo <= 0.0 else agente.progress + 1.0 / costo
        if self.deliveries:
            agente.drain()
        self._llega_si_toca(agente, destino)

    def _mueve_la_carga(self, agente: Agent) -> None:
        """La caja que lleva encima va donde va el AGV, como el pallet de M3."""
        if not agente.carrying:
            return
        caja = self.inventory.get(agente.carrying)
        if caja is not None:
            caja.node = agente.current_node

    def _llega_si_toca(self, agente: Agent, destino: str) -> None:
        """Si el progreso paso de 1.0, planta al agente en el nodo y suelta el otro."""
        if agente.progress < 1.0:
            return

        if self.occupancy.get(agente.current_node) == agente.id:
            del self.occupancy[agente.current_node]

        agente.current_node = destino
        agente.path_index += 1
        agente.progress = 0.0
        self._mueve_la_carga(agente)
        if agente.has_arrived() and not self._empieza_maniobra(agente):
            agente.state = State.DONE

    def _cede_el_paso(self, agente: Agent) -> None:
        """Le toca esperar: no se mueve y suma un tick al reloj de la espera.

        `wait_time` **acumula**, no descuenta. Es el tiempo perdido de todo el
        AGV en la corrida, que es la medida con la que se comparan las politicas.
        """
        agente.state = State.WAITING
        agente.wait_time += 1


    def _recalcula(
        self, agente: Agent
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """Encarece lo que el AGV tiene delante y le pide a A* otra ruta."""
        if agente.next_node() == agente.target_node:
            log.debug(
                "AGV %s: no hay reroute que esquive %s, que es su destino",
                agente.id,
                agente.target_node,
            )
            return None

        if self.step < self._proximo_reroute.get(agente.id, 0):
            return None
        self._proximo_reroute[agente.id] = self.step + REROUTE_COOLDOWN

        vieja = tuple(agente.path)
        for clave, cuanto in conflicts.reroute_penalties(agente).items():
            self.penalties.add(clave, cuanto, step=self.step)

        nueva = conflicts.reroute(agente, self.graph, penalties=self.penalties)
        if nueva is None:
            return None

        log.debug(
            "paso %3d | AGV %s | REROUTE desde %s | %s -> %s",
            self.step,
            agente.id,
            agente.current_node,
            " -> ".join(vieja),
            " -> ".join(nueva),
        )
        return vieja, tuple(nueva)


    def _fuerza(
        self,
        agente: Agent,
        accion: str,
        intenciones: dict[int, str],
        acciones: dict[int, str],
        *,
        ceder: bool,
    ) -> None:
        """Saca a un AGV del reparto normal del tick: el motor ya decidio por el."""
        intenciones.pop(agente.id, None)
        acciones.pop(agente.id, None)
        self._recuento[accion] += 1
        self._forzados += 1
        if ceder:
            self._cede_el_paso(agente)
        self._anota(agente, accion, forced=True)

    def _desatasca(
        self, intenciones: dict[int, str], acciones: dict[int, str]
    ) -> set[int]:
        """Cuando alguien lleva demasiado sin poder moverse, el motor manda."""
        if DEADLOCK_FORCE_TICKS <= 0:
            return set()

        umbral = (
            DEADLOCK_FORCE_TICKS
            if self._ticks_sin_avance >= DEADLOCK_FORCE_TICKS
            else STARVED_TICKS
        )
        parados = {
            agent_id
            for agent_id in intenciones
            if self._parado.get(agent_id, 0) >= umbral
        }
        if not parados:
            return set()

        libre = {
            agent_id: self._puede_entrar(self._por_id[agent_id], intenciones[agent_id])
            for agent_id in parados
        }

        if any(
            acciones[agent_id] == conflicts.Intent.ADVANCE and esta_libre
            for agent_id, esta_libre in libre.items()
        ):
            return set()

        for agent_id in sorted(parados):
            if acciones[agent_id] == conflicts.Intent.ADVANCE:
                continue
            if not libre[agent_id]:
                continue
            acciones[agent_id] = conflicts.Intent.ADVANCE
            self._forzados += 1
            log.warning(
                "paso %3d | DESATASCO | el AGV %s lleva %d ticks sin poder moverse, "
                "le fuerzo el paso a %s",
                self.step,
                agent_id,
                self._parado.get(agent_id, 0),
                intenciones[agent_id],
            )
            return {agent_id}

        atascados = sorted(
            agent_id for agent_id, esta_libre in libre.items() if not esta_libre
        )
        if not atascados:
            return set()

        for agent_id in reversed(atascados):
            agente = self._por_id[agent_id]
            destino = intenciones[agent_id]
            vieja, indice = list(agente.path), agente.path_index
            tramo = (agente.current_node, destino)
            self.penalties.ban(destino, step=self.step)
            self.penalties.ban(tramo, step=self.step)

            conflicts.reroute(agente, self.graph, penalties=self.penalties)
            salida = agente.next_node()
            if (
                salida is not None
                and destino not in agente.path
                and self._puede_entrar(agente, salida)
            ):
                log.warning(
                    "paso %3d | DESATASCO | fuerzo el REROUTE del AGV %s: %s -> %s",
                    self.step,
                    agent_id,
                    " -> ".join(vieja),
                    " -> ".join(agente.path),
                )
                self._fuerza(
                    agente, conflicts.Intent.REROUTE, intenciones, acciones, ceder=True
                )
                return set()

            agente.path, agente.path_index = vieja, indice
            self.penalties.discard(destino)
            self.penalties.discard(tramo)

        for agent_id in atascados:
            estorbo = self.occupancy.get(intenciones[agent_id])
            if estorbo is None or estorbo == agent_id:
                continue

            cadena = self._hueco_mas_cercano(self._por_id[estorbo].current_node)
            if cadena is None:
                continue

            mover, hueco = cadena
            apartado = self._por_id[mover]
            if not self._aparta(apartado, hacia=hueco):
                continue

            self._reservas[apartado.current_node] = (
                agent_id,
                self.step + YIELD_TICKS,
            )
            log.warning(
                "paso %3d | DESATASCO | el AGV %s se aparta a %s (le deja %s al "
                "AGV %s) para descongestionar %s",
                self.step,
                mover,
                hueco,
                apartado.current_node,
                agent_id,
                self._por_id[estorbo].current_node,
            )
            self._fuerza(
                apartado, conflicts.Intent.REROUTE, intenciones, acciones, ceder=False
            )
            return set()

        log.warning(
            "paso %3d | DESATASCO | sin salida: no queda un solo nodo libre al que ir",
            self.step,
        )
        return set()

    def _hueco_mas_cercano(self, desde: str) -> tuple[int, str] | None:
        """El nodo libre mas cercano, y el AGV que tiene que meterse en el."""
        vistos = {desde}
        cola: deque[str] = deque([desde])
        while cola:
            nodo = cola.popleft()
            for vecino in self.graph.neighbors(nodo):
                if vecino in vistos:
                    continue
                vistos.add(vecino)
                if self.occupancy.get(vecino) is not None:
                    cola.append(vecino)
                    continue

                ocupante = self.occupancy.get(nodo)
                if ocupante is None:
                    continue
                return ocupante, vecino
        return None

    def _aparta(self, agente: Agent, *, hacia: str) -> bool:
        """Le da al que estorba una ruta de un tramo hasta el hueco de al lado."""
        if agente.progress > 0.0 or not self.graph.has_edge(agente.current_node, hacia):
            return False

        siguiente = agente.next_node()
        if siguiente is not None and self._puede_entrar(agente, siguiente):
            return False

        if agente.state == State.DONE or agente.target_node is None:
            agente.path = [agente.current_node, hacia]
            agente.target_node = hacia
        else:
            self.penalties.add(
                agente.current_node, config.REROUTE_PENALTY, step=self.step
            )
            resto = astar(self.graph, hacia, agente.target_node, self.penalties)
            if resto is None:
                return False
            agente.path = [agente.current_node, *resto]

        agente.path_index = 0
        agente.progress = 0.0
        agente.state = State.MOVING
        return True


    def _huella_por_agente(self) -> dict[int, tuple[str, float]]:
        """Donde esta cada AGV, uno a uno. Para saber quien no se ha movido."""
        return {
            agente.id: (agente.current_node, agente.progress) for agente in self.agents
        }

    def _cuenta_los_parados(self, antes: dict[int, tuple[str, float]]) -> None:
        """Suma un tick al que no se movio nada, y pone a cero al que si.

        El que llego, el que no tiene ruta y el que esta recogiendo o dejando una
            caja no cuentan como parados: quedarse quieto es justo lo que toca.
        """
        for agente in self.agents:
            if agente.state in (
                State.DONE, State.IDLE, State.PICKING,
                State.DROPPING, State.CHARGING,
            ):
                self._parado[agente.id] = 0
                continue
            if antes.get(agente.id) != (agente.current_node, agente.progress):
                self._parado[agente.id] = 0
                continue
            self._parado[agente.id] = self._parado.get(agente.id, 0) + 1

    def _huella(self) -> tuple[tuple[int, str, float], ...]:
        """Donde esta cada agente activo. Si no cambia en un tick, nadie avanzo."""
        return tuple(
            (agente.id, agente.current_node, agente.progress)
            for agente in self.agents
            if agente.state in (State.MOVING, State.WAITING)
        )

    def _vigila_el_deadlock(self, huella_antes: tuple[tuple[int, str, float], ...]) -> None:
        """Corta la corrida si nadie avanza durante `config.DEADLOCK_TICKS` ticks."""
        if self.finished_reason is not None:
            return

        if not huella_antes:
            self._ticks_sin_avance = 0
            return

        if self._huella() != huella_antes:
            self._ticks_sin_avance = 0
            return

        self._ticks_sin_avance += 1
        if self._ticks_sin_avance < config.DEADLOCK_TICKS:
            return

        self.finished_reason = FINISHED_DEADLOCK
        self.deadlocks += 1
        self._ticks_sin_avance = 0
        log.warning(
            "deadlock en el paso %d: %d ticks seguidos sin que avance nadie (%s)",
            self.step,
            config.DEADLOCK_TICKS,
            ", ".join(
                f"AGV {agente.id} en {agente.current_node}"
                for agente in self.agents
                if agente.state in (State.MOVING, State.WAITING)
            ),
        )


    def _describe(self, agente: Agent) -> dict[str, object]:
        """Un agente tal y como lo ve Unity."""
        registro = self._acciones.get(
            agente.id, ActionRecord(conflicts.Intent.WAIT, self.step)
        )
        px, py = self._posicion(agente)
        x, y, z = to_unity(px, py)
        return {
            "id": agente.id,
            "x": x,
            "y": y,
            "z": z,
            "rotation": self._rotacion(agente),
            "state": agente.state,
            "node": agente.current_node,
            "next_node": agente.next_node(),
            "path": list(agente.path),
            "task": agente.task,
            "wait_time": agente.wait_time,
            "action": registro.action,
            "blocked": registro.blocked,
            "leg": agente.leg,
            "mission": agente.mission,
            "box": agente.box,
            "destination": agente.destination,
            "carrying": agente.carrying,
            "busy": agente.busy,
            "battery": round(agente.battery, 1),
        }

    def _posicion(self, agente: Agent) -> tuple[float, float]:
        """Posicion logica, interpolada entre el nodo actual y el siguiente."""
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
        """Rumbo del agente en grados sobre el eje vertical de Unity."""
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
