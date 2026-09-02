"""La simulacion de verdad: AGVs recorriendo el grafo del almacen.

Sustituye a la `FakeSimulation` de la fase 1 y cumple el mismo contrato,
`protocol.Simulation` (`get_snapshot()` + `reset()`), asi que el servidor la
acepta sin cambiar una linea de transporte.

El movimiento es continuo: un agente tarda `cost(a, b)` ticks en cruzar un tramo
y su `progress` avanza `1/cost` por tick, asi que el snapshot lleva la posicion
ya interpolada entre los dos nodos y Unity no ve teletransportes.

Desde la fase 5 el tick va **en dos fases**, las dos dentro del mismo paso:

    FASE A  cada agente parado en un nodo declara a cual quiere entrar
    FASE B  se detectan los conflictos, la politica decide quien cede,
            y solo entonces se mueve a nadie

Que las dos fases quepan en un tick no es un detalle: declarar la intencion no
puede costar un paso extra, o un AGV solo tardaria el doble en cruzar el almacen
y las medidas de la fase 3 dejarian de valer.

La politica es **intercambiable** (`conflicts.Policy`). Por defecto entra
`BaselinePolicy`, que es la referencia experimental; el Q-Learning de la fase 8
se enchufa en el constructor sin tocar el motor.
"""

import math
import random
import threading
from collections.abc import Sequence
from typing import Any

import config
import conflicts
import protocol
from agent import STATE_DONE, STATE_IDLE, STATE_MOVING, STATE_WAITING, Agent
from graph import WarehouseGraph
from logs import get_logger

log = get_logger("simulation")

# Razon por la que una corrida se da por terminada antes de tiempo.
FINISHED_DEADLOCK: str = "deadlock"

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

    La invariante del almacen: **un nodo no tiene nunca dos agentes**. La
    sostiene `occupancy`, que es `nodo -> agent_id`. Lo contrario si vale: un
    agente a media travesia retiene los dos extremos del tramo y suelta el de
    salida solo cuando llega (reserva doble, explicada en `conflicts`).
    """

    def __init__(
        self,
        graph: WarehouseGraph,
        n_agents: int = 1,
        *,
        origin: str | None = None,
        target: str | None = None,
        seed: int = config.RANDOM_SEED,
        policy: conflicts.Policy | None = None,
        routes: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        if routes is not None:
            n_agents = len(routes)
        if n_agents < 1:
            raise ValueError(f"hace falta al menos un agente, no {n_agents}")

        nodos = graph.nodes()
        if n_agents > len(nodos):
            # No es un capricho: dos AGVs no pueden arrancar en el mismo nodo, y
            # sin nodos de sobra no hay reparto posible que respete la invariante.
            raise ValueError(
                f"no caben {n_agents} agentes en un mapa de {len(nodos)} nodo(s): "
                f"cada AGV necesita un nodo de salida para el solo"
            )

        self.graph: WarehouseGraph = graph
        self.step: int = 0
        self.seed: int = seed
        self.policy: conflicts.Policy = (
            policy if policy is not None else conflicts.BaselinePolicy()
        )
        self._lock = threading.RLock()

        por_defecto = default_route(graph)
        self._origen: str = origin if origin is not None else por_defecto[0]
        self._destino: str = target if target is not None else por_defecto[1]
        self._rutas: list[tuple[str, str]] = (
            self._comprueba_rutas(routes)
            if routes is not None
            else self._planea_rutas(n_agents)
        )

        self.agents: list[Agent] = [
            Agent(numero, graph, origen)
            for numero, (origen, _) in enumerate(self._rutas, start=1)
        ]

        # Estado de la fase 5. `deadlocks` es el unico que sobrevive al reset:
        # cuenta los atascos de la sesion entera, no los de la corrida.
        self.occupancy: dict[str, int] = {}
        self.conflicts: conflicts.ConflictLog = conflicts.ConflictLog()
        self.run: int = 0
        self.deadlocks: int = 0
        self.finished_reason: str | None = None
        self._ticks_sin_avance: int = 0
        self._zonas: frozenset[str] = frozenset()

        self.reset()

    def __repr__(self) -> str:
        return (
            f"Simulation(map={self.graph.name!r}, agents={len(self.agents)}, "
            f"step={self.step}, policy={self.policy.name!r})"
        )

    @property
    def done(self) -> bool:
        """True cuando ya ningun agente tiene nada que hacer, o hay deadlock."""
        with self._lock:
            if self.finished_reason is not None:
                return True
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
            self._cierra_los_que_llegaron()

            huella = self._huella()
            intenciones = self._fase_a_intenciones()
            self._fase_b_resuelve_y_aplica(intenciones)
            self._vigila_el_deadlock(huella)

            return self.step

    def get_snapshot(self) -> protocol.Snapshot:
        """Avanza un paso y devuelve el estado completo en coordenadas de Unity.

        Que la peticion sea la que mueve el mundo es el contrato de la fase 1:
        Python no empuja nada por su cuenta, y dos clientes a la vez ven pasos
        distintos porque cada `GET_STATE` consume su propio tick.

        Si la corrida anterior murio atascada se arranca otra aqui. El snapshot
        del atasco ya se entrego una vez, con `stats.finished_reason` puesto, asi
        que Unity llego a verlo; dejarlo congelado para siempre solo serviria
        para que pareciera que el servidor se ha colgado.
        """
        with self._lock:
            if self.finished_reason == FINISHED_DEADLOCK:
                log.warning(
                    "la corrida %d acabo en deadlock, arranco la %d",
                    self.run,
                    self.run + 1,
                )
                self.reset()

            self.tick()
            return {
                "step": self.step,
                "agents": [self._describe(agente) for agente in self.agents],
                # Lo que agrega la fase 5, por encima del formato congelado.
                "stats": self.stats(),
            }

    def stats(self) -> dict[str, Any]:
        """Los numeros de la corrida, que son con los que se compara el baseline.

        Todo aqui dentro es determinista y serializable: dos simulaciones con la
        misma semilla tienen que producir snapshots identicos, `stats` incluido.
        """
        with self._lock:
            return {
                "run": self.run,
                "policy": self.policy.name,
                "conflicts": self.conflicts.total,
                "conflicts_by_type": self.conflicts.by_type,
                "deadlocks": self.deadlocks,
                "waiting": sum(
                    1 for agente in self.agents if agente.state == STATE_WAITING
                ),
                "total_wait_time": sum(agente.wait_time for agente in self.agents),
                "finished_reason": self.finished_reason,
            }

    def conflict_records(self) -> list[dict[str, Any]]:
        """Los conflictos de la corrida en JSON: paso, tipo y agentes de cada uno."""
        with self._lock:
            return self.conflicts.records()

    def reset(self) -> None:
        """Vuelve al paso cero y reparte otra vez las mismas tareas.

        Determinista: con la misma semilla y el mismo mapa, dos simulaciones
        recien reiniciadas producen exactamente la misma secuencia de snapshots.
        Lo unico que no se borra es `deadlocks`, que cuenta atascos de la sesion.
        """
        with self._lock:
            self.step = 0
            self.run += 1
            self.finished_reason = None
            self._ticks_sin_avance = 0
            self._zonas = frozenset()
            self.conflicts.clear()
            self.occupancy = {}

            for agente, (origen, destino) in zip(self.agents, self._rutas):
                agente.reset()
                agente.assign_task(origen, destino, task=agente.id)
                self.occupancy[agente.current_node] = agente.id

            log.info(
                "simulacion reiniciada: mapa %s, %d agente(s), politica %s, corrida %d",
                self.graph.name or "(sin nombre)",
                len(self.agents),
                self.policy.name,
                self.run,
            )

    def _comprueba_rutas(
        self, routes: Sequence[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Acepta un reparto de tareas escrito a mano, si es que se sostiene.

        Es el gancho para montar escenarios: un cruce de frente, un embudo, lo
        que haga falta demostrar, sin depender de lo que saque la semilla. Los
        origenes tienen que ser distintos por la misma razon que en
        `_planea_rutas()`: dos AGVs en el mismo nodo rompen la invariante antes
        de que la simulacion llegue a mover nada.
        """
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
        """Reparte origen y destino, siempre igual para la misma semilla.

        El primer agente hace la ruta fija del mapa, que es la que se demuestra.
        Los demas salen del generador sembrado con `seed`.

        Origenes **sin repetir**, que es lo que cambia en la fase 5: si dos AGVs
        arrancan en el mismo nodo la invariante nace rota, antes de mover nada.
        Los destinos tampoco se repiten: un AGV aparcado encima del destino de
        otro lo bloquea para siempre, y eso seria un deadlock del reparto, no del
        trafico, que es lo que aqui interesa medir.
        """
        rutas = [(self._origen, self._destino)]
        nodos = self.graph.nodes()
        if len(nodos) < 2:
            # Un mapa de un solo nodo: no hay de donde sacar pares distintos.
            return rutas * n_agents
        if n_agents == 1:
            return rutas

        rng = random.Random(self.seed)
        origenes = rng.sample([n for n in nodos if n != self._origen], n_agents - 1)
        destinos = rng.sample([n for n in nodos if n != self._destino], n_agents - 1)
        rutas.extend(zip(origenes, destinos))
        return rutas

    # --- El tick, en dos fases ----------------------------------------------

    def _cierra_los_que_llegaron(self) -> None:
        """Pasa a `done` al que ya no tiene siguiente nodo. Antes de declarar nada."""
        for agente in self.agents:
            if agente.state == STATE_MOVING and agente.next_node() is None:
                agente.state = STATE_DONE
                agente.progress = 0.0

    def _fase_a_intenciones(self) -> dict[int, str]:
        """FASE A: cada agente parado dice a que nodo quiere entrar este tick.

        Solo declaran los que estan **parados en un nodo** (`progress == 0`). El
        que va a media travesia ya tiene su reserva concedida de un tick anterior
        y no vuelve a pedirla; el que espera si declara, porque sigue queriendo
        pasar y en cuanto le dejen, pasa.
        """
        intenciones: dict[int, str] = {}
        for agente in self.agents:
            if agente.state not in (STATE_MOVING, STATE_WAITING):
                continue
            if agente.progress > 0.0:
                continue

            siguiente = agente.next_node()
            if siguiente is None:
                continue

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
                continue

            intenciones[agente.id] = siguiente
        return intenciones

    def _fase_b_resuelve_y_aplica(self, intenciones: dict[int, str]) -> None:
        """FASE B: detectar -> resolver -> aplicar. En ese orden y sin saltarselo.

        El conflicto se ve **antes** de mover a nadie: la deteccion trabaja sobre
        las intenciones y sobre la ocupacion tal y como estaban al empezar el
        tick, asi que el perdedor de un choque termina el tick exactamente donde
        empezo.
        """
        detectados = conflicts.detect_conflicts(
            intenciones,
            self.graph,
            occupancy=self.occupancy,
            agents=self.agents,
            step=self.step,
            previous_zones=self._zonas,
        )
        self._registra(detectados)

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

        # Los que ya iban a media travesia se apuntan ahora, antes de tocar la
        # ocupacion: se mueven al final del tick para que las decisiones de los
        # que estan parados se tomen todas contra la misma foto del almacen. Si
        # se movieran primero, el que llega soltaria su nodo de salida a mitad de
        # tick y otro podria colarse detras: eso es el following, y no se permite.
        en_travesia = [agente for agente in self.agents if agente.progress > 0.0]

        for agente in self.agents:
            destino = intenciones.get(agente.id)
            if destino is None:
                continue

            accion = self.policy.decide(
                agente, self._estado_local(agente, destino, bloqueado_por, suyos)
            )
            # El gate fisico: diga lo que diga la politica, en un nodo ocupado no
            # se entra. Es lo que hace la invariante inviolable venga la politica
            # que venga, incluida la que aprenda la fase 8.
            if accion == conflicts.ACTION_GO and self._puede_entrar(agente, destino):
                self._empieza_travesia(agente, destino)
            else:
                self._cede_el_paso(agente)

        for agente in en_travesia:
            self._avanza(agente)

        self._zonas = conflicts.congested_zones(self.agents, self.graph)

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

    # --- Movimiento ----------------------------------------------------------

    def _puede_entrar(self, agente: Agent, destino: str) -> bool:
        """Dice si el nodo esta libre para este agente. Sin excepciones."""
        ocupante = self.occupancy.get(destino)
        return ocupante is None or ocupante == agente.id

    def _empieza_travesia(self, agente: Agent, destino: str) -> None:
        """Le concede el paso: reserva el destino y da el primer trozo de tramo.

        Reserva doble: se queda tambien con el nodo del que sale, y lo suelta
        solo al llegar. Un tramo que se cruza en un tick llega aqui mismo.
        """
        costo = self.graph.cost(agente.current_node, destino)
        self.occupancy[destino] = agente.id
        agente.state = STATE_MOVING
        # Un tramo de costo cero se cruza en el mismo tick, sin dividir por cero.
        agente.progress = 1.0 if costo <= 0.0 else 1.0 / costo
        self._llega_si_toca(agente, destino)

    def _avanza(self, agente: Agent) -> None:
        """Un tick mas de travesia para el que ya iba por el tramo."""
        destino = agente.next_node()
        if destino is None:
            return
        costo = self.graph.cost(agente.current_node, destino)
        agente.progress = 1.0 if costo <= 0.0 else agente.progress + 1.0 / costo
        self._llega_si_toca(agente, destino)

    def _llega_si_toca(self, agente: Agent, destino: str) -> None:
        """Si el progreso paso de 1.0, planta al agente en el nodo y suelta el otro."""
        if agente.progress < 1.0:
            return

        if self.occupancy.get(agente.current_node) == agente.id:
            del self.occupancy[agente.current_node]

        agente.current_node = destino
        agente.path_index += 1
        agente.progress = 0.0
        if agente.has_arrived():
            agente.state = STATE_DONE

    def _cede_el_paso(self, agente: Agent) -> None:
        """Le toca esperar: no se mueve y suma un tick al reloj de la espera.

        `wait_time` **acumula**, no descuenta. Es el tiempo perdido de todo el
        AGV en la corrida, que es la medida con la que se comparan las politicas.
        """
        agente.state = STATE_WAITING
        agente.wait_time += 1

    # --- Deadlock ------------------------------------------------------------

    def _huella(self) -> tuple[tuple[int, str, float], ...]:
        """Donde esta cada agente activo. Si no cambia en un tick, nadie avanzo."""
        return tuple(
            (agente.id, agente.current_node, agente.progress)
            for agente in self.agents
            if agente.state in (STATE_MOVING, STATE_WAITING)
        )

    def _vigila_el_deadlock(self, huella_antes: tuple[tuple[int, str, float], ...]) -> None:
        """Corta la corrida si nadie avanza durante `config.DEADLOCK_TICKS` ticks.

        Sin agentes activos no hay deadlock: que hayan llegado todos no es un
        atasco, es el final feliz. Y una simulacion colgada para siempre no es un
        resultado experimental, es un bug de la corrida.

        Una corrida ya declarada muerta no se vuelve a declarar: si no, seguir
        tickeando sobre un atasco sumaria un deadlock cada `DEADLOCK_TICKS` y el
        contador de la sesion dejaria de contar atascos para contar ticks.
        """
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
                if agente.state in (STATE_MOVING, STATE_WAITING)
            ),
        )

    # --- Como lo ve Unity ----------------------------------------------------

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
            # Lo que agrega la fase 5.
            "wait_time": agente.wait_time,
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
