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
`BaselinePolicy`, que es la referencia experimental; el Q-Learning entra por el
mismo hueco, y desde la fase 8 basta con nombrarlo:

    Simulation(grafo, 4, policy="qlearning", model="python/models/q_table.json")

--- Fase 8: A* dice POR DONDE, Q-Learning dice QUE HACER AHORA ---

El bucle de cada AGV en cada tick, entero:

    1. recibe la tarea origen -> destino          Simulation._planea_rutas / routes
    2. A* traza el path                           Agent.assign_task -> astar.astar
    3. consulta el siguiente nodo del path        FASE A, _fase_a_intenciones
    4. construye el estado local (5 enteros)      qlearning.get_local_state
    5. la politica elige ADVANCE/WAIT/REROUTE     policy.decide   <- lo unico que cambia de modo
    6. ADVANCE y es seguro  -> se mueve           _puede_entrar + _empieza_travesia
    7. WAIT                 -> espera y acumula   _cede_el_paso
    8. REROUTE              -> penaliza y A* otra vez   _recalcula
    9. repetir hasta completar la tarea

**Ninguna accion elige un nodo.** La ruta la traza A*; lo que se aprende es solo
que conviene hacer ahora con la ruta que ya se tiene. Y una accion es una
INTENCION, no una garantia: el motor sigue siendo la autoridad y el gate fisico
puede negarla, con lo que el AGV se queda donde estaba y (al entrenar) cobra el
-20 de haberlo intentado.

**`policy` es la unica variable experimental.** Con `baseline` y con `qlearning`
corre exactamente el mismo motor: mismo mapa, mismas rutas, misma semilla, misma
deteccion de conflictos, mismo desatasco y misma caducidad de penalizaciones. Si
cambiara algo mas, comparar las dos corridas no mediria la politica.
"""

import math
import random
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astar
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


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Que decidio un AGV en un tick, y que le concedio el motor.

    Los dos lados van juntos a proposito: `action` es lo que la politica QUISO y
    `blocked` es lo que el motor CONTESTO. Esa diferencia es la fase 8 entera, y
    es lo que sale al snapshot para que Unity la pueda pintar y lo que el
    entrenamiento cobra a -20.

    `forced` marca al que el desatasco obligo a hacer algo distinto de lo que
    eligio. `reroute` lleva la (ruta vieja, ruta nueva) si este tick hubo
    recalculo, que es lo que `qlearning.is_useless_reroute()` necesita.
    """

    action: str
    step: int
    blocked: bool = False
    forced: bool = False
    reroute: tuple[tuple[str, ...], tuple[str, ...]] | None = None


def make_policy(
    name: str,
    *,
    model: str | Path | None = None,
    seed: int = config.RANDOM_SEED,
) -> conflicts.Policy:
    """Monta una politica por su nombre: `"baseline"` o `"qlearning"`.

    Es lo que permite que el modo sea un parametro y no un import: `main.py serve
    --policy qlearning` y el comando `SET_MODE` pasan por aqui, y ninguno de los
    dos sabe nada de Q-tables.

    El import de `qlearning` es **perezoso** y no puede ser de otra forma:
    `qlearning` importa este modulo arriba del todo, asi que al reves solo cabe
    diferido. Cuando esta funcion corre, `qlearning` ya esta cargado o se carga
    sin ciclo, porque `simulation` ya termino de importarse.

    Sin modelo legible lanza `ValueError` en vez de servir una Q-table vacia: una
    tabla a ceros siempre avanza, o sea que pareceria funcionar sin haber
    aprendido nada, y eso es peor que un error.
    """
    nombre = str(name).strip().lower()

    if nombre == config.POLICY_BASELINE:
        return conflicts.BaselinePolicy()

    if nombre == config.POLICY_QLEARNING:
        import qlearning  # perezoso: ver el docstring

        ruta = Path(model) if model is not None else config.Q_TABLE_FILE
        if not ruta.is_file():
            raise ValueError(
                f"no existe la Q-table {ruta}; entrenala antes con: "
                f"python3 python/main.py train --map {config.DEFAULT_MAP}"
            )
        # La politica se sirve con el action set con el que se ENTRENO, no con
        # el que diga `config.ENABLE_REROUTE`. Servir un modelo fuera de su
        # action set no da error: da una politica que elige a ciegas la columna
        # que nunca aprendio, porque esa sigue a ceros y el cero le gana a todo
        # lo aprendido (la recompensa del almacen es casi toda negativa).
        entrenada_con_reroute = qlearning.trained_enable_reroute(ruta)
        if entrenada_con_reroute is False and config.ENABLE_REROUTE:
            log.info(
                "%s se entreno sin REROUTE: lo dejo fuera tambien al servir",
                ruta,
            )

        visitas = qlearning.load_action_visits(ruta)
        if not visitas and config.SERVE_MIN_VISITS > 0:
            log.warning(
                "%s no guarda las visitas por celda: sin ellas no se puede "
                "filtrar lo que la tabla no llego a aprender (reentrenala)",
                ruta,
            )

        return qlearning.QLearningPolicy(
            qlearning.QTable.load(ruta),
            epsilon=config.SERVE_EPSILON,
            seed=seed,
            enable_reroute=entrenada_con_reroute,
            visits=visitas,
            min_visits=config.SERVE_MIN_VISITS,
        )

    raise ValueError(
        f"no conozco la politica {name!r}; las que hay son "
        + ", ".join(config.POLICIES)
    )


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
        policy: conflicts.Policy | str | None = None,
        model: str | Path | None = None,
        routes: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        """`policy` acepta el nombre del modo (`"baseline"` / `"qlearning"`) o un
        objeto que cumpla `conflicts.Policy`. Con un nombre lo monta
        `make_policy()` y la simulacion se lo ata con `bind()`, para que el
        estado local salga exacto; con un objeto se usa tal cual y atarlo es
        cosa de quien lo construyo (asi sigue valiendo lo de la fase 6/7).
        """
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
        self._model: str | Path | None = model
        self.policy: conflicts.Policy = conflicts.BaselinePolicy()
        # `mode` es el nombre de la politica activa, y es lo que sale al
        # snapshot: la UNICA variable experimental de la fase 8.
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

        # Estado de la fase 8. Las penalizaciones son UNA tabla compartida por
        # todo el almacen: que G este congestionado es un hecho del mapa, no una
        # opinion de un AGV. Y caducan, o el mapa se degradaria para siempre.
        self.penalties: astar.TemporaryPenalties = astar.TemporaryPenalties()
        self._acciones: dict[int, ActionRecord] = {}
        # nodo -> (a quien se le reservo, en que paso caduca). Ver `_puede_entrar`.
        self._reservas: dict[str, tuple[int, int]] = {}
        # Ticks seguidos que lleva cada AGV sin moverse ni un poco. Es lo que
        # dispara el desatasco, y va por agente y no por almacen: un AGV puede
        # morirse de hambre en un rincon mientras los demas dan vueltas, y un
        # contador global de "no se movio nadie" no lo veria nunca.
        self._parado: dict[int, int] = {}
        # En que paso podra volver a recalcular cada AGV. Ver `_recalcula`.
        self._proximo_reroute: dict[int, int] = {}
        self._recuento: dict[str, int] = dict.fromkeys(conflicts.INTENTS, 0)
        self._forzados: int = 0
        self._por_id: dict[int, Agent] = {a.id: a for a in self.agents}

        # Lo ultimo: `bind()` tiene que recibir una simulacion ya montada.
        if policy is not None:
            self._monta_politica(policy)

        self.reset()

    def _monta_politica(self, policy: conflicts.Policy | str) -> None:
        """Deja lista la politica, venga por nombre o ya construida."""
        if isinstance(policy, str):
            self.policy = make_policy(policy, model=self._model, seed=self.seed)
            # Solo se ata la que monta la simulacion: una politica que le llega
            # ya construida es de quien la construyo, y atarla por detras le
            # cambiaria el estado que ve sin habermelo pedido.
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
            por_agente = self._huella_por_agente()
            intenciones = self._fase_a_intenciones()
            self._fase_b_resuelve_y_aplica(intenciones)
            self._cuenta_los_parados(por_agente)
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
                # Lo que agrega la fase 8: que politica esta corriendo. Es lo
                # mismo que `stats.policy`, pero arriba del todo, que es donde
                # Unity lo quiere para pintarlo en pantalla sin bucear.
                "mode": self.mode,
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
                # Lo que agrega la fase 8.
                "actions": dict(self._recuento),
                "forced": self._forzados,
                "penalties": len(self.penalties),
            }

    def conflict_records(self) -> list[dict[str, Any]]:
        """Los conflictos de la corrida en JSON: paso, tipo y agentes de cada uno."""
        with self._lock:
            return self.conflicts.records()

    def set_mode(self, mode: str, *, model: str | Path | None = None) -> str:
        """Cambia de politica **en caliente** y arranca una corrida limpia.

        Es lo que hay detras del comando `SET_MODE` del protocolo: en mitad de
        una demo se pasa de `baseline` a `qlearning` sin reiniciar el servidor ni
        que Unity se entere de nada mas que de un `run` nuevo.

        **Siempre resetea**, incluso si el modo pedido es el que ya estaba: media
        corrida con una politica y media con otra no es una corrida de ninguna de
        las dos, y no habria forma de leer sus numeros. Lo unico que sobrevive es
        `deadlocks`, que cuenta los de la sesion.

        Lanza `ValueError` si el modo no existe o si falta la Q-table, y en ese
        caso **no toca nada**: la corrida que estaba en marcha sigue como estaba.
        """
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

            # Lo de la fase 8. Las penalizaciones no se heredan de una corrida a
            # otra: el atasco que las provoco ya no existe.
            self.penalties.clear()
            self._acciones.clear()
            self._reservas.clear()
            self._parado.clear()
            self._proximo_reroute.clear()
            self._recuento = dict.fromkeys(conflicts.INTENTS, 0)
            self._forzados = 0

            for agente, (origen, destino) in zip(self.agents, self._rutas):
                agente.reset()
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
        """FASE B: detectar -> decidir -> desatascar -> aplicar. En ese orden.

        El conflicto se ve **antes** de mover a nadie: la deteccion trabaja sobre
        las intenciones y sobre la ocupacion tal y como estaban al empezar el
        tick, asi que el perdedor de un choque termina el tick exactamente donde
        empezo.

        Los cuatro tramos, y quien manda en cada uno:

            deteccion      el motor, igual que en la fase 5
            decision       LA POLITICA: advance / wait / reroute   <- lo unico que cambia de modo
            desatasco      el motor, que puede pisar la decision
            aplicacion     el motor, con el gate fisico por delante

        Que la decision este en medio y no al final es la fase 8 entera: la
        politica propone y el motor dispone.
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
        bloqueado_por, suyos = self._arbitra(detectados)

        # Los que ya iban a media travesia se apuntan ahora, antes de tocar la
        # ocupacion: se mueven al final del tick para que las decisiones de los
        # que estan parados se tomen todas contra la misma foto del almacen. Si
        # se movieran primero, el que llega soltaria su nodo de salida a mitad de
        # tick y otro podria colarse detras: eso es el following, y no se permite.
        en_travesia = [agente for agente in self.agents if agente.progress > 0.0]

        # PASOS 4 y 5: cada AGV parado construye su estado local y elige. Es una
        # INTENCION: aqui no se mueve nadie todavia.
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

        # El motor manda: si lleva K ticks sin avanzar nadie, pisa la intencion.
        forzados = self._desatasca(intenciones, acciones)

        # PASOS 6, 7 y 8: el motor aplica lo que se pueda aplicar.
        for agente in self.agents:
            destino = intenciones.get(agente.id)
            if destino is None:
                continue

            accion = acciones[agente.id]
            self._recuento[accion] += 1

            # PASO 8: el REROUTE lo ejecuta el MOTOR, no la politica. No mueve al
            # AGV en este tick (la intencion ya se fijo en la fase A), asi que la
            # ruta nueva empieza a valer en el siguiente y este cuesta una espera.
            recalculo = (
                self._recalcula(agente)
                if accion == conflicts.INTENT_REROUTE
                else None
            )

            # PASO 6: el gate fisico. Diga lo que diga la politica, en un nodo
            # ocupado no se entra. Es lo que hace la invariante inviolable venga
            # la politica que venga, y lo que convierte ADVANCE en una intencion.
            quiere_pasar = accion == conflicts.INTENT_ADVANCE
            pasa = quiere_pasar and self._puede_entrar(agente, destino)
            if pasa:
                self._empieza_travesia(agente, destino)
            else:
                # PASO 7: se queda, y el reloj de la espera corre.
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

        # El que no decidio nada tambien sale al snapshot: el que venia cruzando
        # un tramo esta EJECUTANDO un avance —tambien el que lo termina en este
        # tick, que si no aparecería esperando justo al llegar—, y el que llego o
        # no tiene ruta no esta haciendo nada, que en el vocabulario de las
        # acciones es esperar.
        cruzando = {agente.id for agente in en_travesia}
        for agente in self.agents:
            registro = self._acciones.get(agente.id)
            if registro is not None and registro.step == self.step:
                continue
            self._anota(
                agente,
                conflicts.INTENT_ADVANCE
                if agente.id in cruzando or agente.progress > 0.0
                else conflicts.INTENT_WAIT,
            )

        # Las penalizaciones del REROUTE caducan. Sin esto el mapa se degrada
        # para siempre y A* acaba esquivando pasillos que llevan cien ticks
        # libres solo porque una vez hubo alguien delante.
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

    # --- Movimiento ----------------------------------------------------------

    def _puede_entrar(self, agente: Agent, destino: str) -> bool:
        """Dice si el nodo esta libre **para este agente**. Sin excepciones.

        Dos motivos para que no lo este: que lo ocupe otro, que es la invariante
        de siempre, o que el motor se lo tenga reservado a otro. La reserva la
        pone el desatasco: cuando manda apartarse a un AGV, le guarda el hueco al
        que estaba esperando durante `config.YIELD_TICKS` ticks.

        Sin la reserva el desatasco no serviria de nada en el caso mas comun: el
        que se aparta vuelve al tick siguiente, gana el desempate por id menor y
        el atasco se rehace igual. Apartarse y volver es no apartarse.
        """
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

    # --- El REROUTE, ejecutado por el motor ----------------------------------

    def _recalcula(
        self, agente: Agent
    ) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """Encarece lo que el AGV tiene delante y le pide a A* otra ruta.

        Devuelve (ruta vieja, ruta nueva), o None si no habia por donde o si el
        agente iba a media travesia. Aqui esta el reparto de trabajo de la fase 8
        en dos lineas: **la penalizacion dice que evitar y A* dice por donde**;
        el Q-Learning no elige ni un nodo.

        La penalizacion va a la tabla compartida de la simulacion, no a un dict
        de usar y tirar: asi dura `config.PENALTY_TTL` ticks y el AGV no se vuelve
        a meter en el mismo atasco en el tick siguiente. Y caduca, que es la otra
        mitad de la idea.
        """
        # Si el nodo que tengo delante ES mi destino, no hay ruta que lo
        # esquive: todas acaban ahi. Recalcular solo daria una vuelta larga para
        # volver al mismo sitio, y con dos AGVs sentados en el destino del otro
        # eso es una persecucion en circulo que no termina nunca. Se cobra el
        # tick de espera igual, que elegir REROUTE aqui fue un error.
        if agente.next_node() == agente.target_node:
            log.debug(
                "AGV %s: no hay reroute que esquive %s, que es su destino",
                agente.id,
                agente.target_node,
            )
            return None

        # Recalcular cuesta, y dos recalculos seguidos no llevan a ningun sitio:
        # el primero penaliza el nodo de delante y A* da la otra salida, el
        # segundo penaliza esa y devuelve la primera. Un ir y venir que ademas
        # encarece medio mapa. Durante la pausa, un REROUTE es esperar.
        if self.step < self._proximo_reroute.get(agente.id, 0):
            return None
        self._proximo_reroute[agente.id] = self.step + config.REROUTE_COOLDOWN

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

    # --- Desatasco: el motor no deja que el almacen se pare -------------------

    def _fuerza(
        self,
        agente: Agent,
        accion: str,
        intenciones: dict[int, str],
        acciones: dict[int, str],
        *,
        ceder: bool,
    ) -> None:
        """Saca a un AGV del reparto normal del tick: el motor ya decidio por el.

        Hay que sacarlo, no solo cambiarle la accion: al desatascar se le reescribe
        la ruta, y la intencion que declaro en la FASE A apuntaba a la ruta vieja.
        Dejarsela puesta le haria entrar en un nodo que ya no es el suyo, y ahi se
        rompe la invariante del almacen.
        """
        intenciones.pop(agente.id, None)
        acciones.pop(agente.id, None)
        self._recuento[accion] += 1
        self._forzados += 1
        if ceder:
            # El que queria pasar y acaba recalculando pierde el tick igual.
            self._cede_el_paso(agente)
        self._anota(agente, accion, forced=True)

    def _desatasca(
        self, intenciones: dict[int, str], acciones: dict[int, str]
    ) -> set[int]:
        """Cuando alguien lleva demasiado sin poder moverse, el motor manda.

        Mide "no se ha movido" en vez de "eligio WAIT", que cubre lo mismo y
        ademas el caso que la otra lectura se deja fuera: todos eligiendo ADVANCE
        contra un bloqueo mutuo. Son **dos atascos distintos y dos umbrales**:

        - El almacen entero parado, `DEADLOCK_FORCE_TICKS` (8) ticks sin que se
          mueva nadie. Salta muy antes de los `DEADLOCK_TICKS` (20) con los que
          la corrida se da por muerta: llegados a 20 ya no hay nada que salvar, y
          de lo que se trata es de no llegar.
        - Un AGV solo, clavado `STARVED_TICKS` (45) ticks mientras los demas
          circulan. El contador global no lo ve nunca, porque el almacen SI se
          esta moviendo, y ese se muere de hambre en silencio.

        Con `DEADLOCK_FORCE_TICKS` a 0 el desatasco se apaga entero y el motor
        vuelve a ser el de la fase 5.

        La escalada tiene tres peldaños, y se prueban en orden hasta que uno
        cambia algo:

            1. pasa el de id menor que tenga el nodo libre    (el desempate por prioridad)
            2. veto temporal y REROUTE al de id mayor         (si de verdad hay otra ruta)
            3. el que estorba se aparta a un hueco libre      (aunque ya haya terminado)

        El peldaño 3 no es un primitivo de movimiento nuevo: es una ruta de un
        tramo que pasa por el gate como cualquier otra, asi que la invariante
        "un nodo, un AGV" sigue intacta.

        Devuelve los ids a los que el motor forzo algo en el bucle de aplicacion
        (el peldaño 1); los otros dos se apuntan ellos mismos con `_fuerza`. Todos
        quedan marcados `forced` para que el entrenamiento les cobre el -20: la
        leccion es que quedarse parado esperando a que pase algo sale caro.

        Lo que **no** resuelve: si no queda un solo nodo libre en todo ese lado
        del mapa no hay a donde apartarse, y la corrida acaba muriendo como en la
        fase 5. Hace falta llenar un componente entero para llegar ahi.
        """
        if config.DEADLOCK_FORCE_TICKS <= 0:
            return set()

        # Dos atascos distintos, dos umbrales:
        #
        #   - El almacen entero parado: nadie se movio en DEADLOCK_FORCE_TICKS.
        #     Es el atasco clasico y hay que deshacerlo ya.
        #   - Un AGV solo, clavado mientras los demas circulan: el contador
        #     global no lo ve, porque el almacen SI se mueve. Ese se muere de
        #     hambre en silencio, y por eso hay tambien un contador por AGV, con
        #     el umbral mas alto: esperar un rato es normal, esperar treinta
        #     ticks es que nadie te va a dejar pasar nunca.
        umbral = (
            config.DEADLOCK_FORCE_TICKS
            if self._ticks_sin_avance >= config.DEADLOCK_FORCE_TICKS
            else config.STARVED_TICKS
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

        # Si el atascado ya va a poder pasar por su cuenta, aqui no hay nada que
        # hacer: meter mano seria pisarle la ruta al que ya iba bien. Pasa en el
        # tick siguiente a un desatasco, con el AGV apartado saliendo al hueco.
        if any(
            acciones[agent_id] == conflicts.INTENT_ADVANCE and esta_libre
            for agent_id, esta_libre in libre.items()
        ):
            return set()

        # 1. El desempate por prioridad: gana el id menor, igual que en la
        #    baseline. Basta con uno moviendose para que el almacen se descongele.
        for agent_id in sorted(parados):
            if acciones[agent_id] == conflicts.INTENT_ADVANCE:
                continue  # ya lo estaba intentando; forzarselo no cambiaria nada
            if not libre[agent_id]:
                continue
            acciones[agent_id] = conflicts.INTENT_ADVANCE
            self._forzados += 1
            log.warning(  # el resto del tick lo aplica el bucle de siempre
                "paso %3d | DESATASCO | el AGV %s lleva %d ticks sin poder moverse, "
                "le fuerzo el paso a %s",
                self.step,
                agent_id,
                self._parado.get(agent_id, 0),
                intenciones[agent_id],
            )
            return {agent_id}

        # De aqui en adelante, solo los que estan bloqueados de verdad: al que
        # tiene el nodo libre delante no hay que recalcularle nada.
        atascados = sorted(
            agent_id for agent_id, esta_libre in libre.items() if not esta_libre
        )
        if not atascados:
            return set()

        # 2. Nadie puede pasar: que el de menos prioridad busque otra ruta, con
        #    veto sobre lo que tiene delante para que A* no se la devuelva igual.
        #
        #    Sirve solo si la ruta nueva **esquiva el nodo en disputa del todo**
        #    y ademas sale por uno libre. Que la ruta cambie no basta: en un mapa
        #    con un cuello de botella A* devuelve encantado otro camino que
        #    vuelve a pasar por el mismo sitio ocupado, y con eso los dos AGVs se
        #    pasan el dia dando vueltas alrededor del que estorba en vez de
        #    desatascarse. Cuando el nodo en disputa es el destino del AGV no hay
        #    ruta que lo esquive, y ahi el unico arreglo es el peldaño 3.
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
                    agente, conflicts.INTENT_REROUTE, intenciones, acciones, ceder=True
                )
                return set()

            # No habia salida: se deshace todo. La ruta vuelve a ser la que era y
            # el veto se retira en vez de dejarlo caducar, o encareceria el mapa
            # para los demas durante PENALTY_TTL ticks a cambio de nada.
            agente.path, agente.path_index = vieja, indice
            self.penalties.discard(destino)
            self.penalties.discard(tramo)

        # 3. Ultimo recurso: que se aparte el que estorba, aunque ya haya
        #    terminado su tarea. Un AGV aparcado encima de G bloquea el almacen
        #    entero, y por ahi no hay ruta alternativa que valga.
        #
        #    Casi nunca puede apartarse el mismo: en un atasco de verdad tiene
        #    los vecinos ocupados tambien. Por eso se busca el hueco libre mas
        #    cercano y se empuja al ULTIMO de la fila, que es el unico que cabe
        #    en el. La fila se acorta en un AGV por cada desatasco, y a la
        #    segunda o la tercera le toca al que estorbaba.
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

            # El hueco que deja se le guarda al que esperaba: si no, el que se
            # aparta vuelve, gana por id menor y todo esto no habra servido.
            self._reservas[apartado.current_node] = (
                agent_id,
                self.step + config.YIELD_TICKS,
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
            # Sin `ceder`: no estaba esperando a nadie, le han mandado moverse.
            self._fuerza(
                apartado, conflicts.INTENT_REROUTE, intenciones, acciones, ceder=False
            )
            return set()

        log.warning(
            "paso %3d | DESATASCO | sin salida: no queda un solo nodo libre al que ir",
            self.step,
        )
        return set()

    def _hueco_mas_cercano(self, desde: str) -> tuple[int, str] | None:
        """El nodo libre mas cercano, y el AGV que tiene que meterse en el.

        Un BFS por el mapa desde `desde`, atravesando **solo nodos ocupados**:
        es literalmente recorrer la fila de AGVs atascados hasta encontrar donde
        se acaba. Devuelve `(id del ultimo de la fila, nodo libre)`, que es el
        unico par que puede moverse ahora mismo; los de mas atras no caben en
        ningun sitio hasta que este se quite.

        None si el componente entero esta lleno, que es el atasco que ya no tiene
        arreglo: sin un hueco no hay a donde apartarse.
        """
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

                # `nodo` es el ultimo ocupado de la fila y `vecino` el hueco.
                ocupante = self.occupancy.get(nodo)
                if ocupante is None:
                    continue
                return ocupante, vecino
        return None

    def _aparta(self, agente: Agent, *, hacia: str) -> bool:
        """Le da al que estorba una ruta de un tramo hasta el hueco de al lado.

        Es la "marcha atras" que el proyecto no tenia, y esta escrita como lo que
        es: **una ruta**, no un movimiento especial. El AGV la recorre en el tick
        siguiente pasando por el gate como cualquier otro, asi que la invariante
        "un nodo, un AGV" no se toca y nadie se teletransporta.

        Al que sigue en marcha se le pega el resto de su ruta detras, recalculada
        desde el hueco, para que no pierda la tarea. Al que ya habia terminado se
        le mueve el destino con el: se aparca en el hueco y ahi se queda, que es
        exactamente lo que hace un AGV real cuando le piden el pasillo.

        Devuelve False si iba a media travesia (ya se esta quitando de en medio),
        si el hueco no es vecino suyo o si desde alli no hay forma de llegar a su
        destino.
        """
        if agente.progress > 0.0 or not self.graph.has_edge(agente.current_node, hacia):
            return False

        # Si ya tiene por donde salir, no hace falta mandarle nada: se va solo en
        # cuanto le toque. Repetirle la orden cada tick seria peor que no darla,
        # porque cada vez se le saca del reparto del tick y no llega a moverse
        # nunca: se quedaria apartandose eternamente sin apartarse.
        siguiente = agente.next_node()
        if siguiente is not None and self._puede_entrar(agente, siguiente):
            return False

        if agente.state == STATE_DONE or agente.target_node is None:
            # Ya habia terminado: se aparca en el hueco y ahi se queda.
            agente.path = [agente.current_node, hacia]
            agente.target_node = hacia
        else:
            # Se penaliza el nodo que deja **antes** de trazar la vuelta, o A*
            # le devolveria la misma ruta y volveria derecho al atasco del que
            # acaba de sacarsele: apartarse y volver es no apartarse.
            self.penalties.add(
                agente.current_node, config.REROUTE_PENALTY, step=self.step
            )
            resto = astar.astar(self.graph, hacia, agente.target_node, self.penalties)
            if resto is None:
                return False
            agente.path = [agente.current_node, *resto]

        agente.path_index = 0
        agente.progress = 0.0
        agente.state = STATE_MOVING
        return True

    # --- Deadlock ------------------------------------------------------------

    def _huella_por_agente(self) -> dict[int, tuple[str, float]]:
        """Donde esta cada AGV, uno a uno. Para saber quien no se ha movido."""
        return {
            agente.id: (agente.current_node, agente.progress) for agente in self.agents
        }

    def _cuenta_los_parados(self, antes: dict[int, tuple[str, float]]) -> None:
        """Suma un tick al que no se movio nada, y pone a cero al que si.

        El que ya llego o no tiene ruta no cuenta como parado: no esta atascado,
        es que no tiene nada que hacer. Al que se aparta se le perdona tambien el
        contador, que bastante tiene con la vuelta que le han mandado dar.
        """
        for agente in self.agents:
            if agente.state in (STATE_DONE, STATE_IDLE):
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
            if agente.state in (STATE_MOVING, STATE_WAITING)
        )

    def _vigila_el_deadlock(self, huella_antes: tuple[tuple[int, str, float], ...]) -> None:
        """Corta la corrida si nadie avanza durante `config.DEADLOCK_TICKS` ticks.

        Este contador tiene **dos umbrales** desde la fase 8, y el que importa es
        el primero: a los `DEADLOCK_FORCE_TICKS` (8) el motor desatasca a la
        fuerza (`_desatasca`), y solo si aun asi no se mueve nadie se llega a los
        `DEADLOCK_TICKS` (20) y la corrida se da por muerta. O sea que llegar
        aqui deberia ser raro, y cuando pasa es que ni un AGV tenia un vecino
        libre al que apartarse.

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
        registro = self._acciones.get(
            agente.id, ActionRecord(conflicts.INTENT_WAIT, self.step)
        )
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
            # Lo que agrega la fase 8: lo que QUISO hacer y lo que el motor le
            # concedio. Que las dos cosas puedan no coincidir es el punto.
            "action": registro.action,
            "blocked": registro.blocked,
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
