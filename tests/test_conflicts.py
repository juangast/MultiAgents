"""CRITERIOS DE LA FASE 5: conflictos, politica baseline y deadlock.

Lo que se demuestra aqui:

1. El conflicto se detecta **antes** de mover a los dos agentes, no despues.
2. Nunca hay dos AGVs en el mismo nodo (500 ticks con 6 agentes, tick a tick).
3. `wait_time` acumula y sale en el snapshot.
4. Los conflictos se cuentan por corrida.
5. Un cruce de frente montado a proposito se detecta como `edge conflict`.
"""

import dataclasses
import unittest
from unittest import mock

import agent
import config
import conflicts
import graph
import simulation


def agv(
    agent_id: int,
    grafo: graph.WarehouseGraph,
    nodo: str,
    *,
    progress: float = 0.0,
    state: str = agent.STATE_MOVING,
    wait_time: int = 0,
) -> agent.Agent:
    """Un AGV colocado a mano, para probar los detectores sin simulacion."""
    uno = agent.Agent(agent_id, grafo, nodo)
    uno.current_node = nodo
    uno.progress = progress
    uno.state = state
    uno.wait_time = wait_time
    return uno


# La fase 8 le puso al motor un desatasco que salta a los
# `config.DEADLOCK_FORCE_TICKS` ticks sin que avance nadie, asi que un atasco ya
# no llega a los `DEADLOCK_TICKS` y las pruebas de deadlock no verian nunca lo
# que vienen a probar. Se apaga donde estorba, que es justo para lo que existe el
# 0 en esa constante: asi estas siguen midiendo el motor de la fase 5, que es de
# lo que van. Con el desatasco encendido, en `tests/test_phase8.py`.
sin_desatasco = mock.patch.object(config, "DEADLOCK_FORCE_TICKS", 0)


def cruce_frontal() -> simulation.Simulation:
    """Dos AGVs de frente sobre el mismo tramo: 1 va de A a C, 2 va de B a A.

    En el mapa `simple`, A y B son vecinos y la ruta de 1 es A -> B -> C, asi que
    en el primer tick los dos quieren el nodo del otro. Es el escenario del
    criterio 5, y no depende de la semilla: las rutas van escritas.
    """
    return simulation.Simulation(graph.simple_graph(), routes=[("A", "C"), ("B", "A")])


class TestConflict(unittest.TestCase):
    """El objeto Conflict: inmutable, hasheable y con su tipo comprobado."""

    def test_lleva_tipo_agentes_paso_y_sitio(self) -> None:
        choque = conflicts.Conflict(conflicts.TYPE_VERTEX, (1, 2), 7, node="G")
        self.assertEqual(choque.type, "vertex")
        self.assertEqual(choque.agents, (1, 2))
        self.assertEqual(choque.step, 7)
        self.assertEqual(choque.node, "G")
        self.assertIsNone(choque.edge)

    def test_no_se_puede_editar_despues(self) -> None:
        choque = conflicts.Conflict(conflicts.TYPE_VERTEX, (1, 2), 7, node="G")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            choque.step = 8  # type: ignore[misc]

    def test_dos_detecciones_del_mismo_choque_son_el_mismo_conflicto(self) -> None:
        uno = conflicts.Conflict(conflicts.TYPE_VERTEX, (1, 2), 7, node="G")
        otro = conflicts.Conflict(conflicts.TYPE_VERTEX, (1, 2), 7, node="G")
        self.assertEqual(uno, otro)
        self.assertEqual(len({uno, otro}), 1)

    def test_un_tipo_que_no_existe_revienta(self) -> None:
        with self.assertRaises(ValueError):
            conflicts.Conflict("chocazo", (1, 2), 7, node="G")

    def test_as_dict_sale_en_json(self) -> None:
        choque = conflicts.Conflict(conflicts.TYPE_EDGE, (1, 2), 3, edge=("A", "B"))
        self.assertEqual(
            choque.as_dict(),
            {"step": 3, "type": "edge", "agents": [1, 2], "node": None, "edge": ["A", "B"]},
        )


class TestDetectConflicts(unittest.TestCase):
    """Los detectores, sin montar una simulacion entera."""

    def setUp(self) -> None:
        self.grafo = graph.simple_graph()

    def test_dos_agentes_al_mismo_nodo_es_un_vertex(self) -> None:
        detectados = conflicts.detect_conflicts({1: "B", 2: "B"}, self.grafo, step=4)
        self.assertEqual(len(detectados), 1)
        self.assertEqual(detectados[0].type, conflicts.TYPE_VERTEX)
        self.assertEqual(detectados[0].agents, (1, 2))
        self.assertEqual(detectados[0].node, "B")
        self.assertEqual(detectados[0].step, 4)

    def test_tres_al_mismo_nodo_son_un_conflicto_de_tres_y_no_tres_conflictos(self) -> None:
        detectados = conflicts.detect_conflicts({3: "B", 1: "B", 2: "B"}, self.grafo)
        self.assertEqual(len(detectados), 1)
        self.assertEqual(detectados[0].agents, (1, 2, 3))

    def test_intenciones_a_nodos_distintos_no_chocan(self) -> None:
        self.assertEqual(conflicts.detect_conflicts({1: "B", 2: "D"}, self.grafo), [])

    def test_le_bastan_los_dos_argumentos_del_contrato(self) -> None:
        # Sin `occupancy` ni `agents` no hay forma de saber donde esta nadie, asi
        # que solo se ven los vertices. Y aun asi tiene que funcionar.
        self.assertEqual(len(conflicts.detect_conflicts({1: "B", 2: "B"}, self.grafo)), 1)

    def test_el_que_ya_esta_en_el_nodo_cuenta_como_contendiente(self) -> None:
        agentes = [agv(1, self.grafo, "A"), agv(2, self.grafo, "B", state=agent.STATE_DONE)]
        detectados = conflicts.detect_conflicts(
            {1: "B"}, self.grafo, occupancy={"B": 2}, agents=agentes
        )
        self.assertEqual([c.type for c in detectados], [conflicts.TYPE_VERTEX])
        self.assertEqual(detectados[0].agents, (1, 2))

    def test_el_cruce_de_frente_es_un_edge(self) -> None:
        agentes = [agv(1, self.grafo, "A"), agv(2, self.grafo, "B")]
        detectados = conflicts.detect_conflicts(
            {1: "B", 2: "A"},
            self.grafo,
            occupancy={"A": 1, "B": 2},
            agents=agentes,
            step=9,
        )
        self.assertEqual(len(detectados), 1)
        self.assertEqual(detectados[0].type, conflicts.TYPE_EDGE)
        self.assertEqual(detectados[0].agents, (1, 2))
        self.assertEqual(detectados[0].edge, ("A", "B"))
        self.assertIsNone(detectados[0].node)

    def test_el_cruce_de_frente_no_se_cuenta_ademas_como_vertex(self) -> None:
        # Los dos nodos estan "disputados", pero es un solo choque, no tres.
        agentes = [agv(1, self.grafo, "A"), agv(2, self.grafo, "B")]
        detectados = conflicts.detect_conflicts(
            {1: "B", 2: "A"}, self.grafo, occupancy={"A": 1, "B": 2}, agents=agentes
        )
        self.assertNotIn(conflicts.TYPE_VERTEX, [c.type for c in detectados])

    def test_el_edge_solo_cuenta_sobre_una_arista_de_verdad(self) -> None:
        # A y F no son vecinos en `simple`: eso no es un cruce, es una ruta rota.
        agentes = [agv(1, self.grafo, "A"), agv(2, self.grafo, "F")]
        detectados = conflicts.detect_conflicts(
            {1: "F", 2: "A"}, self.grafo, occupancy={"A": 1, "F": 2}, agents=agentes
        )
        self.assertNotIn(conflicts.TYPE_EDGE, [c.type for c in detectados])

    def test_entrar_en_el_nodo_que_otro_esta_dejando_es_un_following(self) -> None:
        # El 2 va por el tramo B -> C, asi que retiene los dos: B no esta libre.
        agentes = [agv(1, self.grafo, "A"), agv(2, self.grafo, "B", progress=0.5)]
        detectados = conflicts.detect_conflicts(
            {1: "B"}, self.grafo, occupancy={"B": 2, "C": 2}, agents=agentes
        )
        self.assertEqual([c.type for c in detectados], [conflicts.TYPE_FOLLOWING])
        self.assertEqual(detectados[0].agents, (1, 2))
        self.assertEqual(detectados[0].node, "B")

    def test_pedir_el_nodo_al_que_otro_entra_no_es_following_sino_vertex(self) -> None:
        # El 2 va hacia C: C esta reservado, pero el 2 no lo esta dejando.
        agentes = [agv(1, self.grafo, "B"), agv(2, self.grafo, "B", progress=0.5)]
        agentes[0].current_node = "F"
        detectados = conflicts.detect_conflicts(
            {1: "C"}, self.grafo, occupancy={"B": 2, "C": 2}, agents=agentes
        )
        self.assertEqual([c.type for c in detectados], [conflicts.TYPE_VERTEX])

    def test_la_salida_va_siempre_en_el_mismo_orden(self) -> None:
        agentes = [agv(1, self.grafo, "A"), agv(2, self.grafo, "B"), agv(3, self.grafo, "D")]
        detectados = conflicts.detect_conflicts(
            {1: "B", 2: "A", 3: "E"}, self.grafo, occupancy={"A": 1, "B": 2, "D": 3},
            agents=agentes,
        )
        self.assertEqual(
            detectados, conflicts.detect_conflicts(
                {3: "E", 2: "A", 1: "B"}, self.grafo,
                occupancy={"A": 1, "B": 2, "D": 3}, agents=agentes,
            )
        )


class TestCongestion(unittest.TestCase):
    """Atascos: por agente que lleva mucho esperando, y por zona colapsada."""

    def setUp(self) -> None:
        self.grafo = graph.warehouse_graph()

    def test_cruzar_el_umbral_de_espera_marca_congestion(self) -> None:
        esperando = agv(
            1, self.grafo, "G",
            state=agent.STATE_WAITING,
            wait_time=config.CONFLICT_WAIT_THRESHOLD,
        )
        detectados = conflicts.detect_congestion([esperando], self.grafo, 12)
        self.assertEqual([c.type for c in detectados], [conflicts.TYPE_CONGESTION])
        self.assertEqual(detectados[0].agents, (1,))
        self.assertEqual(detectados[0].node, "G")

    def test_no_se_repite_en_los_ticks_siguientes(self) -> None:
        # Un atasco de cincuenta ticks es un conflicto, no cincuenta: si no, el
        # conteo por corrida dejaria de significar nada.
        for espera in (
            config.CONFLICT_WAIT_THRESHOLD - 1,
            config.CONFLICT_WAIT_THRESHOLD + 1,
        ):
            with self.subTest(wait_time=espera):
                atascado = agv(
                    1, self.grafo, "G", state=agent.STATE_WAITING, wait_time=espera
                )
                self.assertEqual(
                    conflicts.detect_congestion([atascado], self.grafo, 12), []
                )

    def test_varios_esperando_en_la_misma_zona_marcan_la_zona(self) -> None:
        # G y sus vecinos son la zona del cuello de botella del almacen.
        cola = [
            agv(numero, self.grafo, nodo, state=agent.STATE_WAITING)
            for numero, nodo in enumerate(("G", "S3", "N3"), start=1)
        ]
        self.assertIn("G", conflicts.congested_zones(cola, self.grafo))

        detectados = conflicts.detect_congestion(cola, self.grafo, 30)
        del_cuello = [c for c in detectados if c.node == "G"]
        self.assertEqual(len(del_cuello), 1)
        self.assertEqual(del_cuello[0].type, conflicts.TYPE_CONGESTION)
        self.assertEqual(del_cuello[0].agents, (1, 2, 3))

    def test_una_zona_ya_congestionada_no_se_vuelve_a_contar(self) -> None:
        cola = [
            agv(numero, self.grafo, nodo, state=agent.STATE_WAITING)
            for numero, nodo in enumerate(("G", "S3", "N3"), start=1)
        ]
        zonas = conflicts.congested_zones(cola, self.grafo)
        self.assertEqual(
            conflicts.detect_congestion(cola, self.grafo, 31, previous_zones=zonas), []
        )

    def test_los_que_se_mueven_no_hacen_zona(self) -> None:
        pasando = [
            agv(numero, self.grafo, nodo)
            for numero, nodo in enumerate(("G", "S3", "N3"), start=1)
        ]
        self.assertEqual(conflicts.congested_zones(pasando, self.grafo), frozenset())


class TestResolveBaseline(unittest.TestCase):
    """La politica base: gana el id menor, y ya."""

    def test_gana_el_id_menor(self) -> None:
        resuelto = conflicts.resolve_baseline(
            conflicts.Conflict(conflicts.TYPE_VERTEX, (2, 5, 3), 1, node="G")
        )
        self.assertEqual(resuelto.winner, 2)
        self.assertEqual(resuelto.losers, (3, 5))

    def test_en_un_cruce_de_frente_tambien(self) -> None:
        resuelto = conflicts.resolve_baseline(
            conflicts.Conflict(conflicts.TYPE_EDGE, (4, 1), 1, edge=("A", "B"))
        )
        self.assertEqual(resuelto.winner, 1)
        self.assertEqual(resuelto.losers, (4,))

    def test_la_congestion_no_tiene_nada_que_arbitrar(self) -> None:
        resuelto = conflicts.resolve_baseline(
            conflicts.Conflict(conflicts.TYPE_CONGESTION, (1, 2, 3), 9, node="G")
        )
        self.assertIsNone(resuelto.winner)
        self.assertEqual(resuelto.losers, ())

    def test_es_pura_y_no_toca_el_conflicto(self) -> None:
        choque = conflicts.Conflict(conflicts.TYPE_VERTEX, (2, 1), 1, node="G")
        conflicts.resolve_baseline(choque)
        self.assertEqual(choque.agents, (2, 1))


class TestCruceFrontal(unittest.TestCase):
    """CRITERIO 5: dos AGVs de frente, y el edge conflict se ve."""

    def setUp(self) -> None:
        self.simulacion = cruce_frontal()

    def test_las_rutas_del_escenario_son_las_que_se_pidieron(self) -> None:
        self.assertEqual(self.simulacion.agents[0].path, ["A", "B", "C"])
        self.assertEqual(self.simulacion.agents[1].path, ["B", "A"])

    def test_en_el_primer_tick_se_detecta_el_edge_conflict(self) -> None:
        self.simulacion.tick()
        detectados = list(self.simulacion.conflicts)
        self.assertEqual(len(detectados), 1)
        self.assertEqual(detectados[0].type, conflicts.TYPE_EDGE)
        self.assertEqual(detectados[0].agents, (1, 2))
        self.assertEqual(detectados[0].edge, ("A", "B"))
        self.assertEqual(detectados[0].step, 1)

    def test_se_detecta_antes_de_mover_a_ninguno_de_los_dos(self) -> None:
        # CRITERIO 1: al acabar el tick del choque los dos siguen exactamente
        # donde estaban. Si se detectara despues, uno habria avanzado ya.
        self.simulacion.tick()
        uno, otro = self.simulacion.agents
        self.assertEqual((uno.current_node, uno.progress), ("A", 0.0))
        self.assertEqual((otro.current_node, otro.progress), ("B", 0.0))
        self.assertEqual(uno.state, agent.STATE_WAITING)
        self.assertEqual(otro.state, agent.STATE_WAITING)

    def test_el_ganador_tampoco_pasa_porque_el_perdedor_sigue_ahi(self) -> None:
        # El baseline nombra ganador al id menor, pero el 2 no se aparta: no
        # sabe. El gate fisico frena tambien al 1 y el tramo queda muerto. Eso
        # es el baseline, y es justo lo que el Q-Learning tiene que batir.
        for _ in range(5):
            self.simulacion.tick()
        self.assertEqual([a.current_node for a in self.simulacion.agents], ["A", "B"])
        self.assertEqual([a.wait_time for a in self.simulacion.agents], [5, 5])

    @sin_desatasco
    def test_el_cruce_de_frente_acaba_en_deadlock(self) -> None:
        while not self.simulacion.done and self.simulacion.step < 200:
            self.simulacion.tick()
        self.assertEqual(self.simulacion.finished_reason, simulation.FINISHED_DEADLOCK)
        self.assertEqual(self.simulacion.step, config.DEADLOCK_TICKS)
        self.assertEqual(self.simulacion.deadlocks, 1)


class TestFollowing(unittest.TestCase):
    """La reserva doble: nadie entra en el nodo que otro esta dejando."""

    def setUp(self) -> None:
        # El 1 cruza A -> B mientras el 2 viene de D queriendo entrar en A.
        self.simulacion = simulation.Simulation(
            graph.simple_graph(), routes=[("A", "C"), ("D", "A")]
        )

    def test_el_que_viene_detras_espera_a_que_suelte_el_nodo(self) -> None:
        self.simulacion.tick()   # el 1 arranca A -> B (A->B cuesta 2, va al 50%)
        self.simulacion.tick()   # el 2 pide A, que el 1 todavia retiene

        tipos = [choque.type for choque in self.simulacion.conflicts]
        self.assertEqual(tipos, [conflicts.TYPE_VERTEX, conflicts.TYPE_FOLLOWING])

        seguidor = self.simulacion.agents[1]
        self.assertEqual(seguidor.current_node, "D")
        self.assertEqual(seguidor.state, agent.STATE_WAITING)

    def test_en_cuanto_lo_suelta_el_de_detras_pasa(self) -> None:
        # No es un bloqueo permanente: el 1 llega a B en el paso 2 y suelta A,
        # asi que en el 3 el 2 ya puede arrancar.
        for _ in range(3):
            self.simulacion.tick()
        seguidor = self.simulacion.agents[1]
        self.assertEqual(seguidor.state, agent.STATE_MOVING)
        self.assertGreater(seguidor.progress, 0.0)
        self.assertEqual(self.simulacion.occupancy["A"], seguidor.id)


class TestInvarianteDeOcupacion(unittest.TestCase):
    """CRITERIO 2: nunca hay dos AGVs en el mismo nodo. Tick a tick."""

    def revisa(self, simulacion: simulation.Simulation) -> None:
        """La invariante, comprobada sobre los agentes y sobre la ocupacion."""
        nodos = [agente.current_node for agente in simulacion.agents]
        self.assertEqual(
            len(set(nodos)),
            len(nodos),
            f"dos AGVs en el mismo nodo en el paso {simulacion.step}: {nodos}",
        )
        for agente in simulacion.agents:
            self.assertEqual(
                simulacion.occupancy.get(agente.current_node),
                agente.id,
                f"el AGV {agente.id} no figura como dueño de {agente.current_node}",
            )

    def test_500_ticks_con_6_agentes(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 6)
        self.revisa(simulacion)
        for _ in range(500):
            simulacion.tick()
            self.revisa(simulacion)

    @sin_desatasco
    def test_500_snapshots_con_6_agentes_y_reinicios_por_medio(self) -> None:
        # Por este camino el servidor arranca corridas nuevas cuando la anterior
        # muere atascada, asi que la invariante se prueba tambien en los reinicios.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 6)
        for _ in range(500):
            instantanea = simulacion.get_snapshot()
            nodos = [agente["node"] for agente in instantanea["agents"]]
            self.assertEqual(len(set(nodos)), len(nodos), f"colision: {nodos}")
            self.revisa(simulacion)
        self.assertGreater(simulacion.stats()["run"], 1, "no llego a reiniciarse ninguna vez")

    def test_ningun_nodo_tiene_dos_dueños(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 6)
        for _ in range(120):
            simulacion.tick()
            # Un agente retiene como mucho dos nodos: de donde sale y a donde va.
            self.assertLessEqual(
                len(simulacion.occupancy), 2 * len(simulacion.agents)
            )
            self.assertEqual(
                len(set(simulacion.occupancy.values())), len(simulacion.agents)
            )

    def test_dos_agentes_no_pueden_salir_del_mismo_nodo(self) -> None:
        with self.assertRaises(ValueError):
            simulation.Simulation(graph.simple_graph(), routes=[("A", "F"), ("A", "C")])

    def test_no_caben_mas_agentes_que_nodos(self) -> None:
        with self.assertRaises(ValueError):
            simulation.Simulation(graph.simple_graph(), 7)

    def test_el_reparto_por_semilla_da_origenes_distintos(self) -> None:
        for cuantos in range(1, 14):
            with self.subTest(agentes=cuantos):
                simulacion = simulation.Simulation(graph.warehouse_graph(), cuantos)
                origenes = [agente.start_node for agente in simulacion.agents]
                self.assertEqual(len(set(origenes)), cuantos)


class TestPoliticaIntercambiable(unittest.TestCase):
    """La politica entra por el constructor y el motor no se entera de cual es."""

    def test_la_de_serie_es_el_baseline(self) -> None:
        simulacion = simulation.Simulation(graph.simple_graph(), 1)
        self.assertIsInstance(simulacion.policy, conflicts.BaselinePolicy)
        self.assertEqual(simulacion.stats()["policy"], "baseline")

    def test_el_baseline_cumple_el_contrato(self) -> None:
        self.assertIsInstance(conflicts.BaselinePolicy(), conflicts.Policy)

    @sin_desatasco
    def test_una_politica_que_siempre_espera_para_a_todo_el_mundo(self) -> None:
        class Quieta:
            name = "quieta"

            def decide(self, agent, local_state) -> str:
                return conflicts.ACTION_WAIT

        simulacion = simulation.Simulation(graph.simple_graph(), 1, policy=Quieta())
        for _ in range(10):
            simulacion.tick()
        agv_uno = simulacion.agents[0]
        self.assertEqual(agv_uno.current_node, "A")
        self.assertEqual(agv_uno.state, agent.STATE_WAITING)
        self.assertEqual(agv_uno.wait_time, 10)
        self.assertEqual(simulacion.stats()["policy"], "quieta")

    def test_una_politica_temeraria_no_puede_romper_la_invariante(self) -> None:
        # El gate fisico esta por debajo de la politica a proposito: la fase 8
        # podra proponer lo que quiera y seguira sin meter dos AGVs en un nodo.
        class Temeraria:
            name = "temeraria"

            def decide(self, agent, local_state) -> str:
                return conflicts.ACTION_GO

        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 6, policy=Temeraria()
        )
        for _ in range(200):
            simulacion.tick()
            nodos = [agente.current_node for agente in simulacion.agents]
            self.assertEqual(len(set(nodos)), len(nodos), f"colision: {nodos}")


class TestStats(unittest.TestCase):
    """CRITERIOS 3 y 4: la espera y el conteo de conflictos salen en el snapshot."""

    def test_el_snapshot_trae_el_conteo_de_conflictos(self) -> None:
        simulacion = simulation.Simulation(graph.simple_graph(), 1)
        numeros = simulacion.get_snapshot()["stats"]
        self.assertIsInstance(numeros["conflicts"], int)
        self.assertEqual(numeros["conflicts"], 0)

    def test_el_conteo_sube_cuando_chocan(self) -> None:
        simulacion = cruce_frontal()
        for esperado in (1, 2, 3):
            self.assertEqual(
                simulacion.get_snapshot()["stats"]["conflicts"], esperado
            )

    def test_el_conteo_por_tipo_cuadra_con_el_total(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 6)
        for _ in range(60):
            simulacion.tick()
        numeros = simulacion.stats()
        self.assertEqual(sum(numeros["conflicts_by_type"].values()), numeros["conflicts"])
        self.assertEqual(set(numeros["conflicts_by_type"]), set(conflicts.TYPES))

    def test_el_wait_time_acumula_y_sale_en_el_snapshot(self) -> None:
        simulacion = cruce_frontal()
        for esperado in (1, 2, 3, 4):
            agentes = simulacion.get_snapshot()["agents"]
            self.assertEqual([a["wait_time"] for a in agentes], [esperado, esperado])

    def test_la_espera_total_suma_la_de_todos(self) -> None:
        simulacion = cruce_frontal()
        for _ in range(6):
            simulacion.tick()
        self.assertEqual(simulacion.stats()["total_wait_time"], 12)

    def test_cuenta_a_los_que_estan_esperando_ahora_mismo(self) -> None:
        simulacion = cruce_frontal()
        self.assertEqual(simulacion.stats()["waiting"], 0)
        simulacion.tick()
        self.assertEqual(simulacion.stats()["waiting"], 2)

    def test_el_registro_de_la_corrida_sale_en_json(self) -> None:
        simulacion = cruce_frontal()
        simulacion.tick()
        self.assertEqual(
            simulacion.conflict_records(),
            [{"step": 1, "type": "edge", "agents": [1, 2], "node": None, "edge": ["A", "B"]}],
        )

    def test_el_registro_es_por_corrida_y_se_vacia_al_reiniciar(self) -> None:
        simulacion = cruce_frontal()
        for _ in range(3):
            simulacion.tick()
        self.assertEqual(simulacion.stats()["conflicts"], 3)
        simulacion.reset()
        self.assertEqual(simulacion.stats()["conflicts"], 0)
        self.assertEqual(simulacion.conflict_records(), [])

    def test_los_campos_congelados_siguen_donde_estaban(self) -> None:
        # `stats` (fase 5) y `mode` (fase 8) se agregan al primer nivel; no tocan
        # ni `step` ni `agents`, que son los de la fase 1.
        instantanea = simulation.Simulation(graph.warehouse_graph(), 1).get_snapshot()
        self.assertEqual(set(instantanea), {"step", "agents", "stats", "mode"})
        self.assertEqual(instantanea["step"], 1)


@sin_desatasco
class TestDeadlock(unittest.TestCase):
    """La corrida no se queda colgada: si nadie avanza, se corta y se dice."""

    def test_arranca_sin_razon_de_final(self) -> None:
        simulacion = simulation.Simulation(graph.simple_graph(), 1)
        self.assertIsNone(simulacion.finished_reason)
        self.assertEqual(simulacion.stats()["deadlocks"], 0)

    def test_llegar_al_destino_no_es_un_deadlock(self) -> None:
        # `simple` de A a F son 7 ticks; despues no se mueve nadie, pero porque
        # ya no hay nada que hacer. Eso es el final feliz, no un atasco.
        simulacion = simulation.Simulation(graph.simple_graph(), 1)
        for _ in range(7 + config.DEADLOCK_TICKS * 2):
            simulacion.tick()
        self.assertTrue(simulacion.done)
        self.assertIsNone(simulacion.finished_reason)
        self.assertEqual(simulacion.stats()["deadlocks"], 0)

    def test_k_ticks_sin_avanzar_marcan_deadlock(self) -> None:
        simulacion = cruce_frontal()
        for _ in range(config.DEADLOCK_TICKS - 1):
            simulacion.tick()
        self.assertIsNone(simulacion.finished_reason)

        simulacion.tick()
        self.assertEqual(simulacion.finished_reason, simulation.FINISHED_DEADLOCK)
        self.assertTrue(simulacion.done)
        self.assertEqual(simulacion.stats()["finished_reason"], "deadlock")

    def test_no_se_cuenta_el_mismo_atasco_una_y_otra_vez(self) -> None:
        simulacion = cruce_frontal()
        for _ in range(config.DEADLOCK_TICKS * 4):
            simulacion.tick()
        self.assertEqual(simulacion.stats()["deadlocks"], 1)

    def test_el_servidor_arranca_otra_corrida_en_vez_de_quedarse_clavado(self) -> None:
        simulacion = cruce_frontal()
        for _ in range(config.DEADLOCK_TICKS):
            simulacion.tick()
        self.assertEqual(simulacion.finished_reason, simulation.FINISHED_DEADLOCK)

        # El snapshot del atasco ya se pudo entregar; el siguiente es otra corrida.
        numeros = simulacion.get_snapshot()["stats"]
        self.assertEqual(numeros["run"], 2)
        self.assertIsNone(numeros["finished_reason"])
        self.assertEqual(simulacion.step, 1)

    def test_los_deadlocks_de_la_sesion_sobreviven_al_reinicio(self) -> None:
        simulacion = cruce_frontal()
        for _ in range(config.DEADLOCK_TICKS):
            simulacion.tick()
        simulacion.reset()
        self.assertEqual(simulacion.stats()["deadlocks"], 1)
        self.assertIsNone(simulacion.finished_reason)


if __name__ == "__main__":
    unittest.main()
