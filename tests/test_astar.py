"""Tests de la fase 3: A*, los agentes y la simulacion de verdad.

Los cinco criterios de aceptacion de la fase estan marcados con "CRITERIO N" en
el docstring de la clase que los cubre.
"""

import itertools
import json
import threading
import unittest
from unittest import mock

import agent
import astar
import config
import graph
import main
import protocol
import simulation
from tests.fake_unity_client import CAMPOS_AGENTE, validar_snapshot


def sin_el_numero_de_corrida(instantanea: dict) -> dict:
    """El snapshot sin `stats.run`, para comparar dos corridas entre si."""
    copia = dict(instantanea)
    copia["stats"] = {
        clave: valor for clave, valor in copia["stats"].items() if clave != "run"
    }
    return copia


def mapas_del_repo() -> list[graph.WarehouseGraph]:
    """Los dos mapas de verdad del proyecto."""
    return [graph.simple_graph(), graph.warehouse_graph()]


def coste_por_fuerza_bruta(
    grafo: graph.WarehouseGraph, origen: str, destino: str
) -> float | None:
    """Costo minimo explorando TODAS las rutas simples, sin heuristica ninguna.

    Es la referencia contra la que se compara A*: lenta y tonta, pero imposible
    que se equivoque. La poda por costo es valida porque no hay aristas
    negativas, asi que alargar una ruta nunca la abarata.
    """
    mejor: float | None = None

    def anda(nodo: str, visitados: frozenset[str], costo: float) -> None:
        nonlocal mejor
        if nodo == destino:
            if mejor is None or costo < mejor:
                mejor = costo
            return
        if mejor is not None and costo >= mejor:
            return
        for vecino in grafo.neighbors(nodo):
            if vecino not in visitados:
                anda(vecino, visitados | {vecino}, costo + grafo.cost(nodo, vecino))

    anda(origen, frozenset({origen}), 0.0)
    return mejor


def grafo_partido() -> graph.WarehouseGraph:
    """Dos islas que no se tocan: de la primera no se llega a la segunda."""
    return graph.WarehouseGraph(
        adjacency={"A": {"B": 1.0}, "B": {"A": 1.0}, "X": {"Y": 1.0}, "Y": {"X": 1.0}},
        positions={"A": (0.0, 0.0), "B": (1.0, 0.0), "X": (9.0, 0.0), "Y": (10.0, 0.0)},
        name="partido",
    )


def grafo_con_atajo_barato() -> graph.WarehouseGraph:
    """Un mapa donde un tramo cuesta MUCHO menos que lo que mide.

    X --1-- Y ------1------ Z      (Y->Z mide 19 y cuesta 1: una cinta)
     \\--5-- W ------5------/       (la ruta "geometricamente razonable")

    La ruta buena es X->Y->Z (costo 2) pero la euclidiana en crudo la descarta:
    desde X, Y parece estar a 19 de la meta y W solo a 10. Es justo el caso que
    el factor de la heuristica existe para arreglar.
    """
    return graph.WarehouseGraph(
        adjacency={
            "X": {"Y": 1.0, "W": 5.0},
            "Y": {"X": 1.0, "Z": 1.0},
            "W": {"X": 5.0, "Z": 5.0},
            "Z": {"Y": 1.0, "W": 5.0},
        },
        positions={
            "X": (0.0, 0.0),
            "Y": (1.0, 0.0),
            "W": (10.0, 0.0),
            "Z": (20.0, 0.0),
        },
        name="atajo",
    )


class TestHeuristica(unittest.TestCase):
    """El factor que mantiene la heuristica admisible."""

    def test_en_los_mapas_del_repo_el_factor_es_uno(self) -> None:
        # Ninguna arista de simple ni de warehouse cuesta menos que su longitud,
        # asi que hoy la heuristica es la euclidiana tal cual.
        for grafo in mapas_del_repo():
            with self.subTest(mapa=grafo.name):
                self.assertAlmostEqual(astar.heuristic_factor(grafo), 1.0)

    def test_el_factor_nunca_pasa_de_uno(self) -> None:
        # Aunque todos los tramos sean carisimos: inflar la heuristica por encima
        # de la distancia real solo puede hacerla sobreestimar.
        caro = graph.WarehouseGraph(
            adjacency={"A": {"B": 100.0}, "B": {"A": 100.0}},
            positions={"A": (0.0, 0.0), "B": (1.0, 0.0)},
        )
        self.assertLessEqual(astar.heuristic_factor(caro), 1.0)

    def test_un_tramo_mas_barato_que_su_longitud_baja_el_factor(self) -> None:
        factor = astar.heuristic_factor(grafo_con_atajo_barato())
        self.assertLess(factor, 1.0)
        self.assertAlmostEqual(factor, 1.0 / 19.0)

    def test_con_el_factor_encuentra_el_optimo_donde_la_euclidiana_falla(self) -> None:
        grafo = grafo_con_atajo_barato()
        self.assertEqual(astar.astar(grafo, "X", "Z"), ["X", "Y", "Z"])
        self.assertAlmostEqual(
            astar.path_cost(grafo, ["X", "Y", "Z"]),
            coste_por_fuerza_bruta(grafo, "X", "Z"),
        )

    def test_sin_el_factor_ese_mismo_mapa_daria_una_ruta_peor(self) -> None:
        # Con el factor clavado a 1.0 (o sea, euclidiana en crudo) A* se traga el
        # camino caro. Sirve para comprobar que el factor no es decorativo.
        grafo = grafo_con_atajo_barato()
        with mock.patch.object(astar, "heuristic_factor", return_value=1.0):
            enganado = astar.astar(grafo, "X", "Z")
        self.assertEqual(enganado, ["X", "W", "Z"])
        self.assertGreater(
            astar.path_cost(grafo, enganado),
            astar.path_cost(grafo, ["X", "Y", "Z"]),
        )

    def test_un_grafo_sin_aristas_no_revienta(self) -> None:
        vacio = graph.WarehouseGraph(adjacency={"A": {}}, positions={"A": (0.0, 0.0)})
        self.assertEqual(astar.heuristic_factor(vacio), 1.0)


class TestRutaOptima(unittest.TestCase):
    """CRITERIO 1: A* encuentra A->F y es la de menor costo."""

    def test_a_f_en_el_mapa_simple(self) -> None:
        grafo = graph.simple_graph()
        ruta = astar.astar(grafo, "A", "F")
        self.assertEqual(ruta, ["A", "B", "E", "F"])
        self.assertAlmostEqual(astar.path_cost(grafo, ruta), 7.0)

    def test_a_f_es_mas_barata_que_cualquier_otra_ruta(self) -> None:
        grafo = graph.simple_graph()
        ruta = astar.astar(grafo, "A", "F")
        self.assertAlmostEqual(
            astar.path_cost(grafo, ruta), coste_por_fuerza_bruta(grafo, "A", "F")
        )

    def test_contra_la_busqueda_exhaustiva_en_los_dos_mapas(self) -> None:
        # Todos los pares ordenados de los dos mapas: 30 en simple, 156 en
        # warehouse. Se compara el costo, no la lista, porque puede haber varias
        # rutas empatadas y cualquiera de ellas es igual de optima.
        for grafo in mapas_del_repo():
            for origen, destino in itertools.permutations(grafo.nodes(), 2):
                with self.subTest(mapa=grafo.name, de=origen, a=destino):
                    ruta = astar.astar(grafo, origen, destino)
                    self.assertIsNotNone(ruta)
                    self.assertAlmostEqual(
                        astar.path_cost(grafo, ruta),
                        coste_por_fuerza_bruta(grafo, origen, destino),
                    )

    def test_el_origen_y_el_destino_son_los_pedidos(self) -> None:
        for grafo in mapas_del_repo():
            for origen, destino in itertools.permutations(grafo.nodes(), 2):
                with self.subTest(mapa=grafo.name, de=origen, a=destino):
                    ruta = astar.astar(grafo, origen, destino)
                    self.assertEqual(ruta[0], origen)
                    self.assertEqual(ruta[-1], destino)

    def test_toda_ruta_que_cruza_el_almacen_pasa_por_el_cuello_de_botella(self) -> None:
        # G es nodo de articulacion, asi que no hay forma de ir de un lado al
        # otro sin pasar por el.
        almacen = graph.warehouse_graph()
        for origen, destino in (("S1", "N6"), ("N1", "S6"), ("S2", "S5")):
            with self.subTest(de=origen, a=destino):
                self.assertIn(graph.BOTTLENECK, astar.astar(almacen, origen, destino))


class TestAristasReales(unittest.TestCase):
    """CRITERIO 3: la ruta nunca usa una arista que no existe."""

    def comprueba(self, grafo: graph.WarehouseGraph, ruta: list[str]) -> None:
        for anterior, siguiente in zip(ruta, ruta[1:]):
            self.assertTrue(
                grafo.has_edge(anterior, siguiente),
                f"{anterior} -> {siguiente} no es una arista de {grafo.name}",
            )

    def test_todos_los_pares_de_los_dos_mapas(self) -> None:
        for grafo in mapas_del_repo():
            for origen, destino in itertools.permutations(grafo.nodes(), 2):
                with self.subTest(mapa=grafo.name, de=origen, a=destino):
                    self.comprueba(grafo, astar.astar(grafo, origen, destino))

    def test_tambien_con_penalizaciones(self) -> None:
        almacen = graph.warehouse_graph()
        castigos = {graph.BOTTLENECK: 50.0, ("S1", "S2"): 20.0, "N5": 7.5}
        for origen, destino in itertools.permutations(almacen.nodes(), 2):
            with self.subTest(de=origen, a=destino):
                self.comprueba(almacen, astar.astar(almacen, origen, destino, castigos))

    def test_la_ruta_no_repite_nodos(self) -> None:
        # Un ciclo en la ruta seria costo tirado a la basura.
        for grafo in mapas_del_repo():
            for origen, destino in itertools.permutations(grafo.nodes(), 2):
                with self.subTest(mapa=grafo.name, de=origen, a=destino):
                    ruta = astar.astar(grafo, origen, destino)
                    self.assertEqual(len(ruta), len(set(ruta)))

    def test_el_path_del_agente_tambien_recorre_aristas_reales(self) -> None:
        almacen = graph.warehouse_graph()
        agv = agent.Agent(1, almacen, "S1")
        agv.assign_task("S1", "N6")
        self.comprueba(almacen, agv.path)


class TestSinRuta(unittest.TestCase):
    """CRITERIO 4: sin ruta devuelve None y el agente queda idle, sin reventar."""

    def test_un_grafo_partido_devuelve_none(self) -> None:
        self.assertIsNone(astar.astar(grafo_partido(), "A", "Y"))

    def test_un_nodo_desconocido_devuelve_none(self) -> None:
        grafo = graph.simple_graph()
        for origen, destino in (("A", "Z"), ("Z", "A"), ("Z", "W")):
            with self.subTest(de=origen, a=destino):
                self.assertIsNone(astar.astar(grafo, origen, destino))

    def test_origen_igual_a_destino_devuelve_solo_ese_nodo(self) -> None:
        self.assertEqual(astar.astar(graph.simple_graph(), "C", "C"), ["C"])

    def test_el_agente_se_queda_idle_y_no_lanza(self) -> None:
        agv = agent.Agent(1, grafo_partido(), "A")
        with self.assertLogs(level="WARNING"):
            self.assertFalse(agv.assign_task("A", "Y"))
        self.assertEqual(agv.state, agent.STATE_IDLE)
        self.assertEqual(agv.path, [])
        self.assertEqual(agv.path_index, 0)
        self.assertEqual(agv.progress, 0.0)
        self.assertIsNone(agv.task)
        self.assertIsNone(agv.next_node())

    def test_un_destino_que_no_existe_tampoco_lanza(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "A")
        with self.assertLogs(level="WARNING"):
            self.assertFalse(agv.assign_task("A", "Atlantida"))
        self.assertEqual(agv.state, agent.STATE_IDLE)
        # Se queda donde estaba: apuntarlo a un nodo inexistente lo dejaria en un
        # sitio que no se puede ni dibujar.
        self.assertEqual(agv.current_node, "A")

    def test_la_simulacion_aguanta_un_mapa_sin_ruta(self) -> None:
        with self.assertLogs(level="WARNING"):
            simulacion = simulation.Simulation(grafo_partido(), 1)
        for _ in range(20):
            instantanea = simulacion.get_snapshot()
        self.assertEqual(validar_snapshot(instantanea), "")
        self.assertEqual(instantanea["agents"][0]["state"], agent.STATE_IDLE)
        self.assertEqual(instantanea["step"], 20)


class TestDestinoNuevo(unittest.TestCase):
    """CRITERIO 2: cambiar el destino recalcula la ruta."""

    def test_reasignar_recalcula(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "A")
        agv.assign_task("A", "F")
        self.assertEqual(agv.path, ["A", "B", "E", "F"])

        agv.assign_task("A", "D")
        self.assertEqual(agv.path, ["A", "D"])
        self.assertEqual(agv.target_node, "D")
        self.assertEqual(agv.path_index, 0)
        self.assertEqual(agv.progress, 0.0)

    def test_reasignar_a_media_ruta_arranca_de_cero(self) -> None:
        almacen = graph.warehouse_graph()
        simulacion = simulation.Simulation(almacen, 1)
        for _ in range(10):
            simulacion.tick()

        agv = simulacion.agents[0]
        self.assertGreater(agv.path_index, 0)

        agv.assign_task(agv.current_node, "S6")
        self.assertEqual(agv.path[0], agv.current_node)
        self.assertEqual(agv.path[-1], "S6")
        self.assertEqual(agv.path_index, 0)
        self.assertEqual(agv.progress, 0.0)
        self.assertEqual(agv.state, agent.STATE_MOVING)

    def test_desde_el_destino_mismo_queda_done(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "F")
        self.assertTrue(agv.assign_task("F", "F"))
        self.assertEqual(agv.path, ["F"])
        self.assertEqual(agv.state, agent.STATE_DONE)
        self.assertTrue(agv.has_arrived())


class TestPenalizaciones(unittest.TestCase):
    """El gancho del REROUTE de la fase 8."""

    def setUp(self) -> None:
        self.grafo = graph.simple_graph()

    def test_penalizar_un_nodo_desvia_la_ruta(self) -> None:
        self.assertEqual(astar.astar(self.grafo, "A", "F"), ["A", "B", "E", "F"])
        self.assertEqual(
            astar.astar(self.grafo, "A", "F", {"E": 5.0}), ["A", "B", "C", "F"]
        )

    def test_penalizar_una_arista_desvia_la_ruta(self) -> None:
        self.assertEqual(
            astar.astar(self.grafo, "A", "F", {("B", "E"): 5.0}), ["A", "B", "C", "F"]
        )

    def test_en_un_grafo_no_dirigido_la_arista_vale_en_los_dos_sentidos(self) -> None:
        # La ruta cruza B->E, pero la penalizacion viene escrita al reves.
        self.assertEqual(
            astar.astar(self.grafo, "A", "F", {("E", "B"): 5.0}), ["A", "B", "C", "F"]
        )

    def test_en_un_grafo_dirigido_cada_sentido_va_por_su_cuenta(self) -> None:
        dirigido = graph.WarehouseGraph(
            adjacency={"A": {"B": 1.0}, "B": {"A": 1.0}},
            positions={"A": (0.0, 0.0), "B": (1.0, 0.0)},
            directed=True,
        )
        self.assertAlmostEqual(astar.path_cost(dirigido, ["A", "B"], {("B", "A"): 9.0}), 1.0)
        self.assertAlmostEqual(astar.path_cost(dirigido, ["A", "B"], {("A", "B"): 9.0}), 10.0)

    def test_una_penalizacion_de_cero_no_cambia_nada(self) -> None:
        sin = astar.astar(self.grafo, "A", "F")
        con = astar.astar(self.grafo, "A", "F", {"E": 0.0, ("B", "E"): 0.0})
        self.assertEqual(sin, con)

    def test_sin_penalizaciones_es_lo_mismo_que_con_el_diccionario_vacio(self) -> None:
        self.assertEqual(
            astar.astar(self.grafo, "A", "F"), astar.astar(self.grafo, "A", "F", {})
        )

    def test_al_nodo_de_partida_no_se_le_cobra(self) -> None:
        # No se "entra" en el origen: el agente ya estaba ahi.
        ruta = astar.astar(self.grafo, "A", "F", {"A": 1000.0})
        self.assertEqual(ruta, ["A", "B", "E", "F"])
        self.assertAlmostEqual(astar.path_cost(self.grafo, ruta, {"A": 1000.0}), 7.0)

    def test_la_penalizacion_suma_al_costo(self) -> None:
        castigos = {"E": 2.5}
        self.assertAlmostEqual(
            astar.path_cost(self.grafo, ["A", "B", "E", "F"], castigos), 9.5
        )

    def test_penalizar_el_cuello_de_botella_no_parte_el_almacen(self) -> None:
        # G es la unica union entre las dos mitades: por caro que se ponga, la
        # ruta sigue existiendo y sigue pasando por el.
        almacen = graph.warehouse_graph()
        ruta = astar.astar(almacen, "S1", "N6", {graph.BOTTLENECK: 1000.0})
        self.assertIsNotNone(ruta)
        self.assertIn(graph.BOTTLENECK, ruta)

    def test_el_agente_acepta_penalizaciones(self) -> None:
        agv = agent.Agent(1, self.grafo, "A")
        agv.assign_task("A", "F", penalties={"E": 5.0})
        self.assertEqual(agv.path, ["A", "B", "C", "F"])


class TestDeterminismo(unittest.TestCase):
    """Dos corridas con la misma semilla dan exactamente lo mismo."""

    def test_astar_siempre_devuelve_la_misma_ruta(self) -> None:
        almacen = graph.warehouse_graph()
        primera = astar.astar(almacen, "S1", "N6")
        for _ in range(50):
            self.assertEqual(astar.astar(almacen, "S1", "N6"), primera)

    def test_desempata_por_nombre_de_nodo(self) -> None:
        # Dos rutas de costo identico (A->B->C y A->D->C, ambas 2.0). Gana la que
        # pasa por el nombre menor, y gana siempre.
        empate = graph.WarehouseGraph(
            adjacency={
                "A": {"B": 1.0, "D": 1.0},
                "B": {"A": 1.0, "C": 1.0},
                "D": {"A": 1.0, "C": 1.0},
                "C": {"B": 1.0, "D": 1.0},
            },
            positions={
                "A": (0.0, 0.0),
                "B": (1.0, 1.0),
                "D": (1.0, -1.0),
                "C": (2.0, 0.0),
            },
        )
        self.assertEqual(astar.astar(empate, "A", "C"), ["A", "B", "C"])

    def test_dos_simulaciones_con_la_misma_semilla_van_a_la_par(self) -> None:
        una = simulation.Simulation(graph.warehouse_graph(), 3)
        otra = simulation.Simulation(graph.warehouse_graph(), 3)
        for _ in range(40):
            self.assertEqual(una.get_snapshot(), otra.get_snapshot())

    def test_reset_repite_la_corrida_clavada(self) -> None:
        # `stats.run` es lo unico que cambia entre corridas, y a proposito: es el
        # contador que deja a Unity distinguir un reinicio de un `step` estancado.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 1)
        primera = [sin_el_numero_de_corrida(simulacion.get_snapshot()) for _ in range(30)]
        simulacion.reset()
        segunda = [sin_el_numero_de_corrida(simulacion.get_snapshot()) for _ in range(30)]
        self.assertEqual(primera, segunda)

    def test_el_numero_de_corrida_sube_en_cada_reset(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 1)
        self.assertEqual(simulacion.get_snapshot()["stats"]["run"], 1)
        simulacion.reset()
        self.assertEqual(simulacion.get_snapshot()["stats"]["run"], 2)

    def test_la_semilla_sale_de_config(self) -> None:
        self.assertEqual(
            simulation.Simulation(graph.simple_graph(), 1).seed, config.RANDOM_SEED
        )


class TestAgent(unittest.TestCase):
    """La clase Agent: campos, estados y rutas propias."""

    CAMPOS = {
        "id",
        "current_node",
        "target_node",
        "path",
        "path_index",
        "state",
        "wait_time",
        "task",
        "progress",
    }

    def test_tiene_los_campos_pedidos_con_su_nombre_exacto(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "A")
        self.assertTrue(self.CAMPOS.issubset(vars(agv)))

    def test_arranca_parado_y_sin_tarea(self) -> None:
        agv = agent.Agent(7, graph.simple_graph(), "B")
        self.assertEqual(agv.id, 7)
        self.assertEqual(agv.current_node, "B")
        self.assertIsNone(agv.target_node)
        self.assertEqual(agv.path, [])
        self.assertEqual(agv.path_index, 0)
        self.assertEqual(agv.state, agent.STATE_IDLE)
        self.assertEqual(agv.wait_time, 0)
        self.assertIsNone(agv.task)
        self.assertEqual(agv.progress, 0.0)

    def test_los_estados_son_los_cuatro_del_contrato(self) -> None:
        self.assertEqual(agent.STATES, ("idle", "moving", "waiting", "done"))

    def test_dos_agentes_no_comparten_la_lista_de_ruta(self) -> None:
        grafo = graph.simple_graph()
        uno = agent.Agent(1, grafo, "A")
        otro = agent.Agent(2, grafo, "A")

        self.assertIsNot(uno.path, otro.path)
        uno.assign_task("A", "F")
        otro.assign_task("A", "F")
        self.assertEqual(uno.path, otro.path)
        self.assertIsNot(uno.path, otro.path)

        uno.path.append("BASURA")
        self.assertNotIn("BASURA", otro.path)

    def test_la_ruta_no_es_la_misma_lista_que_devolvio_astar(self) -> None:
        grafo = graph.simple_graph()
        agv = agent.Agent(1, grafo, "A")
        ruta = astar.astar(grafo, "A", "F")
        with mock.patch.object(astar, "astar", return_value=ruta):
            agv.assign_task("A", "F")
        self.assertIsNot(agv.path, ruta)

    def test_next_node_y_has_arrived(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "A")
        agv.assign_task("A", "F")

        self.assertEqual(agv.next_node(), "B")
        self.assertFalse(agv.has_arrived())
        self.assertIsNone(agv.previous_node())

        agv.path_index = len(agv.path) - 1
        agv.current_node = agv.path[-1]
        self.assertIsNone(agv.next_node())
        self.assertTrue(agv.has_arrived())
        self.assertEqual(agv.previous_node(), "E")

    def test_un_agente_sin_ruta_no_tiene_siguiente_ni_ha_llegado(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "A")
        self.assertIsNone(agv.next_node())
        self.assertFalse(agv.has_arrived())

    def test_reset_lo_deja_como_recien_creado(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "A")
        agv.assign_task("A", "F", task=3)
        agv.path_index, agv.progress, agv.wait_time = 2, 0.5, 4
        agv.state = agent.STATE_WAITING

        agv.reset()
        self.assertEqual(agv.current_node, "A")
        self.assertIsNone(agv.target_node)
        self.assertEqual(agv.path, [])
        self.assertEqual(agv.path_index, 0)
        self.assertEqual(agv.state, agent.STATE_IDLE)
        self.assertEqual(agv.wait_time, 0)
        self.assertIsNone(agv.task)
        self.assertEqual(agv.progress, 0.0)

    def test_guarda_el_id_de_la_tarea(self) -> None:
        agv = agent.Agent(1, graph.simple_graph(), "A")
        agv.assign_task("A", "F", task=42)
        self.assertEqual(agv.task, 42)


class TestSimulation(unittest.TestCase):
    """La simulacion: ticks, progreso continuo y reinicio."""

    def setUp(self) -> None:
        self.simulacion = simulation.Simulation(graph.simple_graph(), 1)

    def test_cumple_el_contrato_que_pide_el_servidor(self) -> None:
        self.assertIsInstance(self.simulacion, protocol.Simulation)

    def test_arranca_en_el_paso_cero(self) -> None:
        self.assertEqual(self.simulacion.step, 0)

    def test_tick_incrementa_el_paso(self) -> None:
        for esperado in (1, 2, 3):
            self.assertEqual(self.simulacion.tick(), esperado)
            self.assertEqual(self.simulacion.step, esperado)

    def test_get_snapshot_avanza_un_paso(self) -> None:
        self.assertEqual(self.simulacion.get_snapshot()["step"], 1)
        self.assertEqual(self.simulacion.get_snapshot()["step"], 2)

    def test_el_progreso_avanza_una_fraccion_del_costo(self) -> None:
        # A->B cuesta 2.0, asi que cada tick suma medio tramo.
        agv = self.simulacion.agents[0]
        self.assertEqual(agv.current_node, "A")
        self.simulacion.tick()
        self.assertAlmostEqual(agv.progress, 0.5)
        self.assertEqual(agv.current_node, "A")
        self.simulacion.tick()
        # Al completar el tramo salta de nodo y el progreso vuelve a cero.
        self.assertEqual(agv.current_node, "B")
        self.assertEqual(agv.progress, 0.0)

    def test_tarda_los_ticks_que_cuesta_la_ruta(self) -> None:
        # simple: A->B(2) + B->E(3) + E->F(2) = 7 ticks.
        for _ in range(7):
            self.simulacion.tick()
        agv = self.simulacion.agents[0]
        self.assertEqual(agv.current_node, "F")
        self.assertEqual(agv.state, agent.STATE_DONE)
        self.assertTrue(self.simulacion.done)

    def test_en_el_almacen_cruza_entero_y_llega(self) -> None:
        # warehouse: 4+4+5.7+5.7+4+4, y los tramos de 5.7 se comen 6 ticks cada
        # uno porque el tick que cruza el 1.0 es el que llega.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 1)
        while not simulacion.done and simulacion.step < 200:
            simulacion.tick()
        agv = simulacion.agents[0]
        self.assertEqual(simulacion.step, 28)
        self.assertEqual(agv.current_node, "N6")
        self.assertEqual(agv.state, agent.STATE_DONE)

    def test_el_paso_sigue_creciendo_cuando_ya_llegaron_todos(self) -> None:
        # El cliente de Unity exige que `step` no se estanque nunca.
        for _ in range(7):
            self.simulacion.tick()
        self.assertTrue(self.simulacion.done)
        self.assertEqual(self.simulacion.tick(), 8)
        self.assertEqual(self.simulacion.get_snapshot()["step"], 9)

    def test_la_posicion_se_interpola_entre_los_dos_nodos(self) -> None:
        agv = self.simulacion.agents[0]
        self.simulacion.tick()  # medio tramo A->B
        instantanea = self.simulacion.get_snapshot()  # y otro medio: llega a B
        self.assertEqual(agv.current_node, "B")

        simulacion = simulation.Simulation(graph.simple_graph(), 1)
        simulacion.tick()
        agente_medio = simulacion._describe(simulacion.agents[0])
        # A=(0,0) y B=(2,0): a mitad de camino tiene que estar en x=1.
        self.assertAlmostEqual(agente_medio["x"], 1.0)
        self.assertAlmostEqual(agente_medio["z"], 0.0)
        self.assertEqual(instantanea["agents"][0]["node"], "B")

    def test_no_hay_teletransportes(self) -> None:
        # Entre dos snapshots seguidos el AGV nunca salta mas de lo que mide el
        # tramo mas largo del mapa.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 1)
        anterior = simulacion.get_snapshot()["agents"][0]
        for _ in range(40):
            actual = simulacion.get_snapshot()["agents"][0]
            salto = abs(actual["x"] - anterior["x"]) + abs(actual["z"] - anterior["z"])
            self.assertLessEqual(salto, 8.0)
            anterior = actual

    def test_reset_deja_el_contador_en_cero_y_al_agente_en_su_sitio(self) -> None:
        for _ in range(5):
            self.simulacion.tick()
        self.simulacion.reset()
        self.assertEqual(self.simulacion.step, 0)
        agv = self.simulacion.agents[0]
        self.assertEqual(agv.current_node, "A")
        self.assertEqual(agv.path_index, 0)
        self.assertEqual(agv.progress, 0.0)
        self.assertEqual(agv.state, agent.STATE_MOVING)
        self.assertEqual(self.simulacion.get_snapshot()["step"], 1)

    def test_respeta_el_origen_y_el_destino_que_le_pasan(self) -> None:
        simulacion = simulation.Simulation(
            graph.simple_graph(), 1, origin="D", target="C"
        )
        agv = simulacion.agents[0]
        self.assertEqual(agv.path[0], "D")
        self.assertEqual(agv.path[-1], "C")

    def test_la_ruta_por_defecto_de_cada_mapa(self) -> None:
        self.assertEqual(simulation.default_route(graph.simple_graph()), ("A", "F"))
        self.assertEqual(
            simulation.default_route(graph.warehouse_graph()), ("S1", "N6")
        )

    def test_un_mapa_desconocido_usa_el_primer_y_el_ultimo_nodo(self) -> None:
        self.assertEqual(simulation.default_route(grafo_partido()), ("A", "Y"))

    def test_no_admite_cero_agentes(self) -> None:
        with self.assertRaises(ValueError):
            simulation.Simulation(graph.simple_graph(), 0)

    def test_varios_agentes_tienen_cada_uno_su_ruta(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 3)
        self.assertEqual(len(simulacion.agents), 3)
        self.assertEqual([a.id for a in simulacion.agents], [1, 2, 3])
        for uno, otro in itertools.combinations(simulacion.agents, 2):
            self.assertIsNot(uno.path, otro.path)

    def test_es_segura_entre_hilos(self) -> None:
        # El servidor comparte una sola simulacion entre todos los clientes: ni
        # un paso repetido ni uno perdido.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 1)
        pasos: list[int] = []
        cerrojo = threading.Lock()

        def tirar() -> None:
            for _ in range(200):
                paso = simulacion.get_snapshot()["step"]
                with cerrojo:
                    pasos.append(paso)

        hilos = [threading.Thread(target=tirar) for _ in range(4)]
        for hilo in hilos:
            hilo.start()
        for hilo in hilos:
            hilo.join()

        self.assertEqual(len(pasos), 800)
        self.assertEqual(len(set(pasos)), 800)


class TestSnapshot(unittest.TestCase):
    """CRITERIO 5: el snapshot sigue valiendo para el cliente de la fase 1."""

    def setUp(self) -> None:
        self.simulacion = simulation.Simulation(graph.warehouse_graph(), 1)

    def test_lo_valida_el_cliente_de_prueba_de_la_fase_1(self) -> None:
        for _ in range(30):
            self.assertEqual(validar_snapshot(self.simulacion.get_snapshot()), "")

    def test_conserva_los_seis_campos_congelados_con_su_tipo(self) -> None:
        agente = self.simulacion.get_snapshot()["agents"][0]
        self.assertEqual(set(CAMPOS_AGENTE) - set(agente), set())
        self.assertIsInstance(agente["id"], int)
        self.assertIsInstance(agente["state"], str)
        for eje in ("x", "y", "z", "rotation"):
            self.assertIsInstance(agente[eje], float)

    def test_el_paso_es_un_entero_que_empieza_en_uno(self) -> None:
        instantanea = self.simulacion.get_snapshot()
        self.assertIsInstance(instantanea["step"], int)
        self.assertEqual(instantanea["step"], 1)

    def test_agrega_los_cuatro_campos_de_la_fase_3(self) -> None:
        agente = self.simulacion.get_snapshot()["agents"][0]
        self.assertEqual(agente["node"], "S1")
        self.assertEqual(agente["next_node"], "S2")
        self.assertEqual(agente["path"][0], "S1")
        self.assertEqual(agente["path"][-1], "N6")
        self.assertEqual(agente["task"], 1)

    def test_el_path_es_una_lista_de_strings(self) -> None:
        agente = self.simulacion.get_snapshot()["agents"][0]
        self.assertIsInstance(agente["path"], list)
        for nodo in agente["path"]:
            self.assertIsInstance(nodo, str)

    def test_el_estado_es_uno_de_los_del_contrato(self) -> None:
        for _ in range(30):
            for agente in self.simulacion.get_snapshot()["agents"]:
                self.assertIn(agente["state"], agent.STATES)

    def test_la_altura_siempre_es_cero(self) -> None:
        # La altura la aplica Unity con el prefab, Python solo manda el suelo.
        for _ in range(10):
            self.assertEqual(self.simulacion.get_snapshot()["agents"][0]["y"], 0.0)

    def test_se_serializa_en_una_sola_linea(self) -> None:
        linea = protocol.encode_snapshot(self.simulacion.get_snapshot())
        self.assertTrue(linea.endswith("\n"))
        self.assertNotIn("\n", linea[:-1])
        self.assertEqual(json.loads(linea)["step"], 1)

    def test_el_path_sobrevive_al_ida_y_vuelta_por_json(self) -> None:
        instantanea = self.simulacion.get_snapshot()
        redondo = json.loads(protocol.encode_snapshot(instantanea))
        self.assertEqual(redondo, instantanea)

    def test_next_node_es_nulo_cuando_ya_llego(self) -> None:
        while not self.simulacion.done:
            instantanea = self.simulacion.get_snapshot()
        agente = instantanea["agents"][0]
        self.assertIsNone(agente["next_node"])
        self.assertEqual(agente["state"], agent.STATE_DONE)
        self.assertEqual(agente["node"], "N6")

    def test_el_snapshot_no_deja_tocar_la_ruta_del_agente_por_dentro(self) -> None:
        instantanea = self.simulacion.get_snapshot()
        instantanea["agents"][0]["path"].append("BASURA")
        self.assertNotIn("BASURA", self.simulacion.agents[0].path)


class TestSimulateCLI(unittest.TestCase):
    """El subcomando `simulate`."""

    def test_llega_al_destino_en_el_mapa_simple(self) -> None:
        with self.assertLogs(level="INFO") as capturado:
            self.assertEqual(
                main.main(["simulate", "--map", "simple", "--steps", "50", "--headless"]),
                0,
            )
        salida = "\n".join(capturado.output)
        self.assertIn("A -> B -> E -> F", salida)
        self.assertIn("done en F", salida)

    def test_cruza_el_almacen_por_el_cuello_de_botella(self) -> None:
        with self.assertLogs(level="INFO") as capturado:
            self.assertEqual(
                main.main(
                    ["simulate", "--map", "warehouse", "--agents", "1",
                     "--steps", "100", "--headless"]
                ),
                0,
            )
        salida = "\n".join(capturado.output)
        self.assertIn("S1 -> S2 -> S3 -> G -> N4 -> N5 -> N6", salida)
        self.assertIn("pasos dados : 28 de 100", salida)

    def test_acepta_from_y_to(self) -> None:
        with self.assertLogs(level="INFO") as capturado:
            self.assertEqual(
                main.main(
                    ["simulate", "--map", "simple", "--from", "D", "--to", "C",
                     "--headless"]
                ),
                0,
            )
        self.assertIn("D -> ", "\n".join(capturado.output))

    def test_un_nodo_que_no_existe_devuelve_dos(self) -> None:
        with self.assertLogs(level="ERROR") as capturado:
            self.assertEqual(
                main.main(["simulate", "--map", "simple", "--to", "Z", "--headless"]), 2
            )
        self.assertIn("no es un nodo", capturado.output[0])

    def test_un_mapa_que_no_existe_devuelve_dos(self) -> None:
        with self.assertLogs(level="ERROR"):
            self.assertEqual(
                main.main(["simulate", "--map", "atlantida", "--headless"]), 2
            )

    def test_sin_ruta_devuelve_uno_y_no_revienta(self) -> None:
        with mock.patch.object(main, "_abre_mapa", return_value=(grafo_partido(), "test", 0)):
            with self.assertLogs(level="WARNING") as capturado:
                self.assertEqual(main.main(["simulate", "--headless"]), 1)
        self.assertIn("no hay ruta", "\n".join(capturado.output))

    def test_sin_headless_avisa_pero_corre(self) -> None:
        with self.assertLogs(level="WARNING") as capturado:
            self.assertEqual(main.main(["simulate", "--map", "simple"]), 0)
        self.assertIn("headless", "\n".join(capturado.output))

    def test_con_varios_agentes_gestiona_los_conflictos(self) -> None:
        # Hasta la fase 4 esto avisaba de que los AGVs se cruzarian sin verse.
        # Ahora se ven: el resumen trae el conteo de conflictos y por que acabo.
        with self.assertLogs(level="INFO") as capturado:
            self.assertEqual(
                main.main(
                    ["simulate", "--map", "warehouse", "--agents", "3",
                     "--steps", "40", "--headless"]
                ),
                0,
            )
        salida = "\n".join(capturado.output)
        self.assertIn("conflictos  :", salida)
        self.assertIn("final       :", salida)
        self.assertIn("espera total:", salida)
        self.assertNotIn("colisiones", salida)

    def test_mas_agentes_que_nodos_no_arranca(self) -> None:
        # Cada AGV necesita un nodo de salida para el solo, o la invariante de
        # ocupacion nace rota antes de mover nada.
        with self.assertLogs(level="ERROR") as capturado:
            self.assertEqual(
                main.main(
                    ["simulate", "--map", "simple", "--agents", "7", "--headless"]
                ),
                2,
            )
        self.assertIn("no caben", "\n".join(capturado.output))

    def test_los_valores_por_defecto_del_parser(self) -> None:
        args = main.build_parser().parse_args(["simulate"])
        self.assertEqual(args.map, config.DEFAULT_MAP)
        self.assertEqual(args.agents, 1)
        self.assertEqual(args.steps, 100)
        self.assertFalse(args.headless)
        self.assertIsNone(args.origen)
        self.assertIsNone(args.destino)


if __name__ == "__main__":
    unittest.main()
