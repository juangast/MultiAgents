"""Tests de A*: que la ruta que devuelve es la mas barata que hay.

La prueba fuerte es `TestRutaOptima`, que compara A* contra una **busqueda
exhaustiva** sobre los 186 pares ordenados de nodos de los dos mapas del repo:
si hubiera una ruta mas barata, la fuerza bruta la encontraria.

El agente esta en `test_agent.py` y la simulacion en `test_simulation.py`.
"""

import itertools
import unittest
from unittest import mock

import agent
import astar
import graph
import simulation
from tests.fake_unity_client import validar_snapshot


def mapas_del_repo() -> list[graph.WarehouseGraph]:
    """Los dos mapas de verdad del proyecto."""
    return [graph.simple_graph(), graph.warehouse_graph()]


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
