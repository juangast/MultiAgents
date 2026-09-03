"""Tests del AGV: sus campos, su ciclo de vida y que su estado es SUYO.

Lo que mas se prueba aqui es lo segundo, porque es el error clasico de este
modelo: dos agentes compartiendo la misma lista de ruta sin querer. `Agent`
crea `path` en cada `__init__` y `assign_task()` se queda con una **copia** de
lo que devuelve A*, nunca con la lista misma.
"""

import itertools
import unittest
from unittest import mock

import agent
import astar
import graph
import simulation


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


class TestEstadoIndependiente(unittest.TestCase):
    """CRITERIO: el estado de un AGV es suyo y de nadie mas.

    Es el fallo clasico de un modelo multiagente: dos agentes acaban apuntando a
    la misma lista y mover a uno mueve al otro. Aqui se comprueba nodo a nodo,
    porque un `path` compartido no da error: da una simulacion que miente.
    """

    def dos_agentes(self) -> tuple[agent.Agent, agent.Agent, graph.WarehouseGraph]:
        grafo = graph.warehouse_graph()
        return agent.Agent(1, grafo, "S1"), agent.Agent(2, grafo, "S2"), grafo

    def test_cada_uno_nace_con_su_propia_lista_de_ruta(self) -> None:
        uno, otro, _ = self.dos_agentes()
        self.assertIsNot(uno.path, otro.path, "comparten la lista recien creados")
        uno.path.append("S2")
        self.assertEqual(otro.path, [])

    def test_la_misma_ruta_de_astar_no_se_comparte(self) -> None:
        # Los dos piden exactamente la misma ruta: si `assign_task()` guardara lo
        # que devuelve A* en vez de una copia, seria la misma lista para los dos.
        uno, otro, _ = self.dos_agentes()
        for quien in (uno, otro):
            quien.assign_task("S1", "N6", task=quien.id)
        self.assertEqual(uno.path, otro.path)
        self.assertIsNot(uno.path, otro.path)

        uno.path[0] = "TOCADO"
        self.assertEqual(otro.path[0], "S1")

    def test_avanzar_uno_no_mueve_al_otro(self) -> None:
        uno, otro, _ = self.dos_agentes()
        for quien in (uno, otro):
            quien.assign_task("S1", "N6", task=quien.id)

        uno.path_index = 3
        uno.current_node = uno.path[3]
        uno.wait_time = 11
        uno.progress = 0.5

        self.assertEqual(otro.path_index, 0)
        self.assertEqual(otro.current_node, "S1")
        self.assertEqual(otro.wait_time, 0)
        self.assertEqual(otro.progress, 0.0)

    def test_reset_de_uno_no_toca_al_otro(self) -> None:
        uno, otro, _ = self.dos_agentes()
        for quien in (uno, otro):
            quien.assign_task("S1", "N6", task=quien.id)
        otro.wait_time = 7

        uno.reset()

        self.assertEqual(uno.path, [])
        self.assertEqual(uno.state, agent.STATE_IDLE)
        self.assertNotEqual(otro.path, [])
        self.assertEqual(otro.state, agent.STATE_MOVING)
        self.assertEqual(otro.wait_time, 7)

    def test_en_la_simulacion_tampoco_se_comparte_nada(self) -> None:
        # La comprobacion de verdad: 6 AGVs sobre el mapa real, 120 ticks, y
        # ningun par comparte objeto en ninguno de sus campos mutables.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 6)
        for _ in range(120):
            simulacion.tick()
            for uno, otro in itertools.combinations(simulacion.agents, 2):
                self.assertIsNot(uno.path, otro.path)
            ids = [uno.id for uno in simulacion.agents]
            self.assertEqual(len(set(ids)), len(ids))

    def test_el_grafo_si_se_comparte_y_esta_bien(self) -> None:
        # El mapa es lo unico que SI es el mismo objeto para todos: es de solo
        # lectura y copiarlo por agente solo gastaria memoria.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 4)
        for uno in simulacion.agents:
            self.assertIs(uno.graph, simulacion.graph)


if __name__ == "__main__":
    unittest.main()
