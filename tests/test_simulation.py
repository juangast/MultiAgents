"""Tests de la simulacion: el tick, el snapshot y el determinismo.

Los dos criterios que se prueban aqui:

- **Determinismo por semilla.** Dos simulaciones con la misma semilla y el mismo
  mapa producen la MISMA secuencia de snapshots, tick a tick. Sin eso no hay
  experimento que valga: la fase 10 compara dos politicas suponiendo que lo
  unico distinto entre dos corridas es la politica.
- **El snapshot es valido.** Se comprueba con `validar_snapshot()`, que es
  exactamente la misma funcion que usa el cliente falso de Unity, para que lo
  que pasa el test sea lo que Unity va a poder leer.
"""

import itertools
import json
import threading
import unittest

import agent
import config
import graph
import protocol
import simulation
from tests.fake_unity_client import CAMPOS_AGENTE, validar_snapshot


def grafo_partido() -> graph.WarehouseGraph:
    """Dos islas que no se tocan: de la primera no se llega a la segunda."""
    return graph.WarehouseGraph(
        adjacency={"A": {"B": 1.0}, "B": {"A": 1.0}, "X": {"Y": 1.0}, "Y": {"X": 1.0}},
        positions={"A": (0.0, 0.0), "B": (1.0, 0.0), "X": (9.0, 0.0), "Y": (10.0, 0.0)},
        name="partido",
    )


def sin_el_numero_de_corrida(instantanea: dict) -> dict:
    """El snapshot sin `stats.run`, para comparar dos corridas entre si."""
    copia = dict(instantanea)
    copia["stats"] = {
        clave: valor for clave, valor in copia["stats"].items() if clave != "run"
    }
    return copia


class TestDeterminismo(unittest.TestCase):
    """CRITERIO: la misma semilla da exactamente la misma corrida."""

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


class TestDeterminismoPorSemilla(unittest.TestCase):
    """CRITERIO: la semilla manda, y semillas distintas dan corridas distintas.

    Las dos mitades importan. Que la misma semilla repita es lo que hace
    comparable un experimento; que semillas distintas NO repitan es lo que hace
    que veinte corridas sean veinte corridas y no una veinte veces. Sin la
    segunda, un test de reproducibilidad pasa sin medir nada.
    """

    def secuencia(self, semilla: int, pasos: int = 60) -> list[dict]:
        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 4, seed=semilla
        )
        return [sin_el_numero_de_corrida(simulacion.get_snapshot()) for _ in range(pasos)]

    def test_la_misma_semilla_da_la_misma_secuencia_tick_a_tick(self) -> None:
        self.assertEqual(self.secuencia(7), self.secuencia(7))

    def test_semillas_distintas_dan_corridas_distintas(self) -> None:
        self.assertNotEqual(self.secuencia(7), self.secuencia(8))

    def test_la_semilla_reparte_las_rutas_y_no_mueve_el_motor(self) -> None:
        # Lo unico que la semilla decide es el reparto de tareas: el AGV 1 hace
        # siempre la ruta fija del mapa, pase lo que pase.
        for semilla in (1, 2, 3, 99):
            simulacion = simulation.Simulation(
                graph.warehouse_graph(), 4, seed=semilla
            )
            self.assertEqual(simulacion.agents[0].current_node, "S1")
            self.assertEqual(simulacion.agents[0].target_node, "N6")

    def test_los_dos_modos_arrancan_igual_con_la_misma_semilla(self) -> None:
        # La invariante de la fase 8, comprobada desde aqui: entre `baseline` y
        # `qlearning` lo unico distinto es la politica, nunca el escenario.
        def rutas(modo: str) -> list[tuple]:
            simulacion = simulation.Simulation(
                graph.warehouse_graph(), 4, seed=5, policy=modo
            )
            return [
                (uno.id, uno.current_node, uno.target_node, tuple(uno.path))
                for uno in simulacion.agents
            ]

        if not config.Q_TABLE_FILE.is_file():
            self.skipTest(f"hace falta {config.Q_TABLE_FILE}")
        self.assertEqual(rutas("baseline"), rutas("qlearning"))


class TestSnapshotValido(unittest.TestCase):
    """CRITERIO: el snapshot cumple el contrato en cada tick, no solo el primero."""

    def test_200_snapshots_pasan_el_validador_del_cliente_de_unity(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 6)
        anterior = 0
        for _ in range(200):
            instantanea = simulacion.get_snapshot()
            problema = validar_snapshot(instantanea)
            self.assertEqual(problema, "", f"paso {instantanea['step']}: {problema}")
            self.assertGreater(instantanea["step"], anterior)
            anterior = instantanea["step"]

    def test_es_json_de_una_sola_linea(self) -> None:
        # El contrato es "una linea entra, una linea sale": si un snapshot
        # llevara un salto de linea dentro, el cliente perderia el emparejado.
        simulacion = simulation.Simulation(graph.warehouse_graph(), 6)
        for _ in range(50):
            linea = protocol.encode_snapshot(simulacion.get_snapshot())
            self.assertEqual(linea.count("\n"), 1)
            self.assertTrue(linea.endswith("\n"))
            json.loads(linea)


if __name__ == "__main__":
    unittest.main()
