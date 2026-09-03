"""Tests de integracion: el sistema entero, de punta a punta.

Los otros ficheros prueban piezas. Este prueba que las piezas encajan, y lo hace
por el mismo camino que va a usar Unity: **socket de verdad, comandos de verdad,
JSON de verdad**. Si algo de aqui falla, lo que esta roto es la entrega, aunque
todas las piezas pasen sus tests por separado.

Dos corridas completas, una por politica, sobre el MISMO escenario: es la
condicion de la fase 9 (lo unico distinto entre dos corridas es la politica) y es
lo que hace que comparar los dos numeros signifique algo.
"""

import json
import socket
import threading
import unittest

import config
import graph
import metrics
import protocol
import scenarios
import server
import simulation
from agent import STATE_DONE
from tests.fake_unity_client import validar_snapshot

TIMEOUT: float = 5.0

hay_modelo = config.Q_TABLE_FILE.is_file()
sin_modelo = unittest.skipUnless(
    hay_modelo, f"hace falta {config.Q_TABLE_FILE}: python3 python/main.py train"
)


class ClienteDePrueba:
    """Un cliente TCP minimo, que lee por lineas como hay que leer."""

    def __init__(self, direccion: tuple[str, int]) -> None:
        self.sock = socket.create_connection(direccion, timeout=TIMEOUT)
        self._buffer = bytearray()

    def pide(self, comando: str) -> dict:
        self.sock.sendall(f"{comando}\n".encode(config.ENCODING))
        while b"\n" not in self._buffer:
            dato = self.sock.recv(4096)
            if not dato:
                raise AssertionError("el servidor cerro la conexion")
            self._buffer.extend(dato)
        corte = self._buffer.index(b"\n")
        linea = bytes(self._buffer[:corte])
        del self._buffer[: corte + 1]
        return json.loads(linea.decode(config.ENCODING))

    def close(self) -> None:
        self.sock.close()


class ServidorLevantado(unittest.TestCase):
    """Monta el servidor de verdad con la simulacion que le pase cada test."""

    def levanta(self, simulacion) -> ClienteDePrueba:
        servidor = server.AGVServer((config.HOST, 0), simulacion)
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        cliente = ClienteDePrueba(servidor.server_address)

        def cerrar() -> None:
            cliente.close()
            servidor.shutdown()
            servidor.server_close()
            hilo.join(timeout=TIMEOUT)

        self.addCleanup(cerrar)
        return cliente


class TestCorridaCompletaPorSocket(ServidorLevantado):
    """Una corrida entera pedida por el socket, como la pediria Unity."""

    def corre(self, modo: str, pasos: int = 300) -> dict:
        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 4, policy=modo, model=config.Q_TABLE_FILE
        )
        cliente = self.levanta(simulacion)

        ultimo: dict = {}
        for esperado in range(1, pasos + 1):
            ultimo = cliente.pide(config.CMD_GET_STATE)
            self.assertEqual(validar_snapshot(ultimo), "")
            self.assertEqual(ultimo["step"], esperado)
            self.assertEqual(ultimo["mode"], modo)
            if simulacion.done:
                break
        return ultimo

    def test_baseline_de_punta_a_punta(self) -> None:
        ultimo = self.corre("baseline")
        self.assertGreater(ultimo["stats"]["actions"]["advance"], 0)
        self.assertIsNone(ultimo["stats"]["finished_reason"])

    @sin_modelo
    def test_qlearning_de_punta_a_punta(self) -> None:
        ultimo = self.corre("qlearning")
        recuento = ultimo["stats"]["actions"]
        self.assertGreater(recuento["advance"], 0)
        # La politica aprendida cede y rodea; si solo avanzara, no seria ella.
        self.assertGreater(recuento["wait"] + recuento["reroute"], 0)
        self.assertIsNone(ultimo["stats"]["finished_reason"])

    @sin_modelo
    def test_las_dos_politicas_por_el_mismo_socket(self) -> None:
        # SET_MODE cambia de politica en caliente y arranca corrida limpia: es
        # como se ensena la diferencia en la demo sin reiniciar el servidor.
        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 4, policy="baseline", model=config.Q_TABLE_FILE
        )
        cliente = self.levanta(simulacion)

        for _ in range(40):
            self.assertEqual(cliente.pide(config.CMD_GET_STATE)["mode"], "baseline")

        respuesta = cliente.pide(f"{config.CMD_SET_MODE} qlearning")
        self.assertTrue(respuesta["ok"])
        self.assertEqual(respuesta["mode"], "qlearning")

        instantanea = cliente.pide(config.CMD_GET_STATE)
        self.assertEqual(instantanea["mode"], "qlearning")
        self.assertEqual(instantanea["step"], 1, "la corrida nueva empieza en 1")
        self.assertEqual(validar_snapshot(instantanea), "")

    def test_los_cuatro_comandos_contestan(self) -> None:
        cliente = self.levanta(simulation.Simulation(graph.simple_graph(), 1))
        self.assertEqual(cliente.pide(config.CMD_PING), {"ok": True})
        self.assertEqual(cliente.pide(config.CMD_GET_STATE)["step"], 1)
        self.assertEqual(cliente.pide(config.CMD_RESET), {"ok": True})
        self.assertEqual(cliente.pide(config.CMD_GET_STATE)["step"], 1)
        self.assertTrue(cliente.pide(f"{config.CMD_SET_MODE} baseline")["ok"])
        self.assertEqual(
            cliente.pide("BASURA")["error"], protocol.ERROR_UNKNOWN_COMMAND
        )
        # Y despues de todo eso el servidor sigue vivo.
        self.assertEqual(cliente.pide(config.CMD_PING), {"ok": True})


class TestEscenarioCompleto(unittest.TestCase):
    """El escenario con cola de tareas, corrido con las dos politicas."""

    def test_el_escenario_de_baja_congestion_despacha_su_cola(self) -> None:
        spec = scenarios.get("A")
        escenario = spec.build(spec.seed)
        medidas = metrics.run_once(
            spec.graph(), escenario, "baseline", max_steps=spec.max_steps
        )
        numeros = medidas.to_dict()
        self.assertEqual(numeros["completed_tasks"], escenario.n_tasks)
        self.assertTrue(medidas.all_completed)
        self.assertGreater(medidas.makespan, 0)
        self.assertIsNone(numeros["finished_reason"])

    @sin_modelo
    def test_las_dos_politicas_ven_exactamente_el_mismo_trabajo(self) -> None:
        # La invariante de la fase 9. Si esto falla, ninguna comparacion del
        # proyecto vale nada, porque las politicas no estarian midiendo lo mismo.
        spec = scenarios.get("B")
        semillas = spec.seeds(2)
        resultados = metrics.run_comparison(
            spec.graph(),
            spec.n_agents,
            spec.n_tasks,
            semillas,
            model=config.Q_TABLE_FILE,
            max_steps=spec.max_steps,
            builder=spec.build,
        )

        self.assertEqual(set(resultados), set(config.POLICIES))
        for semilla in semillas:
            uno, otro = (spec.build(semilla), spec.build(semilla))
            self.assertEqual(uno.routes, otro.routes)
            self.assertEqual(uno.pending, otro.pending)

        for nombre, corridas in resultados.items():
            self.assertEqual(len(corridas), len(semillas), nombre)
            for una in corridas:
                self.assertEqual(una.policy, nombre)
                self.assertEqual(una.n_agents, spec.n_agents)

    @sin_modelo
    def test_del_escenario_salen_metricas_comparables(self) -> None:
        spec = scenarios.get("A")
        resultados = metrics.run_comparison(
            spec.graph(),
            spec.n_agents,
            spec.n_tasks,
            spec.seeds(2),
            model=config.Q_TABLE_FILE,
            max_steps=spec.max_steps,
            builder=spec.build,
        )
        for nombre, corridas in resultados.items():
            resumen = metrics.summarize(corridas)
            self.assertEqual(resumen["runs"], 2, nombre)
            for campo in metrics.METRIC_FIELDS:
                self.assertIn(campo, resumen["metrics"], f"{nombre} no trae {campo}")

        lineas = metrics.comparison_lines(resultados)
        self.assertTrue(any("makespan" in linea for linea in lineas))


class TestSimulacionCompletaSinServidor(unittest.TestCase):
    """La misma corrida sin socket: el motor no depende del transporte."""

    def test_la_corrida_headless_llega_al_final(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 4)
        for _ in range(600):
            simulacion.tick()
            if simulacion.done:
                break
        self.assertTrue(
            all(uno.state == STATE_DONE for uno in simulacion.agents),
            [(uno.id, uno.state) for uno in simulacion.agents],
        )
        self.assertIsNone(simulacion.finished_reason)


if __name__ == "__main__":
    unittest.main()
