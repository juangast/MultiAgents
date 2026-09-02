"""Tests del servidor TCP contra un socket de verdad en un puerto efimero."""

import json
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest

import config
import protocol
import server
from tests.fake_unity_client import LineReader, validar_snapshot

TIMEOUT: float = 5.0


class ServidorEnPrueba(unittest.TestCase):
    """Levanta un AGVServer en el puerto 0 y lo cierra al terminar."""

    def setUp(self) -> None:
        self.simulacion = server.FakeSimulation()
        self.servidor = server.AGVServer((config.HOST, 0), self.simulacion)
        self.hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
        self.hilo.start()
        self.direccion = self.servidor.server_address

    def tearDown(self) -> None:
        self.servidor.shutdown()
        self.servidor.server_close()
        self.hilo.join(timeout=TIMEOUT)

    def conectar(self) -> socket.socket:
        sock = socket.create_connection(self.direccion, timeout=TIMEOUT)
        self.addCleanup(sock.close)
        return sock

    def pedir(self, sock: socket.socket, comando: str) -> dict:
        """Manda un comando y devuelve la respuesta ya decodificada."""
        lector = LineReader(sock)
        sock.sendall(f"{comando}\n".encode(config.ENCODING))
        return json.loads(lector.read_line())


class TestContrato(ServidorEnPrueba):
    def test_get_state_cumple_la_forma(self) -> None:
        payload = self.pedir(self.conectar(), config.CMD_GET_STATE)
        self.assertEqual(validar_snapshot(payload), "")
        self.assertEqual(payload["step"], 1)
        self.assertEqual(payload["agents"][0]["y"], 0.0)

    def test_step_incrementa_en_cada_peticion(self) -> None:
        sock = self.conectar()
        lector = LineReader(sock)
        pasos = []
        for _ in range(3):
            sock.sendall(b"GET_STATE\n")
            pasos.append(json.loads(lector.read_line())["step"])
        self.assertEqual(pasos, [1, 2, 3])

    def test_reset_vuelve_el_step_a_cero(self) -> None:
        sock = self.conectar()
        lector = LineReader(sock)
        for _ in range(3):
            sock.sendall(b"GET_STATE\n")
            lector.read_line()

        sock.sendall(b"RESET\n")
        self.assertEqual(json.loads(lector.read_line()), {"ok": True})

        sock.sendall(b"GET_STATE\n")
        self.assertEqual(json.loads(lector.read_line())["step"], 1)

    def test_ping(self) -> None:
        self.assertEqual(self.pedir(self.conectar(), config.CMD_PING), {"ok": True})

    def test_fake_simulation_cumple_el_protocolo(self) -> None:
        self.assertIsInstance(self.simulacion, protocol.Simulation)


class TestRobustez(ServidorEnPrueba):
    def test_comando_basura_no_tumba_la_conexion(self) -> None:
        sock = self.conectar()
        lector = LineReader(sock)

        sock.sendall(b"BASURA\n")
        self.assertEqual(
            json.loads(lector.read_line()),
            {"error": "unknown_command", "command": "BASURA"},
        )

        sock.sendall(b"GET_STATE\n")
        self.assertEqual(validar_snapshot(json.loads(lector.read_line())), "")

    def test_bytes_no_utf8_no_truenan(self) -> None:
        sock = self.conectar()
        lector = LineReader(sock)

        sock.sendall(b"\xff\xfe\n")
        self.assertEqual(json.loads(lector.read_line())["error"], "unknown_command")

        sock.sendall(b"PING\n")
        self.assertEqual(json.loads(lector.read_line()), {"ok": True})

    def test_linea_partida_en_dos_envios(self) -> None:
        sock = self.conectar()
        lector = LineReader(sock)

        sock.sendall(b"GET_ST")
        sock.sendall(b"ATE\n")
        self.assertEqual(json.loads(lector.read_line())["step"], 1)

        # Si el buffer hubiera contestado de mas, aqui llegaria esa respuesta
        # sobrante en vez del ok del PING.
        sock.sendall(b"PING\n")
        self.assertEqual(json.loads(lector.read_line()), {"ok": True})

    def test_varios_comandos_en_un_solo_envio(self) -> None:
        sock = self.conectar()
        lector = LineReader(sock)

        sock.sendall(b"PING\nBASURA\nGET_STATE\n")
        primera = json.loads(lector.read_line())
        segunda = json.loads(lector.read_line())
        tercera = json.loads(lector.read_line())

        self.assertEqual(primera, {"ok": True})
        self.assertEqual(segunda["error"], "unknown_command")
        self.assertEqual(validar_snapshot(tercera), "")

    def test_linea_sin_fin_cierra_la_conexion(self) -> None:
        sock = self.conectar()
        sock.sendall(b"A" * (server.MAX_LINE_BYTES + 1024))
        sock.settimeout(TIMEOUT)
        self.assertEqual(sock.recv(1024), b"")

    def test_dos_clientes_a_la_vez(self) -> None:
        uno, dos = self.conectar(), self.conectar()
        pasos = {
            self.pedir(uno, config.CMD_GET_STATE)["step"],
            self.pedir(dos, config.CMD_GET_STATE)["step"],
        }
        self.assertEqual(pasos, {1, 2})

    def test_el_servidor_sobrevive_a_un_corte_brusco(self) -> None:
        bruto = socket.create_connection(self.direccion, timeout=TIMEOUT)
        bruto.sendall(b"GET_STATE\n")
        # SO_LINGER en 0 hace que el cierre mande RST en vez de FIN: el peor
        # caso para el servidor, el mismo que un kill -9 al cliente.
        bruto.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        bruto.close()

        nuevo = self.pedir(self.conectar(), config.CMD_GET_STATE)
        self.assertEqual(validar_snapshot(nuevo), "")


class TestCierrePorSenal(unittest.TestCase):
    """El proceso real tiene que cerrar limpio con Ctrl+C y con kill."""

    MAIN = str(config.PROJECT_ROOT / "python" / "main.py")
    ESCUCHANDO = re.compile(r"escuchando en (\S+):(\d+)")

    def arrancar(self) -> tuple[subprocess.Popen, int]:
        """Levanta el servidor en un puerto efimero y espera a que escuche."""
        proceso = subprocess.Popen(
            [sys.executable, self.MAIN, "serve", "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.addCleanup(self._rematar, proceso)

        limite = time.monotonic() + TIMEOUT
        while time.monotonic() < limite:
            linea = proceso.stdout.readline()
            if not linea:
                break
            encontrado = self.ESCUCHANDO.search(linea)
            if encontrado:
                return proceso, int(encontrado.group(2))
        self.fail("el servidor no dijo por donde escucha")

    @staticmethod
    def _rematar(proceso: subprocess.Popen) -> None:
        if proceso.poll() is None:
            proceso.kill()
            proceso.wait(timeout=TIMEOUT)

    def comprobar_cierre(self, senal: signal.Signals) -> None:
        proceso, puerto = self.arrancar()

        # Un cliente conectado y pidiendo mientras llega la senal.
        sock = socket.create_connection((config.HOST, puerto), timeout=TIMEOUT)
        self.addCleanup(sock.close)
        sock.sendall(b"PING\n")
        self.assertIn(b'"ok":true', sock.recv(1024))

        proceso.send_signal(senal)
        salida = proceso.communicate(timeout=TIMEOUT)[0]

        self.assertEqual(proceso.returncode, 0, salida)
        self.assertIn("servidor cerrado", salida)

    def test_sigint_cierra_limpio(self) -> None:
        self.comprobar_cierre(signal.SIGINT)

    def test_sigterm_cierra_limpio(self) -> None:
        self.comprobar_cierre(signal.SIGTERM)


class TestFakeSimulation(unittest.TestCase):
    def test_avanza_en_linea_recta(self) -> None:
        sim = server.FakeSimulation()
        primero = sim.get_snapshot()["agents"][0]
        segundo = sim.get_snapshot()["agents"][0]
        self.assertGreater(segundo["x"], primero["x"])
        self.assertEqual(segundo["z"], primero["z"])
        self.assertEqual(segundo["y"], 0.0)

    def test_reset_deja_el_contador_en_cero(self) -> None:
        sim = server.FakeSimulation()
        sim.get_snapshot()
        sim.reset()
        self.assertEqual(sim.step, 0)
        self.assertEqual(sim.get_snapshot()["step"], 1)

    def test_es_segura_entre_hilos(self) -> None:
        sim = server.FakeSimulation()
        pasos: list[int] = []
        cerrojo = threading.Lock()

        def tirar() -> None:
            for _ in range(200):
                paso = sim.get_snapshot()["step"]
                with cerrojo:
                    pasos.append(paso)

        hilos = [threading.Thread(target=tirar) for _ in range(4)]
        for hilo in hilos:
            hilo.start()
        for hilo in hilos:
            hilo.join()

        self.assertEqual(len(pasos), 800)
        self.assertEqual(len(set(pasos)), 800)  # ningun paso repetido ni perdido


if __name__ == "__main__":
    unittest.main()
