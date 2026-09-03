"""Tests del contrato PULL: los comandos, la serializacion y las coordenadas.

Casi todo se prueba sin sockets, sobre el modulo puro. La excepcion es
`TestFragmentacionTCP`, que **si** abre un socket de verdad: TCP es un flujo de
bytes y no respeta los limites de los mensajes, asi que la regla "una linea
entra, una linea sale" solo significa algo si se comprueba contra un socket que
parte y pega los envios como le da la gana.
"""

import json
import socket
import threading
import unittest
from unittest import mock

import config
import graph
import protocol
import server
import simulation


class SimulacionStub:
    """Doble de simulacion que cuenta las llamadas que recibe."""

    def __init__(self) -> None:
        self.snapshot: protocol.Snapshot = {"step": 7, "agents": []}
        self.llamadas_snapshot = 0
        self.llamadas_reset = 0

    def get_snapshot(self) -> protocol.Snapshot:
        self.llamadas_snapshot += 1
        return self.snapshot

    def reset(self) -> None:
        self.llamadas_reset += 1


class TestParseCommand(unittest.TestCase):
    def test_comando_simple(self) -> None:
        self.assertEqual(protocol.parse_command("GET_STATE\n"), ("GET_STATE", []))

    def test_con_argumentos(self) -> None:
        self.assertEqual(protocol.parse_command("MOVE 1 2\n"), ("MOVE", ["1", "2"]))

    def test_normaliza_a_mayusculas(self) -> None:
        self.assertEqual(protocol.parse_command("get_state")[0], "GET_STATE")

    def test_aguanta_crlf(self) -> None:
        self.assertEqual(protocol.parse_command("PING\r\n"), ("PING", []))

    def test_aguanta_espacios_de_mas(self) -> None:
        self.assertEqual(protocol.parse_command("   RESET   "), ("RESET", []))

    def test_linea_vacia(self) -> None:
        self.assertEqual(protocol.parse_command("   \r\n"), ("", []))


class TestEncode(unittest.TestCase):
    def test_termina_en_salto_de_linea(self) -> None:
        self.assertTrue(protocol.encode_snapshot({"step": 1}).endswith("\n"))

    def test_sin_saltos_internos(self) -> None:
        linea = protocol.encode_snapshot({"step": 1, "agents": [{"state": "a\nb"}]})
        self.assertEqual(linea.count("\n"), 1)

    def test_round_trip(self) -> None:
        snapshot = {"step": 3, "agents": [{"id": 1, "x": 0.25}]}
        self.assertEqual(json.loads(protocol.encode_snapshot(snapshot)), snapshot)

    def test_ok_payload(self) -> None:
        self.assertEqual(protocol.encode_line(protocol.OK_PAYLOAD), '{"ok":true}\n')


class TestToUnity(unittest.TestCase):
    def test_aplica_la_escala_a_x_y_a_z(self) -> None:
        with mock.patch.object(config, "UNITY_SCALE", 2.0):
            self.assertEqual(protocol.to_unity(3.0, 4.0), (6.0, 0.0, 8.0))

    def test_la_altura_la_pone_unity(self) -> None:
        self.assertEqual(protocol.to_unity(1.0, 1.0)[1], 0.0)

    def test_el_segundo_eje_del_plano_va_a_z(self) -> None:
        _x, _y, z = protocol.to_unity(0.0, 5.0)
        self.assertEqual(z, 5.0 * config.UNITY_SCALE)


class TestHandleLine(unittest.TestCase):
    def setUp(self) -> None:
        self.sim = SimulacionStub()

    def responder(self, linea: str) -> dict:
        respuesta = protocol.handle_line(linea, self.sim)
        self.assertTrue(respuesta.endswith("\n"))
        self.assertEqual(respuesta.count("\n"), 1)
        return json.loads(respuesta)

    def test_get_state_devuelve_el_snapshot(self) -> None:
        self.assertEqual(self.responder("GET_STATE\n"), self.sim.snapshot)
        self.assertEqual(self.sim.llamadas_snapshot, 1)

    def test_reset_reinicia_y_confirma(self) -> None:
        self.assertEqual(self.responder("RESET\n"), {"ok": True})
        self.assertEqual(self.sim.llamadas_reset, 1)

    def test_ping_confirma_sin_tocar_la_simulacion(self) -> None:
        self.assertEqual(self.responder("PING\n"), {"ok": True})
        self.assertEqual(self.sim.llamadas_snapshot, 0)
        self.assertEqual(self.sim.llamadas_reset, 0)

    def test_comando_desconocido_hace_eco(self) -> None:
        self.assertEqual(
            self.responder("BASURA algo\n"),
            {"error": "unknown_command", "command": "BASURA"},
        )

    def test_linea_vacia_tambien_responde(self) -> None:
        self.assertEqual(
            self.responder("\n"),
            {"error": "unknown_command", "command": ""},
        )

    def test_el_stub_cumple_el_protocolo(self) -> None:
        self.assertIsInstance(self.sim, protocol.Simulation)


class TestFragmentacionTCP(unittest.TestCase):
    """El contrato aguanta como TCP parta los bytes por el camino.

    TCP entrega un **flujo**, no mensajes: un `send()` del cliente puede llegar
    en tres trozos, y tres `send()` pueden llegar pegados en uno. Un cliente que
    de por hecho que un `recv()` trae exactamente una respuesta funciona en
    localhost y se rompe en cuanto hay red de por medio. Estas tres pruebas son
    las que el cliente de Unity tiene que poder pasar.
    """

    def setUp(self) -> None:
        self.simulacion = simulation.Simulation(graph.simple_graph(), 1)
        self.servidor = server.AGVServer((config.HOST, 0), self.simulacion)
        self.hilo = threading.Thread(target=self.servidor.serve_forever, daemon=True)
        self.hilo.start()
        self.sock = socket.create_connection(self.servidor.server_address, timeout=5.0)
        self.addCleanup(self.cerrar)

    def cerrar(self) -> None:
        self.sock.close()
        self.servidor.shutdown()
        self.servidor.server_close()
        self.hilo.join(timeout=5.0)

    def lee_una_linea(self) -> dict:
        """Lee hasta el primer salto de linea. Asi es como se hace bien."""
        trozos = bytearray()
        while b"\n" not in trozos:
            dato = self.sock.recv(4096)
            if not dato:
                self.fail("el servidor cerro sin contestar")
            trozos.extend(dato)
        linea, _, resto = bytes(trozos).partition(b"\n")
        self.assertEqual(resto, b"", "llego mas de una linea de golpe")
        return json.loads(linea.decode(config.ENCODING))

    def test_un_comando_partido_en_varios_envios(self) -> None:
        # El caso que rompe a un cliente ingenuo: la respuesta no puede salir
        # hasta que llegue el salto de linea, por muchos trozos que hagan falta.
        for trozo in (b"GET_", b"STA", b"TE", b"\n"):
            self.sock.sendall(trozo)
        self.assertEqual(self.lee_una_linea()["step"], 1)

        self.sock.sendall(b"PI")
        self.sock.sendall(b"NG\n")
        self.assertEqual(self.lee_una_linea(), {"ok": True})

    def test_varios_comandos_pegados_en_un_solo_envio(self) -> None:
        # Y el contrario: tres comandos en un `send()` son tres respuestas, en
        # orden y cada una en su linea.
        self.sock.sendall(b"PING\nBASURA\nPING\n")

        trozos = bytearray()
        while trozos.count(b"\n") < 3:
            trozos.extend(self.sock.recv(4096))
        lineas = bytes(trozos).split(b"\n")[:3]
        respuestas = [json.loads(una.decode(config.ENCODING)) for una in lineas]

        self.assertEqual(respuestas[0], {"ok": True})
        self.assertEqual(respuestas[1]["error"], protocol.ERROR_UNKNOWN_COMMAND)
        self.assertEqual(respuestas[2], {"ok": True})

    def test_el_emparejado_no_se_pierde_en_100_peticiones(self) -> None:
        # Una peticion, una respuesta, siempre en el mismo orden: si el servidor
        # se saltara una o contestara de mas, `step` dejaria de cuadrar.
        for esperado in range(1, 101):
            self.sock.sendall(b"GET_STATE\n")
            self.assertEqual(self.lee_una_linea()["step"], esperado)


if __name__ == "__main__":
    unittest.main()
