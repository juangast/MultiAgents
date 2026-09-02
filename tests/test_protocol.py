"""Tests del contrato PULL. Sin sockets: aqui solo se prueba el modulo puro."""

import json
import unittest
from unittest import mock

import config
import protocol


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


if __name__ == "__main__":
    unittest.main()
