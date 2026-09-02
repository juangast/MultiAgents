"""Tests del CLI."""

import contextlib
import io
import unittest
from unittest import mock

import main

# serve ya esta implementado; el resto sigue siendo andamiaje.
PENDIENTES = [nombre for nombre in main.COMMANDS if nombre != "serve"]


class TestParser(unittest.TestCase):
    def test_subcomandos(self) -> None:
        parser = main.build_parser()
        for name in main.COMMANDS:
            args = parser.parse_args([name])
            self.assertEqual(args.command, name)

    def test_sin_subcomando(self) -> None:
        args = main.build_parser().parse_args([])
        self.assertIsNone(args.command)

    def test_verbose_antes_del_subcomando(self) -> None:
        args = main.build_parser().parse_args(["--verbose", "serve"])
        self.assertTrue(args.verbose)

    def test_verbose_despues_del_subcomando(self) -> None:
        args = main.build_parser().parse_args(["serve", "--verbose"])
        self.assertTrue(args.verbose)

    def test_serve_acepta_host_y_puerto(self) -> None:
        args = main.build_parser().parse_args(["serve", "--host", "0.0.0.0", "--port", "9999"])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9999)

    def test_serve_usa_los_valores_de_config_por_defecto(self) -> None:
        args = main.build_parser().parse_args(["serve"])
        self.assertEqual(args.host, main.config.HOST)
        self.assertEqual(args.port, main.config.PORT)

    def test_subcomando_desconocido(self) -> None:
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                main.build_parser().parse_args(["volar"])


class TestMain(unittest.TestCase):
    def test_los_subcomandos_pendientes_devuelven_cero(self) -> None:
        for name in PENDIENTES:
            with self.assertLogs(level="WARNING") as captured:
                self.assertEqual(main.main([name]), 0)
            self.assertIn("no implementado", captured.output[0])

    def test_serve_levanta_el_servidor_con_la_simulacion_falsa(self) -> None:
        with mock.patch.object(main.server, "serve_forever", return_value=0) as falso:
            self.assertEqual(main.main(["serve", "--port", "0"]), 0)

        falso.assert_called_once()
        posicionales, nombrados = falso.call_args
        self.assertIsInstance(posicionales[0], main.server.FakeSimulation)
        self.assertEqual(nombrados["port"], 0)
        self.assertEqual(nombrados["host"], main.config.HOST)

    def test_sin_argumentos_devuelve_uno(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            self.assertEqual(main.main([]), 1)
        self.assertIn("subcomando", salida.getvalue())

    def test_verbose_no_truena(self) -> None:
        with self.assertLogs(level="WARNING"):
            self.assertEqual(main.main(["--verbose", "train"]), 0)


if __name__ == "__main__":
    unittest.main()
