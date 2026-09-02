"""Tests del CLI."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import graph
import main
import simulation

# Lo unico que sigue siendo andamiaje es `benchmark`.
IMPLEMENTADOS = {"serve", "map", "simulate", "train", "evaluate"}
PENDIENTES = [nombre for nombre in main.COMMANDS if nombre not in IMPLEMENTADOS]


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

    def test_map_usa_el_mapa_por_defecto_de_config(self) -> None:
        args = main.build_parser().parse_args(["map"])
        self.assertEqual(args.name, config.DEFAULT_MAP)

    def test_map_acepta_otro_nombre(self) -> None:
        args = main.build_parser().parse_args(["map", "--name", "simple"])
        self.assertEqual(args.name, "simple")

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

    def test_serve_levanta_el_servidor_con_la_simulacion_de_verdad(self) -> None:
        with mock.patch.object(main.server, "serve_forever", return_value=0) as falso:
            self.assertEqual(main.main(["serve", "--port", "0"]), 0)

        falso.assert_called_once()
        posicionales, nombrados = falso.call_args
        self.assertIsInstance(posicionales[0], simulation.Simulation)
        self.assertEqual(nombrados["port"], 0)
        self.assertEqual(nombrados["host"], main.config.HOST)

    def test_serve_sirve_el_mapa_que_le_pidan(self) -> None:
        with mock.patch.object(main.server, "serve_forever", return_value=0) as falso:
            self.assertEqual(main.main(["serve", "--map", "simple", "--port", "0"]), 0)

        simulacion = falso.call_args[0][0]
        self.assertEqual(simulacion.graph.name, "simple")

    def test_serve_con_un_mapa_que_no_existe_devuelve_dos(self) -> None:
        with mock.patch.object(main.server, "serve_forever") as falso:
            with self.assertLogs(level="ERROR"):
                self.assertEqual(main.main(["serve", "--map", "atlantida"]), 2)
        falso.assert_not_called()

    def test_map_muestra_los_dos_mapas_y_devuelve_cero(self) -> None:
        for nombre in graph.BUILTIN_MAPS:
            with self.subTest(mapa=nombre):
                with self.assertLogs(level="INFO") as capturado:
                    self.assertEqual(main.main(["map", "--name", nombre]), 0)
                salida = "\n".join(capturado.output)
                self.assertIn(f"mapa {nombre}", salida)
                self.assertIn("validate(): OK", salida)

    def test_map_muestra_las_dos_coordenadas_de_cada_nodo(self) -> None:
        with self.assertLogs(level="INFO") as capturado:
            main.main(["map", "--name", "simple"])
        salida = "\n".join(capturado.output)
        # D esta en (0, 3) logicas, o sea (0, 0, 3) en Unity: la Y va a la Z.
        self.assertIn("(0, 3)", salida)
        self.assertIn("(0, 0, 3)", salida)

    def test_map_de_un_mapa_que_no_existe_devuelve_dos(self) -> None:
        with self.assertLogs(level="ERROR") as capturado:
            self.assertEqual(main.main(["map", "--name", "atlantida"]), 2)
        self.assertIn("no conozco el mapa", capturado.output[0])

    def test_map_tira_del_mapa_interno_si_falta_el_fichero(self) -> None:
        with tempfile.TemporaryDirectory() as vacia:
            with mock.patch.object(config, "MAPS_DIR", Path(vacia)):
                with self.assertLogs(level="WARNING") as capturado:
                    self.assertEqual(main.main(["map", "--name", "warehouse"]), 0)
        self.assertIn("mapa interno", capturado.output[0])

    def test_map_devuelve_uno_si_el_mapa_no_es_valido(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            roto = Path(carpeta) / "roto.json"
            roto.write_text(
                '{"adjacency": {"A": {"B": 1}}, "positions": {"A": [0, 0]}}',
                encoding=config.ENCODING,
            )
            with mock.patch.object(config, "MAPS_DIR", Path(carpeta)):
                with self.assertLogs(level="ERROR") as capturado:
                    self.assertEqual(main.main(["map", "--name", "roto"]), 1)
        self.assertIn("validate()", capturado.output[0])

    def test_sin_argumentos_devuelve_uno(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            self.assertEqual(main.main([]), 1)
        self.assertIn("subcomando", salida.getvalue())

    def test_verbose_no_truena(self) -> None:
        with self.assertLogs(level="WARNING"):
            self.assertEqual(main.main(["--verbose", "benchmark"]), 0)


if __name__ == "__main__":
    unittest.main()
