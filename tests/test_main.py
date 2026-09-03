"""Tests del CLI."""

import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import graph
import main
import metrics
import qlearning
import scenarios
import simulation

# Desde la fase 10 estan los siete: `scenario` es el ultimo.
IMPLEMENTADOS = {
    "serve", "map", "simulate", "train", "evaluate", "benchmark", "scenario",
}
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
    def test_ya_no_queda_ningun_subcomando_por_implementar(self) -> None:
        """La fase 10 cierra `scenario`, que era el ultimo.

        El bucle se queda puesto por si alguna fase vuelve a dejar andamiaje:
        entonces `PENDIENTES` deja de estar vacio y esto vuelve a medir algo.
        """
        self.assertEqual(PENDIENTES, [])
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
        # `benchmark` de verdad escribe ficheros, asi que va contra un tempdir y
        # con una sola semilla del mapa pequeno: aqui se prueba el CLI, no la
        # calidad de la comparacion.
        with tempfile.TemporaryDirectory() as carpeta:
            with self.assertLogs(level="WARNING"):
                self.assertEqual(
                    main.main(_benchmark_corto(carpeta, "--verbose")), 0
                )


def _benchmark_corto(carpeta: str, *antes: str) -> list[str]:
    """Un `benchmark` minimo: mapa pequeno, una semilla y sin graficas."""
    return [
        *antes,
        "benchmark",
        "--map", "simple",
        "--agents", "2",
        "--tasks", "4",
        "--seeds", "1",
        "--policies", "baseline",
        "--out", carpeta,
        "--no-plots",
    ]


class TestBenchmark(unittest.TestCase):
    """La fase 9 por el CLI."""

    def test_escribe_el_csv_y_el_json_en_donde_le_digan(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            self.assertEqual(main.main(_benchmark_corto(carpeta)), 0)
            destino = Path(carpeta)
            self.assertTrue((destino / "baseline.csv").is_file())
            self.assertTrue((destino / "comparison.json").is_file())
            filas = metrics.read_runs_csv(destino / "baseline.csv")
            self.assertEqual(len(filas), 1)
            self.assertEqual(filas[0]["policy"], "baseline")
            self.assertEqual(filas[0]["seed"], "1")

    def test_sin_q_table_avisa_y_devuelve_dos(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            with self.assertLogs(level="ERROR") as capturado:
                codigo = main.main([
                    "benchmark", "--map", "simple", "--agents", "2",
                    "--seeds", "1", "--out", carpeta,
                    "--model", str(Path(carpeta) / "no-existe.json"),
                ])
            self.assertEqual(codigo, 2)
            self.assertIn("no existe el modelo", capturado.output[0])

    def test_el_parser_de_semillas(self) -> None:
        self.assertEqual(main._semillas("1-5"), [1, 2, 3, 4, 5])
        self.assertEqual(main._semillas("3,1,2"), [3, 1, 2])
        self.assertEqual(main._semillas("1-3,7"), [1, 2, 3, 7])
        # Repetidas fuera: correr dos veces la misma semilla es correr dos veces
        # el mismo trabajo y contarlo como dos medidas.
        self.assertEqual(main._semillas("1,1,2"), [1, 2])

    def test_una_semilla_que_no_se_entiende_no_pasa(self) -> None:
        for texto in ("hola", "5-1", "", "1-x"):
            with self.subTest(seeds=texto):
                with self.assertRaises(argparse.ArgumentTypeError):
                    main._semillas(texto)


class TestScenario(unittest.TestCase):
    """La fase 10 por el CLI."""

    def setUp(self) -> None:
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        self.carpeta = Path(carpeta.name)

    def _corre(self, *extra: str) -> int:
        return main.main(
            ["scenario", "--runs", "2", "--out", str(self.carpeta), *extra]
        )

    def test_un_escenario_escribe_su_csv_y_la_tabla_resumen(self) -> None:
        self.assertEqual(self._corre("--name", "A"), 0)
        for politica in config.POLICIES:
            self.assertTrue((self.carpeta / f"scenario_A_{politica}.csv").is_file())
        self.assertTrue((self.carpeta / "summary_table.csv").is_file())

    def test_una_sola_politica_escribe_solo_su_csv(self) -> None:
        self.assertEqual(self._corre("--name", "A", "--policy", "baseline"), 0)
        self.assertTrue((self.carpeta / "scenario_A_baseline.csv").is_file())
        self.assertFalse((self.carpeta / "scenario_A_qlearning.csv").is_file())

    def test_all_corre_los_cinco_con_las_dos_politicas(self) -> None:
        self.assertEqual(self._corre("--all"), 0)
        for letra in scenarios.LETTERS:
            for politica in config.POLICIES:
                ruta = self.carpeta / f"scenario_{letra}_{politica}.csv"
                self.assertTrue(ruta.is_file(), f"falta {ruta}")
        filas = scenarios.read_summary_table(self.carpeta / "summary_table.csv")
        self.assertEqual(len(filas), len(scenarios.LETTERS) * len(config.POLICIES))
        self.assertEqual([f["scenario"] for f in filas[::2]], list(scenarios.LETTERS))

    def test_hace_falta_name_o_all(self) -> None:
        """Parsea, pero no corre: avisa por el log y sale con 2, como el resto."""
        args = main.build_parser().parse_args(["scenario"])
        self.assertIsNone(args.name)
        self.assertFalse(args.all)
        with self.assertLogs(level="ERROR") as capturado:
            self.assertEqual(main.main(["scenario"]), 2)
        self.assertIn("--name", capturado.output[0])

    def test_name_y_all_a_la_vez_no_pasan(self) -> None:
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                main.build_parser().parse_args(["scenario", "--name", "A", "--all"])

    def test_un_escenario_que_no_existe_devuelve_dos(self) -> None:
        with self.assertLogs(level="ERROR") as capturado:
            self.assertEqual(self._corre("--name", "Z"), 2)
        self.assertIn("no conozco el escenario", capturado.output[0])

    def test_sin_q_table_avisa_y_devuelve_dos(self) -> None:
        with self.assertLogs(level="ERROR") as capturado:
            codigo = self._corre(
                "--name", "A", "--model", str(self.carpeta / "no-existe.json")
            )
        self.assertEqual(codigo, 2)
        self.assertIn("no existe el modelo", capturado.output[0])
        # Y no ha escrito nada: se comprueban los modelos ANTES de correr.
        self.assertEqual(list(self.carpeta.glob("*.csv")), [])

    def test_sin_la_q_table_del_escenario_dice_como_entrenarla(self) -> None:
        with mock.patch.object(config, "MODELS_DIR", self.carpeta):
            with self.assertLogs(level="ERROR") as capturado:
                codigo = self._corre("--name", "D", "--per-scenario-model")
        self.assertEqual(codigo, 2)
        self.assertIn("train --scenario D", "\n".join(capturado.output))

    def test_no_summary_deja_la_tabla_sin_tocar(self) -> None:
        self.assertEqual(self._corre("--name", "A", "--no-summary"), 0)
        self.assertFalse((self.carpeta / "summary_table.csv").exists())

    def test_solo_la_baseline_no_necesita_modelo(self) -> None:
        codigo = self._corre(
            "--name", "A", "--policy", "baseline",
            "--model", str(self.carpeta / "no-existe.json"),
        )
        self.assertEqual(codigo, 0)

    def test_dos_corridas_iguales_dan_los_mismos_csv(self) -> None:
        """El criterio de la fase: misma semilla, mismo resultado."""
        self.assertEqual(self._corre("--name", "D"), 0)
        primera = (self.carpeta / "scenario_D_qlearning.csv").read_text(
            encoding=config.ENCODING
        )
        self.assertEqual(self._corre("--name", "D"), 0)
        segunda = (self.carpeta / "scenario_D_qlearning.csv").read_text(
            encoding=config.ENCODING
        )
        self.assertEqual(primera, segunda)


class TestTrainPorEscenario(unittest.TestCase):
    """`train --scenario X` entrena en el escenario y no pisa la tabla general."""

    def test_entrena_en_el_mapa_del_escenario_y_escribe_su_modelo(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta)
            with mock.patch.object(config, "MODELS_DIR", destino):
                codigo = main.main([
                    "train", "--scenario", "E", "--episodes", "10",
                    "--max-steps", "100", "--no-curve",
                    "--log", str(destino / "log.csv"),
                ])
            self.assertEqual(codigo, 0)
            modelo = destino / "q_table_E.json"
            self.assertTrue(modelo.is_file())
            metadata = qlearning.load_metadata(modelo)
            # El escenario manda sobre --map: E corre en la rejilla.
            self.assertEqual(metadata["map"], "grid")
            self.assertEqual(metadata["hyperparameters"]["scenario"], "E")

    def test_un_escenario_que_no_existe_devuelve_dos(self) -> None:
        with self.assertLogs(level="ERROR") as capturado:
            self.assertEqual(main.main(["train", "--scenario", "Z"]), 2)
        self.assertIn("no conozco el escenario", capturado.output[0])


if __name__ == "__main__":
    unittest.main()
