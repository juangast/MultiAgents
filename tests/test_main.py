"""Tests del CLI."""

import contextlib
import io
import unittest

import main


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

    def test_subcomando_desconocido(self) -> None:
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                main.build_parser().parse_args(["volar"])


class TestMain(unittest.TestCase):
    def test_cada_subcomando_devuelve_cero(self) -> None:
        for name in main.COMMANDS:
            with self.assertLogs(level="WARNING") as captured:
                self.assertEqual(main.main([name]), 0)
            self.assertIn("no implementado", captured.output[0])

    def test_sin_argumentos_devuelve_uno(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()) as salida:
            self.assertEqual(main.main([]), 1)
        self.assertIn("subcomando", salida.getvalue())

    def test_verbose_no_truena(self) -> None:
        with self.assertLogs(level="WARNING"):
            self.assertEqual(main.main(["--verbose", "train"]), 0)


if __name__ == "__main__":
    unittest.main()
