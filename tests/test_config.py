"""Tests de las constantes de config."""

import unittest
from pathlib import Path

import config


class TestConfig(unittest.TestCase):
    def test_red(self) -> None:
        self.assertEqual(config.HOST, "127.0.0.1")
        self.assertEqual(config.PORT, 5000)
        self.assertEqual(config.CMD_GET_STATE, "GET_STATE")
        self.assertEqual(config.CMD_RESET, "RESET")
        self.assertEqual(config.CMD_PING, "PING")
        self.assertIsInstance(config.ENCODING, str)

    def test_tick(self) -> None:
        self.assertIsInstance(config.TICK_RATE, int)
        self.assertGreater(config.TICK_RATE, 0)
        self.assertAlmostEqual(config.TICK_DURATION, 1.0 / config.TICK_RATE)

    def test_unity(self) -> None:
        self.assertIsInstance(config.UNITY_SCALE, float)
        self.assertIsInstance(config.AGV_HEIGHT, float)
        self.assertGreater(config.UNITY_SCALE, 0.0)

    def test_semilla(self) -> None:
        self.assertIsInstance(config.RANDOM_SEED, int)

    def test_rutas(self) -> None:
        self.assertIsInstance(config.PROJECT_ROOT, Path)
        self.assertEqual(config.RESULTS_DIR, config.PROJECT_ROOT / "results")
        self.assertEqual(config.MAPS_DIR, config.PROJECT_ROOT / "python" / "maps")
        self.assertTrue((config.PROJECT_ROOT / "python").is_dir())
        self.assertTrue(config.MAPS_DIR.is_dir())

    def test_mapa_por_defecto(self) -> None:
        self.assertIsInstance(config.DEFAULT_MAP, str)
        self.assertTrue((config.MAPS_DIR / f"{config.DEFAULT_MAP}.json").is_file())


if __name__ == "__main__":
    unittest.main()
