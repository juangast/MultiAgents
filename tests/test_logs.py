"""Tests del setup del logging."""

import logging
import unittest

import logs


class TestLogs(unittest.TestCase):
    def setUp(self) -> None:
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._level = root.level
        root.handlers.clear()

    def tearDown(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()
        root.handlers.extend(self._handlers)
        root.setLevel(self._level)

    def test_nivel_normal(self) -> None:
        logs.setup_logging()
        self.assertEqual(logging.getLogger().level, logging.INFO)

    def test_nivel_verbose(self) -> None:
        logs.setup_logging(verbose=True)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_no_duplica_handlers(self) -> None:
        logs.setup_logging()
        logs.setup_logging(verbose=True)
        root = logging.getLogger()
        self.assertEqual(len(root.handlers), 1)
        self.assertEqual(root.handlers[0].level, logging.DEBUG)

    def test_get_logger(self) -> None:
        self.assertEqual(logs.get_logger("agv").name, "agv")


if __name__ == "__main__":
    unittest.main()
