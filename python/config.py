"""Constantes del proyecto."""

from pathlib import Path

HOST: str = "127.0.0.1"
PORT: int = 5000
ENCODING: str = "utf-8"
CMD_GET_STATE: str = "GET_STATE"
CMD_RESET: str = "RESET"
CMD_PING: str = "PING"

TICK_RATE: int = 10
TICK_DURATION: float = 1.0 / TICK_RATE

UNITY_SCALE: float = 1.0
AGV_HEIGHT: float = 0.5

RANDOM_SEED: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
