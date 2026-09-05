"""Constantes del proyecto y configuracion del logging."""

import logging
import sys
from pathlib import Path

HOST: str = "127.0.0.1"
PORT: int = 5000
ENCODING: str = "utf-8"
CMD_GET_STATE: str = "GET_STATE"
CMD_RESET: str = "RESET"
CMD_PING: str = "PING"
CMD_SET_MODE: str = "SET_MODE"

UNITY_SCALE: float = 1.0
RANDOM_SEED: int = 42

DEADLOCK_TICKS: int = 20

POLICY_BASELINE: str = "baseline"
POLICY_QLEARNING: str = "qlearning"
POLICIES: tuple[str, ...] = (POLICY_BASELINE, POLICY_QLEARNING)
DEFAULT_POLICY: str = POLICY_BASELINE


ENABLE_REROUTE: bool = True


REROUTE_PENALTY: float = 10.0


BATTERY_FULL: float = 100.0
BATTERY_DRAIN: float = 0.8
BATTERY_THRESHOLD: float = 25.0
BATTERY_CHARGE_RATE: float = 5.0
BATTERY_RESERVE: float = 10.0
BATTERY_DETOUR: float = 1.5


REWARD_TASK_COMPLETE: float = 100.0
REWARD_PICKED: float = 50.0
REWARD_PROGRESS: float = 2.0
REWARD_WAIT: float = -1.0
REWARD_CONFLICT: float = -20.0
REWARD_DEADLOCK: float = -50.0
REWARD_USELESS_REROUTE: float = -3.0

ALPHA: float = 0.2
GAMMA: float = 0.95
EPSILON_START: float = 1.0
EPSILON_END: float = 0.05
EPSILON_DECAY: float = 0.995
EPISODES: int = 1000
MAX_STEPS_PER_EPISODE: int = 200
TRAIN_AGENTS: int = 4

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MAPS_DIR: Path = PROJECT_ROOT / "python" / "maps"
MODELS_DIR: Path = PROJECT_ROOT / "python" / "models"
DEFAULT_MAP: str = "warehouse"

Q_TABLE_FILE: Path = MODELS_DIR / "q_table.json"
TRAINING_LOG_FILE: Path = RESULTS_DIR / "training_log.csv"

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_HANDLER_NAME = "agv_console"

def setup_logging(verbose: bool = False) -> None:
    """Deja el logging escribiendo a stderr. Con verbose usa DEBUG."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    for handler in root.handlers:
        if handler.get_name() == _HANDLER_NAME:
            handler.setLevel(level)
            return

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)

def get_logger(name: str) -> logging.Logger:
    """Devuelve el logger con ese nombre."""
    return logging.getLogger(name)
