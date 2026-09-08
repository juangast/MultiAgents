"""Constantes del proyecto y configuracion del logging."""

import logging
import sys
from pathlib import Path

HOST: str = "127.0.0.1"
PORT: int = 5000
ENCODING: str = "utf-8"
UNITY_SCALE: float = 1.0
RANDOM_SEED: int = 42

DEADLOCK_TICKS: int = 20

POLICY_QLEARNING: str = "qlearning"
POLICIES: tuple[str, ...] = (POLICY_QLEARNING,)
DEFAULT_POLICY: str = POLICY_QLEARNING


ENABLE_REROUTE: bool = True


REROUTE_PENALTY: float = 10.0


BATTERY_FULL: float = 100.0
BATTERY_DRAIN: float = 0.8
BATTERY_THRESHOLD: float = 25.0
BATTERY_CHARGE_RATE: float = 5.0
BATTERY_RESERVE: float = 30.0


REWARD_TASK_COMPLETE: float = 100.0
REWARD_PICKED: float = 50.0
# Lo que vale dejar la caja en su muelle, que es el objetivo del reto entero.
# Faltaba. En modo entregas los AGV encadenan misiones y nunca pasan a DONE, asi
# que REWARD_TASK_COMPLETE no se cobra nunca (1500 episodios cerraron con
# completed_tasks=0) y la unica señal positiva era recoger: se entrenaban AGVs
# que levantaban cajas y no las llevaban a ningun sitio.
REWARD_DELIVERED: float = 100.0
REWARD_PROGRESS: float = 2.0
REWARD_WAIT: float = -1.0
# Un conflicto cuesta, de verdad, un tick de retraso: lo mismo que esperar. A
# -20 costaba veinte, y como salen ~1200 por episodio contra ~10 entregas, la
# suma de penalizaciones (-24000) sepultaba a la de entregas (+1000): la
# politica optima pasaba a ser "no te metas en lios", o sea no moverse.
REWARD_CONFLICT: float = -2.0
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
DEFAULT_MAP: str = "almacen_reto"

# La tabla del estado de dos bits. Con 4 estados y 3 acciones el espacio de
# politicas tiene 81 elementos, asi que se pudo evaluar entero sobre el objetivo
# real (cajas entregadas, en 4 escenarios) en vez de aproximarlo: 23.0 de media,
# contra 20.0 de la vieja de 144 estados y 14.2 de la que sale de entrenar.
# `q_table.json` se deja como estaba, con el estado de seis campos.
Q_TABLE_FILE: Path = MODELS_DIR / "q_table_busqueda.json"
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
