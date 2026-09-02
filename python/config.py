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

# Gestion de conflictos (fase 5). El baseline es una referencia experimental,
# asi que estos tres numeros se tocan a menudo para comparar corridas.
CONFLICT_WAIT_THRESHOLD: int = 5   # ticks esperando antes de marcar congestion
CONGESTION_ZONE_AGENTS: int = 3    # agentes esperando en una zona para que cuente
DEADLOCK_TICKS: int = 20           # ticks seguidos sin que avance nadie

# Q-Learning (fase 6). Aqui se define el entorno; el entrenamiento es la 7.
# Q-Learning NO sustituye a A*: A* sigue trazando la ruta, y lo que se aprende
# es solo que hacer AHORA ante un riesgo de conflicto.
ENABLE_REROUTE: bool = True        # si es False la politica solo elige ADVANCE/WAIT

# Cortes del bucket de distancia, contados en NODOS que faltan de la ruta, nunca
# en distancia euclidiana: cerca <= NEAR < medio <= MID < lejos.
DISTANCE_NEAR_NODES: int = 3
DISTANCE_MID_NODES: int = 8

# Cuanto se encarece el nodo (y el tramo hacia el) que se esta esquivando cuando
# la politica elige REROUTE. Va a `astar.astar(..., penalties=...)`.
REROUTE_PENALTY: float = 10.0

# Recompensas del Q-Learning. Estan aqui para poder ajustarlas sin abrir
# qlearning.py; quien las lee es `qlearning.reward(event)`, y nadie mas.
REWARD_TASK_COMPLETE: float = 100.0   # completar la tarea
REWARD_PROGRESS: float = 2.0          # avanzar hacia el destino (path_index subio)
REWARD_WAIT: float = -1.0             # esperar un tick
REWARD_CONFLICT: float = -20.0        # intentar entrar en conflicto
REWARD_DEADLOCK: float = -50.0        # provocar un deadlock
REWARD_USELESS_REROUTE: float = -3.0  # recalcular sin ganar nada

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MAPS_DIR: Path = PROJECT_ROOT / "python" / "maps"
DEFAULT_MAP: str = "warehouse"
