"""Constantes del proyecto."""

from pathlib import Path

HOST: str = "127.0.0.1"
PORT: int = 5000
ENCODING: str = "utf-8"
CMD_GET_STATE: str = "GET_STATE"
CMD_RESET: str = "RESET"
CMD_PING: str = "PING"
CMD_SET_MODE: str = "SET_MODE"

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

# Las dos politicas que la fase 8 puede montar por nombre. Es la UNICA variable
# experimental: con `baseline` y con `qlearning` corre exactamente el mismo motor.
POLICY_BASELINE: str = "baseline"
POLICY_QLEARNING: str = "qlearning"
POLICIES: tuple[str, ...] = (POLICY_BASELINE, POLICY_QLEARNING)
DEFAULT_POLICY: str = POLICY_BASELINE

# Desatasco de la fase 8. Salta MUY antes que DEADLOCK_TICKS a proposito: cuando
# se llega a los 20 la corrida ya esta muerta, y lo que hace falta es que no
# llegue. Ponerlo a 0 apaga el desatasco y devuelve el motor al de la fase 5.
DEADLOCK_FORCE_TICKS: int = 8      # ticks sin que avance nadie antes de que el motor mande
# Cuando el motor manda a un AGV apartarse de un nodo, ese nodo le queda
# reservado al que esperaba durante estos ticks. Sin la reserva, el que se
# aparto vuelve, gana el desempate por id menor y el atasco se rehace: apartarse
# y volver es no apartarse.
YIELD_TICKS: int = 10
# Un AGV solo, clavado en el sitio mientras los demas circulan, es el otro
# atasco: el contador global no lo ve porque el almacen SI se esta moviendo. El
# umbral va mas alto que el de arriba porque esperar es normal: cruzar un tramo
# cuesta entre 4 y 8 ticks, asi que ceder el paso dos veces seguidas ya son 16.
STARVED_TICKS: int = 45

# Epsilon con el que se sirve una Q-table ya entrenada. Greedy puro: al servir no
# se explora, que lo que se enseña es lo aprendido, no como se aprendio.
SERVE_EPSILON: float = 0.0

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

# Las penalizaciones del REROUTE CADUCAN (fase 8). Sin caducidad el mapa se
# degrada para siempre: A* acabaria esquivando pasillos que llevan cien ticks
# libres solo porque una vez hubo alguien delante.
PENALTY_TTL: int = 15        # ticks que dura una penalizacion antes de expirar
PENALTY_MAX: float = 40.0    # tope al acumular: insistir sube el precio, no lo hace infinito
# El veto del desatasco: caro de verdad, pero FINITO. Con `inf` un nodo de paso
# obligado dejaria a A* sin ruta que devolver, y sin ruta no hay desatasco.
PENALTY_BAN: float = 1000.0
# Recalcular cuesta, asi que no se recalcula en cada tick. Sin esta pausa un AGV
# bloqueado pide ruta nueva 20 veces seguidas: cada recalculo penaliza el nodo
# que tiene delante, A* le da la otra salida, al tick siguiente penaliza esa y
# vuelve a la primera. Un ir y venir que no lleva a ningun sitio y que ademas
# encarece medio mapa. Con la pausa, un REROUTE seguido de otro se queda en
# esperar, que es lo que de verdad estaba haciendo.
REROUTE_COOLDOWN: int = 8

# Recompensas del Q-Learning. Estan aqui para poder ajustarlas sin abrir
# qlearning.py; quien las lee es `qlearning.reward(event)`, y nadie mas.
REWARD_TASK_COMPLETE: float = 100.0   # completar la tarea
REWARD_PROGRESS: float = 2.0          # avanzar hacia el destino (path_index subio)
REWARD_WAIT: float = -1.0             # esperar un tick
REWARD_CONFLICT: float = -20.0        # intentar entrar en conflicto
REWARD_DEADLOCK: float = -50.0        # provocar un deadlock
REWARD_USELESS_REROUTE: float = -3.0  # recalcular sin ganar nada

# Entrenamiento del Q-Learning (fase 7). La fase 6 define el entorno; estos son
# los numeros del bucle que aprende encima de el, y los lee `qlearning.Trainer`.
ALPHA: float = 0.2          # cuanto pesa lo nuevo frente a lo que ya sabia
GAMMA: float = 0.95         # cuanto vale el futuro; casi 1 porque el +100 esta al final
EPSILON_START: float = 1.0  # se empieza explorando del todo
EPSILON_END: float = 0.05   # y nunca se deja de explorar un poco
EPSILON_DECAY: float = 0.995  # exponencial: epsilon <- max(END, epsilon * DECAY)
EPISODES: int = 1000
# Tope de ticks por episodio. Un tramo del `warehouse` cuesta entre 4 y 8 ticks,
# asi que una ruta entera son ~30 y 200 deja sitio de sobra para atascarse y aun
# asi llegar. Sin tope, un episodio del principio (epsilon 1.0) no termina nunca.
MAX_STEPS_PER_EPISODE: int = 200
TRAIN_AGENTS: int = 4
# Cada cuantos episodios se imprime una fila del resumen por consola.
REPORT_EVERY: int = 100

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = PROJECT_ROOT / "results"
MAPS_DIR: Path = PROJECT_ROOT / "python" / "maps"
MODELS_DIR: Path = PROJECT_ROOT / "python" / "models"
DEFAULT_MAP: str = "warehouse"

# Lo que produce y consume la fase 7.
Q_TABLE_FILE: Path = MODELS_DIR / "q_table.json"
TRAINING_LOG_FILE: Path = RESULTS_DIR / "training_log.csv"
LEARNING_CURVE_FILE: Path = RESULTS_DIR / "learning_curve.png"
