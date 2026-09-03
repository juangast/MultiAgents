"""Los cinco escenarios de la fase 10, el runner que los corre y la tabla resumen.

La fase 9 dejo un solo experimento: `warehouse`, 4 AGVs, 16 tareas, 20 semillas.
Con eso se sabe que el Q-Learning **no gana de media**, pero no se sabe *donde*
falla ni *donde* podria aportar. Esta fase monta cinco escenarios que barren ese
rango, de un almacen casi vacio a un cuello de botella, y los corre con las dos
politicas bajo condiciones identicas.

    A  baja congestion    2 AGVs, cada uno en su mitad, no se cruzan
    B  congestion media   4 AGVs cuyas rutas comparten la interseccion S3
    C  alta congestion    6 AGVs en 13 nodos, destinos por todo el mapa
    D  cuello de botella  toda tarea cruza G, que es el unico paso
    E  rutas alternativas la rejilla 4x4, donde SI hay una ruta igual de buena

El par D/E es el que contesta la pregunta del proyecto. Los dos escenarios piden
el mismo trabajo (cruzar el mapa de lado a lado una y otra vez); lo unico que
cambia es si existe o no una ruta alternativa. Si el REROUTE del Q-Learning vale
para algo, tiene que verse ahi y no en D.

--- Que es reproducible y que no ---

De un escenario **no** se sortea nada estructural: el mapa, cuantos AGVs hay,
donde arranca cada uno y de que conjunto de nodos salen las tareas estan
escritos a mano en la `ScenarioSpec`. Lo unico que depende de la semilla es
**que destinos concretos** toca esta vez. Por eso `spec.build(k)` devuelve
siempre el mismo `metrics.Scenario` campo a campo, y por eso las N corridas de
`--runs N` (semillas `seed`, `seed+1`, ...) no son la misma corrida N veces.

--- Y que se mantiene de la fase 9 ---

Nada del motor cambia. El escenario se sigue construyendo **una vez por semilla
y fuera del bucle de politicas** (`metrics.run_comparison()`), que es la
invariante que sostiene toda la comparacion: las dos politicas ven exactamente
el mismo mapa, los mismos AGVs, los mismos origenes, los mismos destinos y la
misma cola. Lo unico que cambia entre una corrida y la otra es el nombre de la
politica.
"""

import csv
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import config
import graph as graph_mod
import metrics
import qlearning
from graph import WarehouseGraph
from logs import get_logger

log = get_logger("scenarios")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """La configuracion completa de un escenario, sin una sola cosa sorteada.

    | Campo             | Que es                                                  |
    |-------------------|---------------------------------------------------------|
    | `letter`          | La letra con la que se pide por el CLI: A..E             |
    | `name`            | Como se llama en el reporte                              |
    | `map_name`        | `warehouse` o `grid`                                     |
    | `starts`          | Donde arranca cada AGV. **Fijo** y sin repetir           |
    | `first_targets`   | El primer destino de cada AGV. **Fijo**                  |
    | `pools`           | De donde salen las tareas siguientes; la cola CICLA      |
    | `tasks_per_agent` | Tareas totales = `n_agents * tasks_per_agent`            |
    | `seed`            | Semilla base: las corridas son seed, seed+1, seed+2...   |
    | `max_steps`       | Tope de ticks por corrida                                |
    | `tests`           | Que prueba y que se espera ver, para el log y el reporte |

    `pools` es una tupla de listas y la cola de tareas **cicla** entre ellas: la
    tarea 0 sale de `pools[0]`, la 1 de `pools[1]`, y vuelta a empezar. Con un
    solo pool es un sorteo normal dentro de los destinos del escenario. Con dos
    (D y E) la cola alterna lado, y como los AGVs arrancan alternando lado
    tambien, a cada uno le toca sistematicamente el lado contrario al que acaba
    de servir: es lo que hace que las tareas **sigan** cruzando el mapa en vez
    de cruzarlo una vez y quedarse.
    """

    letter: str
    name: str
    map_name: str
    starts: tuple[str, ...]
    first_targets: tuple[str, ...]
    pools: tuple[tuple[str, ...], ...]
    tasks_per_agent: int
    tests: str
    seed: int = config.SCENARIO_SEED
    max_steps: int = config.SCENARIO_MAX_STEPS

    @property
    def n_agents(self) -> int:
        """Cuantos AGVs corren este escenario."""
        return len(self.starts)

    @property
    def n_tasks(self) -> int:
        """Tareas totales, la primera de cada AGV incluida."""
        return self.n_agents * self.tasks_per_agent

    def graph(self) -> WarehouseGraph:
        """Carga el mapa del escenario, del disco o del constructor interno."""
        return load_map(self.map_name)

    def build(self, seed: int) -> metrics.Scenario:
        """El `metrics.Scenario` de esta semilla. Determinista, y sin politica.

        Los origenes y los primeros destinos son los de la spec, siempre. Lo que
        la semilla decide es la cola: `n_tasks - n_agents` destinos sacados de
        los pools, ciclando entre ellos. Dos llamadas con la misma semilla
        devuelven el mismo escenario campo a campo.
        """
        rng = random.Random(seed)
        pendientes = tuple(
            rng.choice(self.pools[i % len(self.pools)])
            for i in range(self.n_tasks - self.n_agents)
        )
        return metrics.Scenario(
            seed=int(seed),
            routes=tuple(zip(self.starts, self.first_targets)),
            pending=pendientes,
        )

    def seeds(self, runs: int) -> list[int]:
        """Las semillas de N corridas: la base y las que siguen."""
        if runs < 1:
            raise ValueError(f"hacen falta corridas, no {runs}")
        return [self.seed + i for i in range(runs)]

    def routes_factory(self) -> Callable[[random.Random], list[tuple[str, str]]]:
        """El generador de rutas para entrenar EN este escenario (fase 10).

        Se le pasa a `qlearning.TrainingEnv` en lugar de `random_routes()`, que
        sortea por todo el mapa. Asi la Q-table de un escenario aprende en sus
        posiciones de salida y con sus destinos, que es lo que el README de la
        fase 9 dejo apuntado: entrenar en el regimen en el que se evalua.

        Los origenes son los de la spec pero **repartidos al azar entre los
        AGVs**, y los destinos salen de los pools. Sin ese barajado los mil
        episodios serian el mismo reparto mil veces y la tabla se lo aprenderia
        de memoria en vez de aprender a ceder el paso.
        """

        def fabrica(rng: random.Random) -> list[tuple[str, str]]:
            origenes = list(self.starts)
            rng.shuffle(origenes)
            rutas: list[tuple[str, str]] = []
            for i, origen in enumerate(origenes):
                pool = self.pools[i % len(self.pools)]
                opciones = [nodo for nodo in pool if nodo != origen] or list(pool)
                rutas.append((origen, rng.choice(opciones)))
            return rutas

        return fabrica

    def check(self) -> None:
        """Revienta con ValueError si el escenario no se sostiene.

        Se comprueba aqui y no en la simulacion porque un escenario mal escrito
        no da error: da una corrida rara. Dos AGVs en el mismo nodo rompen la
        invariante antes de mover nada; un AGV que arranca en su propio destino
        cierra la tarea en el paso cero; y dos AGVs con el mismo primer destino
        acaban con uno aparcado encima del destino del otro, que lo bloquea para
        siempre.
        """
        problemas: list[str] = []
        grafo = self.graph()
        nodos = set(grafo.adjacency)

        if len(self.starts) != len(self.first_targets):
            problemas.append(
                f"{len(self.starts)} origenes y {len(self.first_targets)} destinos"
            )
        if len(set(self.starts)) != len(self.starts):
            problemas.append(f"dos AGVs saldrian del mismo nodo: {self.starts}")
        if len(set(self.first_targets)) != len(self.first_targets):
            problemas.append(
                f"dos AGVs tendrian el mismo primer destino: {self.first_targets}"
            )
        for origen, destino in zip(self.starts, self.first_targets):
            if origen == destino:
                problemas.append(f"el AGV que sale de {origen} ya esta en su destino")
        if not self.pools or any(not pool for pool in self.pools):
            problemas.append("hay un pool de destinos vacio")
        if self.tasks_per_agent < 1:
            problemas.append(f"{self.tasks_per_agent} tareas por AGV")

        sueltos = sorted(
            {nodo for nodo in self.starts + self.first_targets if nodo not in nodos}
            | {nodo for pool in self.pools for nodo in pool if nodo not in nodos}
        )
        if sueltos:
            problemas.append(
                f"estos nodos no estan en el mapa {self.map_name!r}: "
                + ", ".join(repr(nodo) for nodo in sueltos)
            )

        if problemas:
            raise ValueError(
                f"el escenario {self.letter} no se sostiene:\n"
                + "\n".join(f"  - {problema}" for problema in problemas)
            )

    def header_lines(self) -> list[str]:
        """La ficha del escenario, para el log. Es lo que va al reporte."""
        return [
            f"--- escenario {self.letter}: {self.name} ---",
            f"mapa          : {self.map_name}",
            f"AGVs          : {self.n_agents} ({', '.join(self.starts)})",
            f"primer destino: {', '.join(self.first_targets)}",
            f"tareas        : {self.n_tasks} ({self.tasks_per_agent} por AGV)",
            f"destinos      : "
            + " | ".join("{" + ", ".join(pool) + "}" for pool in self.pools),
            f"semilla base  : {self.seed}, tope {self.max_steps} ticks",
            f"que prueba    : {self.tests}",
        ]


def load_map(name: str) -> WarehouseGraph:
    """Carga un mapa por nombre: del JSON si esta, y si no del constructor interno.

    Es la misma politica que `main._abre_mapa()`, pero devolviendo el grafo o
    lanzando: aqui quien llama no tiene que traducir un codigo de salida.
    """
    ruta = graph_mod.map_path(name)
    if ruta.exists():
        grafo = graph_mod.load_graph(ruta)
    elif name in graph_mod.BUILTIN_MAPS:
        log.warning("no existe %s, tiro del mapa interno", ruta)
        grafo = graph_mod.BUILTIN_MAPS[name]()
    else:
        raise ValueError(
            f"no conozco el mapa {name!r} (internos: "
            + ", ".join(sorted(graph_mod.BUILTIN_MAPS))
            + ")"
        )
    grafo.validate()
    return grafo


# --- Los cinco escenarios ------------------------------------------------------
#
# El mapa `warehouse` visto desde arriba, que es donde caen A, B, C y D:
#
#   N1--N2--N3            N4--N5--N6
#    |       | \          / |       |
#    |       |   >  G  <    |       |
#    |       | /          \ |       |
#   S1--S2--S3            S4--S5--S6
#
# G es punto de articulacion: **toda** ruta que cruce el almacen pasa por el.
# Los pools de "oeste" y "este" son las dos mitades que G separa.

OESTE: tuple[str, ...] = ("S1", "S2", "S3", "N1", "N2", "N3")
ESTE: tuple[str, ...] = ("S4", "S5", "S6", "N4", "N5", "N6")
TODO_EL_ALMACEN: tuple[str, ...] = OESTE + ESTE + ("G",)


def scenario_a() -> ScenarioSpec:
    """A - Baja congestion: 2 AGVs, cada uno en su mitad, casi sin cruzarse.

    QUE PRUEBA: el suelo del experimento. Un AGV trabaja el rincon oeste
    (S1-S2-N1-N2) y el otro el este (S5-S6-N5-N6); los dos conjuntos de nodos
    son disjuntos y ninguno toca S3, N3, S4, N4 ni G. La cola alterna
    oeste/este y las dos rutas iniciales cuestan lo mismo (12), asi que los AGVs
    llegan a la vez, `_reparte()` los atiende por id y a cada uno le toca
    sistematicamente su propia mitad.

    QUE SE ESPERA VER: practicamente cero conflictos y cero espera, y las dos
    politicas EMPATADAS. Medido con la baseline sobre 20 semillas: 0.10
    conflictos por tick (contra 1.7 en B, D y E) y 19 de 20 corridas completas.
    Si el Q-Learning saliera distinto aqui seria ruido o un rodeo gratis: sin
    nadie con quien chocar no hay nada que decidir, y tener esa comparacion es
    justo lo que hace que las otras cuatro signifiquen algo.

    (No es un aislamiento perfecto: si un tick de espera desordena las llegadas,
    a un AGV le puede tocar una tarea de la otra mitad y cruzar una vez. Pasa en
    alguna semilla y se ve en el CSV; con 2 AGVs en 13 nodos no cambia nada.)
    """
    return ScenarioSpec(
        letter="A",
        name="Baja congestion",
        map_name="warehouse",
        starts=("S1", "S6"),
        first_targets=("N2", "N5"),
        pools=(("S1", "S2", "N1", "N2"), ("S5", "S6", "N5", "N6")),
        tasks_per_agent=3,
        tests="dos AGVs en mitades disjuntas: casi nadie se cruza con nadie",
    )


def scenario_b() -> ScenarioSpec:
    """B - Congestion media: 4 AGVs cuyas rutas arrancan cruzandose en S3.

    QUE PRUEBA: un solo nodo disputado. S3 une S2, N3 y G, y las cuatro rutas
    iniciales pasan por el:

        S1 -> G  : S1->S2->S3->G          N1 -> S3 : N1->N2->N3->S3
        S2 -> N3 : S2->S3->N3             N3 -> S2 : N3->S3->S2

    S2->N3 y N3->S2 son ademas una pareja de frente: el conflicto de arista de
    manual, en el mismo sitio donde los otros dos se cruzan.

    Las tareas siguientes salen de **todo el mapa** y no del bloque oeste, y eso
    no es un descuido. Con un pool de 7 nodos y 4 AGVs, la mitad de las corridas
    acababan sin poder terminar: el AGV que se queda sin tareas **aparca**, y un
    AGV aparcado encima del destino de otro lo bloquea para siempre (es el
    mismo riesgo que documenta `Simulation._planea_rutas`). Con el mapa entero
    se pasa de 12 a 16 corridas completas de 20 sin perder la congestion:
    1.73 conflictos por tick, los mismos que D.

    QUE SE ESPERA VER: conflictos de nodo y de arista de verdad, esperas
    repartidas y ningun deadlock (para eso esta el desatasco de la fase 8). Es
    el escenario donde ceder el paso a tiempo deberia notarse, porque hay algo
    que ceder y sitio para hacerlo.
    """
    return ScenarioSpec(
        letter="B",
        name="Congestion media",
        map_name="warehouse",
        starts=("S1", "S2", "N1", "N3"),
        first_targets=("G", "N3", "S3", "S2"),
        pools=(TODO_EL_ALMACEN,),
        tasks_per_agent=3,
        tests="cuatro rutas que se cruzan en S3, dos de ellas de frente",
    )


def scenario_c() -> ScenarioSpec:
    """C - Alta congestion: 6 AGVs en 13 nodos y destinos por todo el mapa.

    QUE PRUEBA: el almacen por encima de su capacidad. Seis AGVs ocupan casi la
    mitad de los nodos, las tareas salen de los trece y muchas cruzan G. Es el
    regimen de la fase 9 pero peor, y el escenario donde el motor tiene que
    tirar del desatasco todo el rato (200-300 movimientos forzados por corrida).

    OJO CON EL TOPE, que aqui es 2000 ticks y no 800 como en los demas, y hay
    una razon medida: con 6 AGVs el almacen **se satura**. Con el tope de 800 la
    baseline no completa ni una de 20 corridas y despacha el 25.6 % de las
    tareas; con 2000 despacha el 93.9 %. No es lentitud del programa, es la
    capacidad del mapa: 18 tareas con 6 AGVs cuestan ~1300 ticks, contra los
    ~310 que cuestan 12 tareas con 4 AGVs en el escenario B.

    Y AQUI EL MAKESPAN NO SIRVE PARA COMPARAR, que es lo que hay que saber
    antes de leer la tabla. Con la baseline, 15 de las 20 corridas se quedan a
    **1, 2 o 3 tareas** del final y ya no avanzan nunca: el AGV que se queda sin
    trabajo aparca, y con 6 AGVs sobre 13 nodos es casi seguro que uno acabe
    aparcado justo encima del ultimo destino que queda por servir. Subir el tope
    de 2000 a 5000 no cambia ni una: siguen siendo 5 completas y el 93.9 % de
    tareas. Asi que en C el makespan de esas 15 corridas es el tope que se
    elija, no una medida de nada, y lo que compara de verdad a las dos politicas
    es **`task_rate` y `throughput`** (cuanto trabajo despacha cada una antes de
    atascarse). El veredicto de `scenario_verdict()` lo tiene en cuenta.

    Es un limite del montaje, no un fallo del motor ni de las politicas: las dos
    lo sufren igual y el escenario sigue siendo pareado. Sacarlo aqui por
    escrito es preferible a publicar un makespan que solo mide el tope.

    QUE SE ESPERA VER: muchos conflictos por tick (2.5 contra 1.7 en B), muchos
    desatascos forzados (200-300 por corrida), y la cola sin despachar del todo.
    Al leerlo hay que mirar **primero** `task_rate` y `throughput`: los totales
    crudos premian a la politica que muere antes.
    """
    return ScenarioSpec(
        letter="C",
        name="Alta congestion",
        map_name="warehouse",
        starts=("S1", "N2", "S3", "N4", "S5", "N6"),
        first_targets=("N6", "S5", "N4", "S3", "N2", "S1"),
        pools=(TODO_EL_ALMACEN,),
        tasks_per_agent=3,
        max_steps=2000,
        tests="seis AGVs en trece nodos: el almacen por encima de su capacidad",
    )


def scenario_d() -> ScenarioSpec:
    """D - Cuello de botella: todas las tareas cruzan G, que es el unico paso.

    QUE PRUEBA: que pasa cuando **no hay ruta alternativa**. G es punto de
    articulacion del `warehouse`, asi que ir de una mitad a la otra pasa por el
    a la fuerza. Los cuatro AGVs arrancan alternando lado (S1 oeste, S6 este,
    N1 oeste, N6 este) y la cola alterna pool: al que acaba de entregar en el
    este le toca un destino del oeste y al reves, asi que las tareas siguen
    cruzando G una y otra vez en vez de cruzarlo una vez y quedarse.

    QUE SE ESPERA VER: G saturado, mucha espera y muchos conflictos de nodo
    (1.74 por tick con la baseline). Y sobre todo: **el REROUTE no tiene a donde
    ir**. Penalizar G no le da a A* una ruta alternativa, le da una peor o
    ninguna, asi que recalcular aqui solo cuesta ticks. Es el escenario donde la
    mania de rerutear de la politica aprendida deberia salirle mas cara.
    """
    return ScenarioSpec(
        letter="D",
        name="Cuello de botella",
        map_name="warehouse",
        starts=("S1", "S6", "N1", "N6"),
        first_targets=("S5", "S2", "N5", "N2"),
        pools=(OESTE, ESTE),
        tasks_per_agent=4,
        tests="toda tarea cruza G, y G es el unico paso entre las dos mitades",
    )


def scenario_e() -> ScenarioSpec:
    """E - Rutas alternativas: el mismo trabajo que D, pero en un mapa redundante.

    QUE PRUEBA: es D con la unica diferencia que importa. El mapa `grid` es una
    rejilla 4x4 con todos los tramos del mismo costo, asi que entre dos nodos
    cualesquiera hay **varias rutas de costo minimo**: penalizar un nodo deja a
    A* una alternativa igual de buena en vez de un rodeo. Los AGVs arrancan
    alternando columna extrema (A1, D4, A3, D2) y la cola alterna entre la
    columna A y la D, asi que cruzan la rejilla de lado a lado sin parar, igual
    que en D. La presion es la misma: 1.70 conflictos por tick contra los 1.74
    de D.

    QUE SE ESPERA VER: si el REROUTE del Q-Learning vale para algo, tiene que
    verse AQUI y no en D: mismo trabajo, misma presion, pero con una salida. El
    par D/E es lo que convierte "el Q-Learning rerutea mucho" en una respuesta
    sobre si rerutear sirve o no.
    """
    return ScenarioSpec(
        letter="E",
        name="Rutas alternativas",
        map_name="grid",
        starts=("A1", "D4", "A3", "D2"),
        first_targets=("D3", "A2", "D1", "A4"),
        pools=(("D1", "D2", "D3", "D4"), ("A1", "A2", "A3", "A4")),
        tasks_per_agent=4,
        tests="el mismo cruce constante que D, pero con rutas de igual costo",
    )


SCENARIOS: dict[str, Callable[[], ScenarioSpec]] = {
    "A": scenario_a,
    "B": scenario_b,
    "C": scenario_c,
    "D": scenario_d,
    "E": scenario_e,
}

LETTERS: tuple[str, ...] = tuple(SCENARIOS)


def get(letter: str) -> ScenarioSpec:
    """El escenario de esa letra, ya comprobado. Insensible a mayusculas."""
    clave = str(letter).strip().upper()
    if clave not in SCENARIOS:
        raise ValueError(
            f"no conozco el escenario {letter!r}; los que hay son "
            + ", ".join(LETTERS)
        )
    spec = SCENARIOS[clave]()
    spec.check()
    return spec


def all_scenarios() -> list[ScenarioSpec]:
    """Los cinco, en orden y ya comprobados."""
    return [get(letra) for letra in LETTERS]


# --- El runner -----------------------------------------------------------------


def model_for(spec: ScenarioSpec, *, per_scenario: bool, model: str | Path | None) -> Path:
    """Que Q-table le toca a este escenario: la general o la suya."""
    if per_scenario:
        return config.scenario_model(spec.letter)
    return Path(model) if model is not None else config.Q_TABLE_FILE


def run_scenario(
    spec: ScenarioSpec,
    policies: Sequence[str] = config.POLICIES,
    *,
    runs: int = config.SCENARIO_RUNS,
    seeds: Sequence[int] | None = None,
    model: str | Path | None = None,
    max_steps: int | None = None,
    out_dir: str | Path | None = config.RESULTS_DIR,
) -> dict[str, list[metrics.RunMetrics]]:
    """Corre UN escenario con las politicas que se pidan y escribe sus CSV.

    Reutiliza `metrics.run_comparison()` entero: lo unico que cambia es que el
    escenario de cada semilla lo construye `spec.build` en vez del sorteo de la
    fase 9. La invariante se mantiene intacta — un escenario por semilla,
    construido **fuera** del bucle de politicas — y por eso las dos corridas
    siguen siendo comparables.

    Con `out_dir=None` no escribe nada, que es como lo llaman los tests.
    """
    grafo = spec.graph()
    semillas = list(seeds) if seeds else spec.seeds(runs)
    tope = spec.max_steps if max_steps is None else int(max_steps)

    for linea in spec.header_lines():
        log.info("%s", linea)

    resultados = metrics.run_comparison(
        grafo,
        spec.n_agents,
        spec.n_tasks,
        semillas,
        list(policies),
        model=model,
        max_steps=tope,
        builder=spec.build,
    )

    if out_dir is not None:
        destino = Path(out_dir)
        for nombre, corridas in resultados.items():
            metrics.write_runs_csv(
                corridas, destino / config.scenario_csv(spec.letter, nombre).name
            )
    return resultados


# --- La tabla resumen ----------------------------------------------------------

# Lo que identifica una fila, antes de las metricas.
SUMMARY_HEAD: tuple[str, ...] = (
    "scenario",
    "name",
    "map",
    "agents",
    "tasks",
    "policy",
    "model",
    "runs",
    "completion_rate",
    "task_rate",
    "deadlock_free_rate",
)

# Y lo que va detras de las metricas: el pareado contra la baseline.
SUMMARY_TAIL: tuple[str, ...] = (
    "makespan_median",
    "makespan_stdev",
    "wins_vs_baseline",
    "losses_vs_baseline",
    "ties_vs_baseline",
)

# El contrato del fichero. Las columnas del medio son la MEDIA de cada metrica
# sobre las N corridas, con el mismo nombre que en los CSV por corrida.
SUMMARY_COLUMNS: tuple[str, ...] = SUMMARY_HEAD + metrics.METRIC_FIELDS + SUMMARY_TAIL


def summary_rows(
    spec: ScenarioSpec,
    results: dict[str, list[metrics.RunMetrics]],
    *,
    model: str | Path | None = None,
) -> list[dict[str, object]]:
    """Una fila por politica de este escenario, con todo ya promediado.

    `wins_vs_baseline` sale de `metrics.paired_wins()` sobre el makespan, que es
    lo que la media sola no dice: dos politicas pueden empatar de media porque
    una gana casi siempre por poco y pierde una vez por mucho. En la fila de la
    propia baseline va vacio, que no tiene sentido compararla consigo misma.
    """
    referencia = results.get(config.POLICY_BASELINE, [])
    filas: list[dict[str, object]] = []

    for nombre, corridas in results.items():
        if not corridas:
            continue
        resumen = metrics.summarize(corridas)
        fila: dict[str, object] = {
            "scenario": spec.letter,
            "name": spec.name,
            "map": spec.map_name,
            "agents": spec.n_agents,
            "tasks": spec.n_tasks,
            "policy": nombre,
            "model": (
                Path(model).name
                if model is not None and nombre == config.POLICY_QLEARNING
                else ""
            ),
            "runs": resumen["runs"],
            "completion_rate": resumen["completion_rate"],
            "task_rate": resumen["task_rate"],
            "deadlock_free_rate": resumen["deadlock_free_rate"],
        }
        for campo in metrics.METRIC_FIELDS:
            fila[campo] = resumen["metrics"][campo]["mean"]
        fila["makespan_median"] = resumen["metrics"]["makespan"]["median"]
        fila["makespan_stdev"] = resumen["metrics"]["makespan"]["stdev"]

        if nombre != config.POLICY_BASELINE and referencia:
            gana_base, gana_otro, empates = metrics.paired_wins(
                referencia, corridas, metrics.MAIN_FIELD
            )
            fila["wins_vs_baseline"] = gana_otro
            fila["losses_vs_baseline"] = gana_base
            fila["ties_vs_baseline"] = empates
        else:
            fila["wins_vs_baseline"] = ""
            fila["losses_vs_baseline"] = ""
            fila["ties_vs_baseline"] = ""
        filas.append(fila)
    return filas


def write_summary_table(
    rows: Sequence[dict[str, object]], path: str | Path = config.SUMMARY_TABLE_FILE
) -> Path:
    """Escribe `results/summary_table.csv`: una fila por (escenario, politica).

    Es el fichero que se pega en el reporte, asi que va plano y con coma: se
    abre en Excel tal cual y lo lee `csv.DictReader` sin configurar nada.

    La tabla es siempre la de **esta** invocacion y no se acumula: mezclar filas
    de dos corridas distintas daria una tabla con escenarios medidos con otro
    modelo o con otro numero de semillas, y eso en un reporte es peor que no
    tenerla. Lo que si se avisa es cuando la que se escribe tiene menos filas que
    la que habia, que es lo que pasa al correr `--name X` despues de un `--all`.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.is_file():
        try:
            antes = len(read_summary_table(destino))
        except (OSError, csv.Error):
            antes = 0
        if antes > len(rows):
            log.warning(
                "%s tenia %d fila(s) y esta corrida escribe %d: la tabla es la de "
                "esta invocacion, no se acumula (usa --no-summary o --out para "
                "no pisarla)",
                destino,
                antes,
                len(rows),
            )
    with destino.open("w", encoding=config.ENCODING, newline="") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=list(SUMMARY_COLUMNS))
        escritor.writeheader()
        for fila in rows:
            escritor.writerow({clave: fila.get(clave, "") for clave in SUMMARY_COLUMNS})
    log.info("%s: %d fila(s), %d columnas", destino, len(rows), len(SUMMARY_COLUMNS))
    return destino


def read_summary_table(
    path: str | Path = config.SUMMARY_TABLE_FILE,
) -> list[dict[str, str]]:
    """Lee la tabla resumen tal cual, en texto. Para los tests y para mirarla."""
    with Path(path).open("r", encoding=config.ENCODING, newline="") as fichero:
        return list(csv.DictReader(fichero))


# --- El veredicto por escenario ------------------------------------------------

# Por debajo de este porcentaje de corridas completas, la media del makespan es
# el tope de ticks disfrazado de medida y el veredicto pasa a decidirse por el
# trabajo despachado. Ver el docstring del escenario C.
SATURATED_BELOW: float = 50.0


def scenario_verdict(
    spec: ScenarioSpec, results: dict[str, list[metrics.RunMetrics]]
) -> str:
    """Una linea: si el Q-Learning mejora en este escenario, y en que se nota.

    Se decide con **tres** cosas y no con una: el makespan medio, en cuantas
    semillas gana pareado, y cuantas corridas completa. Una politica que baja el
    makespan medio porque termina menos corridas no ha mejorado nada, y esa es
    justo la trampa que la fase 9 dejo documentada.
    """
    base = results.get(config.POLICY_BASELINE, [])
    otra = results.get(config.POLICY_QLEARNING, [])
    if not base or not otra:
        return f"{spec.letter}: falta una de las dos politicas, no hay veredicto"

    resumen_base = metrics.summarize(base)
    resumen_otra = metrics.summarize(otra)
    mk_base = resumen_base["metrics"]["makespan"]["mean"]
    mk_otra = resumen_otra["metrics"]["makespan"]["mean"]
    gana_base, gana_otra, empates = metrics.paired_wins(base, otra, metrics.MAIN_FIELD)
    completa_base = resumen_base["completion_rate"]
    completa_otra = resumen_otra["completion_rate"]

    # Cuando casi ninguna corrida termina, el makespan de las que no terminan es
    # el tope de ticks: comparar esas medias seria comparar el tope consigo
    # mismo. Ahi manda el trabajo despachado, que es lo que si distingue a una
    # politica de la otra (ver el docstring del escenario C).
    saturado = max(completa_base, completa_otra) < SATURATED_BELOW
    if saturado:
        base_valor, otra_valor = resumen_base["task_rate"], resumen_otra["task_rate"]
        mejor = otra_valor > base_valor
        peor = otra_valor < base_valor
    else:
        base_valor, otra_valor = mk_base, mk_otra
        mejor = mk_otra < mk_base and gana_otra >= gana_base and completa_otra >= completa_base
        peor = mk_otra > mk_base and gana_otra <= gana_base

    if mejor:
        juicio = "MEJORA"
    elif peor:
        juicio = "NO APORTA"
    elif base_valor == otra_valor and gana_base == gana_otra:
        juicio = "EMPATE"
    else:
        juicio = "MIXTO"

    cabeza = (
        f"tareas despachadas {otra_valor:.1f}% contra {base_valor:.1f}% "
        f"(el makespan aqui es el tope, no dice nada)"
        if saturado
        else (
            f"makespan {mk_otra:.1f} contra {mk_base:.1f} "
            f"({metrics._diferencia(mk_base, mk_otra)})"
        )
    )
    return (
        f"{spec.letter} ({spec.name}): {juicio}. {cabeza}, "
        f"gana {gana_otra} de {gana_base + gana_otra + empates} semillas, "
        f"completa {completa_otra:.0f}% contra {completa_base:.0f}%"
    )


# --- Entrenar una Q-table por escenario ----------------------------------------


def train_scenario(
    spec: ScenarioSpec,
    cfg: qlearning.TrainingConfig,
    *,
    model_path: str | Path | None = None,
    log_path: str | Path | None = None,
    curve_path: str | Path | None = None,
) -> qlearning.Trainer:
    """Entrena una Q-table EN este escenario y la escribe en su propio fichero.

    Es el `train` de la fase 7 con una sola diferencia: los episodios se juegan
    en las posiciones de salida y con los destinos del escenario
    (`spec.routes_factory()`) en vez de en el sorteo abierto de todo el mapa.

    **Lo que esto sigue sin arreglar**, y hay que decirlo: un episodio de
    entrenamiento es UNA tarea por AGV y `max_steps` ticks, mientras que la
    evaluacion son varias tareas por AGV con AGVs que terminan y se quedan
    aparcados ocupando nodos. La cola de tareas vive en `metrics.py` y no en
    `TrainingEnv`, asi que ese salto de regimen sigue ahi.
    """
    destino = (
        Path(model_path) if model_path is not None else config.scenario_model(spec.letter)
    )
    return qlearning.train(
        spec.graph(),
        cfg,
        model_path=destino,
        log_path=log_path,
        curve_path=curve_path,
        routes_factory=spec.routes_factory(),
    )
