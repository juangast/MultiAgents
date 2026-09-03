"""CLI del servidor de simulacion de AGVs."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import astar
import config
import graph
import metrics
import qlearning
import scenarios
import server
import simulation
from logs import get_logger, setup_logging

log = get_logger("main")


def cmd_serve(args: argparse.Namespace) -> int:
    """Levanta el servidor TCP para Unity con la simulacion de verdad.

    `--policy` es la unica variable experimental: con `baseline` y con
    `qlearning` corre exactamente el mismo motor. Y se puede cambiar en caliente
    desde el socket con `SET_MODE`, sin reiniciar nada.
    """
    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    try:
        simulacion = simulation.Simulation(
            grafo, args.agents, policy=args.policy, model=args.model
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    log.info(
        "sirviendo el mapa %s con %d agente(s) en modo %s",
        grafo.name or "(sin nombre)",
        len(simulacion.agents),
        simulacion.mode,
    )
    return server.serve_forever(simulacion, host=args.host, port=args.port)


def _abre_mapa(nombre: str) -> tuple[graph.WarehouseGraph | None, str, int]:
    """Carga un mapa del disco (o el interno) y lo valida.

    Devuelve (grafo, de donde salio, codigo de salida). El grafo es None si algo
    fallo, y entonces el codigo es el que debe devolver el proceso: 1 si el mapa
    esta roto, 2 si ni siquiera existe. Lo comparten `map`, `simulate` y `serve`,
    que necesitan exactamente lo mismo.
    """
    ruta = graph.map_path(nombre)
    constructor = graph.BUILTIN_MAPS.get(nombre)

    try:
        if ruta.exists():
            grafo = graph.load_graph(ruta)
            origen = str(ruta)
        elif constructor is not None:
            log.warning("no existe %s, tiro del mapa interno", ruta)
            grafo = constructor()
            origen = "codigo"
        else:
            log.error(
                "no conozco el mapa %r: no existe %s y tampoco hay uno interno "
                "con ese nombre (internos: %s)",
                nombre,
                ruta,
                ", ".join(sorted(graph.BUILTIN_MAPS)),
            )
            return None, "", 2
    except graph.GraphError as exc:
        log.error("%s", exc)
        return None, "", 1

    # Se valida antes de devolverlo: sin un grafo valido no se puede ni construir
    # la tabla de coordenadas (un nodo puede no tener posicion), y el error de
    # validate() ya dice todo lo que le pasa al mapa.
    try:
        grafo.validate()
    except graph.GraphError as exc:
        log.error("validate(): %s", exc)
        return None, "", 1

    return grafo, origen, 0


def cmd_map(args: argparse.Namespace) -> int:
    """Muestra el mapa logico y lo valida."""
    grafo, origen, codigo = _abre_mapa(args.name)
    if grafo is None:
        return codigo

    _informa_mapa(grafo, origen)
    log.info("validate(): OK")
    return 0


def _informa_mapa(grafo: graph.WarehouseGraph, origen: str) -> None:
    """Escupe el mapa por el log: cabecera, nodos con sus dos coordenadas y aristas."""
    exportado = grafo.to_unity_dict()
    flecha = "->" if grafo.directed else "--"

    log.info("--- mapa %s ---", grafo.name or "(sin nombre)")
    log.info("origen        : %s", origen)
    log.info("nodos         : %d", len(exportado["nodes"]))
    log.info("aristas       : %d", len(exportado["edges"]))
    log.info("dirigido      : %s", "si" if grafo.directed else "no")
    log.info("UNITY_SCALE   : %s", config.UNITY_SCALE)

    log.info("--- nodos: logicas (x, y) -> Unity (x, y, z) ---")
    for nodo in exportado["nodes"]:
        px, py = grafo.positions[nodo["id"]]
        log.info(
            "%-4s %14s  ->  %s",
            nodo["id"],
            f"({px:g}, {py:g})",
            f"({nodo['x']:g}, {nodo['y']:g}, {nodo['z']:g})",
        )

    log.info("--- aristas ---")
    for arista in exportado["edges"]:
        log.info(
            "%-4s %s %-4s  costo %g", arista["from"], flecha, arista["to"], arista["cost"]
        )


def cmd_simulate(args: argparse.Namespace) -> int:
    """Corre la simulacion sin servidor y cuenta paso a paso lo que hace."""
    if not args.headless:
        log.warning("simulate solo tiene modo headless por ahora, corro igual")

    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    por_defecto = simulation.default_route(grafo)
    origen = args.origen if args.origen is not None else por_defecto[0]
    destino = args.destino if args.destino is not None else por_defecto[1]

    for bandera, nodo in (("--from", origen), ("--to", destino)):
        if nodo not in grafo.adjacency:
            log.error(
                "%s %r no es un nodo de %s (nodos: %s)",
                bandera,
                nodo,
                grafo.name or "(sin nombre)",
                ", ".join(grafo.nodes()),
            )
            return 2

    try:
        simulacion = simulation.Simulation(
            grafo,
            args.agents,
            origin=origen,
            target=destino,
            policy=args.policy,
            model=args.model,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    return _corre_simulacion(simulacion, args.steps)


def _corre_simulacion(simulacion: simulation.Simulation, pasos: int) -> int:
    """Tickea hasta `pasos`, o hasta que lleguen todos, contandolo por el log.

    Devuelve 0 aunque la corrida muera en deadlock: un baseline que se atasca es
    un resultado experimental valido, no un fallo del programa. Lo unico que se
    considera error es que un AGV no tenga ni ruta que seguir.
    """
    grafo = simulacion.graph
    log.info(
        "--- simulacion: mapa %s, %d agente(s), modo %s, %d pasos como mucho ---",
        grafo.name or "(sin nombre)",
        len(simulacion.agents),
        simulacion.mode,
        pasos,
    )

    sin_ruta = False
    for agente in simulacion.agents:
        if not agente.path:
            sin_ruta = True
            log.error(
                "AGV %s: no hay ruta hasta %s, se queda %s en %s",
                agente.id,
                agente.target_node,
                agente.state,
                agente.current_node,
            )
            continue
        log.info(
            "AGV %s: %s -> %s | costo %g | %s",
            agente.id,
            agente.path[0],
            agente.path[-1],
            astar.path_cost(grafo, agente.path),
            " -> ".join(agente.path),
        )

    log.info("--- pasos ---")
    while simulacion.step < pasos and not simulacion.done:
        simulacion.tick()
        for agente in simulacion.agents:
            registro = simulacion.action_record(agente.id)
            log.info(
                "paso %3d | AGV %s | %-7s | %-7s%-1s | %-4s -> %-4s | %3.0f%% | "
                "tramo %d/%d | espera %3d | tarea %s",
                simulacion.step,
                agente.id,
                agente.state,
                "-" if registro is None else registro.action,
                "!" if registro is not None and registro.blocked else " ",
                agente.current_node,
                agente.next_node() or "-",
                agente.progress * 100.0,
                agente.path_index,
                max(len(agente.path) - 1, 0),
                agente.wait_time,
                "-" if agente.task is None else agente.task,
            )

    numeros = simulacion.stats()
    log.info("--- resumen ---")
    log.info("pasos dados : %d de %d", simulacion.step, pasos)
    log.info("final       : %s", _razon_del_final(simulacion))
    log.info(
        "conflictos  : %d (%s)",
        numeros["conflicts"],
        ", ".join(
            f"{tipo} {cuantos}"
            for tipo, cuantos in numeros["conflicts_by_type"].items()
        ),
    )
    log.info("espera total: %d ticks entre todos", numeros["total_wait_time"])
    log.info(
        "acciones    : %s (desatascos forzados: %d)",
        ", ".join(f"{accion} {cuantas}" for accion, cuantas in numeros["actions"].items()),
        numeros["forced"],
    )
    for agente in simulacion.agents:
        log.info(
            "AGV %s      : %s en %s, tramo %d/%d, %d ticks esperando",
            agente.id,
            agente.state,
            agente.current_node,
            agente.path_index,
            max(len(agente.path) - 1, 0),
            agente.wait_time,
        )

    if sin_ruta:
        return 1
    return 0


def _razon_del_final(simulacion: simulation.Simulation) -> str:
    """Por que se paro la corrida, en una linea para el resumen."""
    if simulacion.finished_reason == simulation.FINISHED_DEADLOCK:
        return (
            f"deadlock, nadie avanzo en {config.DEADLOCK_TICKS} ticks seguidos "
            f"(el baseline no sabe deshacerlo: para eso esta el Q-Learning)"
        )
    if simulacion.done:
        return "llegaron todos"
    return "se acabaron los pasos antes de que llegaran todos"


def cmd_train(args: argparse.Namespace) -> int:
    """Modo TRAIN: entrena la Q-table y escribe el modelo, el CSV y la curva.

    **No levanta el servidor ni habla con Unity.** Mil episodios son unos cientos
    de miles de ticks: meter un socket en medio multiplicaria el tiempo por el
    ping y no le daria al algoritmo ni un dato mas. Unity entra despues, con
    `serve`, a ver correr lo aprendido.

    Con `--scenario X` entrena **en** ese escenario de la fase 10: su mapa, sus
    AGVs y sus posiciones de salida, en vez del sorteo abierto de todo el mapa.
    Es la mitad "una Q-table por escenario" del experimento de la fase; la otra
    mitad es la tabla general que ya esta en `python/models/q_table.json`.
    """
    if getattr(args, "scenario", None):
        return _entrena_escenario(args)

    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    ajustes = _ajustes(args, grafo)
    try:
        entrenador = qlearning.train(
            grafo,
            ajustes,
            model_path=args.model,
            log_path=args.log,
            curve_path=None if args.no_curve else args.curve,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    _informa_modelo(args.model, entrenador.metadata())
    return 0


def _entrena_escenario(args: argparse.Namespace) -> int:
    """`train --scenario X`: entrena en las salidas y los destinos de ese escenario.

    El mapa y el numero de AGVs los manda el escenario, no el CLI: entrenar en
    `--map simple` una tabla que se va a servir en el `warehouse` del escenario
    D no mediria lo que la fase pregunta. Y el modelo va por defecto a su propio
    fichero (`q_table_<letra>.json`) para no pisar la tabla general, que es
    contra la que hay que compararlo.
    """
    try:
        spec = scenarios.get(args.scenario)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    destino = (
        Path(args.model)
        if args.model != str(config.Q_TABLE_FILE)
        else config.scenario_model(spec.letter)
    )
    log.info(
        "entrenando en el escenario %s (%s): mapa %s, %d AGVs, salidas %s -> %s",
        spec.letter,
        spec.name,
        spec.map_name,
        spec.n_agents,
        ", ".join(spec.starts),
        destino,
    )

    try:
        ajustes = qlearning.TrainingConfig(
            map_name=spec.map_name,
            agents=spec.n_agents,
            episodes=args.episodes,
            seed=args.seed,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=args.epsilon_decay,
            max_steps=args.max_steps,
            enable_reroute=False if args.no_reroute else None,
            scenario=spec.letter,
        )
        entrenador = scenarios.train_scenario(
            spec,
            ajustes,
            model_path=destino,
            log_path=args.log,
            curve_path=None if args.no_curve else args.curve,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    _informa_modelo(destino, entrenador.metadata())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Modo EVALUATE: carga la Q-table del disco y juega greedy puro.

    `epsilon = 0` y la tabla **no se toca**: no explora, no aprende y no guarda
    nada. Corre ademas la baseline de la fase 5 sobre los mismos escenarios, que
    es contra lo que hay que comparar para que los numeros signifiquen algo.
    """
    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    modelo = Path(args.model)
    if not modelo.is_file():
        log.error(
            "no existe el modelo %s; entrenalo antes con: "
            "python3 python/main.py train --map %s --agents %d",
            modelo,
            args.map,
            args.agents,
        )
        return 2

    try:
        aprendida, referencia = qlearning.evaluate(
            grafo, _ajustes(args, grafo), model_path=modelo, episodes=args.episodes
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    _informa_modelo(modelo, qlearning.load_metadata(modelo))
    for linea in qlearning.compare_lines(aprendida.history, referencia):
        log.info("%s", linea)
    if args.log:
        qlearning.write_training_log(aprendida.history, args.log)
    return 0


def _ajustes(args: argparse.Namespace, grafo: graph.WarehouseGraph) -> qlearning.TrainingConfig:
    """Arma la `TrainingConfig` con lo que venga por CLI, y el resto de config.py."""
    return qlearning.TrainingConfig(
        map_name=grafo.name or args.map,
        agents=args.agents,
        episodes=getattr(args, "episodes", config.EPISODES),
        seed=args.seed,
        alpha=getattr(args, "alpha", config.ALPHA),
        gamma=getattr(args, "gamma", config.GAMMA),
        epsilon_start=getattr(args, "epsilon_start", config.EPSILON_START),
        epsilon_end=getattr(args, "epsilon_end", config.EPSILON_END),
        epsilon_decay=getattr(args, "epsilon_decay", config.EPSILON_DECAY),
        max_steps=args.max_steps,
        enable_reroute=False if getattr(args, "no_reroute", False) else None,
    )


def _informa_modelo(path: str | Path, metadata: dict[str, object]) -> None:
    """Escupe por el log con que se entreno una Q-table."""
    log.info("--- modelo %s ---", path)
    if not metadata:
        log.warning("sin metadata: no hay forma de saber con que se entreno")
        return
    for clave, valor in metadata.items():
        if isinstance(valor, dict):
            log.info(
                "%-16s: %s",
                clave,
                ", ".join(f"{sub}={dato}" for sub, dato in valor.items()),
            )
        else:
            log.info("%-16s: %s", clave, valor)


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Modo BENCHMARK: enfrenta las politicas semilla a semilla y escribe results/.

    La regla de la fase: para cada semilla se construye UN escenario y se corre
    con TODAS las politicas. Mismo mapa, mismos AGVs, mismos origenes, mismos
    destinos, misma cola de tareas y misma semilla; lo unico que cambia es la
    politica. Si cambiara algo mas, la comparacion no mediria la politica.

    Devuelve 0 aunque el Q-Learning pierda: un resultado malo es un resultado, y
    lo que aqui seria un fallo es no poder correr el experimento.
    """
    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    politicas = list(dict.fromkeys(args.policies))
    modelo = Path(args.model)
    if config.POLICY_QLEARNING in politicas and not modelo.is_file():
        log.error(
            "no existe el modelo %s; entrenalo antes con: "
            "python3 python/main.py train --map %s --agents %d",
            modelo,
            args.map,
            args.agents,
        )
        return 2

    semillas = args.seeds if args.seeds else list(range(1, args.runs + 1))
    if len(semillas) < config.BENCHMARK_RUNS:
        log.warning(
            "%d semillas: con menos de %d la media y la desviacion dicen poco",
            len(semillas),
            config.BENCHMARK_RUNS,
        )

    try:
        resultados = metrics.run_comparison(
            grafo,
            args.agents,
            args.tasks,
            semillas,
            politicas,
            model=modelo,
            max_steps=args.max_steps,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    destino = Path(args.out)
    for nombre, corridas in resultados.items():
        metrics.write_runs_csv(corridas, destino / f"{nombre}.csv")
    metrics.write_comparison_json(
        resultados,
        destino / config.COMPARISON_JSON.name,
        header={
            "map": grafo.name or args.map,
            "agents": args.agents,
            "tasks": resultados[politicas[0]][0].n_tasks,
            "seeds": semillas,
            "policies": politicas,
            "model": str(modelo),
            "max_steps": args.max_steps,
        },
    )
    if not args.no_plots:
        metrics.save_comparison_plot(resultados, destino / config.COMPARISON_PLOT.name)

    for linea in metrics.comparison_lines(resultados):
        log.info("%s", linea)
    return 0


def cmd_scenario(args: argparse.Namespace) -> int:
    """Modo SCENARIO: corre los escenarios de la fase 10 y escribe la tabla resumen.

    Cada escenario se corre con cada politica sobre las MISMAS semillas, con el
    escenario construido una vez por semilla y fuera del bucle de politicas
    (es la invariante de la fase 9, aqui intacta). Escribe un
    `results/scenario_<letra>_<policy>.csv` por cada par y un
    `results/summary_table.csv` con una fila por (escenario, politica).

    Devuelve 0 aunque el Q-Learning pierda en los cinco: un resultado malo es un
    resultado. Lo que aqui es un error es no poder correr el experimento.
    """
    if not args.all and not args.name:
        log.error(
            "hace falta decir que escenario: --name {%s} o --all para los cinco",
            "|".join(scenarios.LETTERS),
        )
        return 2

    try:
        specs = scenarios.all_scenarios() if args.all else [scenarios.get(args.name)]
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    politicas = [args.policy] if args.policy else list(config.POLICIES)
    if config.POLICY_QLEARNING in politicas:
        codigo = _comprueba_modelos(specs, args)
        if codigo:
            return codigo

    destino = Path(args.out)
    filas: list[dict[str, object]] = []
    veredictos: list[str] = []

    for spec in specs:
        modelo = scenarios.model_for(
            spec, per_scenario=args.per_scenario_model, model=args.model
        )
        try:
            resultados = scenarios.run_scenario(
                spec,
                politicas,
                runs=args.runs,
                seeds=args.seeds,
                model=modelo,
                max_steps=args.max_steps,
                out_dir=destino,
            )
        except ValueError as exc:
            log.error("escenario %s: %s", spec.letter, exc)
            return 2

        for linea in metrics.comparison_lines(resultados):
            log.info("%s", linea)
        filas.extend(scenarios.summary_rows(spec, resultados, model=modelo))
        if len(politicas) > 1:
            veredictos.append(scenarios.scenario_verdict(spec, resultados))

    if not args.no_summary:
        scenarios.write_summary_table(filas, destino / config.SUMMARY_TABLE_FILE.name)

    if veredictos:
        log.info("--- veredicto por escenario ---")
        for linea in veredictos:
            log.info("%s", linea)
    return 0


def _comprueba_modelos(
    specs: Sequence[scenarios.ScenarioSpec], args: argparse.Namespace
) -> int:
    """Que existan las Q-tables que hacen falta, antes de correr nada.

    Se comprueban **todas** de golpe y no una por una segun toca: con `--all`,
    reventar en el escenario D despues de haber corrido A, B y C dejaria
    `results/` a medias y con una tabla resumen incompleta.
    """
    faltan: list[Path] = []
    for spec in specs:
        modelo = scenarios.model_for(
            spec, per_scenario=args.per_scenario_model, model=args.model
        )
        if not modelo.is_file() and modelo not in faltan:
            faltan.append(modelo)

    if not faltan:
        return 0

    for modelo in faltan:
        if args.per_scenario_model:
            letra = modelo.stem.rsplit("_", 1)[-1]
            log.error(
                "no existe el modelo %s; entrenalo antes con: "
                "python3 python/main.py train --scenario %s",
                modelo,
                letra,
            )
        else:
            log.error(
                "no existe el modelo %s; entrenalo antes con: "
                "python3 python/main.py train --map %s",
                modelo,
                config.DEFAULT_MAP,
            )
    return 2


COMMANDS: dict[str, str] = {
    "serve": "Levanta el servidor TCP para Unity",
    "map": "Muestra el mapa logico y lo valida",
    "simulate": "Corre la simulacion sin servidor",
    "train": "Entrena los agentes con Q-Learning",
    "evaluate": "Evalua una politica ya entrenada",
    "benchmark": "Mide el rendimiento de la simulacion",
    "scenario": "Corre los escenarios de la fase 10 y su tabla resumen",
}

HANDLERS = {
    "serve": cmd_serve,
    "map": cmd_map,
    "simulate": cmd_simulate,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "benchmark": cmd_benchmark,
    "scenario": cmd_scenario,
}


def build_parser() -> argparse.ArgumentParser:
    """Arma el parser con los subcomandos."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Muestra los mensajes de nivel DEBUG",
    )

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Servidor Python de la simulacion multiagente de AGVs.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="command", metavar="subcomando")

    for name, help_text in COMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text, parents=[common])
        if name == "serve":
            sub.add_argument(
                "--map",
                default=config.DEFAULT_MAP,
                help=f"Mapa a servir (por defecto {config.DEFAULT_MAP})",
            )
            sub.add_argument(
                "--host",
                default=config.HOST,
                help=f"Direccion donde escuchar (por defecto {config.HOST})",
            )
            sub.add_argument(
                "--port",
                type=int,
                default=config.PORT,
                help=f"Puerto donde escuchar (por defecto {config.PORT})",
            )
            sub.add_argument(
                "--agents",
                type=int,
                default=1,
                help="Cuantos AGVs servir (por defecto 1)",
            )
            _argumentos_de_politica(sub)
        elif name == "map":
            sub.add_argument(
                "--name",
                default=config.DEFAULT_MAP,
                help=(
                    f"Mapa a mostrar, de python/maps/ "
                    f"(por defecto {config.DEFAULT_MAP})"
                ),
            )
        elif name == "simulate":
            sub.add_argument(
                "--map",
                default=config.DEFAULT_MAP,
                help=f"Mapa por el que moverse (por defecto {config.DEFAULT_MAP})",
            )
            sub.add_argument(
                "--agents",
                type=int,
                default=1,
                help="Cuantos AGVs correr (por defecto 1)",
            )
            sub.add_argument(
                "--steps",
                type=int,
                default=100,
                help="Tope de pasos, corta antes si llegan todos (por defecto 100)",
            )
            sub.add_argument(
                "--headless",
                action="store_true",
                help="Corre sin servidor, que es el unico modo por ahora",
            )
            # "from" es palabra reservada, de ahi el dest explicito.
            sub.add_argument(
                "--from",
                dest="origen",
                default=None,
                help="Nodo de salida (por defecto el de la ruta del mapa)",
            )
            sub.add_argument(
                "--to",
                dest="destino",
                default=None,
                help="Nodo de destino (por defecto el de la ruta del mapa)",
            )
            _argumentos_de_politica(sub)
        elif name in ("train", "evaluate"):
            _argumentos_de_aprendizaje(sub, name)
        elif name == "benchmark":
            _argumentos_de_benchmark(sub)
        elif name == "scenario":
            _argumentos_de_escenario(sub)

    return parser


def _argumentos_de_politica(sub: argparse.ArgumentParser) -> None:
    """`--policy` y `--model`, que son los de la fase 8.

    Los comparten `serve` y `simulate` porque son la misma pregunta: con que
    politica se corre. **`--policy` es la unica variable experimental**: lo demas
    (mapa, agentes, semilla, conflictos, desatasco) es identico en los dos modos,
    y si no lo fuera, comparar las dos corridas no mediria la politica.
    """
    sub.add_argument(
        "--policy",
        choices=list(config.POLICIES),
        default=config.DEFAULT_POLICY,
        help=(
            f"Politica con la que correr (por defecto {config.DEFAULT_POLICY}); "
            f"con {config.POLICY_QLEARNING} hace falta un modelo entrenado"
        ),
    )
    sub.add_argument(
        "--model",
        default=str(config.Q_TABLE_FILE),
        help=(
            f"Q-table a cargar con --policy {config.POLICY_QLEARNING} "
            f"(por defecto {config.Q_TABLE_FILE})"
        ),
    )


def _argumentos_de_aprendizaje(sub: argparse.ArgumentParser, name: str) -> None:
    """Los argumentos de `train` y `evaluate`. Los defaults salen de `config.py`.

    Lo comun va primero (mapa, agentes, semilla, tope de ticks) y despues lo de
    cada modo: `train` puede tocar los hiperparametros y elige donde escribir;
    `evaluate` solo dice que modelo cargar y cuantos episodios jugar.
    """
    verbo = "entrenar" if name == "train" else "evaluar"
    sub.add_argument(
        "--map",
        default=config.DEFAULT_MAP,
        help=f"Mapa sobre el que {verbo} (por defecto {config.DEFAULT_MAP})",
    )
    sub.add_argument(
        "--agents",
        type=int,
        default=config.TRAIN_AGENTS,
        help=f"Cuantos AGVs por episodio (por defecto {config.TRAIN_AGENTS})",
    )
    sub.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help=f"Semilla; la misma da la misma corrida (por defecto {config.RANDOM_SEED})",
    )
    sub.add_argument(
        "--max-steps",
        type=int,
        default=config.MAX_STEPS_PER_EPISODE,
        help=f"Tope de ticks por episodio (por defecto {config.MAX_STEPS_PER_EPISODE})",
    )
    sub.add_argument(
        "--model",
        default=str(config.Q_TABLE_FILE),
        help=(
            "Donde se escribe la Q-table"
            if name == "train"
            else "Q-table a cargar"
        )
        + f" (por defecto {config.Q_TABLE_FILE})",
    )

    if name == "train":
        sub.add_argument(
            "--scenario",
            default=None,
            help=(
                "Entrena EN un escenario de la fase 10 ("
                + ", ".join(scenarios.LETTERS)
                + "): su mapa, sus AGVs y sus posiciones de salida. El modelo va "
                "a python/models/q_table_<letra>.json salvo que se de --model"
            ),
        )
        sub.add_argument(
            "--episodes",
            type=int,
            default=config.EPISODES,
            help=f"Cuantos episodios entrenar (por defecto {config.EPISODES})",
        )
        sub.add_argument("--alpha", type=float, default=config.ALPHA, help=f"Tasa de aprendizaje (por defecto {config.ALPHA})")
        sub.add_argument("--gamma", type=float, default=config.GAMMA, help=f"Descuento del futuro (por defecto {config.GAMMA})")
        sub.add_argument("--epsilon-start", type=float, default=config.EPSILON_START, help=f"Exploracion inicial (por defecto {config.EPSILON_START})")
        sub.add_argument("--epsilon-end", type=float, default=config.EPSILON_END, help=f"Suelo de la exploracion (por defecto {config.EPSILON_END})")
        sub.add_argument("--epsilon-decay", type=float, default=config.EPSILON_DECAY, help=f"Factor exponencial por episodio (por defecto {config.EPSILON_DECAY})")
        sub.add_argument(
            "--no-reroute",
            action="store_true",
            help="Deja fuera la accion REROUTE: solo ADVANCE y WAIT",
        )
        sub.add_argument(
            "--log",
            default=str(config.TRAINING_LOG_FILE),
            help=f"CSV con una fila por episodio (por defecto {config.TRAINING_LOG_FILE})",
        )
        sub.add_argument(
            "--curve",
            default=str(config.LEARNING_CURVE_FILE),
            help=f"PNG de la curva de aprendizaje (por defecto {config.LEARNING_CURVE_FILE})",
        )
        sub.add_argument(
            "--no-curve",
            action="store_true",
            help="No dibuja la curva aunque haya matplotlib",
        )
    else:
        sub.add_argument(
            "--episodes",
            type=int,
            default=100,
            help="Cuantos episodios evaluar (por defecto 100)",
        )
        sub.add_argument(
            "--log",
            default=None,
            help="CSV opcional con los episodios de la evaluacion",
        )


def _semillas(valor: str) -> list[int]:
    """Parsea `--seeds`: `1-20`, `1,2,3` o `1-5,10`. Sin repetidas y en orden.

    Un rango se escribe `a-b` con `a <= b`. Las semillas son la identidad de cada
    escenario, asi que repetirlas seria correr dos veces el mismo trabajo y
    contarlo como dos medidas: se quitan las duplicadas y se avisa por el error
    si el texto no se entiende.
    """
    numeros: list[int] = []
    for trozo in valor.split(","):
        trozo = trozo.strip()
        if not trozo:
            continue
        try:
            if "-" in trozo.lstrip("-"):
                desde, hasta = trozo.split("-", 1)
                arranque, final = int(desde), int(hasta)
                if arranque > final:
                    raise argparse.ArgumentTypeError(
                        f"el rango {trozo!r} va al reves: {arranque} > {final}"
                    )
                numeros.extend(range(arranque, final + 1))
            else:
                numeros.append(int(trozo))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{trozo!r} no es una semilla ni un rango de semillas"
            ) from None

    unicas = list(dict.fromkeys(numeros))
    if not unicas:
        raise argparse.ArgumentTypeError(f"{valor!r} no tiene ni una semilla")
    return unicas


def _argumentos_de_benchmark(sub: argparse.ArgumentParser) -> None:
    """Los argumentos de `benchmark`. Definen el escenario, no la politica.

    Todos menos `--policies` se aplican **identicos** a las dos politicas: esa es
    la condicion de que la comparacion mida algo. `--seeds` manda sobre `--runs`
    cuando se dan los dos, porque decir cuales es mas concreto que decir cuantas.
    """
    sub.add_argument(
        "--map",
        default=config.DEFAULT_MAP,
        help=f"Mapa sobre el que medir (por defecto {config.DEFAULT_MAP})",
    )
    sub.add_argument(
        "--agents",
        type=int,
        default=config.TRAIN_AGENTS,
        help=f"Cuantos AGVs por corrida (por defecto {config.TRAIN_AGENTS})",
    )
    sub.add_argument(
        "--tasks",
        type=int,
        default=None,
        help=(
            f"Tareas totales por corrida "
            f"(por defecto {config.BENCHMARK_TASKS_PER_AGENT} por AGV)"
        ),
    )
    sub.add_argument(
        "--runs",
        type=int,
        default=config.BENCHMARK_RUNS,
        help=f"Cuantas semillas correr si no se dan con --seeds (por defecto {config.BENCHMARK_RUNS})",
    )
    sub.add_argument(
        "--seeds",
        type=_semillas,
        default=None,
        help="Semillas concretas: 1-20, 1,2,3 o 1-5,10 (manda sobre --runs)",
    )
    sub.add_argument(
        "--policies",
        nargs="+",
        choices=list(config.POLICIES),
        default=list(config.POLICIES),
        help="Politicas a enfrentar (por defecto las dos)",
    )
    sub.add_argument(
        "--model",
        default=str(config.Q_TABLE_FILE),
        help=f"Q-table del modo {config.POLICY_QLEARNING} (por defecto {config.Q_TABLE_FILE})",
    )
    sub.add_argument(
        "--max-steps",
        type=int,
        default=config.BENCHMARK_MAX_STEPS,
        help=f"Tope de ticks por corrida (por defecto {config.BENCHMARK_MAX_STEPS})",
    )
    sub.add_argument(
        "--out",
        default=str(config.RESULTS_DIR),
        help=f"Carpeta donde escribir los CSV y el JSON (por defecto {config.RESULTS_DIR})",
    )
    sub.add_argument(
        "--no-plots",
        action="store_true",
        help="No dibuja las graficas aunque haya matplotlib",
    )


def _argumentos_de_escenario(sub: argparse.ArgumentParser) -> None:
    """Los argumentos de `scenario` (fase 10).

    `--name` y `--all` son excluyentes y hace falta uno de los dos: correr "el
    escenario por defecto" no significa nada cuando el sentido de la fase es
    justo que son cinco distintos. Que falten los dos lo comprueba `cmd_scenario`
    y no argparse, para que salga por el log y con codigo de salida como el
    resto de los errores del CLI.

    `--policy` **omitido corre las dos**, que es lo que pide la fase. Sigue
    siendo la unica variable experimental: mapa, AGVs, salidas, cola y semillas
    son identicos para las dos, y por eso la comparacion mide la politica.
    """
    cual = sub.add_mutually_exclusive_group()
    cual.add_argument(
        "--name",
        help="Escenario a correr: " + ", ".join(scenarios.LETTERS),
    )
    cual.add_argument(
        "--all",
        action="store_true",
        help="Corre los cinco escenarios seguidos",
    )
    sub.add_argument(
        "--policy",
        choices=list(config.POLICIES),
        default=None,
        help="Politica con la que correr (omitido: corre las dos y las compara)",
    )
    sub.add_argument(
        "--runs",
        type=int,
        default=config.SCENARIO_RUNS,
        help=f"Semillas por escenario (por defecto {config.SCENARIO_RUNS})",
    )
    sub.add_argument(
        "--seeds",
        type=_semillas,
        default=None,
        help="Semillas concretas: 1-20, 1,2,3 o 1-5,10 (manda sobre --runs)",
    )
    sub.add_argument(
        "--model",
        default=str(config.Q_TABLE_FILE),
        help=f"La Q-table general (por defecto {config.Q_TABLE_FILE})",
    )
    sub.add_argument(
        "--per-scenario-model",
        action="store_true",
        help=(
            "Usa la Q-table de cada escenario (python/models/q_table_<letra>.json) "
            "en vez de la general; se entrenan con train --scenario"
        ),
    )
    sub.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Tope de ticks por corrida (por defecto, el de cada escenario)",
    )
    sub.add_argument(
        "--out",
        default=str(config.RESULTS_DIR),
        help=f"Carpeta donde escribir los CSV (por defecto {config.RESULTS_DIR})",
    )
    sub.add_argument(
        "--no-summary",
        action="store_true",
        help=f"No escribe {config.SUMMARY_TABLE_FILE.name}",
    )

def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))

    if args.command is None:
        parser.print_help()
        return 1

    return HANDLERS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
