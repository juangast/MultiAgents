"""CLI del servidor de simulacion de AGVs."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import config
import graph
import qlearning
import server
import simulation
from config import get_logger, setup_logging

log = get_logger("main")


def cmd_serve(args: argparse.Namespace) -> int:
    """Levanta el servidor TCP para Unity con la simulacion de verdad."""
    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    try:
        simulacion = simulation.Simulation(
            grafo, args.agents, policy=args.policy, model=args.model,
            deliveries=args.deliveries,
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
    """Carga un mapa de `maps/<nombre>.json` y lo valida."""
    ruta = graph.map_path(nombre)
    if not ruta.exists():
        disponibles = sorted(p.stem for p in config.MAPS_DIR.glob("*.json"))
        log.error(
            "no existe el mapa %s (los que hay: %s)", ruta, ", ".join(disponibles) or "ninguno"
        )
        return None, "", 2

    try:
        grafo = graph.load_graph(ruta)
        grafo.validate()
    except graph.GraphError as exc:
        log.error("%s", exc)
        return None, "", 1

    return grafo, str(ruta), 0


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
    if grafo.boxes:
        log.info("cajas         : %d", len(grafo.boxes))
    for rol in sorted(grafo.roles):
        nodos = grafo.nodes_with_role(rol)
        if nodos:
            log.info("%-14s: %s", rol, ", ".join(nodos))

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
            deliveries=args.deliveries,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    codigo = _corre_simulacion(simulacion, args.steps)
    if getattr(args, "bus", False) and simulacion.manager is not None:
        log.info("--- la negociacion, mensaje a mensaje ---")
        for mensaje in simulacion.bus.history:
            log.info("%s", mensaje)
    return codigo


def _corre_simulacion(simulacion: simulation.Simulation, pasos: int) -> int:
    """Tickea hasta `pasos`, o hasta que lleguen todos, contandolo por el log.

    Devuelve 0 aunque la corrida muera en deadlock: un baseline que se atasca es
    un resultado valido, no un fallo. Solo es error que un AGV no tenga ni ruta.
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
            if simulacion.deliveries:
                log.info("AGV %s: libre en %s, a la espera de mision", agente.id, agente.current_node)
                continue
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
            graph.path_cost(grafo, agente.path),
            " -> ".join(agente.path),
        )

    log.info("--- pasos ---")
    while simulacion.step < pasos and not simulacion.done():
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
                agente.progress * 100,
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
    if numeros["deliveries"]:
        log.info(
            "cajas       : %d entregada(s) de %d",
            numeros["boxes_delivered"],
            len(simulacion.inventory),
        )
        log.info(
            "misiones    : %d abiertas en total, %d servidas, %d en la bolsa",
            numeros["missions_total"],
            numeros["delivered"],
            numeros["missions_pending"],
        )
        log.info(
            "negociacion : %d mensajes en el bus, %s",
            numeros["messages"],
            ", ".join(
                f"AGV {a.id} sirvio {a.completed}" for a in simulacion.agents
            ),
        )
    for agente in simulacion.agents:
        log.info(
            "AGV %s      : %s en %s, tramo %d/%d, %d ticks esperando%s",
            agente.id,
            agente.state,
            agente.current_node,
            agente.path_index,
            max(len(agente.path) - 1, 0),
            agente.wait_time,
            f", lleva {agente.carrying}" if agente.carrying else "",
        )

    if getattr(simulacion, "manager", None) is not None:
        log.info("--- las misiones ---")
        for mision in simulacion.manager.missions.values():
            log.info(
                "%-4s %-18s caja %-6s %-3s nivel %d -> %-3s | %-11s | AGV %s",
                mision.id, mision.flow, mision.box, mision.node, mision.level,
                mision.destination, mision.status,
                mision.agv_id if mision.agv_id else "-",
            )

    return 1 if sin_ruta else 0


def _razon_del_final(simulacion: simulation.Simulation) -> str:
    """Por que se paro la corrida, en una linea para el resumen."""
    if simulacion.finished_reason == simulation.FINISHED_DEADLOCK:
        return (
            f"deadlock, nadie avanzo en {config.DEADLOCK_TICKS} ticks seguidos "
            f"(el baseline no sabe deshacerlo: para eso esta el Q-Learning)"
        )
    if simulacion.done():
        return "llegaron todos"
    return "se acabaron los pasos antes de que llegaran todos"


def cmd_train(args: argparse.Namespace) -> int:
    """Modo TRAIN: entrena la Q-table y escribe el modelo y el CSV.

    No levanta el servidor ni habla con Unity: mil episodios son cientos de
    miles de ticks, y un socket en medio no le daria al algoritmo ni un dato mas.
    """
    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    try:
        entrenador = qlearning.train(
            grafo, _ajustes(args, grafo), model_path=args.model, log_path=args.log
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    _informa_modelo(args.model, entrenador.metadata())
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Modo EVALUATE: carga la Q-table del disco y juega greedy puro.

    `epsilon = 0` y la tabla no se toca. Corre ademas la baseline sobre los
    mismos escenarios, que es contra lo que hay que comparar.
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
        deliveries=getattr(args, "deliveries", False),
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


COMMANDS = {
    "serve": "Levanta el servidor TCP para Unity",
    "map": "Muestra el mapa logico y lo valida",
    "simulate": "Corre la simulacion sin servidor",
    "train": "Entrena los agentes con Q-Learning",
    "evaluate": "Evalua una politica ya entrenada",
}

HANDLERS = {
    "serve": cmd_serve,
    "map": cmd_map,
    "simulate": cmd_simulate,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
}


def _argumentos_de_politica(sub: argparse.ArgumentParser) -> None:
    """`--policy` y `--model`, que comparten `serve` y `simulate`.

    `--policy` es la unica variable experimental: lo demas es identico en los dos
    modos, y si no lo fuera, comparar las dos corridas no mediria la politica.
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


def _argumentos_de_entregas(sub: argparse.ArgumentParser, verbo: str) -> None:
    """`--deliveries`, que comparten los cuatro modos que corren la simulacion."""
    sub.add_argument(
        "--deliveries",
        action="store_true",
        help=(
            f"{verbo} con entregas: cada AGV recoge una caja y la lleva a un "
            "muelle. Pide un mapa con 'boxes' y con nodos de rol 'dock'"
        ),
    )


def _argumentos_de_aprendizaje(sub: argparse.ArgumentParser, name: str) -> None:
    """Los argumentos de `train` y `evaluate`. Los defaults salen de `config.py`."""
    verbo = "Entrena" if name == "train" else "Evalua"
    sub.add_argument(
        "--map",
        default=config.DEFAULT_MAP,
        help=f"Mapa sobre el que correr (por defecto {config.DEFAULT_MAP})",
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
        help=("Donde se escribe la Q-table" if name == "train" else "Q-table a cargar")
        + f" (por defecto {config.Q_TABLE_FILE})",
    )
    _argumentos_de_entregas(sub, verbo)

    if name == "train":
        sub.add_argument("--episodes", type=int, default=config.EPISODES, help=f"Cuantos episodios entrenar (por defecto {config.EPISODES})")
        sub.add_argument("--alpha", type=float, default=config.ALPHA, help=f"Tasa de aprendizaje (por defecto {config.ALPHA})")
        sub.add_argument("--gamma", type=float, default=config.GAMMA, help=f"Descuento del futuro (por defecto {config.GAMMA})")
        sub.add_argument("--epsilon-start", type=float, default=config.EPSILON_START, help=f"Exploracion inicial (por defecto {config.EPSILON_START})")
        sub.add_argument("--epsilon-end", type=float, default=config.EPSILON_END, help=f"Suelo de la exploracion (por defecto {config.EPSILON_END})")
        sub.add_argument("--epsilon-decay", type=float, default=config.EPSILON_DECAY, help=f"Factor exponencial por episodio (por defecto {config.EPSILON_DECAY})")
        sub.add_argument("--no-reroute", action="store_true", help="Deja fuera la accion REROUTE: solo ADVANCE y WAIT")
        sub.add_argument("--log", default=str(config.TRAINING_LOG_FILE), help=f"CSV con una fila por episodio (por defecto {config.TRAINING_LOG_FILE})")
    else:
        sub.add_argument("--episodes", type=int, default=100, help="Cuantos episodios evaluar (por defecto 100)")
        sub.add_argument("--log", default=None, help="CSV opcional con los episodios de la evaluacion")


def build_parser() -> argparse.ArgumentParser:
    """Arma el parser con los cinco subcomandos."""
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
            sub.add_argument("--map", default=config.DEFAULT_MAP, help=f"Mapa a servir (por defecto {config.DEFAULT_MAP})")
            sub.add_argument("--host", default=config.HOST, help=f"Direccion donde escuchar (por defecto {config.HOST})")
            sub.add_argument("--port", type=int, default=config.PORT, help=f"Puerto donde escuchar (por defecto {config.PORT})")
            sub.add_argument("--agents", type=int, default=1, help="Cuantos AGVs servir (por defecto 1)")
            _argumentos_de_politica(sub)
            _argumentos_de_entregas(sub, "Sirve")

        elif name == "map":
            sub.add_argument("--name", default=config.DEFAULT_MAP, help=f"Mapa a mostrar (por defecto {config.DEFAULT_MAP})")

        elif name == "simulate":
            sub.add_argument("--map", default=config.DEFAULT_MAP, help=f"Mapa por el que moverse (por defecto {config.DEFAULT_MAP})")
            sub.add_argument("--agents", type=int, default=1, help="Cuantos AGVs correr (por defecto 1)")
            sub.add_argument("--steps", type=int, default=100, help="Tope de pasos, corta antes si llegan todos (por defecto 100)")
            sub.add_argument("--from", dest="origen", default=None, help="Nodo de salida del primer AGV")
            sub.add_argument("--to", dest="destino", default=None, help="Nodo de destino del primer AGV")
            _argumentos_de_politica(sub)
            _argumentos_de_entregas(sub, "Corre")
            sub.add_argument(
                "--bus",
                action="store_true",
                help="Escupe la negociacion entera: quien publico, quien pujo y quien gano",
            )

        else:
            _argumentos_de_aprendizaje(sub, name)

    return parser


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
