"""CLI del servidor de simulacion de AGVs."""

import argparse
from collections.abc import Sequence

import astar
import config
import graph
import server
import simulation
from logs import get_logger, setup_logging

log = get_logger("main")


def cmd_serve(args: argparse.Namespace) -> int:
    """Levanta el servidor TCP para Unity con la simulacion de verdad."""
    grafo, _origen, codigo = _abre_mapa(args.map)
    if grafo is None:
        return codigo

    try:
        simulacion = simulation.Simulation(grafo, args.agents)
    except ValueError as exc:
        log.error("%s", exc)
        return 2

    log.info(
        "sirviendo el mapa %s con %d agente(s) y politica %s",
        grafo.name or "(sin nombre)",
        len(simulacion.agents),
        simulacion.policy.name,
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
            grafo, args.agents, origin=origen, target=destino
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
        "--- simulacion: mapa %s, %d agente(s), politica %s, %d pasos como mucho ---",
        grafo.name or "(sin nombre)",
        len(simulacion.agents),
        simulacion.policy.name,
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
            log.info(
                "paso %3d | AGV %s | %-7s | %-4s -> %-4s | %3.0f%% | tramo %d/%d "
                "| espera %3d | tarea %s",
                simulacion.step,
                agente.id,
                agente.state,
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
    """Entrena los agentes con Q-Learning."""
    log.warning("train: no implementado")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evalua una politica ya entrenada."""
    log.warning("evaluate: no implementado")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Mide el rendimiento de la simulacion."""
    log.warning("benchmark: no implementado")
    return 0


COMMANDS: dict[str, str] = {
    "serve": "Levanta el servidor TCP para Unity",
    "map": "Muestra el mapa logico y lo valida",
    "simulate": "Corre la simulacion sin servidor",
    "train": "Entrena los agentes con Q-Learning",
    "evaluate": "Evalua una politica ya entrenada",
    "benchmark": "Mide el rendimiento de la simulacion",
}

HANDLERS = {
    "serve": cmd_serve,
    "map": cmd_map,
    "simulate": cmd_simulate,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "benchmark": cmd_benchmark,
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
