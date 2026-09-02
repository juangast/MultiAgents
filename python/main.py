"""CLI del servidor de simulacion de AGVs."""

import argparse
from collections.abc import Sequence

import config
import graph
import server
from logs import get_logger, setup_logging

log = get_logger("main")


def cmd_serve(args: argparse.Namespace) -> int:
    """Levanta el servidor TCP para Unity."""
    simulacion = server.FakeSimulation()
    return server.serve_forever(simulacion, host=args.host, port=args.port)


def cmd_map(args: argparse.Namespace) -> int:
    """Muestra el mapa logico y lo valida."""
    ruta = graph.map_path(args.name)
    constructor = graph.BUILTIN_MAPS.get(args.name)

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
                args.name,
                ruta,
                ", ".join(sorted(graph.BUILTIN_MAPS)),
            )
            return 2
    except graph.GraphError as exc:
        log.error("%s", exc)
        return 1

    # Se valida antes de informar: sin un grafo valido la tabla de coordenadas
    # no se puede ni construir (un nodo puede no tener posicion), y el error de
    # validate() ya dice todo lo que le pasa al mapa.
    try:
        grafo.validate()
    except graph.GraphError as exc:
        log.error("validate(): %s", exc)
        return 1

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
    """Corre la simulacion sin servidor."""
    log.warning("simulate: no implementado")
    return 0


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
        elif name == "map":
            sub.add_argument(
                "--name",
                default=config.DEFAULT_MAP,
                help=(
                    f"Mapa a mostrar, de python/maps/ "
                    f"(por defecto {config.DEFAULT_MAP})"
                ),
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
