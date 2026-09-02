"""CLI del servidor de simulacion de AGVs."""

import argparse
from collections.abc import Sequence

import config
import server
from logs import get_logger, setup_logging

log = get_logger("main")


def cmd_serve(args: argparse.Namespace) -> int:
    """Levanta el servidor TCP para Unity."""
    simulacion = server.FakeSimulation()
    return server.serve_forever(simulacion, host=args.host, port=args.port)


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
    "simulate": "Corre la simulacion sin servidor",
    "train": "Entrena los agentes con Q-Learning",
    "evaluate": "Evalua una politica ya entrenada",
    "benchmark": "Mide el rendimiento de la simulacion",
}

HANDLERS = {
    "serve": cmd_serve,
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
