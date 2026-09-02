"""Configuracion del logging del proyecto."""

import logging
import sys

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
