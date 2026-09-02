"""Contrato de la comunicacion PULL con Unity: comandos, formato y despacho.

Modulo puro: no abre sockets ni guarda estado. `server.py` se encarga del
transporte y llama aqui para traducir cada linea recibida en su respuesta.

Regla del contrato: una linea entra, una linea sale. Siempre. Encoding utf-8,
delimitador "\\n", y el JSON de respuesta nunca lleva saltos de linea internos.
"""

import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import config

# La altura la aplica Unity con el prefab, por eso la coordenada vertical del
# contrato es 0.0 y no config.AGV_HEIGHT. Python solo manda el plano del suelo.
UNITY_Y: float = 0.0

OK_PAYLOAD: dict[str, Any] = {"ok": True}
ERROR_UNKNOWN_COMMAND: str = "unknown_command"

Snapshot = dict[str, Any]


@runtime_checkable
class Simulation(Protocol):
    """Lo que el servidor necesita de una simulacion para poder atenderla.

    Es el contrato de la inyeccion de dependencia: en la fase 1 se le pasa un
    `server.FakeSimulation` y mas adelante la simulacion de verdad, sin tocar
    ni el servidor ni este modulo.
    """

    def get_snapshot(self) -> Snapshot:
        """Devuelve el estado completo de la simulacion en este momento."""
        ...

    def reset(self) -> None:
        """Deja la simulacion en su estado inicial."""
        ...


def to_unity(px: float, py: float) -> tuple[float, float, float]:
    """Pasa una posicion (px, py) del plano de la simulacion a coordenadas de Unity.

    Unity usa Y como eje vertical, asi que el segundo eje del plano va a Z.
    Esta es la unica conversion del proyecto: no la repitas en otro sitio.
    """
    return (px * config.UNITY_SCALE, UNITY_Y, py * config.UNITY_SCALE)


def encode_line(payload: Mapping[str, Any]) -> str:
    """Serializa un payload como una sola linea JSON terminada en salto de linea."""
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n"


def encode_snapshot(snapshot: Snapshot) -> str:
    """Serializa el snapshot en el formato que espera Unity."""
    return encode_line(snapshot)


def parse_command(line: str) -> tuple[str, list[str]]:
    """Parte una linea del cliente en (comando, argumentos).

    El comando vuelve en mayusculas y sin espacios sobrantes; el `strip()`
    tambien se come el "\\r" de los clientes que mandan CRLF. Una linea vacia
    devuelve ("", []).
    """
    partes = line.strip().split()
    if not partes:
        return "", []
    return partes[0].upper(), partes[1:]


def unknown_command_payload(command: str) -> dict[str, Any]:
    """Respuesta para un comando que el servidor no conoce."""
    return {"error": ERROR_UNKNOWN_COMMAND, "command": command}


def handle_line(line: str, simulation: Simulation) -> str:
    """Traduce una linea del cliente en la linea de respuesta ya serializada.

    Devuelve siempre exactamente una linea, incluso si el comando es
    desconocido o la linea venia vacia: asi el cliente nunca pierde el
    emparejamiento entre lo que pide y lo que recibe.
    """
    command, _args = parse_command(line)

    if command == config.CMD_GET_STATE:
        return encode_snapshot(simulation.get_snapshot())

    if command == config.CMD_RESET:
        simulation.reset()
        return encode_line(OK_PAYLOAD)

    if command == config.CMD_PING:
        return encode_line(OK_PAYLOAD)

    return encode_line(unknown_command_payload(command))
