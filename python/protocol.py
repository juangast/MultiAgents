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
# Los tres finales posibles de un SET_MODE que no sale bien.
ERROR_BAD_MODE: str = "bad_mode"                    # ese modo no existe
ERROR_MODE_NOT_SUPPORTED: str = "mode_not_supported"  # esta simulacion no cambia de modo
ERROR_SET_MODE_FAILED: str = "set_mode_failed"      # existe, pero no se pudo montar

Snapshot = dict[str, Any]


@runtime_checkable
class Simulation(Protocol):
    """Lo que el servidor necesita de una simulacion para poder atenderla.

    Es el contrato de la inyeccion de dependencia: el servidor atiende a
    cualquiera que lo cumpla. Desde la fase 3 se le pasa una
    `simulation.Simulation`, sin tocar ni el servidor ni este modulo.
    """

    def get_snapshot(self) -> Snapshot:
        """Devuelve el estado completo de la simulacion en este momento."""
        ...

    def reset(self) -> None:
        """Deja la simulacion en su estado inicial."""
        ...

    # `set_mode` NO va en el Protocol a proposito: es **opcional**. Una
    # simulacion que no sepa cambiar de politica tiene que poder servirse igual,
    # y `SET_MODE` le contesta `mode_not_supported` en vez de reventar el hilo
    # del cliente. Es la misma regla que con un comando desconocido: se responde
    # y se sigue.


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

    if command == config.CMD_SET_MODE:
        return encode_line(set_mode_payload(simulation, _args))

    return encode_line(unknown_command_payload(command))


def set_mode_payload(simulation: Simulation, args: list[str]) -> dict[str, Any]:
    """Cambia la politica en caliente y contesta como fue.

    Es el unico comando con argumento: `SET_MODE baseline` o `SET_MODE
    qlearning`. Sirve para cambiar de politica en mitad de una demo sin
    reiniciar el servidor, y **arranca una corrida limpia**: media corrida con
    una politica y media con otra no es una corrida de ninguna de las dos.

    Ninguno de los cuatro finales lanza. Un comando que tumba el hilo del
    cliente rompe el contrato de "una linea entra, una linea sale", y de eso no
    se salva ni el que viene mal escrito.
    """
    modo = args[0].lower() if args else ""
    if modo not in config.POLICIES:
        return {
            "error": ERROR_BAD_MODE,
            "command": config.CMD_SET_MODE,
            "mode": modo,
            "modes": list(config.POLICIES),
        }

    cambiar = getattr(simulation, "set_mode", None)
    if not callable(cambiar):
        return {"error": ERROR_MODE_NOT_SUPPORTED, "command": config.CMD_SET_MODE}

    try:
        activo = cambiar(modo)
    except ValueError as exc:
        # El caso tipico: `qlearning` sin Q-table entrenada en el disco.
        return {
            "error": ERROR_SET_MODE_FAILED,
            "command": config.CMD_SET_MODE,
            "mode": modo,
            "detail": str(exc),
        }

    respuesta: dict[str, Any] = {"ok": True, "mode": activo or modo}
    corrida = getattr(simulation, "run", None)
    if isinstance(corrida, int):
        respuesta["run"] = corrida
    return respuesta
