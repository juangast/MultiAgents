"""Cliente falso que hace de Unity mientras Unity no existe.

Se conecta al servidor, pide GET_STATE a un ritmo fijo durante N segundos,
muestra lo que recibe y valida que cada respuesta sea JSON con la forma del
contrato. Al final imprime un resumen y sale con codigo 1 si algo fallo, para
poder comprobar los criterios de aceptacion sin leer el log a ojo.

    python3 tests/fake_unity_client.py --seconds 60 --rate 10

No lo recoge `unittest discover`: el patron de descubrimiento es test*.py.
"""

import argparse
import json
import socket
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Se ejecuta directo, asi que el __init__.py del paquete no corre y hay que
# meter python/ en sys.path a mano, igual que hace tests/__init__.py.
_PYTHON_DIR = Path(__file__).resolve().parent.parent / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import config  # noqa: E402
from logs import get_logger, setup_logging  # noqa: E402

RECV_SIZE: int = 4096
CAMPOS_AGENTE: tuple[str, ...] = ("id", "x", "y", "z", "rotation", "state")


class LineReader:
    """Lee lineas completas de un socket.

    Con su propio buffer: un recv puede traer media respuesta, una entera o
    varias pegadas, igual que del lado del servidor.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = bytearray()

    def read_line(self) -> str:
        """Devuelve la siguiente linea, sin el salto final."""
        while True:
            corte = self._buffer.find(b"\n")
            if corte >= 0:
                linea = bytes(self._buffer[:corte])
                del self._buffer[: corte + 1]
                return linea.decode(config.ENCODING, errors="replace")

            trozo = self._sock.recv(RECV_SIZE)
            if not trozo:
                raise ConnectionError("el servidor cerro la conexion")
            self._buffer.extend(trozo)


def validar_snapshot(payload: Any) -> str:
    """Comprueba la forma del contrato. Devuelve "" si esta bien, o el motivo."""
    if not isinstance(payload, dict):
        return "la respuesta no es un objeto JSON"
    if not isinstance(payload.get("step"), int):
        return "falta el campo 'step' entero"

    agentes = payload.get("agents")
    if not isinstance(agentes, list):
        return "falta el campo 'agents' como lista"

    for indice, agente in enumerate(agentes):
        if not isinstance(agente, dict):
            return f"el agente {indice} no es un objeto"
        faltan = [campo for campo in CAMPOS_AGENTE if campo not in agente]
        if faltan:
            return f"al agente {indice} le faltan campos: {', '.join(faltan)}"
    return ""


class Resumen:
    """Acumula el resultado de la corrida."""

    def __init__(self) -> None:
        self.enviadas = 0
        self.ok = 0
        self.errores_json = 0
        self.errores_forma = 0
        self.errores_red = 0
        self.latencias: list[float] = []
        self.primer_step: int | None = None
        self.ultimo_step: int | None = None
        self.pasos_no_crecientes = 0

    @property
    def hubo_fallos(self) -> bool:
        return bool(
            self.errores_json
            or self.errores_forma
            or self.errores_red
            or self.pasos_no_crecientes
            or self.ok != self.enviadas
        )

    def percentil(self, fraccion: float) -> float:
        """Percentil sencillo sobre las latencias medidas, en ms."""
        if not self.latencias:
            return 0.0
        ordenadas = sorted(self.latencias)
        indice = min(len(ordenadas) - 1, int(fraccion * len(ordenadas)))
        return ordenadas[indice]

    def registrar_step(self, step: int) -> None:
        """Anota el paso recibido y vigila que sea estrictamente creciente."""
        if self.primer_step is None:
            self.primer_step = step
        elif self.ultimo_step is not None and step <= self.ultimo_step:
            self.pasos_no_crecientes += 1
        self.ultimo_step = step


def correr(host: str, port: int, segundos: float, rate: float, log: Any) -> Resumen:
    """Manda GET_STATE al ritmo pedido y valida cada respuesta."""
    resumen = Resumen()
    total = max(1, round(segundos * rate))
    periodo = 1.0 / rate
    peticion = (config.CMD_GET_STATE + "\n").encode(config.ENCODING)

    log.info("conectando a %s:%d", host, port)
    with socket.create_connection((host, port), timeout=5.0) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        lector = LineReader(sock)
        log.info(
            "conectado. %d peticiones a %.1f/s durante %.1fs",
            total,
            rate,
            segundos,
        )

        inicio = time.monotonic()
        for numero in range(1, total + 1):
            espera = (inicio + (numero - 1) * periodo) - time.monotonic()
            if espera > 0:
                time.sleep(espera)

            marca = time.perf_counter()
            try:
                sock.sendall(peticion)
                linea = lector.read_line()
            except (OSError, ConnectionError) as exc:
                resumen.errores_red += 1
                resumen.enviadas += 1
                log.error("peticion %d fallo: %s", numero, exc)
                break
            latencia_ms = (time.perf_counter() - marca) * 1000.0

            resumen.enviadas += 1
            resumen.latencias.append(latencia_ms)
            log.debug("<- %s  (%.2f ms)", linea, latencia_ms)

            try:
                payload = json.loads(linea)
            except json.JSONDecodeError as exc:
                resumen.errores_json += 1
                log.error("peticion %d, respuesta no es JSON (%s): %r", numero, exc, linea)
                continue

            motivo = validar_snapshot(payload)
            if motivo:
                resumen.errores_forma += 1
                log.error("peticion %d, forma invalida: %s -> %r", numero, motivo, linea)
                continue

            resumen.ok += 1
            resumen.registrar_step(payload["step"])

            if numero % max(1, round(rate)) == 0:
                log.info(
                    "%4d/%d ok=%d  %.2f ms  <- %s",
                    numero,
                    total,
                    resumen.ok,
                    latencia_ms,
                    linea,
                )

    return resumen


def informar(resumen: Resumen, log: Any) -> None:
    """Escupe el resumen final de la corrida."""
    log.info("--- resumen ---")
    log.info("enviadas          : %d", resumen.enviadas)
    log.info("respuestas ok     : %d", resumen.ok)
    log.info("errores de JSON   : %d", resumen.errores_json)
    log.info("errores de forma  : %d", resumen.errores_forma)
    log.info("errores de red    : %d", resumen.errores_red)
    log.info("steps no crecientes: %d", resumen.pasos_no_crecientes)
    log.info("step primero/ultimo: %s / %s", resumen.primer_step, resumen.ultimo_step)

    if resumen.latencias:
        media = sum(resumen.latencias) / len(resumen.latencias)
        log.info(
            "latencia ms min/media/p95/max: %.2f / %.2f / %.2f / %.2f",
            min(resumen.latencias),
            media,
            resumen.percentil(0.95),
            max(resumen.latencias),
        )

    log.info("resultado: %s", "FALLO" if resumen.hubo_fallos else "OK")


def build_parser() -> argparse.ArgumentParser:
    """Arma el parser del cliente falso."""
    parser = argparse.ArgumentParser(
        prog="fake_unity_client.py",
        description="Cliente de prueba que imita a Unity contra el servidor PULL.",
    )
    parser.add_argument("--host", default=config.HOST, help="Host del servidor")
    parser.add_argument("--port", type=int, default=config.PORT, help="Puerto del servidor")
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="Cuantos segundos pedir estado"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=float(config.TICK_RATE),
        help=f"Peticiones por segundo (por defecto {config.TICK_RATE})",
    )
    parser.add_argument("--label", default="", help="Etiqueta para distinguir varios clientes")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Muestra todas las respuestas recibidas"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Punto de entrada del cliente falso."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    nombre = f"unity:{args.label}" if args.label else "unity"
    log = get_logger(nombre)

    if args.rate <= 0 or args.seconds <= 0:
        log.error("--rate y --seconds tienen que ser mayores que cero")
        return 2

    try:
        resumen = correr(args.host, args.port, args.seconds, args.rate, log)
    except OSError as exc:
        log.error("no se pudo conectar a %s:%d -> %s", args.host, args.port, exc)
        return 1
    except KeyboardInterrupt:
        log.warning("interrumpido por el usuario")
        return 1

    informar(resumen, log)
    return 1 if resumen.hubo_fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
