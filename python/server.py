"""Servidor TCP que atiende las peticiones PULL de Unity.

Solo transporte: acepta conexiones, arma las lineas que llegan partidas y
delega en `protocol.handle_line`. La simulacion entra por inyeccion de
dependencia, asi que cambiar la falsa por la de verdad no toca este modulo.
"""

import signal
import socket
import socketserver
import threading
from typing import Any

import config
import protocol
from logs import get_logger

log = get_logger("server")

RECV_SIZE: int = 4096
MAX_LINE_BYTES: int = 64 * 1024

FAKE_AGENT_ID: int = 1
FAKE_STEP_SIZE: float = 0.25
FAKE_PATH_LENGTH: float = 10.0
FAKE_ROTATION: float = 0.0
FAKE_STATE: str = "moving"


class FakeSimulation:
    """Simulacion falsa de la fase 1: un AGV que avanza en linea recta sobre +X.

    Sirve para probar la comunicacion sin simulacion real. La posicion es una
    funcion del paso, no un acumulado, asi no hay deriva de coma flotante y
    `reset()` solo tiene que poner el contador a cero.

    Es segura entre hilos: el servidor la comparte entre todos los clientes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._step = 0

    @property
    def step(self) -> int:
        """Numero del ultimo paso entregado."""
        with self._lock:
            return self._step

    def get_snapshot(self) -> protocol.Snapshot:
        """Avanza un paso y devuelve el estado completo en coordenadas de Unity."""
        with self._lock:
            self._step += 1
            paso = self._step

        px = (paso * FAKE_STEP_SIZE) % FAKE_PATH_LENGTH
        x, y, z = protocol.to_unity(px, 0.0)
        agente: dict[str, Any] = {
            "id": FAKE_AGENT_ID,
            "x": x,
            "y": y,
            "z": z,
            "rotation": FAKE_ROTATION,
            "state": FAKE_STATE,
        }
        return {"step": paso, "agents": [agente]}

    def reset(self) -> None:
        """Vuelve al paso cero y al origen."""
        with self._lock:
            self._step = 0
        log.info("simulacion reiniciada")


class AGVRequestHandler(socketserver.BaseRequestHandler):
    """Atiende una conexion: lee lineas con buffer propio y contesta una por una."""

    server: "AGVServer"

    def setup(self) -> None:
        """Ajusta el socket para un ida y vuelta corto y frecuente."""
        try:
            # Sin esto, Nagle junta paquetes y le mete decenas de ms a un
            # protocolo de peticion/respuesta como este.
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError as exc:
            log.debug("no se pudieron ajustar las opciones del socket: %s", exc)

    def handle(self) -> None:
        """Bucle de lectura de la conexion. Nunca deja escapar un error de socket."""
        cliente = _formatea(self.client_address)
        log.info("cliente conectado: %s", cliente)
        buffer = bytearray()

        try:
            while True:
                trozo = self.request.recv(RECV_SIZE)
                if not trozo:
                    break  # el cliente cerro por su lado

                buffer.extend(trozo)

                if b"\n" not in buffer and len(buffer) > MAX_LINE_BYTES:
                    log.warning(
                        "linea sin terminar de %s (%d bytes), cierro la conexion",
                        cliente,
                        len(buffer),
                    )
                    break

                # Un recv puede traer media linea, una, o varias pegadas.
                while True:
                    corte = buffer.find(b"\n")
                    if corte < 0:
                        break
                    linea = bytes(buffer[:corte])
                    del buffer[: corte + 1]
                    self._responder(linea)
        except (ConnectionResetError, BrokenPipeError, TimeoutError) as exc:
            log.info("conexion con %s cortada: %s", cliente, exc)
        except OSError as exc:
            log.warning("error de socket con %s: %s", cliente, exc)
        finally:
            log.info("cliente desconectado: %s", cliente)

    def _responder(self, linea: bytes) -> None:
        """Contesta una linea completa. Los bytes invalidos acaban en unknown_command."""
        texto = linea.decode(config.ENCODING, errors="replace")
        respuesta = protocol.handle_line(texto, self.server.simulation)
        log.debug("%s -> %s", texto.strip(), respuesta.strip())
        self.request.sendall(respuesta.encode(config.ENCODING))


class AGVServer(socketserver.ThreadingTCPServer):
    """Servidor TCP con un hilo por cliente y la simulacion inyectada."""

    allow_reuse_address = True
    daemon_threads = True  # sin esto Ctrl+C se queda esperando a los clientes vivos

    def __init__(
        self,
        server_address: tuple[str, int],
        simulation: protocol.Simulation,
        bind_and_activate: bool = True,
    ) -> None:
        self.simulation = simulation
        super().__init__(server_address, AGVRequestHandler, bind_and_activate)

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Registra el fallo de un cliente en vez de escupir el traceback a stdout."""
        log.exception("fallo atendiendo a %s", _formatea(client_address))


class _Parada(Exception):
    """Peticion de parada ordenada que llego por una senal."""


def _instalar_parada_por_senal() -> None:
    """Hace que SIGTERM cierre tan limpio como Ctrl+C.

    SIGINT ya levanta KeyboardInterrupt por su cuenta, pero SIGTERM (un `kill`
    normal, o el que manda un gestor de servicios) mataria el proceso de golpe,
    sin cerrar el socket ni dejar rastro en el log.
    """

    def parar(signum: int, _frame: Any) -> None:
        raise _Parada(signal.Signals(signum).name)

    try:
        signal.signal(signal.SIGTERM, parar)
    except ValueError:
        # signal.signal solo se puede usar desde el hilo principal.
        log.debug("no se pudo instalar el manejador de SIGTERM")


def _formatea(client_address: Any) -> str:
    """Deja la direccion del cliente como host:puerto."""
    if isinstance(client_address, tuple) and len(client_address) >= 2:
        return f"{client_address[0]}:{client_address[1]}"
    return str(client_address)


def serve_forever(
    simulation: protocol.Simulation,
    host: str = config.HOST,
    port: int = config.PORT,
) -> int:
    """Levanta el servidor y atiende hasta Ctrl+C. Devuelve el codigo de salida."""
    try:
        servidor = AGVServer((host, port), simulation)
    except OSError as exc:
        log.error("no se pudo abrir %s:%d -> %s", host, port, exc)
        return 1

    escucha = _formatea(servidor.server_address)
    log.info("servidor escuchando en %s", escucha)
    log.info("comandos: %s, %s, %s", config.CMD_GET_STATE, config.CMD_RESET, config.CMD_PING)

    _instalar_parada_por_senal()

    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        log.info("Ctrl+C recibido, cerrando el servidor")
    except _Parada as senal:
        log.info("%s recibido, cerrando el servidor", senal)
    finally:
        servidor.server_close()
        log.info("servidor cerrado")

    return 0
