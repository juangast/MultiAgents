"""El enlace con Unity: servidor HTTP con JSON.

Cuatro rutas y nada mas. `GET /state` mira sin tocar el reloj y `POST /step`
avanza un paso: en HTTP el GET no puede tener efectos, y de paso eso arregla que
dos clientes ya no se roben los ticks el uno al otro.

La simulacion entra por inyeccion de dependencia: aqui no hay ni una linea de
logica del almacen.
"""

import json
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import config
from config import get_logger

log = get_logger("server")

MAX_BODY_BYTES: int = 64 * 1024

Snapshot = dict[str, Any]


class Simulation(Protocol):
    """Lo que el servidor necesita de una simulacion para poder atenderla.

    `set_mode` no va aqui a proposito: es opcional, y una simulacion que no sepa
    cambiar de politica tiene que poder servirse igual.
    """

    def snapshot(self) -> Snapshot:
        ...

    def get_snapshot(self) -> Snapshot:
        ...

    def reset(self) -> None:
        ...


class AGVRequestHandler(BaseHTTPRequestHandler):
    """Atiende una peticion. Todas las respuestas son JSON."""

    server: "AGVServer"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        """`GET /state` mira el almacen; `GET /health` comprueba que vive."""
        ruta = self.path.split("?")[0].rstrip("/") or "/"

        if ruta == "/state":
            self._responder(200, self.server.simulation.snapshot())
        elif ruta in ("/health", "/ping"):
            self._responder(200, {"ok": True})
        else:
            self._responder(404, self._desconocida(ruta))

    def do_POST(self) -> None:
        """`POST /step` avanza un paso, `/reset` reinicia y `/mode` cambia de politica."""
        ruta = self.path.split("?")[0].rstrip("/") or "/"
        cuerpo = self._lee_cuerpo()
        if cuerpo is None:
            return

        if ruta == "/step":
            self._responder(200, self.server.simulation.get_snapshot())
        elif ruta == "/reset":
            self.server.simulation.reset()
            self._responder(200, {"ok": True})
        elif ruta == "/mode":
            codigo, payload = set_mode_payload(self.server.simulation, cuerpo)
            self._responder(codigo, payload)
        else:
            self._responder(404, self._desconocida(ruta))

    def _lee_cuerpo(self) -> dict[str, Any] | None:
        """El JSON del cuerpo, o `{}` si venia vacio. None si ya se contesto un error."""
        try:
            largo = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._responder(400, {"error": "bad_content_length"})
            return None

        if largo <= 0:
            return {}
        if largo > MAX_BODY_BYTES:
            self._responder(413, {"error": "body_too_large", "max_bytes": MAX_BODY_BYTES})
            return None

        crudo = self.rfile.read(largo)
        try:
            cuerpo = json.loads(crudo.decode(config.ENCODING))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._responder(400, {"error": "bad_json", "detail": str(exc)})
            return None

        if not isinstance(cuerpo, dict):
            self._responder(400, {"error": "body_must_be_object"})
            return None
        return cuerpo

    def _responder(self, codigo: int, payload: dict[str, Any]) -> None:
        """Una respuesta JSON, con su Content-Length para que valga keep-alive."""
        cuerpo = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode(
            config.ENCODING
        )
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _desconocida(self, ruta: str) -> dict[str, Any]:
        """Respuesta para una ruta que no existe, con la lista de las que si."""
        return {"error": "unknown_route", "path": ruta, "routes": sorted(ROUTES)}

    def log_message(self, formato: str, *args: Any) -> None:
        """Manda el log de acceso al logging del proyecto, no a stderr en crudo."""
        log.debug("%s - %s", self.address_string(), formato % args)


ROUTES: dict[str, str] = {
    "GET /state": "El estado del almacen, sin avanzar el reloj",
    "GET /health": "Comprueba que el servidor vive",
    "POST /step": "Avanza un paso y devuelve el estado",
    "POST /reset": "Reinicia la corrida",
    "POST /mode": "Cambia de politica: {\"mode\": \"baseline\"|\"qlearning\"}",
}

ERROR_BAD_MODE: str = "bad_mode"
ERROR_MODE_NOT_SUPPORTED: str = "mode_not_supported"
ERROR_SET_MODE_FAILED: str = "set_mode_failed"


def set_mode_payload(
    simulation: Simulation, body: dict[str, Any]
) -> tuple[int, dict[str, Any]]:
    """Cambia la politica en caliente. Devuelve (codigo HTTP, payload).

    Arranca una corrida limpia: media corrida con una politica y media con otra
    no es una corrida de ninguna de las dos.
    """
    modo = str(body.get("mode", "")).strip().lower()
    if modo not in config.POLICIES:
        return 400, {
            "error": ERROR_BAD_MODE,
            "mode": modo,
            "modes": list(config.POLICIES),
        }

    cambiar = getattr(simulation, "set_mode", None)
    if not callable(cambiar):
        return 501, {"error": ERROR_MODE_NOT_SUPPORTED}

    try:
        activo = cambiar(modo)
    except ValueError as exc:
        return 409, {"error": ERROR_SET_MODE_FAILED, "mode": modo, "detail": str(exc)}

    respuesta: dict[str, Any] = {"ok": True, "mode": activo or modo}
    corrida = getattr(simulation, "run", None)
    if isinstance(corrida, int):
        respuesta["run"] = corrida
    return 200, respuesta


class AGVServer(ThreadingHTTPServer):
    """Servidor HTTP con un hilo por peticion y la simulacion inyectada."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        simulation: Simulation,
        bind_and_activate: bool = True,
    ) -> None:
        self.simulation = simulation
        super().__init__(server_address, AGVRequestHandler, bind_and_activate)

    def server_bind(self) -> None:
        """Abre el socket sin resolver el nombre de la maquina.

        `HTTPServer.server_bind()` llama a `getfqdn()`, que se va a DNS y puede
        tardar segundos en arrancar. El nombre solo lo usa la cabecera `Server`,
        asi que no vale lo que cuesta.
        """
        super(type(self).__mro__[1], self).server_bind()
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Registra el fallo de un cliente en vez de escupir el traceback."""
        log.exception("fallo atendiendo a %s", client_address)


class _Parada(Exception):
    """Peticion de parada ordenada que llego por una senal."""


def _instalar_parada_por_senal() -> None:
    """Hace que SIGTERM cierre tan limpio como Ctrl+C."""

    def parar(signum: int, _frame: Any) -> None:
        raise _Parada(signal.Signals(signum).name)

    try:
        signal.signal(signal.SIGTERM, parar)
    except ValueError:
        log.debug("no se pudo instalar el manejador de SIGTERM")


def serve_forever(
    simulation: Simulation,
    host: str = config.HOST,
    port: int = config.PORT,
) -> int:
    """Levanta el servidor y atiende hasta Ctrl+C. Devuelve el codigo de salida."""
    try:
        servidor = AGVServer((host, port), simulation)
    except OSError as exc:
        log.error("no se pudo abrir %s:%d -> %s", host, port, exc)
        return 1

    direccion = f"http://{host}:{servidor.server_address[1]}"
    log.info("servidor escuchando en %s", direccion)
    for ruta, que_hace in ROUTES.items():
        log.info("  %-13s %s", ruta, que_hace)

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
