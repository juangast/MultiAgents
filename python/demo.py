"""La demostracion final: el mismo escenario con las dos politicas, en vivo.

Corre el escenario de alta congestion con la Q-table entrenada, ensena las
metricas cada 50 ticks, cambia a la baseline **sobre el mismo escenario** con el
comando `SET_MODE`, lo repite y compara los dos al final.

    python3 python/demo.py                       # C con las dos politicas
    python3 python/demo.py --scenario B          # otro escenario
    python3 python/demo.py --policy qlearning    # solo una
    python3 python/demo.py --rate 10             # al ritmo de Unity

--- Por que va por el socket ---

Podria tickear la simulacion directamente y seria mas corto, pero entonces la
demo no ensenaria el sistema que se entrega, sino un atajo por dentro. Aqui se
levanta el servidor de verdad y el propio script hace de cliente: pide
`GET_STATE`, y **el mundo avanza porque alguien pregunta**, que es el contrato
PULL entero en una linea. Unity puede conectarse al mismo puerto y mirar; cada
`GET_STATE` suyo consume su propio tick, como dice el contrato.

Con `--no-serve` corre sin socket, por si hace falta en un sitio sin red.
"""

import argparse
import json
import socket
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import config
import metrics
import protocol
import scenarios
import server
import simulation
from logs import get_logger, setup_logging

log = get_logger("demo")

# Cada cuantos ticks se imprime una linea de metricas en vivo.
CADA: int = 50


class DemoSimulation:
    """La simulacion de un escenario, servible por el socket.

    Cumple `protocol.Simulation` (`get_snapshot()` + `reset()`) y ademas
    `set_mode()`, asi que el servidor la acepta sin tocarle una linea: es el
    gancho de inyeccion de dependencia que hay desde la fase 1.

    Lo que anade por encima de `Simulation` es lo que `metrics.run_once()` hace
    en su bucle y el motor no sabe hacer solo: **repartir la cola de tareas** al
    que llega y **acumular las metricas** tick a tick. Se reutiliza el mismo
    `metrics._reparte()` que usa el runner de las fases 9 y 10 a proposito: si la
    demo repartiera las tareas por su cuenta, ensenaria numeros que no son los
    que estan en el README.
    """

    def __init__(
        self,
        spec: scenarios.ScenarioSpec,
        policy: str,
        *,
        model: str | Path | None = None,
    ) -> None:
        self.spec = spec
        self.graph = spec.graph()
        self.escenario = spec.build(spec.seed)
        self.model = model
        self._lock = threading.RLock()
        self._arranca(policy)

    def _arranca(self, policy: str) -> None:
        """Monta una corrida limpia con la politica que se le diga."""
        self.sim = simulation.Simulation(
            self.graph,
            routes=list(self.escenario.routes),
            policy=policy,
            model=self.model,
            seed=self.escenario.seed,
        )
        self.metrics = metrics.RunMetrics(
            policy=self.sim.mode,
            seed=self.escenario.seed,
            map_name=self.graph.name or "(sin nombre)",
            n_agents=self.escenario.n_agents,
            n_tasks=self.escenario.n_tasks,
        )
        self.cola: deque[str] = deque(self.escenario.pending)
        self.metrics.start(self.sim)
        # Un AGV puede arrancar ya en su destino: cierra en el paso cero y coge
        # la siguiente de la cola antes del primer tick.
        metrics._reparte(self.sim, self.cola, self.metrics)

    @property
    def mode(self) -> str:
        return self.sim.mode

    @property
    def terminado(self) -> bool:
        """True cuando se despacho la cola entera o el motor no puede seguir."""
        with self._lock:
            return not self.cola and self.sim.done

    def get_snapshot(self) -> protocol.Snapshot:
        """Un tick: avanza, anota lo que paso y reparte lo que toque.

        El orden importa y es el mismo de `metrics.run_once()`: primero se anota
        el tick y solo despues se reparte, o la llegada del AGV quedaria tapada
        por la tarea nueva y el tick de la entrega se perderia.
        """
        with self._lock:
            instantanea = self.sim.get_snapshot()
            self.metrics.observe(self.sim)
            metrics._reparte(self.sim, self.cola, self.metrics)
            return instantanea

    def reset(self) -> None:
        """Vuelve a empezar con la misma politica y el mismo escenario."""
        with self._lock:
            self._arranca(self.sim.mode)

    def set_mode(self, mode: str) -> str:
        """Cambia de politica y arranca de cero **con el mismo escenario**.

        Que el escenario no cambie es la condicion de la fase 9, y es lo unico
        que hace que comparar las dos corridas signifique algo.
        """
        with self._lock:
            self._arranca(mode)
            return self.sim.mode

    def cerrar(self) -> metrics.RunMetrics:
        """Cierra las metricas de la corrida y las devuelve."""
        with self._lock:
            self.metrics.close(self.sim)
            return self.metrics

    def linea_en_vivo(self) -> str:
        """Una linea de metricas para imprimir mientras corre.

        Las tasas se sacan de `sim.stats()` y no de `RunMetrics`: los contadores
        de conflictos y de espera los copia `close()` del motor **al terminar**,
        asi que a media corrida valdrian cero. El motor si los lleva al dia.
        """
        with self._lock:
            numeros = self.sim.stats()
            ticks = max(self.sim.step, 1)
            return (
                f"paso {self.sim.step:>5} | {self.sim.mode:<9} | "
                f"tareas {self.metrics.completed_tasks:>2}/{self.escenario.n_tasks} | "
                f"conf/tick {numeros['conflicts'] / ticks:>5.2f} | "
                f"espera/tick {numeros['total_wait_time'] / ticks:>5.2f} | "
                f"reroutes {numeros['actions']['reroute']:>5} | "
                f"desatascos {numeros['forced']:>3}"
            )


class Cliente:
    """Un cliente TCP minimo: manda una linea y lee una linea."""

    def __init__(self, host: str, port: int) -> None:
        self.sock = socket.create_connection((host, port), timeout=10.0)
        self._buffer = bytearray()

    def pide(self, comando: str) -> dict[str, Any]:
        self.sock.sendall(f"{comando}\n".encode(config.ENCODING))
        # TCP es un flujo: hay que leer hasta el salto de linea, no fiarse de
        # que un recv() traiga exactamente una respuesta.
        while b"\n" not in self._buffer:
            dato = self.sock.recv(4096)
            if not dato:
                raise ConnectionError("el servidor cerro la conexion")
            self._buffer.extend(dato)
        corte = self._buffer.index(b"\n")
        linea = bytes(self._buffer[:corte])
        del self._buffer[: corte + 1]
        return json.loads(linea.decode(config.ENCODING))

    def close(self) -> None:
        self.sock.close()


def corre_una(
    demo: DemoSimulation,
    politica: str,
    *,
    cliente: Cliente | None,
    max_steps: int,
    rate: float,
) -> metrics.RunMetrics:
    """Corre el escenario entero con una politica, contandolo cada 50 ticks."""
    if cliente is not None:
        respuesta = cliente.pide(f"{config.CMD_SET_MODE} {politica}")
        if not respuesta.get("ok"):
            raise RuntimeError(f"SET_MODE fallo: {respuesta}")
    else:
        demo.set_mode(politica)

    log.info("")
    log.info("--- %s | escenario %s: %s ---", politica.upper(), demo.spec.letter, demo.spec.name)

    espera = 1.0 / rate if rate > 0 else 0.0
    for paso in range(1, max_steps + 1):
        if cliente is not None:
            cliente.pide(config.CMD_GET_STATE)
        else:
            demo.get_snapshot()

        if paso % CADA == 0:
            log.info("%s", demo.linea_en_vivo())
        if demo.terminado:
            break
        if espera:
            time.sleep(espera)

    medidas = demo.cerrar()
    log.info("%s  <- final", demo.linea_en_vivo())
    return medidas


def build_parser() -> argparse.ArgumentParser:
    """El CLI de la demo."""
    parser = argparse.ArgumentParser(
        prog="demo.py",
        description=(
            "Demostracion final: el mismo escenario con las dos politicas, "
            "servido por el socket y comparado al final."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="C",
        choices=list(scenarios.LETTERS),
        help="Escenario a demostrar (por defecto C, alta congestion)",
    )
    parser.add_argument(
        "--policy",
        choices=list(config.POLICIES),
        default=None,
        help="Corre solo esta politica (omitido: las dos y las compara)",
    )
    parser.add_argument(
        "--model",
        default=str(config.Q_TABLE_FILE),
        help=f"Q-table a servir (por defecto {config.Q_TABLE_FILE})",
    )
    parser.add_argument(
        "--host", default=config.HOST, help=f"Donde escuchar (por defecto {config.HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=config.PORT, help=f"Puerto (por defecto {config.PORT})"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.0,
        help="Peticiones por segundo; 0 (por defecto) es tan rapido como pueda",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Tope de ticks por corrida (por defecto, el del escenario)",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Sin socket: tickea la simulacion directamente",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Nivel DEBUG")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la demo."""
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)

    modelo = Path(args.model)
    politicas = [args.policy] if args.policy else list(config.POLICIES)
    if config.POLICY_QLEARNING in politicas and not modelo.is_file():
        log.error(
            "no existe la Q-table %s; entrenala antes con: "
            "python3 python/main.py train --map warehouse",
            modelo,
        )
        return 2

    try:
        spec = scenarios.get(args.scenario)
    except KeyError as exc:
        log.error("%s", exc)
        return 2

    tope = args.max_steps if args.max_steps is not None else spec.max_steps
    demo = DemoSimulation(spec, politicas[0], model=modelo)

    log.info("=== DEMO | escenario %s: %s ===", spec.letter, spec.name)
    for linea in spec.header_lines():
        log.info("%s", linea)
    log.info("tope de ticks : %d", tope)

    servidor = None
    cliente = None
    if not args.no_serve:
        try:
            servidor = server.AGVServer((args.host, args.port), demo)
        except OSError as exc:
            log.error("no se pudo abrir %s:%d -> %s", args.host, args.port, exc)
            return 1
        threading.Thread(target=servidor.serve_forever, daemon=True).start()
        direccion = servidor.server_address
        log.info("servidor en %s:%d | Unity puede conectarse y mirar", *direccion[:2])
        cliente = Cliente(direccion[0], direccion[1])

    resultados: dict[str, list[metrics.RunMetrics]] = {}
    try:
        # `_quiet` calla los conflictos de la simulacion: con 6 AGVs son una
        # linea por tick y taparian las metricas, que es lo que hay que leer.
        with metrics.qlearning._quiet("simulation", "agent", "conflicts"):
            for politica in politicas:
                medidas = corre_una(
                    demo, politica, cliente=cliente, max_steps=tope, rate=args.rate
                )
                resultados[politica] = [medidas]
    except (ConnectionError, OSError) as exc:
        log.error("se corto la conexion con el servidor: %s", exc)
        return 1
    finally:
        if cliente is not None:
            cliente.close()
        if servidor is not None:
            servidor.shutdown()
            servidor.server_close()

    if len(resultados) > 1:
        log.info("")
        for linea in metrics.comparison_lines(resultados):
            log.info("%s", linea)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
