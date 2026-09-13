"""Obstaculos que aparecen en sitio libre y estorban de verdad."""

import heapq
import random
from collections.abc import Iterable, Sequence
from typing import Any

import config
from config import get_logger
from graph import ROLE_TRANSIT, WarehouseGraph, to_unity

log = get_logger("obstacles")

OBSTACLE_IDS: tuple[str, ...] = ("Obstaculo_1", "Obstaculo_2", "Obstaculo_3")

OBSTACLE_ROLE: str = ROLE_TRANSIT

# Tope de cuanto puede alargarse el rodeo. Medido sobre 12 corridas: hasta 7x
# se entregaban 7-14 cajas; a 12x se cayo a 4 y acabo en deadlock.
MAX_DETOUR: float = 8.0


class Obstacle:
    """Una caja negra: en que nodo esta y, por tanto, si se ve."""

    def __init__(self, obstacle_id: str, node: str | None = None) -> None:
        self.id: str = str(obstacle_id)
        self.node: str | None = node

    @property
    def active(self) -> bool:
        """Un obstaculo sin nodo no esta en ninguna parte, y Unity lo esconde."""
        return self.node is not None

    def __repr__(self) -> str:
        return f"Obstacle(id={self.id!r}, node={self.node!r})"


class ObstacleField:
    """Los obstaculos de una corrida: su sorteo y los nodos que dejan vetados."""

    def __init__(
        self,
        graph: WarehouseGraph,
        *,
        ids: Sequence[str] = OBSTACLE_IDS,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        self.graph: WarehouseGraph = graph
        self.obstacles: list[Obstacle] = [Obstacle(nombre) for nombre in ids]
        self._rng: random.Random = random.Random(seed)

    def __repr__(self) -> str:
        return f"ObstacleField(obstaculos={len(self.obstacles)}, nodos={sorted(self.nodes())})"

    def nodes(self) -> frozenset[str]:
        """Los nodos tapados ahora mismo. Es lo que hay que vetar en A*."""
        return frozenset(
            obstaculo.node
            for obstaculo in self.obstacles
            if obstaculo.node is not None
        )

    def scatter(self, *, avoid: Iterable[str] = ()) -> frozenset[str]:
        """Reparte los obstaculos por pasillos libres, uno por nodo."""
        prohibidos = set(avoid) | set(self.graph.agv_starts)
        candidatos = [
            nodo
            for nodo in self.graph.nodes_with_role(OBSTACLE_ROLE)
            if nodo not in prohibidos
        ]
        self._rng.shuffle(candidatos)

        puestos: list[str] = []
        for obstaculo in self.obstacles:
            obstaculo.node = self._elige(candidatos, puestos)
            if obstaculo.node is None:
                log.warning(
                    "obstaculo %s sin sitio: no queda pasillo que tapar sin "
                    "partir el mapa, se queda fuera de la escena",
                    obstaculo.id,
                )
                continue
            puestos.append(obstaculo.node)

        log.info(
            "obstaculos repartidos: %s",
            ", ".join(f"{o.id} en {o.node}" for o in self.obstacles if o.active)
            or "(ninguno)",
        )
        return frozenset(puestos)

    def _elige(self, candidatos: list[str], puestos: Sequence[str]) -> str | None:
        """Saca de `candidatos` el primero que no parta el mapa, o None si ninguno."""
        while candidatos:
            candidato = candidatos.pop()
            if _reparto_valido(self.graph, [*puestos, candidato]):
                return candidato
            log.debug("descartado %s: taparlo obliga a rodear media nave", candidato)
        return None

    def clear(self) -> None:
        """Quita todos los obstaculos de la escena."""
        for obstaculo in self.obstacles:
            obstaculo.node = None

    def as_dicts(self) -> list[dict[str, Any]]:
        """Los obstaculos tal y como los lee Unity, con el sitio ya convertido."""
        salida: list[dict[str, Any]] = []
        for obstaculo in self.obstacles:
            punto = (
                self.graph.positions.get(obstaculo.node)
                if obstaculo.node is not None
                else None
            )
            x, _y, z = to_unity(*punto) if punto is not None else (0.0, 0.0, 0.0)
            salida.append(
                {
                    "id": obstaculo.id,
                    "active": obstaculo.active,
                    "node": obstaculo.node,
                    "x": x,
                    "z": z,
                }
            )
        return salida


def _reparto_valido(
    graph: WarehouseGraph,
    tapados: Sequence[str],
    factor_max: float = MAX_DETOUR,
) -> bool:
    """True si con `tapados` puestos se sigue pasando por todos lados sin rodeos absurdos."""
    bloqueados = set(tapados)
    for nodo in tapados:
        vecinos = [v for v in graph.neighbors(nodo) if v not in bloqueados]
        for i, uno in enumerate(vecinos):
            for otro in vecinos[i + 1 :]:
                directo = graph.cost(uno, nodo) + graph.cost(nodo, otro)
                rodeo = _coste_evitando(graph, uno, otro, bloqueados)
                if rodeo is None or rodeo > directo * factor_max:
                    return False
    return True


def _coste_evitando(
    graph: WarehouseGraph,
    origen: str,
    destino: str,
    bloqueados: set[str],
) -> float | None:
    """Lo que cuesta ir de `origen` a `destino` sin pisar `bloqueados`, o None si no se puede."""
    if origen in bloqueados or destino in bloqueados:
        return None

    mejor: dict[str, float] = {origen: 0.0}
    pendientes: list[tuple[float, str]] = [(0.0, origen)]
    while pendientes:
        coste, nodo = heapq.heappop(pendientes)
        if nodo == destino:
            return coste
        if coste > mejor.get(nodo, float("inf")):
            continue
        for vecino in graph.neighbors(nodo):
            if vecino in bloqueados:
                continue
            candidato = coste + graph.cost(nodo, vecino)
            if candidato < mejor.get(vecino, float("inf")):
                mejor[vecino] = candidato
                heapq.heappush(pendientes, (candidato, vecino))
    return None
