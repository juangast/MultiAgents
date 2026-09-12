"""Obstaculos que aparecen en sitio libre y estorban de verdad.

Son las cajas negras que ya estan puestas en la escena de Unity
(`Obstaculo_1`..`Obstaculo_3`). Cada corrida se reparten por nodos de paso
elegidos al azar y el nodo que ocupan queda **vetado para A\\***: no son
decorado, los AGV les dan la vuelta.

Sortear el nodo a ciegas tiene una trampa. En el almacen hay pasillos que son el
unico paso a una zona, y taparlos deja estanterias o muelles incomunicados: A*
se queda sin ruta y la corrida se muere en deadlock. Por eso ningun candidato se
acepta sin probarlo antes: `_deja_el_mapa_entero` comprueba que, quitandolo, se
siga llegando de cualquier nodo libre a todos los demas.

Unity solo pinta lo que se le manda desde aqui: el `id` enlaza con el objeto de
la escena y `x`/`z` dicen donde ponerlo. Que el obstaculo estorbe o no lo decide
este modulo, no el cliente.
"""

import heapq
import random
from collections.abc import Iterable, Sequence
from typing import Any

import config
from config import get_logger
from graph import ROLE_TRANSIT, WarehouseGraph, to_unity

log = get_logger("obstacles")

# Los objetos que ya existen en la escena de Unity. El id **es** el nombre del
# GameObject porque el cliente los busca por nombre: esto es contrato, no una
# etiqueta interna. Anadir otro aqui sin crearlo en la escena no rompe nada,
# simplemente Unity no lo encuentra y lo ignora.
OBSTACLE_IDS: tuple[str, ...] = ("Obstaculo_1", "Obstaculo_2", "Obstaculo_3")

# Solo se tapan pasillos. Las estanterias, los muelles, las bandas y los
# cargadores son puestos de trabajo: un obstaculo encima de uno no bloquea un
# paso, anula la mision que iba a ese sitio.
OBSTACLE_ROLE: str = ROLE_TRANSIT

# Cuanto se le permite alargarse al rodeo, contra el tramo que tapa.
#
# Que haya que dar la vuelta es justo lo que se busca. Lo que no puede pasar es
# que la vuelta sea media nave: hay pasillos de paso (P03 y P30 tienen dos
# vecinos y nada mas) que son el eslabon de una cadena, y taparlos no desconecta
# el mapa —por eso no basta con mirar la conectividad— pero manda a los AGV a
# rodear el almacen entero. Medido sobre 12 corridas: los repartos con rodeos de
# hasta 7x entregaban entre 7 y 14 cajas, y el unico que se fue a 12x se quedo
# en 4 y acabo en deadlock. El corte va justo encima de ese grupo.
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
        # Un generador propio, y que **no** se reinicia en cada `scatter()`: asi
        # cada corrida saca un reparto distinto y a la vez la sesion entera
        # sigue siendo reproducible desde la semilla.
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
        """Reparte los obstaculos por pasillos libres, uno por nodo.

        `avoid` son los nodos que no se pueden tapar por estar ocupados ahora
        mismo (los de salida de los AGV entran solos). Devuelve los nodos que
        han quedado tapados.
        """
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
        """Saca de `candidatos` el primero que no parta el mapa, o None si ninguno.

        Los que no valen se tiran para siempre: `puestos` solo crece, asi que un
        nodo que ya estrangula el mapa con los obstaculos de ahora tampoco va a
        valer con uno mas encima.
        """
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
    """True si con `tapados` puestos se sigue pasando por todos lados sin rodeos absurdos.

    Para cada nodo tapado mira sus vecinos de dos en dos: si para ir de un lado
    al otro del obstaculo hace falta mas de `factor_max` veces lo que media
    cruzarlo, ese nodo era un cuello de botella y el reparto no vale.

    De paso cubre la conectividad, que es el caso extremo de lo mismo: si dos
    vecinos dejan de alcanzarse, el rodeo es infinito y se descarta igual. Y hay
    que repasar **todos** los tapados y no solo el ultimo, porque el rodeo que
    le valia a uno puede pasar justo por donde acaba de caer otro.
    """
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
