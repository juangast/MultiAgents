"""Busqueda de rutas con A* sobre el grafo del almacen.

Modulo puro: no abre sockets ni guarda estado. Le entra un `WarehouseGraph`, un
origen y un destino, y le sale la lista de nodos de la ruta mas barata, o None.

`penalties` es el gancho del REROUTE de la fase 8: encarece nodos y tramos
concretos para esquivar una congestion, sin tocar el mapa ni recargarlo.
"""

import heapq
import math
from collections.abc import Mapping, Sequence

from graph import WarehouseGraph

# Una clave es un nodo ("G", extra por entrar en el) o una arista (("S3", "G"),
# extra por cruzarla). Los dos tipos conviven en el mismo diccionario.
Penalties = Mapping[str | tuple[str, str], float]


def heuristic_factor(graph: WarehouseGraph) -> float:
    """Cuanto hay que encoger la distancia euclidiana para que no sobreestime.

    Es el minimo de costo/longitud sobre todas las aristas, recortado a 1.0 para
    no inflar nunca la heuristica. O(E) y sin cachear a proposito: el mapa se
    puede editar entre llamadas y una copia guardada se quedaria vieja.

    --- Por que hace falta el factor ---

    A* solo garantiza la ruta optima si h(n) nunca sobreestima lo que falta de
    verdad. Aqui h es geometria y el costo NO lo es: en el mapa `simple`,
    A(0,0) -> D(0,3) mide 3 y cuesta 4, y nada impide un mapa donde un tramo
    cueste MENOS que su longitud (una cinta transportadora, por ejemplo). Ahi la
    distancia euclidiana en crudo sobreestimaria y A* podria devolver una ruta
    peor que la optima.

    Con el factor, para toda arista:      cost(u,v) >= factor * dist(u,v)
    y por la desigualdad triangular:      dist(n,goal) <= suma de dist de
                                          cualquier ruta de n a goal
    de donde:  factor*dist(n,goal) <= factor*suma(dist) <= suma(cost) = costo real.

    O sea que h es admisible por construccion, sea cual sea el mapa. Ademas, como
    factor*dist(u,v) <= cost(u,v) para cada arista suelta, h tambien es
    consistente: por eso `astar()` puede cerrar cada nodo una sola vez sin tener
    que reabrirlo nunca.

    En los dos mapas del repo el factor sale exactamente 1.0 (ninguna arista
    cuesta menos que su longitud), asi que hoy h es la euclidiana tal cual; el
    factor solo entra en juego con mapas futuros.
    """
    factor = 1.0
    for origen, destino, costo in graph.edges():
        largo = _distancia(graph, origen, destino)
        if largo <= 0.0:
            # Dos nodos en el mismo punto: la razon costo/largo no existe, y
            # entre ellos h ya vale 0, que nunca sobreestima. No dice nada.
            continue
        factor = min(factor, costo / largo)
    return max(factor, 0.0)


def astar(
    graph: WarehouseGraph,
    start: str,
    goal: str,
    penalties: Penalties | None = None,
) -> list[str] | None:
    """Ruta mas barata de `start` a `goal`, o None si no hay ninguna.

    Devuelve la lista completa de nodos, empezando en `start` y terminando en
    `goal`. Con `start == goal` devuelve `[start]`.

    Nunca lanza: un nodo desconocido, un mapa roto o un destino inalcanzable
    devuelven None. El que llama decide que hacer, no tiene que envolverlo en un
    try.
    """
    if start not in graph.adjacency or goal not in graph.adjacency:
        return None
    if start == goal:
        return [start]

    factor = heuristic_factor(graph)

    def h(nodo: str) -> float:
        return factor * _distancia(graph, nodo, goal)

    procedencia: dict[str, str] = {}
    mejor: dict[str, float] = {start: 0.0}
    cerrados: set[str] = set()
    # El heap guarda (f, nodo): a igual f gana el nombre menor. Con eso, y con
    # los vecinos que `graph.neighbors()` ya devuelve ordenados, dos corridas
    # sobre el mismo mapa dan siempre exactamente la misma ruta.
    pendientes: list[tuple[float, str]] = [(h(start), start)]

    while pendientes:
        _, actual = heapq.heappop(pendientes)
        if actual in cerrados:
            continue
        cerrados.add(actual)

        if actual == goal:
            return _reconstruye(procedencia, start, goal)

        for vecino in graph.neighbors(actual):
            # Un vecino sin fila propia en la adyacencia es un mapa roto que
            # `validate()` cazaria. Aqui se salta, que este modulo no lanza.
            if vecino in cerrados or vecino not in graph.adjacency:
                continue

            candidato = (
                mejor[actual]
                + graph.cost(actual, vecino)
                + _penalizacion_arista(graph, penalties, actual, vecino)
                + _penalizacion_nodo(penalties, vecino)
            )
            if candidato < mejor.get(vecino, math.inf):
                mejor[vecino] = candidato
                procedencia[vecino] = actual
                heapq.heappush(pendientes, (candidato + h(vecino), vecino))

    return None


def path_cost(
    graph: WarehouseGraph,
    path: Sequence[str],
    penalties: Penalties | None = None,
) -> float:
    """Costo de recorrer `path` entero, penalizaciones incluidas.

    Lanza KeyError si dos nodos seguidos no son vecinos: sirve justo para
    detectar una ruta invalida, no para tragarsela.
    """
    total = 0.0
    for anterior, siguiente in zip(path, path[1:]):
        total += (
            graph.cost(anterior, siguiente)
            + _penalizacion_arista(graph, penalties, anterior, siguiente)
            + _penalizacion_nodo(penalties, siguiente)
        )
    return total


def _distancia(graph: WarehouseGraph, a: str, b: str) -> float:
    """Distancia euclidiana entre dos nodos.

    Si a alguno le falta la posicion devuelve 0.0 en vez de lanzar: un cero
    tampoco sobreestima, asi que la heuristica sigue siendo admisible y A* se
    limita a buscar peor en un mapa que `validate()` habria rechazado.
    """
    origen = graph.positions.get(a)
    destino = graph.positions.get(b)
    if origen is None or destino is None:
        return 0.0
    return math.dist(origen, destino)


def _penalizacion_nodo(penalties: Penalties | None, node: str) -> float:
    """Extra por entrar en `node`.

    Al nodo de partida no se le aplica nunca, porque no se entra en el: el
    agente ya estaba ahi.
    """
    if not penalties:
        return 0.0
    extra = penalties.get(node)
    return 0.0 if extra is None else float(extra)


def _penalizacion_arista(
    graph: WarehouseGraph,
    penalties: Penalties | None,
    a: str,
    b: str,
) -> float:
    """Extra por cruzar a -> b.

    En un grafo no dirigido (a, b) y (b, a) son el mismo tramo, asi que si falta
    la clave exacta se prueba la contraria. Con `directed=True` no se prueba:
    ahi cada sentido es un pasillo distinto y penalizar uno no debe penalizar el
    otro.
    """
    if not penalties:
        return 0.0
    extra = penalties.get((a, b))
    if extra is None and not graph.directed:
        extra = penalties.get((b, a))
    return 0.0 if extra is None else float(extra)


def _reconstruye(procedencia: Mapping[str, str], start: str, goal: str) -> list[str]:
    """Rehace la ruta hacia atras desde el destino."""
    ruta = [goal]
    while ruta[-1] != start:
        ruta.append(procedencia[ruta[-1]])
    ruta.reverse()
    return ruta
