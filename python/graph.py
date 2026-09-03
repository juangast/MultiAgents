"""Mapa logico del almacen: el grafo que comparten Python y Unity.

Un nodo es un punto donde un AGV puede estar y una arista es un tramo por el que
puede pasar, con su costo. La simulacion piensa en coordenadas logicas (px, py);
la exportacion a Unity sale de `protocol.to_unity()`, que es la unica conversion
del proyecto y no se copia aqui.

Modulo puro: no abre sockets ni guarda estado global, se puede importar y probar
sin levantar el servidor.
"""

import json
import math
import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import config
import protocol

Adjacency = dict[str, dict[str, float]]
Positions = dict[str, tuple[float, float]]


class GraphError(ValueError):
    """Un grafo, o el fichero que lo describe, no sirve como mapa."""


class WarehouseGraph:
    """Grafo del almacen: nodos con posicion y aristas con costo.

    Por defecto es no dirigido, que es lo normal en un almacen: `validate()`
    exige entonces que cada tramo exista en los dos sentidos y valga lo mismo.
    Con `directed=True` la asimetria es legitima y sirve para pasillos de un solo
    sentido, asi que solo se comprueba que se pueda ir y volver.
    """

    def __init__(
        self,
        adjacency: Mapping[str, Mapping[str, float]],
        positions: Mapping[str, tuple[float, float]],
        *,
        name: str = "",
        directed: bool = False,
    ) -> None:
        # Copia propia de los dos diccionarios: si el grafo se quedara con los
        # que le pasaron, cualquiera podria cambiarle el mapa por detras.
        self.adjacency: Adjacency = {
            str(nodo): {str(vecino): float(costo) for vecino, costo in vecinos.items()}
            for nodo, vecinos in adjacency.items()
        }
        self.positions: Positions = {
            str(nodo): (float(punto[0]), float(punto[1]))
            for nodo, punto in positions.items()
        }
        self.name: str = name
        self.directed: bool = directed

    def __repr__(self) -> str:
        return (
            f"WarehouseGraph(name={self.name!r}, nodes={len(self.adjacency)}, "
            f"edges={len(self.edges())}, directed={self.directed})"
        )

    def __eq__(self, otro: object) -> bool:
        if not isinstance(otro, WarehouseGraph):
            return NotImplemented
        return (
            self.name == otro.name
            and self.directed == otro.directed
            and self.adjacency == otro.adjacency
            and self.positions == otro.positions
        )

    def nodes(self) -> list[str]:
        """Todos los nodos, ordenados para que la salida sea siempre la misma."""
        return sorted(self.adjacency)

    def neighbors(self, node: str) -> list[str]:
        """Vecinos alcanzables desde `node`, ordenados."""
        if node not in self.adjacency:
            raise KeyError(f"nodo desconocido: {node!r}")
        return sorted(self.adjacency[node])

    def has_edge(self, a: str, b: str) -> bool:
        """Dice si existe la arista a -> b. Nunca lanza."""
        return b in self.adjacency.get(a, {})

    def cost(self, a: str, b: str) -> float:
        """Costo de la arista a -> b.

        Lanza KeyError si no existe: usa `has_edge()` como guardia cuando la
        arista puede faltar.
        """
        if not self.has_edge(a, b):
            raise KeyError(f"no hay arista {a!r} -> {b!r}")
        return self.adjacency[a][b]

    def edges(self) -> list[tuple[str, str, float]]:
        """Aristas ordenadas como (origen, destino, costo).

        Si el grafo es no dirigido cada tramo sale una sola vez, con el par ya
        ordenado: la arista inversa es el mismo tramo contado dos veces.
        """
        aristas: list[tuple[str, str, float]] = []
        vistas: set[tuple[str, str]] = set()

        for origen in self.nodes():
            for destino in self.neighbors(origen):
                costo = self.adjacency[origen][destino]
                if self.directed:
                    aristas.append((origen, destino, costo))
                    continue

                # El par ordenado identifica el tramo, venga del sentido que
                # venga: asi un mapa asimetrico por error tampoco pierde aristas
                # al listarlo, y `validate()` puede explicar que le pasa.
                clave = (origen, destino) if origen <= destino else (destino, origen)
                if clave in vistas:
                    continue
                vistas.add(clave)
                aristas.append((clave[0], clave[1], costo))

        aristas.sort(key=lambda arista: (arista[0], arista[1]))
        return aristas

    def to_unity_dict(self) -> dict[str, Any]:
        """Exporta nodos y aristas con las coordenadas ya en el sistema de Unity.

        La conversion sale de `protocol.to_unity()`, no de una copia local, y se
        hace en cada llamada sin cachear: cambiar `config.UNITY_SCALE` cambia
        todas las coordenadas exportadas. Los campos x/y/z llevan los mismos
        nombres que el snapshot, para que Unity lea siempre lo mismo.

        Solo tiene sentido en un grafo que pasa `validate()`: un nodo sin
        posicion lanza KeyError aqui.
        """
        nodos: list[dict[str, Any]] = []
        for nodo in self.nodes():
            px, py = self.positions[nodo]
            unity_x, unity_y, unity_z = protocol.to_unity(px, py)
            nodos.append({"id": nodo, "x": unity_x, "y": unity_y, "z": unity_z})

        return {
            "name": self.name,
            "directed": self.directed,
            "scale": config.UNITY_SCALE,
            "nodes": nodos,
            "edges": [
                {"from": origen, "to": destino, "cost": costo}
                for origen, destino, costo in self.edges()
            ],
        }

    def validate(self) -> None:
        """Revienta con GraphError si el grafo no sirve como mapa.

        Junta todos los problemas en un solo mensaje en vez de parar en el
        primero: asi un mapa mal editado se arregla de una pasada.
        """
        problemas = self._problemas()
        if not problemas:
            return

        cabecera = f"el grafo {self.name!r} no es valido" if self.name else "grafo no valido"
        cuenta = f"{len(problemas)} problema" + ("s" if len(problemas) != 1 else "")
        detalle = "\n".join(f"  - {problema}" for problema in problemas)
        raise GraphError(f"{cabecera} ({cuenta}):\n{detalle}")

    def _problemas(self) -> list[str]:
        """Lista todo lo que esta mal en el grafo, en orden de lectura."""
        if not self.adjacency:
            return ["el grafo no tiene nodos"]

        problemas: list[str] = []
        conocidos = set(self.adjacency)

        for nodo in self.nodes():
            if nodo not in self.positions:
                problemas.append(f"el nodo {nodo!r} no tiene posicion")

        for nodo in sorted(set(self.positions) - conocidos):
            problemas.append(f"hay una posicion para {nodo!r}, que no es un nodo del grafo")

        problemas.extend(self._problemas_de_aristas(conocidos))
        problemas.extend(self._problemas_de_conectividad(conocidos))
        return problemas

    def _problemas_de_aristas(self, conocidos: set[str]) -> list[str]:
        """Aristas colgantes, bucles propios, costos malos y asimetrias."""
        problemas: list[str] = []

        for origen in self.nodes():
            for destino in self.neighbors(origen):
                arista = f"la arista {origen!r} -> {destino!r}"

                if destino == origen:
                    problemas.append(f"el nodo {origen!r} tiene una arista a si mismo")
                    continue
                if destino not in conocidos:
                    problemas.append(f"{arista} apunta a un nodo que no existe")
                    continue

                costo = self.adjacency[origen][destino]
                if not math.isfinite(costo):
                    problemas.append(f"{arista} tiene un costo que no es finito: {costo}")
                    continue
                if costo < 0:
                    problemas.append(f"{arista} tiene un costo negativo: {costo}")

                if self.directed:
                    continue

                # No dirigido: una asimetria aqui casi siempre es un despiste al
                # editar el mapa, no una decision. La comprobacion del costo se
                # hace en un solo sentido para no repetir el mismo aviso.
                inverso = self.adjacency.get(destino, {}).get(origen)
                if inverso is None:
                    problemas.append(
                        f"el grafo es no dirigido pero falta la arista inversa "
                        f"{destino!r} -> {origen!r}"
                    )
                elif origen < destino and not math.isclose(costo, inverso, rel_tol=1e-9):
                    problemas.append(
                        f"{arista} cuesta {costo} pero la inversa cuesta {inverso}"
                    )

        return problemas

    def _problemas_de_conectividad(self, conocidos: set[str]) -> list[str]:
        """Comprueba que se llegue a todos los nodos, y en dirigido que se vuelva."""
        if len(conocidos) < 2:
            return []

        raiz = self.nodes()[0]
        sueltos = sorted(conocidos - _alcanzables(raiz, self.adjacency))
        if sueltos:
            return [
                f"el grafo esta desconectado: desde {raiz!r} no se llega a "
                f"{_lista(sueltos)}"
            ]

        if not self.directed:
            return []

        # Dirigido: llegar no basta, un AGV tambien tiene que poder volver.
        sin_vuelta = sorted(conocidos - _alcanzables(raiz, self._invertida()))
        if sin_vuelta:
            return [
                f"el grafo dirigido no es fuertemente conexo: desde "
                f"{_lista(sin_vuelta)} no se vuelve a {raiz!r}"
            ]
        return []

    def _invertida(self) -> Adjacency:
        """El mismo grafo con todas las aristas al reves."""
        invertida: Adjacency = {nodo: {} for nodo in self.adjacency}
        for origen, vecinos in self.adjacency.items():
            for destino, costo in vecinos.items():
                invertida.setdefault(destino, {})[origen] = costo
        return invertida


def _alcanzables(raiz: str, adyacencia: Adjacency) -> set[str]:
    """Nodos a los que se llega desde `raiz` siguiendo `adyacencia`, con un BFS."""
    vistos = {raiz}
    cola = deque([raiz])
    while cola:
        nodo = cola.popleft()
        for vecino in adyacencia.get(nodo, {}):
            if vecino not in vistos:
                vistos.add(vecino)
                cola.append(vecino)
    return vistos


def _lista(nombres: Iterable[str]) -> str:
    """Junta nombres de nodo para meterlos en un mensaje de error."""
    return ", ".join(repr(nombre) for nombre in nombres)


def _adyacencia_desde_tramos(
    tramos: Iterable[tuple[str, str, float]],
    nodos: Iterable[str],
) -> Adjacency:
    """Arma la adyacencia no dirigida a partir de la lista de tramos.

    Cada tramo se escribe una sola vez y se duplica aqui en los dos sentidos,
    que es la forma de que no se cuele la asimetria que `validate()` persigue.
    """
    adyacencia: Adjacency = {nodo: {} for nodo in nodos}
    for a, b, costo in tramos:
        adyacencia.setdefault(a, {})[b] = costo
        adyacencia.setdefault(b, {})[a] = costo
    return adyacencia


# --- Mapa "simple": el grafo de 6 nodos de la guia, para pruebas rapidas ------
#
# Ojo: los costos no son la distancia entre las posiciones. A(0,0) -> D(0,3)
# cuesta 4 y no 3. Es a proposito: un pasillo puede ser lento sin ser largo, y
# por eso `validate()` no compara costo con geometria.

SIMPLE_ADJACENCY: Adjacency = {
    "A": {"B": 2.0, "D": 4.0},
    "B": {"A": 2.0, "C": 2.0, "E": 3.0},
    "C": {"B": 2.0, "F": 4.0},
    "D": {"A": 4.0, "E": 2.0},
    "E": {"D": 2.0, "B": 3.0, "F": 2.0},
    "F": {"E": 2.0, "C": 4.0},
}

SIMPLE_POSITIONS: Positions = {
    "A": (0.0, 0.0),
    "B": (2.0, 0.0),
    "C": (4.0, 0.0),
    "D": (0.0, 3.0),
    "E": (2.0, 3.0),
    "F": (4.0, 3.0),
}


# --- Mapa "warehouse": 13 nodos con forma de pasillos ------------------------
#
# Dos corredores horizontales (S al sur, N al norte), cuatro conexiones
# verticales y un cuello de botella en G:
#
#   N1--N2--N3            N4--N5--N6      y = 8
#    |       | \          / |       |
#    |       |   >  G  <    |       |     y = 4
#    |       | /          \ |       |
#   S1--S2--S3            S4--S5--S6      y = 0
#    x=0     4   8   12   16  20  24
#
# G es un nodo de articulacion: es la unica union entre la zona izquierda y la
# derecha, asi que toda ruta que cruce el almacen pasa por el a la fuerza. De
# ahi salen los escenarios de congestion de las fases siguientes.

BOTTLENECK: str = "G"

WAREHOUSE_POSITIONS: Positions = {
    "S1": (0.0, 0.0),
    "S2": (4.0, 0.0),
    "S3": (8.0, 0.0),
    "S4": (16.0, 0.0),
    "S5": (20.0, 0.0),
    "S6": (24.0, 0.0),
    "N1": (0.0, 8.0),
    "N2": (4.0, 8.0),
    "N3": (8.0, 8.0),
    "N4": (16.0, 8.0),
    "N5": (20.0, 8.0),
    "N6": (24.0, 8.0),
    "G": (12.0, 4.0),
}

# Cada tramo una sola vez, como (a, b, costo). El costo es la distancia entre
# las dos posiciones redondeada a un decimal.
WAREHOUSE_EDGES: tuple[tuple[str, str, float], ...] = (
    ("S1", "S2", 4.0),
    ("S2", "S3", 4.0),
    ("S4", "S5", 4.0),
    ("S5", "S6", 4.0),
    ("N1", "N2", 4.0),
    ("N2", "N3", 4.0),
    ("N4", "N5", 4.0),
    ("N5", "N6", 4.0),
    ("S1", "N1", 8.0),
    ("S3", "N3", 8.0),
    ("S4", "N4", 8.0),
    ("S6", "N6", 8.0),
    ("S3", "G", 5.7),
    ("N3", "G", 5.7),
    ("G", "S4", 5.7),
    ("G", "N4", 5.7),
)


# --- Mapa "grid": la rejilla 4x4 con caminos redundantes ---------------------
#
# Cuatro columnas (A-D, de oeste a este) por cuatro filas (1-4, de sur a norte),
# todos los tramos del mismo costo:
#
#   A4--B4--C4--D4      y = 12
#    |   |   |   |
#   A3--B3--C3--D3      y = 8
#    |   |   |   |
#   A2--B2--C2--D2      y = 4
#    |   |   |   |
#   A1--B1--C1--D1      y = 0
#   x=0  4   8   12
#
# Es el contrario exacto del `warehouse`: aqui **no hay ningun nodo de
# articulacion**, y entre dos nodos cualesquiera hay varias rutas del MISMO
# costo (todas las de Manhattan que no se desvian). Esa es la condicion para que
# REROUTE pueda aportar algo: en el almacen, penalizar G no da una ruta
# alternativa, da una ruta peor o ninguna; aqui da otra igual de buena.

GRID_COLUMNS: tuple[str, ...] = ("A", "B", "C", "D")
GRID_ROWS: tuple[int, ...] = (1, 2, 3, 4)
GRID_SPACING: float = 4.0

GRID_POSITIONS: Positions = {
    f"{columna}{fila}": (GRID_SPACING * x, GRID_SPACING * y)
    for x, columna in enumerate(GRID_COLUMNS)
    for y, fila in enumerate(GRID_ROWS)
}

# Cada tramo una sola vez, como (a, b, costo): primero los horizontales de cada
# fila y despues los verticales de cada columna. `_adyacencia_desde_tramos()` los
# duplica en los dos sentidos.
GRID_EDGES: tuple[tuple[str, str, float], ...] = tuple(
    [
        (f"{GRID_COLUMNS[i]}{fila}", f"{GRID_COLUMNS[i + 1]}{fila}", GRID_SPACING)
        for fila in GRID_ROWS
        for i in range(len(GRID_COLUMNS) - 1)
    ]
    + [
        (f"{columna}{GRID_ROWS[j]}", f"{columna}{GRID_ROWS[j + 1]}", GRID_SPACING)
        for columna in GRID_COLUMNS
        for j in range(len(GRID_ROWS) - 1)
    ]
)


def simple_graph() -> WarehouseGraph:
    """El grafo de 6 nodos de la guia."""
    return WarehouseGraph(SIMPLE_ADJACENCY, SIMPLE_POSITIONS, name="simple")


def warehouse_graph() -> WarehouseGraph:
    """El almacen de 13 nodos con el cuello de botella en G."""
    return WarehouseGraph(
        _adyacencia_desde_tramos(WAREHOUSE_EDGES, WAREHOUSE_POSITIONS),
        WAREHOUSE_POSITIONS,
        name="warehouse",
    )


def grid_graph() -> WarehouseGraph:
    """La rejilla 4x4 de 16 nodos, sin cuellos de botella y con rutas redundantes."""
    return WarehouseGraph(
        _adyacencia_desde_tramos(GRID_EDGES, GRID_POSITIONS),
        GRID_POSITIONS,
        name="grid",
    )


BUILTIN_MAPS: dict[str, Callable[[], WarehouseGraph]] = {
    "simple": simple_graph,
    "warehouse": warehouse_graph,
    "grid": grid_graph,
}


# --- Ficheros ----------------------------------------------------------------


def map_path(name: str) -> Path:
    """Ruta del fichero JSON del mapa que se llama asi."""
    return config.MAPS_DIR / f"{name}.json"


def save_graph(graph: WarehouseGraph, path: str | Path) -> None:
    """Guarda el grafo como JSON legible, para editar mapas sin tocar codigo.

    Solo van las coordenadas logicas. Las de Unity son derivadas y dependen de
    `config.UNITY_SCALE`, asi que congelarlas en el fichero seria guardar una
    copia condenada a quedarse vieja. El JSON va indentado y ordenado porque
    esto se edita a mano, al reves que las lineas del socket.
    """
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "name": graph.name,
        "directed": graph.directed,
        "positions": {
            nodo: [px, py] for nodo, (px, py) in sorted(graph.positions.items())
        },
        "adjacency": {
            nodo: dict(sorted(graph.adjacency[nodo].items())) for nodo in graph.nodes()
        },
    }
    # Sin sort_keys: las claves ya van en el orden en el que se leen bien
    # (que mapa es, luego donde estan los nodos, luego como se conectan), y
    # dentro de cada una los nodos van ordenados a mano.
    texto = json.dumps(payload, indent=2, ensure_ascii=True)
    destino.write_text(_compacta_puntos(texto) + "\n", encoding=config.ENCODING)


_PUNTO_PARTIDO = re.compile(r"\[\s+(-?[\d.eE+-]+),\s+(-?[\d.eE+-]+)\s+\]")


def _compacta_puntos(texto: str) -> str:
    """Deja cada posicion [x, y] en una sola linea.

    `json.dumps` con indent tambien parte las listas de dos numeros, y un mapa
    que se edita a mano se lee mucho mejor con la posicion entera de un vistazo.
    """
    return _PUNTO_PARTIDO.sub(r"[\1, \2]", texto)


def load_graph(path: str | Path) -> WarehouseGraph:
    """Carga un grafo desde su fichero JSON.

    Comprueba la forma del fichero, pero a proposito no llama a `validate()`:
    hay que poder cargar un mapa roto justo para que `validate()` explique que
    le pasa.
    """
    origen = Path(path)
    try:
        crudo = json.loads(origen.read_text(encoding=config.ENCODING))
    except OSError as exc:
        raise GraphError(f"no se pudo leer el mapa {origen}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GraphError(f"el mapa {origen} no es JSON valido: {exc}") from exc

    if not isinstance(crudo, dict):
        raise GraphError(f"el mapa {origen} tendria que ser un objeto JSON")

    return WarehouseGraph(
        _lee_adyacencia(crudo.get("adjacency"), origen),
        _lee_posiciones(crudo.get("positions"), origen),
        name=str(crudo.get("name") or origen.stem),
        directed=bool(crudo.get("directed", False)),
    )


def _es_numero(valor: Any) -> bool:
    """Numero de verdad. bool es int en Python y aqui no cuela."""
    return isinstance(valor, (int, float)) and not isinstance(valor, bool)


def _lee_adyacencia(crudo: Any, origen: Path) -> Adjacency:
    """Comprueba la forma de la adyacencia que venia en el fichero."""
    if not isinstance(crudo, dict):
        raise GraphError(f"al mapa {origen} le falta 'adjacency' como objeto")

    adyacencia: Adjacency = {}
    for nodo, vecinos in crudo.items():
        if not isinstance(vecinos, dict):
            raise GraphError(
                f"en {origen}, los vecinos de {nodo!r} tendrian que ser un objeto"
            )
        costos: dict[str, float] = {}
        for vecino, costo in vecinos.items():
            if not _es_numero(costo):
                raise GraphError(
                    f"en {origen}, el costo de {nodo!r} -> {vecino!r} no es un "
                    f"numero: {costo!r}"
                )
            costos[str(vecino)] = float(costo)
        adyacencia[str(nodo)] = costos
    return adyacencia


def _lee_posiciones(crudo: Any, origen: Path) -> Positions:
    """Comprueba la forma de las posiciones que venian en el fichero."""
    if not isinstance(crudo, dict):
        raise GraphError(f"al mapa {origen} le falta 'positions' como objeto")

    posiciones: Positions = {}
    for nodo, punto in crudo.items():
        if not isinstance(punto, (list, tuple)) or len(punto) != 2:
            raise GraphError(
                f"en {origen}, la posicion de {nodo!r} tendria que ser [x, y]"
            )
        if not all(_es_numero(valor) for valor in punto):
            raise GraphError(
                f"en {origen}, la posicion de {nodo!r} no son dos numeros: {punto!r}"
            )
        posiciones[str(nodo)] = (float(punto[0]), float(punto[1]))
    return posiciones
