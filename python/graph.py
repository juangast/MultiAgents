"""El mapa del almacen y el ruteo sobre el.

Un nodo es un punto donde un AGV puede estar y una arista un tramo con su costo.
El JSON de `maps/` es la unica fuente, y `to_unity()` la unica conversion de
coordenadas del proyecto.
"""

import heapq
import json
import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import config

Adjacency = dict[str, dict[str, float]]
Positions = dict[str, tuple[float, float]]
Cells = dict[str, tuple[int, int]]
NodeRoles = dict[str, str]

ROLE_PRODUCTION: str = "production"
ROLE_STORAGE: str = "storage"
ROLE_TRANSIT: str = "transit"
ROLE_DOCK: str = "dock"
ROLE_CHARGING: str = "charging"

DEFAULT_ROLE: str = ROLE_TRANSIT

ORIGENES_DE_CAJA: frozenset[str] = frozenset({ROLE_PRODUCTION, ROLE_STORAGE})

UNITY_Y: float = 0.0


def to_unity(px: float, py: float) -> tuple[float, float, float]:
    """Pasa una posicion (px, py) del plano de la simulacion a coordenadas de Unity.

    Es la unica conversion del proyecto: no la repitas en otro sitio.
    """
    return (px * config.UNITY_SCALE, UNITY_Y, py * config.UNITY_SCALE)


class GraphError(ValueError):
    """Un grafo, o el fichero que lo describe, no sirve como mapa."""


class Box:
    """Una caja del almacen: donde esta y a que altura."""

    def __init__(self, id: str, node: str, level: int) -> None:
        self.id = id
        self.node = node
        self.level = level


class WarehouseGraph:
    """Grafo del almacen: nodos con posicion y aristas con costo."""

    def __init__(
        self,
        adjacency: Mapping[str, Mapping[str, float]],
        positions: Mapping[str, tuple[float, float]],
        *,
        name: str = "",
        directed: bool = False,
        cells: Mapping[str, tuple[int, int]] | None = None,
        node_roles: Mapping[str, str] | None = None,
        roles: Mapping[str, str] | None = None,
        boxes: Sequence[Box] | None = None,
        coordinate_system: Mapping[str, Any] | None = None,
    ) -> None:
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

        self.cells: Cells = {
            str(nodo): (int(celda[0]), int(celda[1]))
            for nodo, celda in (cells or {}).items()
        }
        self.node_roles: NodeRoles = {
            str(nodo): str(rol) for nodo, rol in (node_roles or {}).items()
        }
        self.roles: dict[str, str] = {
            str(rol): str(texto) for rol, texto in (roles or {}).items()
        }
        self.boxes: list[Box] = list(boxes or ())
        self.coordinate_system: dict[str, Any] = dict(coordinate_system or {})

    def __repr__(self) -> str:
        return (
            f"WarehouseGraph(name={self.name!r}, nodes={len(self.adjacency)}, "
            f"edges={len(self.edges())}, directed={self.directed})"
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

                clave = (origen, destino) if origen <= destino else (destino, origen)
                if clave in vistas:
                    continue
                vistas.add(clave)
                aristas.append((clave[0], clave[1], costo))

        aristas.sort(key=lambda arista: (arista[0], arista[1]))
        return aristas

    def role_of(self, node: str) -> str:
        """El rol de ese nodo. Un nodo sin rol declarado es de paso."""
        return self.node_roles.get(node, DEFAULT_ROLE)

    def nodes_with_role(self, role: str) -> list[str]:
        """Los nodos que tienen ese rol, en orden alfabetico."""
        return sorted(nodo for nodo in self.nodes() if self.role_of(nodo) == role)

    def boxes_at(self, node: str) -> list[Box]:
        """Las cajas que hay en ese nodo, de abajo arriba."""
        return sorted(
            (caja for caja in self.boxes if caja.node == node),
            key=lambda caja: caja.level,
        )

    def box(self, box_id: str) -> Box | None:
        """La caja con ese id, o None si el mapa no la tiene."""
        for caja in self.boxes:
            if caja.id == box_id:
                return caja
        return None

    def to_unity_dict(self) -> dict[str, Any]:
        """Exporta nodos y aristas con las coordenadas ya en el sistema de Unity."""
        nodos: list[dict[str, Any]] = []
        for nodo in self.nodes():
            px, py = self.positions[nodo]
            unity_x, unity_y, unity_z = to_unity(px, py)
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
        problemas.extend(self._problemas_de_almacen(conocidos))
        return problemas

    def _problemas_de_almacen(self, conocidos: set[str]) -> list[str]:
        """Roles, celdas y cajas: lo que el mapa dice del almacen, no del grafo.

        Un mapa sin ninguno de los tres bloques no da ningun problema: son
        opcionales y los mapas viejos siguen siendo validos.
        """
        problemas: list[str] = []

        for nodo in sorted(set(self.cells) - conocidos):
            problemas.append(f"hay una celda para {nodo!r}, que no es un nodo del grafo")

        for nodo in sorted(set(self.node_roles) - conocidos):
            problemas.append(f"hay un rol para {nodo!r}, que no es un nodo del grafo")

        if self.roles:
            for nodo in self.nodes():
                rol = self.node_roles.get(nodo)
                if rol is not None and rol not in self.roles:
                    problemas.append(
                        f"el nodo {nodo!r} tiene el rol {rol!r}, que no esta en 'roles'"
                    )

        for nodo, (columna, fila) in sorted(self.cells.items()):
            punto = self.positions.get(nodo)
            paso = self.coordinate_system.get("spacing")
            if punto is None or not isinstance(paso, (int, float)):
                continue
            esperado = (columna * float(paso), fila * float(paso))
            if not all(
                math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
                for a, b in zip(punto, esperado)
            ):
                problemas.append(
                    f"la celda de {nodo!r} es {[columna, fila]}, que con spacing "
                    f"{paso} daria {list(esperado)}, pero su posicion es {list(punto)}"
                )

        problemas.extend(self._problemas_de_cajas(conocidos))
        return problemas

    def _problemas_de_cajas(self, conocidos: set[str]) -> list[str]:
        """Cajas colgantes, ids repetidos, huecos ocupados dos veces y niveles malos."""
        problemas: list[str] = []
        niveles = self.coordinate_system.get("levels")
        vistos: dict[str, str] = {}
        huecos: dict[tuple[str, int], str] = {}

        for caja in self.boxes:
            if caja.id in vistos:
                problemas.append(f"hay dos cajas con el id {caja.id!r}")
            vistos[caja.id] = caja.node

            if caja.node not in conocidos:
                problemas.append(
                    f"la caja {caja.id!r} esta en {caja.node!r}, que no es un nodo del grafo"
                )
            elif self.node_roles and self.role_of(caja.node) not in ORIGENES_DE_CAJA:
                problemas.append(
                    f"la caja {caja.id!r} esta en {caja.node!r}, que es "
                    f"{self.role_of(caja.node)!r}; una caja solo nace en "
                    f"{_lista(sorted(ORIGENES_DE_CAJA))}"
                )

            hueco = (caja.node, caja.level)
            if hueco in huecos:
                problemas.append(
                    f"las cajas {huecos[hueco]!r} y {caja.id!r} ocupan el mismo "
                    f"hueco: {caja.node} nivel {caja.level}"
                )
            huecos[hueco] = caja.id

            if isinstance(niveles, list) and niveles and caja.level not in niveles:
                problemas.append(
                    f"la caja {caja.id!r} esta en el nivel {caja.level}, que no "
                    f"esta en {niveles}"
                )

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


def map_path(name: str) -> Path:
    """Ruta del fichero JSON del mapa que se llama asi."""
    return config.MAPS_DIR / f"{name}.json"


def load_graph(path: str | Path) -> WarehouseGraph:
    """Carga un grafo desde su fichero JSON."""
    origen = Path(path)
    try:
        crudo = json.loads(origen.read_text(encoding=config.ENCODING))
    except OSError as exc:
        raise GraphError(f"no se pudo leer el mapa {origen}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GraphError(f"el mapa {origen} no es JSON valido: {exc}") from exc

    if not isinstance(crudo, dict):
        raise GraphError(f"el mapa {origen} tendria que ser un objeto JSON")

    celdas, roles_por_nodo = _lee_nodos(crudo.get("nodes"), origen)
    return WarehouseGraph(
        _lee_adyacencia(crudo.get("adjacency"), origen),
        _lee_posiciones(crudo.get("positions"), origen),
        name=str(crudo.get("name") or origen.stem),
        directed=bool(crudo.get("directed", False)),
        cells=celdas,
        node_roles=roles_por_nodo,
        roles=_lee_roles(crudo.get("roles"), origen),
        boxes=_lee_cajas(crudo.get("boxes"), origen),
        coordinate_system=_lee_sistema(crudo.get("coordinate_system"), origen),
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


def _lee_sistema(crudo: Any, origen: Path) -> dict[str, Any]:
    """El bloque `coordinate_system`, que es opcional."""
    if crudo is None:
        return {}
    if not isinstance(crudo, dict):
        raise GraphError(f"en {origen}, 'coordinate_system' tendria que ser un objeto")
    return dict(crudo)


def _lee_roles(crudo: Any, origen: Path) -> dict[str, str]:
    """La leyenda de roles, que es opcional."""
    if crudo is None:
        return {}
    if not isinstance(crudo, dict):
        raise GraphError(f"en {origen}, 'roles' tendria que ser un objeto")
    return {str(rol): str(texto) for rol, texto in crudo.items()}


def _lee_nodos(crudo: Any, origen: Path) -> tuple[Cells, NodeRoles]:
    """Saca celdas y roles del bloque `nodes`, que es opcional.

    Cada entrada puede traer `cell`, `role` o las dos: un nodo que solo declara
    su rol no necesita celda, y al reves.
    """
    if crudo is None:
        return {}, {}
    if not isinstance(crudo, dict):
        raise GraphError(f"al mapa {origen} le falta 'nodes' como objeto")

    celdas: Cells = {}
    roles: NodeRoles = {}
    for nodo, datos in crudo.items():
        if not isinstance(datos, dict):
            raise GraphError(
                f"en {origen}, los datos del nodo {nodo!r} tendrian que ser un objeto"
            )

        celda = datos.get("cell")
        if celda is not None:
            if not isinstance(celda, (list, tuple)) or len(celda) != 2:
                raise GraphError(
                    f"en {origen}, la celda de {nodo!r} tendria que ser [columna, fila]"
                )
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in celda):
                raise GraphError(
                    f"en {origen}, la celda de {nodo!r} no son dos enteros: {celda!r}"
                )
            celdas[str(nodo)] = (int(celda[0]), int(celda[1]))

        rol = datos.get("role")
        if rol is not None:
            roles[str(nodo)] = str(rol)

    return celdas, roles


def _lee_cajas(crudo: Any, origen: Path) -> list[Box]:
    """El inventario del bloque `boxes`, que es opcional."""
    if crudo is None:
        return []
    if not isinstance(crudo, list):
        raise GraphError(f"en {origen}, 'boxes' tendria que ser una lista")

    cajas: list[Box] = []
    for numero, datos in enumerate(crudo, start=1):
        if not isinstance(datos, dict):
            raise GraphError(f"en {origen}, la caja {numero} tendria que ser un objeto")

        faltan = sorted({"id", "node", "level"} - set(datos))
        if faltan:
            raise GraphError(
                f"en {origen}, a la caja {numero} le falta {_lista(faltan)}"
            )
        nivel = datos["level"]
        if not isinstance(nivel, int) or isinstance(nivel, bool) or nivel < 1:
            raise GraphError(
                f"en {origen}, el nivel de la caja {datos['id']!r} tendria que ser "
                f"un entero desde 1, no {nivel!r}"
            )
        cajas.append(Box(id=str(datos["id"]), node=str(datos["node"]), level=nivel))

    return cajas


PenaltyKey = str | tuple[str, str]
Penalties = Mapping[PenaltyKey, float]


PENALTY_TTL: int = 15
PENALTY_MAX: float = 40.0
PENALTY_BAN: float = 1000.0


class TemporaryPenalties(Mapping):
    """Penalizaciones de ruta que caducan solas.

    Sin caducidad el mapa se degrada para siempre: A* acabaria esquivando
    pasillos que llevan cien ticks libres.
    """

    def __init__(
        self,
        *,
        ttl: int = PENALTY_TTL,
        cap: float = PENALTY_MAX,
    ) -> None:
        self.ttl: int = int(ttl)
        self.cap: float = float(cap)
        self._items: dict[PenaltyKey, tuple[float, int]] = {}

    def __repr__(self) -> str:
        return f"TemporaryPenalties(activas={len(self._items)}, ttl={self.ttl})"

    def __getitem__(self, key: PenaltyKey) -> float:
        return self._items[key][0]

    def __iter__(self) -> Iterator[PenaltyKey]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, key: PenaltyKey, amount: float, *, step: int) -> float:
        """Encarece `key` y (re)arranca su reloj. Devuelve cuanto vale ahora."""
        extra = float(amount)
        if extra <= 0.0:
            return self._items.get(key, (0.0, 0))[0]
        acumulado = min(self._items.get(key, (0.0, 0))[0] + extra, self.cap)
        self._items[key] = (acumulado, int(step) + self.ttl)
        return acumulado

    def ban(self, key: PenaltyKey, *, step: int) -> float:
        """Veta `key` con un precio caro pero finito, o A* se queda sin ruta."""
        self._items[key] = (PENALTY_BAN, int(step) + self.ttl)
        return PENALTY_BAN

    def discard(self, key: PenaltyKey) -> None:
        """Quita la penalizacion de `key`, si la tenia."""
        self._items.pop(key, None)

    def expire(self, step: int) -> int:
        """Borra las que ya caducaron. Devuelve cuantas se fueron."""
        vencidas = [
            clave for clave, (_, caduca) in self._items.items() if caduca <= int(step)
        ]
        for clave in vencidas:
            del self._items[clave]
        return len(vencidas)

    def clear(self) -> None:
        """Deja la tabla vacia."""
        self._items.clear()


def heuristic_factor(graph: WarehouseGraph) -> float:
    """Cuanto hay que encoger la distancia euclidiana para que no sobreestime."""
    factor = 1.0
    for origen, destino, costo in graph.edges():
        largo = _distancia(graph, origen, destino)
        if largo <= 0.0:
            continue
        factor = min(factor, costo / largo)
    return max(factor, 0.0)


def astar(
    graph: WarehouseGraph,
    start: str,
    goal: str,
    penalties: Penalties | None = None,
) -> list[str] | None:
    """La ruta mas barata de `start` a `goal`, o None si no hay. Nunca lanza."""
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
    pendientes: list[tuple[float, str]] = [(h(start), start)]

    while pendientes:
        _, actual = heapq.heappop(pendientes)
        if actual in cerrados:
            continue
        cerrados.add(actual)

        if actual == goal:
            return _reconstruye(procedencia, start, goal)

        for vecino in graph.neighbors(actual):
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
    """Costo de una ruta ya trazada. Lanza KeyError si dos nodos no son vecinos."""
    total = 0.0
    for anterior, siguiente in zip(path, path[1:]):
        total += (
            graph.cost(anterior, siguiente)
            + _penalizacion_arista(graph, penalties, anterior, siguiente)
            + _penalizacion_nodo(penalties, siguiente)
        )
    return total


def _distancia(graph: WarehouseGraph, a: str, b: str) -> float:
    """Distancia euclidiana entre dos nodos. Sin posicion devuelve 0.0."""
    origen = graph.positions.get(a)
    destino = graph.positions.get(b)
    if origen is None or destino is None:
        return 0.0
    return math.dist(origen, destino)


def _penalizacion_nodo(penalties: Penalties | None, node: str) -> float:
    """Extra por entrar en `node`."""
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
    """Extra por cruzar a -> b. En no dirigido, (a,b) y (b,a) son el mismo tramo."""
    if not penalties:
        return 0.0
    extra = penalties.get((a, b))
    if extra is None and not graph.directed:
        extra = penalties.get((b, a))
    return 0.0 if extra is None else float(extra)


def _reconstruye(procedencia: Mapping[str, str], start: str, goal: str) -> list[str]:
    """Camina hacia atras por los padres hasta `start`."""
    ruta = [goal]
    while ruta[-1] != start:
        ruta.append(procedencia[ruta[-1]])
    ruta.reverse()
    return ruta
