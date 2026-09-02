"""Tests del mapa logico: el grafo, su validacion y su ida y vuelta a JSON."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import graph
import protocol


def grafo_de_prueba(**cambios: object) -> graph.WarehouseGraph:
    """Un cuadrado de 4 nodos, valido, para romperlo a gusto en cada test."""
    argumentos: dict = {
        "adjacency": {
            "A": {"B": 1.0, "D": 1.0},
            "B": {"A": 1.0, "C": 1.0},
            "C": {"B": 1.0, "D": 1.0},
            "D": {"C": 1.0, "A": 1.0},
        },
        "positions": {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)},
        "name": "prueba",
    }
    argumentos.update(cambios)
    adyacencia = argumentos.pop("adjacency")
    posiciones = argumentos.pop("positions")
    return graph.WarehouseGraph(adyacencia, posiciones, **argumentos)


def motivo_del_fallo(grafo: graph.WarehouseGraph) -> str:
    """Corre validate() esperando que reviente y devuelve el mensaje."""
    try:
        grafo.validate()
    except graph.GraphError as exc:
        return str(exc)
    raise AssertionError("validate() tenia que haber reventado y no lo hizo")


class TestMapasDelRepo(unittest.TestCase):
    def test_los_dos_mapas_del_repo_pasan_validate(self) -> None:
        for nombre in graph.BUILTIN_MAPS:
            with self.subTest(mapa=nombre):
                ruta = graph.map_path(nombre)
                self.assertTrue(ruta.is_file(), f"falta {ruta}")
                graph.load_graph(ruta).validate()

    def test_los_constructores_pasan_validate(self) -> None:
        for nombre, constructor in graph.BUILTIN_MAPS.items():
            with self.subTest(mapa=nombre):
                constructor().validate()

    def test_los_json_del_repo_coinciden_con_el_codigo(self) -> None:
        # Si alguien edita un mapa a mano y no el codigo (o al reves), esto lo caza.
        for nombre, constructor in graph.BUILTIN_MAPS.items():
            with self.subTest(mapa=nombre):
                self.assertEqual(graph.load_graph(graph.map_path(nombre)), constructor())

    def test_el_json_no_guarda_coordenadas_de_unity(self) -> None:
        # Serian una copia derivada que se queda vieja al cambiar UNITY_SCALE.
        crudo = json.loads(graph.map_path("warehouse").read_text(encoding=config.ENCODING))
        self.assertEqual(set(crudo), {"name", "directed", "positions", "adjacency"})


class TestGrafoSimple(unittest.TestCase):
    def setUp(self) -> None:
        self.grafo = graph.simple_graph()

    def test_es_el_de_la_guia(self) -> None:
        self.assertEqual(self.grafo.nodes(), ["A", "B", "C", "D", "E", "F"])
        self.assertEqual(self.grafo.adjacency, graph.SIMPLE_ADJACENCY)
        self.assertEqual(self.grafo.positions, graph.SIMPLE_POSITIONS)

    def test_costos_de_la_guia(self) -> None:
        self.assertEqual(self.grafo.cost("A", "D"), 4.0)
        self.assertEqual(self.grafo.cost("B", "E"), 3.0)
        self.assertEqual(self.grafo.cost("C", "F"), 4.0)

    def test_el_costo_no_tiene_por_que_ser_la_distancia(self) -> None:
        # A(0,0) -> D(0,3) mide 3 pero cuesta 4, y eso es valido a proposito.
        (ax, ay), (dx, dy) = self.grafo.positions["A"], self.grafo.positions["D"]
        self.assertEqual(math.dist((ax, ay), (dx, dy)), 3.0)
        self.assertEqual(self.grafo.cost("A", "D"), 4.0)
        self.grafo.validate()

    def test_neighbors_cost_y_has_edge(self) -> None:
        self.assertEqual(self.grafo.neighbors("B"), ["A", "C", "E"])
        self.assertTrue(self.grafo.has_edge("A", "B"))
        self.assertFalse(self.grafo.has_edge("A", "C"))
        self.assertFalse(self.grafo.has_edge("A", "Z"))
        self.assertFalse(self.grafo.has_edge("Z", "A"))

    def test_los_nodos_desconocidos_revientan_claro(self) -> None:
        with self.assertRaises(KeyError):
            self.grafo.neighbors("Z")
        with self.assertRaises(KeyError):
            self.grafo.cost("A", "C")

    def test_el_grafo_copia_lo_que_le_pasan(self) -> None:
        adyacencia = {"A": {"B": 1.0}, "B": {"A": 1.0}}
        grafo = graph.WarehouseGraph(adyacencia, {"A": (0.0, 0.0), "B": (1.0, 0.0)})
        adyacencia["A"]["B"] = 99.0
        self.assertEqual(grafo.cost("A", "B"), 1.0)


class TestAlmacen(unittest.TestCase):
    def setUp(self) -> None:
        self.grafo = graph.warehouse_graph()

    def test_tiene_trece_nodos_en_dos_corredores(self) -> None:
        self.assertEqual(len(self.grafo.nodes()), 13)
        sur = [n for n in self.grafo.nodes() if n.startswith("S")]
        norte = [n for n in self.grafo.nodes() if n.startswith("N")]
        self.assertEqual(len(sur), 6)
        self.assertEqual(len(norte), 6)
        # Cada corredor es una linea horizontal: misma y para todos sus nodos.
        self.assertEqual({self.grafo.positions[n][1] for n in sur}, {0.0})
        self.assertEqual({self.grafo.positions[n][1] for n in norte}, {8.0})

    def test_hay_conexiones_verticales(self) -> None:
        for a, b in (("S1", "N1"), ("S3", "N3"), ("S4", "N4"), ("S6", "N6")):
            with self.subTest(vertical=(a, b)):
                self.assertTrue(self.grafo.has_edge(a, b))
                self.assertTrue(self.grafo.has_edge(b, a))

    def test_quitar_el_nodo_G_parte_el_almacen_en_dos(self) -> None:
        # Esta es la propiedad que hace de G un cuello de botella de verdad:
        # es nodo de articulacion, no solo un nodo muy transitado.
        sin_g = graph.WarehouseGraph(
            {
                nodo: {v: c for v, c in vecinos.items() if v != graph.BOTTLENECK}
                for nodo, vecinos in self.grafo.adjacency.items()
                if nodo != graph.BOTTLENECK
            },
            {n: p for n, p in self.grafo.positions.items() if n != graph.BOTTLENECK},
            name="sin-G",
        )
        izquierda = graph._alcanzables("S1", sin_g.adjacency)
        self.assertEqual(izquierda, {"S1", "S2", "S3", "N1", "N2", "N3"})
        self.assertNotIn("S4", izquierda)
        self.assertIn("desconectado", motivo_del_fallo(sin_g))

    def test_G_es_el_unico_puente_entre_las_dos_zonas(self) -> None:
        izquierda = {"S1", "S2", "S3", "N1", "N2", "N3"}
        derecha = {"S4", "S5", "S6", "N4", "N5", "N6"}

        # Ninguna arista salta de una zona a la otra directamente...
        cruces = [
            (nodo, vecino)
            for nodo in self.grafo.nodes()
            for vecino in self.grafo.neighbors(nodo)
            if (nodo in izquierda and vecino in derecha)
            or (nodo in derecha and vecino in izquierda)
        ]
        self.assertEqual(cruces, [])

        # ...y G, que no es de ninguna zona, toca las dos: es el unico paso.
        vecinos_de_g = set(self.grafo.neighbors(graph.BOTTLENECK))
        self.assertTrue(vecinos_de_g & izquierda)
        self.assertTrue(vecinos_de_g & derecha)
        self.assertEqual(vecinos_de_g, {"S3", "N3", "S4", "N4"})

    def test_el_almacen_entero_sigue_conectado(self) -> None:
        self.assertEqual(
            graph._alcanzables("S1", self.grafo.adjacency), set(self.grafo.nodes())
        )


class TestValidate(unittest.TestCase):
    def test_un_grafo_bueno_no_dice_nada(self) -> None:
        self.assertIsNone(grafo_de_prueba().validate())

    def test_detecta_el_grafo_vacio(self) -> None:
        self.assertIn("no tiene nodos", motivo_del_fallo(grafo_de_prueba(adjacency={}, positions={})))

    def test_detecta_un_nodo_sin_posicion(self) -> None:
        motivo = motivo_del_fallo(
            grafo_de_prueba(positions={"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0)})
        )
        self.assertIn("'D' no tiene posicion", motivo)

    def test_detecta_una_posicion_huerfana(self) -> None:
        posiciones = dict(grafo_de_prueba().positions, Z=(9.0, 9.0))
        self.assertIn("'Z'", motivo_del_fallo(grafo_de_prueba(positions=posiciones)))

    def test_detecta_la_asimetria(self) -> None:
        adyacencia = grafo_de_prueba().adjacency
        del adyacencia["B"]["A"]
        motivo = motivo_del_fallo(grafo_de_prueba(adjacency=adyacencia))
        self.assertIn("falta la arista inversa", motivo)
        self.assertIn("'B' -> 'A'", motivo)

    def test_detecta_la_inversa_con_otro_costo(self) -> None:
        adyacencia = grafo_de_prueba().adjacency
        adyacencia["B"]["A"] = 7.0
        motivo = motivo_del_fallo(grafo_de_prueba(adjacency=adyacencia))
        self.assertIn("la inversa cuesta", motivo)
        # El aviso sale una sola vez, no uno por cada sentido.
        self.assertEqual(motivo.count("la inversa cuesta"), 1)

    def test_detecta_un_costo_negativo(self) -> None:
        adyacencia = grafo_de_prueba().adjacency
        adyacencia["A"]["B"] = -1.0
        adyacencia["B"]["A"] = -1.0
        self.assertIn("costo negativo", motivo_del_fallo(grafo_de_prueba(adjacency=adyacencia)))

    def test_detecta_un_costo_que_no_es_finito(self) -> None:
        adyacencia = grafo_de_prueba().adjacency
        adyacencia["A"]["B"] = math.inf
        adyacencia["B"]["A"] = math.inf
        self.assertIn("no es finito", motivo_del_fallo(grafo_de_prueba(adjacency=adyacencia)))

    def test_detecta_un_grafo_desconectado(self) -> None:
        adyacencia = dict(grafo_de_prueba().adjacency, Y={"Z": 1.0}, Z={"Y": 1.0})
        posiciones = dict(grafo_de_prueba().positions, Y=(9.0, 9.0), Z=(9.0, 8.0))
        motivo = motivo_del_fallo(grafo_de_prueba(adjacency=adyacencia, positions=posiciones))
        self.assertIn("desconectado", motivo)

    def test_detecta_una_arista_colgante(self) -> None:
        adyacencia = grafo_de_prueba().adjacency
        adyacencia["A"]["Z"] = 1.0
        self.assertIn("un nodo que no existe", motivo_del_fallo(grafo_de_prueba(adjacency=adyacencia)))

    def test_detecta_un_bucle_propio(self) -> None:
        adyacencia = grafo_de_prueba().adjacency
        adyacencia["A"]["A"] = 1.0
        self.assertIn("arista a si mismo", motivo_del_fallo(grafo_de_prueba(adjacency=adyacencia)))

    def test_junta_todos_los_problemas_en_un_mensaje(self) -> None:
        adyacencia = grafo_de_prueba().adjacency
        adyacencia["A"]["B"] = -1.0
        adyacencia["B"]["A"] = -1.0
        adyacencia["C"]["C"] = 1.0
        motivo = motivo_del_fallo(
            grafo_de_prueba(adjacency=adyacencia, positions={"A": (0.0, 0.0)})
        )
        self.assertIn("problemas", motivo)
        for esperado in ("no tiene posicion", "costo negativo", "arista a si mismo"):
            self.assertIn(esperado, motivo)


class TestDirigido(unittest.TestCase):
    def test_con_directed_la_asimetria_es_legitima(self) -> None:
        # Un anillo de un solo sentido: pasillos de una direccion, sin vuelta atras.
        anillo = graph.WarehouseGraph(
            {"A": {"B": 1.0}, "B": {"C": 1.0}, "C": {"D": 1.0}, "D": {"A": 1.0}},
            {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)},
            name="anillo",
            directed=True,
        )
        anillo.validate()
        self.assertFalse(anillo.has_edge("B", "A"))

    def test_sin_directed_ese_mismo_anillo_es_invalido(self) -> None:
        anillo = graph.WarehouseGraph(
            {"A": {"B": 1.0}, "B": {"C": 1.0}, "C": {"D": 1.0}, "D": {"A": 1.0}},
            {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)},
        )
        self.assertIn("falta la arista inversa", motivo_del_fallo(anillo))

    def test_dirigido_exige_poder_volver(self) -> None:
        # Se llega a C desde A, pero de C no se sale: un AGV se quedaria ahi.
        callejon = graph.WarehouseGraph(
            {"A": {"B": 1.0}, "B": {"C": 1.0}, "C": {}},
            {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0)},
            name="callejon",
            directed=True,
        )
        self.assertIn("fuertemente conexo", motivo_del_fallo(callejon))

    def test_dirigido_exporta_los_dos_sentidos(self) -> None:
        dirigido = graph.WarehouseGraph(
            {"A": {"B": 1.0}, "B": {"A": 2.0}},
            {"A": (0.0, 0.0), "B": (1.0, 0.0)},
            directed=True,
        )
        self.assertEqual(len(dirigido.to_unity_dict()["edges"]), 2)
        self.assertEqual(len(grafo_de_prueba().to_unity_dict()["edges"]), 4)


class TestExportacionAUnity(unittest.TestCase):
    def test_usa_la_funcion_del_protocolo_y_no_una_copia(self) -> None:
        grafo = graph.simple_graph()
        with mock.patch.object(
            protocol, "to_unity", wraps=protocol.to_unity
        ) as conversion:
            exportado = grafo.to_unity_dict()

        self.assertEqual(conversion.call_count, len(grafo.nodes()))
        conversion.assert_any_call(*grafo.positions["D"])
        self.assertEqual(len(exportado["nodes"]), len(grafo.nodes()))

    def test_el_eje_y_logico_va_a_la_z_de_unity(self) -> None:
        nodos = {n["id"]: n for n in graph.simple_graph().to_unity_dict()["nodes"]}
        self.assertEqual((nodos["D"]["x"], nodos["D"]["z"]), (0.0, 3.0))
        self.assertEqual((nodos["C"]["x"], nodos["C"]["z"]), (4.0, 0.0))

    def test_la_altura_la_pone_unity(self) -> None:
        for nodo in graph.warehouse_graph().to_unity_dict()["nodes"]:
            self.assertEqual(nodo["y"], protocol.UNITY_Y)

    def test_cambiar_unity_scale_cambia_todas_las_coordenadas(self) -> None:
        grafo = graph.warehouse_graph()
        original = grafo.to_unity_dict()

        with mock.patch.object(config, "UNITY_SCALE", 4.0):
            escalado = grafo.to_unity_dict()

        self.assertEqual(escalado["scale"], 4.0)
        for antes, despues in zip(original["nodes"], escalado["nodes"]):
            self.assertEqual(despues["id"], antes["id"])
            self.assertEqual(despues["x"], antes["x"] * 4.0)
            self.assertEqual(despues["z"], antes["z"] * 4.0)
            self.assertEqual(despues["y"], protocol.UNITY_Y)

    def test_las_aristas_llevan_los_nodos_por_id_y_su_costo(self) -> None:
        exportado = graph.simple_graph().to_unity_dict()
        identificadores = {nodo["id"] for nodo in exportado["nodes"]}
        for arista in exportado["edges"]:
            self.assertIn(arista["from"], identificadores)
            self.assertIn(arista["to"], identificadores)
            self.assertGreater(arista["cost"], 0.0)

    def test_no_dirigido_exporta_cada_tramo_una_vez(self) -> None:
        exportado = graph.warehouse_graph().to_unity_dict()
        pares = [(a["from"], a["to"]) for a in exportado["edges"]]
        self.assertEqual(len(pares), len(set(pares)))
        self.assertEqual(len(pares), len(graph.WAREHOUSE_EDGES))


class TestFicheros(unittest.TestCase):
    def setUp(self) -> None:
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        self.carpeta = Path(carpeta.name)

    def test_guardar_y_cargar_da_el_mismo_grafo(self) -> None:
        for nombre, constructor in graph.BUILTIN_MAPS.items():
            with self.subTest(mapa=nombre):
                original = constructor()
                ruta = self.carpeta / f"{nombre}.json"
                graph.save_graph(original, ruta)
                self.assertEqual(graph.load_graph(ruta), original)

    def test_guardar_crea_la_carpeta_que_falte(self) -> None:
        ruta = self.carpeta / "sin" / "crear" / "simple.json"
        graph.save_graph(graph.simple_graph(), ruta)
        self.assertTrue(ruta.is_file())

    def test_el_json_se_puede_editar_a_mano(self) -> None:
        ruta = self.carpeta / "simple.json"
        graph.save_graph(graph.simple_graph(), ruta)
        texto = ruta.read_text(encoding=config.ENCODING)
        self.assertIn("\n", texto.strip())  # indentado, no una sola linea
        self.assertTrue(texto.endswith("\n"))
        # Cada posicion entera en una linea, que es como se lee de un vistazo.
        self.assertIn('"D": [0.0, 3.0]', texto)

    def test_el_json_lleva_las_claves_en_orden_de_lectura(self) -> None:
        ruta = self.carpeta / "simple.json"
        graph.save_graph(graph.simple_graph(), ruta)
        texto = ruta.read_text(encoding=config.ENCODING)
        orden = [texto.index(f'"{clave}"') for clave in ("name", "directed", "positions")]
        self.assertEqual(orden, sorted(orden))
        self.assertLess(texto.index('"positions"'), texto.index('"adjacency"'))

    def test_cargar_un_mapa_roto_no_revienta_hasta_validate(self) -> None:
        # A proposito: hay que poder cargarlo para que validate() lo explique.
        ruta = self.carpeta / "roto.json"
        ruta.write_text(
            json.dumps({"positions": {"A": [0, 0]}, "adjacency": {"A": {"B": 1}}}),
            encoding=config.ENCODING,
        )
        roto = graph.load_graph(ruta)
        self.assertIn("un nodo que no existe", motivo_del_fallo(roto))

    def test_map_path_usa_la_carpeta_de_config(self) -> None:
        self.assertEqual(graph.map_path("simple"), config.MAPS_DIR / "simple.json")

    def test_el_nombre_sale_del_fichero_si_no_viene_dentro(self) -> None:
        ruta = self.carpeta / "mi_mapa.json"
        ruta.write_text(
            json.dumps({"positions": {"A": [0, 0]}, "adjacency": {"A": {}}}),
            encoding=config.ENCODING,
        )
        self.assertEqual(graph.load_graph(ruta).name, "mi_mapa")

    def test_un_fichero_que_no_existe_da_un_error_claro(self) -> None:
        with self.assertRaises(graph.GraphError) as capturado:
            graph.load_graph(self.carpeta / "no_estoy.json")
        self.assertIn("no se pudo leer", str(capturado.exception))

    def test_un_fichero_que_no_es_json_da_un_error_claro(self) -> None:
        ruta = self.carpeta / "malo.json"
        ruta.write_text("esto no es json", encoding=config.ENCODING)
        with self.assertRaises(graph.GraphError) as capturado:
            graph.load_graph(ruta)
        self.assertIn("no es JSON valido", str(capturado.exception))

    def test_un_json_con_mala_forma_da_un_error_claro(self) -> None:
        casos = {
            "sin adjacency": {"positions": {"A": [0, 0]}},
            "sin positions": {"adjacency": {"A": {}}},
            "posicion de tres": {"adjacency": {"A": {}}, "positions": {"A": [0, 0, 0]}},
            "posicion no numerica": {"adjacency": {"A": {}}, "positions": {"A": ["x", 0]}},
            "costo no numerico": {"adjacency": {"A": {"B": "caro"}}, "positions": {"A": [0, 0]}},
            "vecinos no objeto": {"adjacency": {"A": []}, "positions": {"A": [0, 0]}},
        }
        for caso, crudo in casos.items():
            with self.subTest(caso=caso):
                ruta = self.carpeta / "caso.json"
                ruta.write_text(json.dumps(crudo), encoding=config.ENCODING)
                with self.assertRaises(graph.GraphError):
                    graph.load_graph(ruta)


if __name__ == "__main__":
    unittest.main()
