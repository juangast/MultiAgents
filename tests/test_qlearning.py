"""CRITERIOS DE LA FASE 6: el entorno de Q-Learning, sin entrenar todavia.

Lo que se demuestra aqui:

1. El espacio de estados es **chico**: 72 estados, no cientos de miles.
2. `get_local_state()` devuelve siempre una tupla del mismo tamaño y tipo,
   hasheable y con cada campo en su rango.
3. La recompensa sale de `config.py`, de una sola funcion y sin numeros sueltos.
4. La Q-table se guarda y se lee en JSON sin perder nada.
5. `QLearningPolicy` se intercambia con la baseline **sin tocar simulation.py**,
   y 200 ticks con ella no rompen nada ni la invariante del almacen.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent
import astar
import config
import conflicts
import graph
import qlearning
import simulation


class Vista:
    """Una simulacion de mentira: lo justo que mira `get_local_state()`.

    Existe para probar el estado sin montar una `Simulation` entera, que es lo
    que promete el Protocol `qlearning.SimulationView`.
    """

    def __init__(
        self,
        grafo: graph.WarehouseGraph,
        agentes: list[agent.Agent],
        ocupacion: dict[str, int] | None = None,
    ) -> None:
        self.graph = grafo
        self.agents = agentes
        self.occupancy = (
            dict(ocupacion)
            if ocupacion is not None
            else {uno.current_node: uno.id for uno in agentes}
        )


def agv(
    agent_id: int,
    grafo: graph.WarehouseGraph,
    origen: str,
    destino: str,
    *,
    state: str = agent.STATE_MOVING,
    wait_time: int = 0,
) -> agent.Agent:
    """Un AGV con su ruta de A* ya trazada, colocado en el origen."""
    uno = agent.Agent(agent_id, grafo, origen)
    uno.assign_task(origen, destino, task=agent_id)
    uno.state = state
    uno.wait_time = wait_time
    return uno


def con_ruta(agent_id: int, grafo: graph.WarehouseGraph, ruta: list[str]) -> agent.Agent:
    """Un AGV con la ruta puesta a mano, para medir buckets sin buscar mapas."""
    uno = agent.Agent(agent_id, grafo, ruta[0])
    uno.path = list(ruta)
    uno.target_node = ruta[-1]
    uno.state = agent.STATE_MOVING
    return uno


def cruce_frontal() -> tuple[graph.WarehouseGraph, agent.Agent, agent.Agent, Vista]:
    """Dos AGVs de frente en el mapa simple: 1 va de A a C, 2 va de B a A."""
    grafo = graph.simple_graph()
    uno = agv(1, grafo, "A", "C")
    dos = agv(2, grafo, "B", "A")
    return grafo, uno, dos, Vista(grafo, [uno, dos])


class TestEspacioDeEstados(unittest.TestCase):
    """CRITERIO 1: el espacio de estados es manejable y esta declarado."""

    def test_son_72_estados(self) -> None:
        self.assertEqual(qlearning.state_space_size(), 72)
        self.assertEqual(qlearning.state_space_size(), 2 * 2 * 3 * 3 * 2)

    def test_no_pasa_de_unos_cientos(self) -> None:
        # El limite del enunciado: si esto salta, el estado dejo de ser local.
        self.assertLess(qlearning.state_space_size(), 500)

    def test_hay_un_tamaño_por_campo(self) -> None:
        self.assertEqual(len(qlearning.STATE_FIELDS), len(qlearning.STATE_SIZES))
        self.assertEqual(len(qlearning.STATE_FIELDS), 5)
        for tamano in qlearning.STATE_SIZES:
            self.assertGreaterEqual(tamano, 2)

    def test_el_informe_dice_el_total(self) -> None:
        texto = "\n".join(qlearning.report_state_space())
        self.assertIn("72", texto)
        for campo in qlearning.STATE_FIELDS:
            self.assertIn(campo, texto)


class TestEstadoLocal(unittest.TestCase):
    """CRITERIO 2: la tupla del estado, campo por campo."""

    def test_es_una_tupla_de_cinco_enteros_hasheable(self) -> None:
        grafo, uno, _, vista = cruce_frontal()
        estado = qlearning.get_local_state(uno, vista)
        self.assertIsInstance(estado, tuple)
        self.assertEqual(len(estado), len(qlearning.STATE_FIELDS))
        for valor in estado:
            self.assertIsInstance(valor, int)
        self.assertEqual(hash(estado), hash(tuple(estado)))
        self.assertEqual({estado: 1}[estado], 1)

    def test_cada_campo_se_queda_en_su_rango(self) -> None:
        grafo, uno, _, vista = cruce_frontal()
        estado = qlearning.get_local_state(uno, vista)
        for campo, valor, tamano in zip(
            qlearning.STATE_FIELDS, estado, qlearning.STATE_SIZES
        ):
            with self.subTest(campo=campo):
                self.assertIn(valor, range(tamano))

    def test_un_agv_sin_ruta_no_es_un_caso_especial(self) -> None:
        grafo = graph.simple_graph()
        parado = agent.Agent(9, grafo, "A")
        estado = qlearning.get_local_state(parado, Vista(grafo, [parado]))
        self.assertEqual(estado, (0, 0, 0, 0, 1))

    def test_el_nodo_siguiente_ocupado_se_ve(self) -> None:
        grafo, uno, _, vista = cruce_frontal()
        self.assertEqual(uno.next_node(), "B")
        self.assertEqual(qlearning.get_local_state(uno, vista)[0], 1)

        libre = Vista(grafo, [uno], ocupacion={"A": 1})
        self.assertEqual(qlearning.get_local_state(uno, libre)[0], 0)

    def test_el_cruce_de_frente_lo_ven_los_dos(self) -> None:
        _, uno, dos, vista = cruce_frontal()
        self.assertEqual(qlearning.get_local_state(uno, vista)[1], 1)
        self.assertEqual(qlearning.get_local_state(dos, vista)[1], 1)

    def test_dos_que_no_se_cruzan_no_tienen_edge_conflict(self) -> None:
        grafo = graph.simple_graph()
        uno = agv(1, grafo, "A", "C")
        dos = agv(2, grafo, "D", "F")
        vista = Vista(grafo, [uno, dos])
        self.assertEqual(qlearning.get_local_state(uno, vista)[1], 0)

    def test_la_cola_cuenta_a_los_que_esperan_en_los_dos_nodos_siguientes(self) -> None:
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6")
        delante = uno.path[1:3]
        esperando = [
            agv(numero, grafo, nodo, "N6", state=agent.STATE_WAITING)
            for numero, nodo in enumerate(delante, start=2)
        ]
        vista = Vista(grafo, [uno, *esperando])
        self.assertEqual(qlearning.get_local_state(uno, vista)[2], 2)

    def test_los_que_no_esperan_no_hacen_cola(self) -> None:
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6")
        moviendose = agv(2, grafo, uno.path[1], "N6", state=agent.STATE_MOVING)
        vista = Vista(grafo, [uno, moviendose])
        self.assertEqual(qlearning.get_local_state(uno, vista)[2], 0)

    def test_la_cola_satura_en_dos(self) -> None:
        # Tres AGVs delante no caben en el almacen de verdad (un nodo, un dueño),
        # asi que la saturacion se prueba aqui, sobre una lista fabricada.
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6")
        amontonados = [
            agv(numero, grafo, uno.path[1], "N6", state=agent.STATE_WAITING)
            for numero in (2, 3, 4)
        ]
        self.assertEqual(qlearning._cola_delante(uno, amontonados), qlearning.QUEUE_CAP)

    def test_el_bucket_de_distancia_cuenta_nodos_no_metros(self) -> None:
        grafo = graph.warehouse_graph()
        cerca = con_ruta(1, grafo, [f"n{i}" for i in range(config.DISTANCE_NEAR_NODES + 1)])
        medio = con_ruta(2, grafo, [f"n{i}" for i in range(config.DISTANCE_MID_NODES + 1)])
        lejos = con_ruta(3, grafo, [f"n{i}" for i in range(config.DISTANCE_MID_NODES + 5)])
        self.assertEqual(qlearning.distance_bucket(cerca), 0)
        self.assertEqual(qlearning.distance_bucket(medio), 1)
        self.assertEqual(qlearning.distance_bucket(lejos), 2)

    def test_el_bucket_baja_segun_avanza_la_ruta(self) -> None:
        grafo = graph.warehouse_graph()
        uno = con_ruta(1, grafo, [f"n{i}" for i in range(config.DISTANCE_MID_NODES + 5)])
        buckets = []
        while uno.path_index < len(uno.path) - 1:
            buckets.append(qlearning.distance_bucket(uno))
            uno.path_index += 1
        self.assertEqual(buckets, sorted(buckets, reverse=True))
        self.assertEqual(qlearning.distance_bucket(uno), 0)

    def test_la_prioridad_es_del_id_menor(self) -> None:
        _, uno, dos, vista = cruce_frontal()
        self.assertEqual(qlearning.get_local_state(uno, vista)[4], 1)
        self.assertEqual(qlearning.get_local_state(dos, vista)[4], 0)

    def test_sin_rivales_tambien_hay_prioridad(self) -> None:
        grafo = graph.simple_graph()
        solo = agv(7, grafo, "A", "C")
        self.assertEqual(qlearning.get_local_state(solo, Vista(grafo, [solo]))[4], 1)

    def test_dos_que_piden_el_mismo_nodo_son_rivales(self) -> None:
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S3", "G")
        dos = agv(2, grafo, "S4", "G")
        self.assertEqual(uno.next_node(), "G")
        self.assertEqual(dos.next_node(), "G")
        vista = Vista(grafo, [uno, dos])
        self.assertEqual(qlearning.get_local_state(uno, vista)[4], 1)
        self.assertEqual(qlearning.get_local_state(dos, vista)[4], 0)

    def test_el_estado_es_el_mismo_en_la_misma_situacion(self) -> None:
        primero = qlearning.get_local_state(*_situacion())
        segundo = qlearning.get_local_state(*_situacion())
        self.assertEqual(primero, segundo)


def _situacion() -> tuple[agent.Agent, Vista]:
    """El cruce frontal, montado desde cero cada vez."""
    _, uno, _, vista = cruce_frontal()
    return uno, vista


class TestEstadoDesdeElMotor(unittest.TestCase):
    """El camino de respaldo: el estado sacado del LocalState de la fase 5."""

    def local(
        self, uno: agent.Agent, vista: Vista, choques: tuple[conflicts.Conflict, ...] = ()
    ) -> conflicts.LocalState:
        return conflicts.LocalState(
            step=1,
            node=uno.current_node,
            intent=uno.next_node(),
            wait_time=uno.wait_time,
            blocked_by=(),
            conflicts=choques,
            occupancy=conflicts.read_only(vista.occupancy),
            neighbors=tuple(vista.graph.neighbors(uno.current_node)),
        )

    def test_tiene_la_misma_forma_que_el_estado_completo(self) -> None:
        _, uno, _, vista = cruce_frontal()
        estado = qlearning.state_from_local(uno, self.local(uno, vista))
        self.assertEqual(len(estado), len(qlearning.STATE_FIELDS))
        for campo, valor, tamano in zip(
            qlearning.STATE_FIELDS, estado, qlearning.STATE_SIZES
        ):
            with self.subTest(campo=campo):
                self.assertIn(valor, range(tamano))

    def test_coincide_con_el_completo_en_los_campos_exactos(self) -> None:
        _, uno, dos, vista = cruce_frontal()
        choque = conflicts.Conflict(conflicts.TYPE_EDGE, (1, 2), 1, edge=("A", "B"))
        completo = qlearning.get_local_state(uno, vista)
        aproximado = qlearning.state_from_local(uno, self.local(uno, vista, (choque,)))
        # Todos menos queue_ahead, que ahi es una aproximacion documentada.
        for indice in (0, 1, 3, 4):
            with self.subTest(campo=qlearning.STATE_FIELDS[indice]):
                self.assertEqual(completo[indice], aproximado[indice])


class TestAcciones(unittest.TestCase):
    """Las tres acciones y su traduccion a lo que el motor entiende."""

    def test_son_tres_y_en_este_orden(self) -> None:
        self.assertEqual(
            [accion.value for accion in qlearning.ACTIONS],
            ["advance", "wait", "reroute"],
        )

    def test_advance_es_go_y_las_otras_dos_esperan(self) -> None:
        self.assertEqual(
            qlearning.to_engine_action(qlearning.Action.ADVANCE), conflicts.ACTION_GO
        )
        self.assertEqual(
            qlearning.to_engine_action(qlearning.Action.WAIT), conflicts.ACTION_WAIT
        )
        self.assertEqual(
            qlearning.to_engine_action(qlearning.Action.REROUTE), conflicts.ACTION_WAIT
        )

    def test_el_flag_deja_fuera_el_reroute(self) -> None:
        self.assertEqual(len(qlearning.enabled_actions(True)), 3)
        self.assertEqual(
            qlearning.enabled_actions(False),
            (qlearning.Action.ADVANCE, qlearning.Action.WAIT),
        )

    def test_el_flag_sale_de_config(self) -> None:
        self.assertIsInstance(config.ENABLE_REROUTE, bool)
        with mock.patch.object(config, "ENABLE_REROUTE", False):
            self.assertNotIn(qlearning.Action.REROUTE, qlearning.enabled_actions())
        with mock.patch.object(config, "ENABLE_REROUTE", True):
            self.assertIn(qlearning.Action.REROUTE, qlearning.enabled_actions())

    def test_el_flag_no_cambia_la_tabla(self) -> None:
        # La fila guarda siempre las tres acciones: apagar el flag no obliga a
        # migrar una tabla ya entrenada.
        with mock.patch.object(config, "ENABLE_REROUTE", False):
            tabla = qlearning.QTable()
            self.assertEqual(set(tabla[(0, 0, 0, 0, 1)]), set(qlearning.ACTIONS))


class TestRecompensa(unittest.TestCase):
    """CRITERIO 3: una sola funcion y los numeros en config.py."""

    def test_los_seis_eventos_valen_lo_que_dice_el_enunciado(self) -> None:
        self.assertEqual(qlearning.reward(qlearning.Event.TASK_COMPLETE), 100.0)
        self.assertEqual(qlearning.reward(qlearning.Event.PROGRESS), 2.0)
        self.assertEqual(qlearning.reward(qlearning.Event.WAIT), -1.0)
        self.assertEqual(qlearning.reward(qlearning.Event.CONFLICT), -20.0)
        self.assertEqual(qlearning.reward(qlearning.Event.DEADLOCK), -50.0)
        self.assertEqual(qlearning.reward(qlearning.Event.USELESS_REROUTE), -3.0)

    def test_acepta_la_cadena_del_evento(self) -> None:
        self.assertEqual(qlearning.reward("wait"), qlearning.reward(qlearning.Event.WAIT))

    def test_un_evento_que_no_existe_revienta(self) -> None:
        with self.assertRaises(ValueError):
            qlearning.reward("premio_gordo")

    def test_se_ajusta_desde_config_sin_tocar_el_codigo(self) -> None:
        with mock.patch.object(config, "REWARD_WAIT", -7.5):
            self.assertEqual(qlearning.reward(qlearning.Event.WAIT), -7.5)
        self.assertEqual(qlearning.reward(qlearning.Event.WAIT), -1.0)

    def test_todos_los_eventos_tienen_valor(self) -> None:
        for evento in qlearning.Event:
            with self.subTest(evento=evento.value):
                self.assertIsInstance(qlearning.reward(evento), float)


class TestReroute(unittest.TestCase):
    """La tercera accion: recalcular A* penalizando lo que hay delante."""

    def test_las_penalizaciones_encarecen_el_nodo_y_el_tramo(self) -> None:
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6")
        castigo = qlearning.reroute_penalties(uno)
        self.assertEqual(castigo[uno.next_node()], config.REROUTE_PENALTY)
        self.assertEqual(
            castigo[(uno.current_node, uno.next_node())], config.REROUTE_PENALTY
        )

    def test_un_agv_sin_ruta_no_penaliza_nada(self) -> None:
        grafo = graph.simple_graph()
        self.assertEqual(qlearning.reroute_penalties(agent.Agent(1, grafo, "A")), {})

    def test_la_ruta_nueva_sale_de_donde_esta_y_llega_al_destino(self) -> None:
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6")
        for _ in range(2):
            uno.path_index += 1
            uno.current_node = uno.path[uno.path_index]

        nueva = qlearning.reroute(uno, grafo)
        self.assertIsNotNone(nueva)
        self.assertEqual(nueva[0], uno.current_node)
        self.assertEqual(nueva[-1], "N6")
        self.assertEqual(uno.path_index, 0)
        for anterior, siguiente in zip(nueva, nueva[1:]):
            self.assertTrue(grafo.has_edge(anterior, siguiente))

    def test_no_borra_el_reloj_de_la_espera_ni_la_tarea(self) -> None:
        # `assign_task()` reinicia `wait_time`, y esa es la medida con la que se
        # comparan las politicas: un reroute no puede ponerla a cero.
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6", wait_time=13)
        uno.state = agent.STATE_WAITING
        qlearning.reroute(uno, grafo)
        self.assertEqual(uno.wait_time, 13)
        self.assertEqual(uno.task, 1)
        self.assertEqual(uno.target_node, "N6")
        self.assertEqual(uno.state, agent.STATE_WAITING)

    def test_no_se_recalcula_a_media_travesia(self) -> None:
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6")
        uno.progress = 0.5
        self.assertIsNone(qlearning.reroute(uno, grafo))

    def test_la_penalizacion_cambia_la_ruta_cuando_hay_por_donde(self) -> None:
        grafo = graph.warehouse_graph()
        uno = agv(1, grafo, "S1", "N6")
        vieja = list(uno.path)
        castigo = {nodo: 1000.0 for nodo in vieja[1:-1]}
        nueva = qlearning.reroute(uno, grafo, penalties=castigo)
        self.assertIsNotNone(nueva)
        self.assertNotEqual(nueva, vieja)

    def test_un_reroute_que_no_mejora_nada_es_innecesario(self) -> None:
        grafo = graph.warehouse_graph()
        ruta = astar.astar(grafo, "S1", "N6")
        self.assertFalse(
            qlearning.is_useless_reroute(grafo, ruta, ruta, avoided_conflict=True)
        )
        self.assertTrue(qlearning.is_useless_reroute(grafo, ruta, ruta))

    def test_un_reroute_mas_barato_nunca_es_innecesario(self) -> None:
        # En el mapa simple A -> F tiene dos rutas de costos distintos: la de
        # A* (7) y la que pasa por C (8). El almacen no vale para esto porque
        # todo pasa por el cuello de botella G y no hay ruta alternativa.
        grafo = graph.simple_graph()
        corta = astar.astar(grafo, "A", "F")
        larga = ["A", "B", "C", "F"]
        self.assertGreater(astar.path_cost(grafo, larga), astar.path_cost(grafo, corta))
        self.assertFalse(qlearning.is_useless_reroute(grafo, larga, corta))
        self.assertTrue(qlearning.is_useless_reroute(grafo, corta, larga))


class TestQTable(unittest.TestCase):
    """CRITERIO 4: la tabla, su defaultdict y su JSON."""

    def test_un_estado_nuevo_nace_a_ceros(self) -> None:
        tabla = qlearning.QTable()
        self.assertEqual(len(tabla), 0)
        fila = tabla[(1, 0, 2, 1, 0)]
        self.assertEqual(fila, {accion: 0.0 for accion in qlearning.ACTIONS})
        self.assertEqual(len(tabla), 1)

    def test_contains_no_crea_la_fila(self) -> None:
        tabla = qlearning.QTable()
        self.assertNotIn((0, 0, 0, 0, 0), tabla)
        self.assertEqual(len(tabla), 0)

    def test_se_escribe_y_se_lee_una_celda(self) -> None:
        tabla = qlearning.QTable()
        tabla.set_value((0, 1, 0, 2, 1), qlearning.Action.WAIT, 3.5)
        self.assertEqual(tabla.value((0, 1, 0, 2, 1), qlearning.Action.WAIT), 3.5)
        self.assertEqual(tabla.value((0, 1, 0, 2, 1), qlearning.Action.ADVANCE), 0.0)

    def test_con_la_tabla_a_ceros_gana_advance(self) -> None:
        tabla = qlearning.QTable()
        self.assertIs(tabla.best_action((0, 0, 0, 0, 1)), qlearning.Action.ADVANCE)

    def test_la_mejor_accion_es_la_de_mas_valor(self) -> None:
        tabla = qlearning.QTable()
        estado = (1, 1, 2, 0, 0)
        tabla.set_value(estado, qlearning.Action.WAIT, 9.0)
        self.assertIs(tabla.best_action(estado), qlearning.Action.WAIT)
        self.assertEqual(tabla.best_value(estado), 9.0)

    def test_se_puede_elegir_entre_un_subconjunto(self) -> None:
        tabla = qlearning.QTable()
        estado = (1, 1, 2, 0, 0)
        tabla.set_value(estado, qlearning.Action.REROUTE, 9.0)
        self.assertIs(tabla.best_action(estado), qlearning.Action.REROUTE)
        self.assertIs(
            tabla.best_action(estado, among=qlearning.enabled_actions(False)),
            qlearning.Action.ADVANCE,
        )

    def test_la_clave_del_json_es_la_tupla_con_barras(self) -> None:
        self.assertEqual(qlearning.encode_state((0, 1, 2, 1, 0)), "0|1|2|1|0")
        self.assertEqual(qlearning.decode_state("0|1|2|1|0"), (0, 1, 2, 1, 0))

    def test_la_clave_va_y_vuelve_para_los_72_estados(self) -> None:
        for primero in range(2):
            for segundo in range(2):
                for tercero in range(3):
                    for cuarto in range(3):
                        for quinto in range(2):
                            estado = (primero, segundo, tercero, cuarto, quinto)
                            texto = qlearning.encode_state(estado)
                            self.assertEqual(qlearning.decode_state(texto), estado)

    def test_una_clave_con_otra_forma_revienta(self) -> None:
        with self.assertRaises(ValueError):
            qlearning.decode_state("0|1|2")
        with self.assertRaises(ValueError):
            qlearning.decode_state("0|1|2|1|x")
        with self.assertRaises(ValueError):
            qlearning.encode_state((0, 1))

    def test_guardar_y_cargar_no_pierde_nada(self) -> None:
        tabla = qlearning.QTable()
        tabla.set_value((0, 1, 2, 1, 0), qlearning.Action.ADVANCE, 1.5)
        tabla.set_value((0, 1, 2, 1, 0), qlearning.Action.WAIT, -0.25)
        tabla.set_value((1, 0, 0, 2, 1), qlearning.Action.REROUTE, 7.0)

        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "sub" / "qtable.json"
            tabla.save(destino)
            self.assertTrue(destino.is_file())

            crudo = json.loads(destino.read_text(encoding="utf-8"))
            self.assertEqual(crudo["format"], qlearning.FORMAT)
            self.assertEqual(crudo["state_fields"], list(qlearning.STATE_FIELDS))
            self.assertEqual(crudo["q"]["0|1|2|1|0"]["advance"], 1.5)

            leida = qlearning.QTable.load(destino)

        self.assertEqual(len(leida), len(tabla))
        for estado in tabla:
            self.assertEqual(leida[estado], tabla[estado])

    def test_no_carga_una_tabla_de_otro_formato(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "vieja.json"
            destino.write_text(json.dumps({"format": "otro/9", "q": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                qlearning.QTable.load(destino)

    def test_no_carga_una_tabla_con_otros_campos_de_estado(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "otra.json"
            destino.write_text(
                json.dumps(
                    {
                        "format": qlearning.FORMAT,
                        "state_fields": ["has_priority", "edge_conflict"],
                        "q": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                qlearning.QTable.load(destino)

    def test_no_carga_una_accion_que_no_existe(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "rara.json"
            destino.write_text(
                json.dumps(
                    {
                        "format": qlearning.FORMAT,
                        "state_fields": list(qlearning.STATE_FIELDS),
                        "q": {"0|0|0|0|0": {"volar": 1.0}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                qlearning.QTable.load(destino)


class TestPoliticaIntercambiable(unittest.TestCase):
    """CRITERIO 5: entra en el hueco de la baseline sin tocar el motor."""

    def revisa(self, simulacion: simulation.Simulation) -> None:
        """La invariante del almacen: un nodo, un AGV."""
        nodos = [agente.current_node for agente in simulacion.agents]
        self.assertEqual(
            len(set(nodos)),
            len(nodos),
            f"dos AGVs en el mismo nodo en el paso {simulacion.step}: {nodos}",
        )

    def test_cumple_el_contrato_de_la_fase_5(self) -> None:
        self.assertIsInstance(qlearning.QLearningPolicy(), conflicts.Policy)
        self.assertEqual(qlearning.QLearningPolicy().name, "qlearning")

    def test_entra_por_el_constructor_como_la_baseline(self) -> None:
        simulacion = simulation.Simulation(
            graph.simple_graph(), 1, policy=qlearning.QLearningPolicy()
        )
        self.assertEqual(simulacion.stats()["policy"], "qlearning")

    def test_devuelve_lo_que_el_motor_entiende(self) -> None:
        simulacion = simulation.Simulation(graph.simple_graph(), 2)
        politica = qlearning.QLearningPolicy(simulation=simulacion)
        uno = simulacion.agents[0]
        estado = conflicts.LocalState(
            step=1,
            node=uno.current_node,
            intent=uno.next_node(),
            wait_time=0,
            blocked_by=(),
            conflicts=(),
            occupancy=conflicts.read_only(simulacion.occupancy),
            neighbors=tuple(simulacion.graph.neighbors(uno.current_node)),
        )
        # Desde la fase 8 la politica declara una INTENCION de las tres, no
        # el `go`/`wait` que entiende el motor: quien traduce es `conflicts`.
        self.assertIn(politica.decide(uno, estado), conflicts.INTENTS)

    def test_con_la_tabla_a_ceros_siempre_avanza(self) -> None:
        simulacion = simulation.Simulation(graph.warehouse_graph(), 4)
        politica = qlearning.QLearningPolicy(simulation=simulacion)
        for estado in ((0, 0, 0, 0, 1), (1, 1, 2, 2, 0)):
            self.assertIs(politica.choose(estado), qlearning.Action.ADVANCE)

    def test_200_ticks_con_la_politica_dummy(self) -> None:
        politica = qlearning.QLearningPolicy()
        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 6, policy=politica
        )
        politica.bind(simulacion)

        for _ in range(200):
            simulacion.tick()
            self.revisa(simulacion)
            for agente in simulacion.agents:
                estado = qlearning.get_local_state(agente, simulacion)
                self.assertEqual(len(estado), len(qlearning.STATE_FIELDS))
                for campo, valor, tamano in zip(
                    qlearning.STATE_FIELDS, estado, qlearning.STATE_SIZES
                ):
                    with self.subTest(paso=simulacion.step, campo=campo):
                        self.assertIn(valor, range(tamano))

        self.assertEqual(simulacion.step, 200)
        self.assertEqual(simulacion.stats()["policy"], "qlearning")

    def test_200_ticks_sin_bind_tampoco_revientan(self) -> None:
        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 6, policy=qlearning.QLearningPolicy()
        )
        for _ in range(200):
            simulacion.tick()
            self.revisa(simulacion)

    def test_200_ticks_explorando_con_reroute_encendido(self) -> None:
        # Con epsilon las tres acciones salen, REROUTE incluido: es la unica que
        # toca el `path` en marcha, asi que aqui se comprueba que no lo rompe.
        politica = qlearning.QLearningPolicy(epsilon=0.3, seed=7)
        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 6, policy=politica
        )
        politica.bind(simulacion)

        for _ in range(200):
            simulacion.tick()
            self.revisa(simulacion)
            for agente in simulacion.agents:
                if not agente.path:
                    continue
                self.assertEqual(agente.path[agente.path_index], agente.current_node)
                self.assertEqual(agente.path[-1], agente.target_node)
                for anterior, siguiente in zip(agente.path, agente.path[1:]):
                    self.assertTrue(simulacion.graph.has_edge(anterior, siguiente))

        # Quien ejecuta el REROUTE es el motor desde la fase 8, asi que la
        # prueba de que salio alguno esta en su registro, no en la politica.
        self.assertGreater(
            simulacion.stats()["actions"]["reroute"],
            0,
            "con epsilon 0.3 y 200 ticks tenia que haber salido algun REROUTE",
        )

    def test_la_exploracion_es_reproducible(self) -> None:
        def corre() -> list[str]:
            politica = qlearning.QLearningPolicy(epsilon=0.5, seed=3)
            simulacion = simulation.Simulation(
                graph.warehouse_graph(), 6, policy=politica
            )
            politica.bind(simulacion)
            for _ in range(50):
                simulacion.tick()
            return [agente.current_node for agente in simulacion.agents]

        self.assertEqual(corre(), corre())

    def test_guarda_la_ultima_decision_de_cada_agv(self) -> None:
        politica = qlearning.QLearningPolicy()
        simulacion = simulation.Simulation(
            graph.warehouse_graph(), 4, policy=politica
        )
        politica.bind(simulacion)
        simulacion.tick()

        for agente in simulacion.agents:
            decision = politica.last_decision(agente.id)
            if decision is None:
                continue
            estado, accion = decision
            self.assertEqual(len(estado), len(qlearning.STATE_FIELDS))
            self.assertIn(accion, qlearning.ACTIONS)

    def test_el_reset_no_borra_lo_aprendido(self) -> None:
        politica = qlearning.QLearningPolicy()
        politica.q.set_value((0, 0, 0, 0, 1), qlearning.Action.WAIT, 4.0)
        politica.reset()
        self.assertEqual(politica.q.value((0, 0, 0, 0, 1), qlearning.Action.WAIT), 4.0)
        self.assertIsNone(politica.last_decision(1))


if __name__ == "__main__":
    unittest.main()
