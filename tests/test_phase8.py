"""CRITERIOS DE LA FASE 8: A* y Q-Learning corriendo juntos en el mismo tick.

Lo que se demuestra aqui, uno por criterio de aceptacion:

1. Con `policy="qlearning"` y la Q-table entrenada, la corrida completa todas
   las tareas.
2. Salen las tres acciones, no solo ADVANCE: hay WAIT y hay REROUTE.
3. `SET_MODE` cambia de politica en caliente y deja la corrida limpia.
4. Cero deadlocks permanentes: 10 corridas de 1000 ticks con 6 AGVs.
5. La politica es la **unica** variable: lo demas es identico en los dos modos.

Y por debajo, las piezas: las penalizaciones que caducan, la accion como
intencion y no como garantia, y los tres peldaños del desatasco.
"""

import unittest
from unittest import mock

import agent
import astar
import config
import conflicts
import graph
import protocol
import qlearning
import simulation

MODELO = config.Q_TABLE_FILE
hay_modelo = MODELO.is_file()
sin_modelo = unittest.skipUnless(
    hay_modelo, f"hace falta {MODELO}: python3 python/main.py train"
)


def warehouse() -> graph.WarehouseGraph:
    return graph.warehouse_graph()


class Quieta:
    """Una politica que no se mueve nunca. Es el caso del criterio 4."""

    name = "quieta"

    def decide(self, agent_, local_state) -> str:
        return conflicts.INTENT_WAIT


class Temeraria:
    """Una politica que siempre quiere pasar. El gate tiene que poder con ella."""

    name = "temeraria"

    def decide(self, agent_, local_state) -> str:
        return conflicts.INTENT_ADVANCE


# --- Las tres intenciones ----------------------------------------------------


class TestIntenciones(unittest.TestCase):
    """`advance`/`wait`/`reroute`, y la traduccion de las politicas de antes."""

    def test_son_tres(self) -> None:
        self.assertEqual(conflicts.INTENTS, ("advance", "wait", "reroute"))

    def test_el_go_de_la_fase_5_sigue_valiendo(self) -> None:
        # Es lo que hace que `BaselinePolicy` entre en la fase 8 sin tocarla.
        self.assertEqual(conflicts.normalize_intent(conflicts.ACTION_GO), "advance")
        self.assertEqual(conflicts.normalize_intent(conflicts.ACTION_WAIT), "wait")

    def test_aguanta_mayusculas_y_espacios(self) -> None:
        self.assertEqual(conflicts.normalize_intent("  ADVANCE "), "advance")

    def test_una_accion_inventada_no_se_traga_en_silencio(self) -> None:
        with self.assertRaises(ValueError) as roto:
            conflicts.normalize_intent("turbo")
        self.assertIn("turbo", str(roto.exception))

    def test_la_politica_de_qlearning_declara_las_tres(self) -> None:
        self.assertEqual(
            {accion.value for accion in qlearning.ACTIONS}, set(conflicts.INTENTS)
        )


# --- Las penalizaciones caducan ----------------------------------------------


class TestPenalizacionesTemporales(unittest.TestCase):
    """CRITERIO: el REROUTE penaliza, pero la penalizacion no es para siempre."""

    def test_encarece_y_caduca(self) -> None:
        castigos = astar.TemporaryPenalties(ttl=10)
        castigos.add("G", 5.0, step=3)
        self.assertEqual(castigos["G"], 5.0)
        self.assertEqual(castigos.expire(12), 0)
        self.assertEqual(castigos.expire(13), 1)
        self.assertEqual(len(castigos), 0)

    def test_repetir_acumula_y_refresca_el_reloj(self) -> None:
        castigos = astar.TemporaryPenalties(ttl=10)
        castigos.add("G", 5.0, step=0)
        castigos.add("G", 5.0, step=8)
        self.assertEqual(castigos["G"], 10.0)
        self.assertEqual(castigos.expire(10), 0, "el segundo add reinicio el reloj")
        self.assertEqual(castigos.expire(18), 1)

    def test_el_tope_impide_que_el_mapa_se_degrade_sin_fin(self) -> None:
        castigos = astar.TemporaryPenalties(ttl=10, cap=12.0)
        for _ in range(10):
            castigos.add("G", 5.0, step=0)
        self.assertEqual(castigos["G"], 12.0)

    def test_un_castigo_de_cero_no_ensucia_la_tabla(self) -> None:
        castigos = astar.TemporaryPenalties()
        castigos.add("G", 0.0, step=1)
        self.assertEqual(len(castigos), 0)

    def test_se_puede_retirar_antes_de_tiempo(self) -> None:
        castigos = astar.TemporaryPenalties()
        castigos.add("G", 5.0, step=1)
        castigos.discard("G")
        castigos.discard("no estaba")  # no lanza
        self.assertEqual(len(castigos), 0)

    def test_es_un_mapping_que_a_star_consume_tal_cual(self) -> None:
        grafo = warehouse()
        castigos = astar.TemporaryPenalties()
        self.assertEqual(astar.astar(grafo, "S1", "S3", castigos), ["S1", "S2", "S3"])

        castigos.add("S2", 30.0, step=1)
        self.assertEqual(
            astar.astar(grafo, "S1", "S3", castigos),
            ["S1", "N1", "N2", "N3", "S3"],
            "A* tiene que esquivar el nodo penalizado sin tocar el mapa",
        )

        castigos.expire(config.PENALTY_TTL + 1)
        self.assertEqual(
            astar.astar(grafo, "S1", "S3", castigos),
            ["S1", "S2", "S3"],
            "y volver a la ruta buena en cuanto la penalizacion caduca",
        )

    def test_el_veto_es_caro_pero_finito(self) -> None:
        # Con `inf` un nodo de paso obligado dejaria a A* sin ruta que devolver.
        grafo = warehouse()
        castigos = astar.TemporaryPenalties()
        castigos.ban("G", step=1)
        self.assertEqual(
            astar.astar(grafo, "S1", "S4", castigos),
            ["S1", "S2", "S3", "G", "S4"],
            "G es el unico paso: mas vale carisimo que ninguna ruta",
        )

    def test_la_simulacion_las_caduca_sola(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 1)
        simulacion.penalties.add("G", 5.0, step=simulacion.step)
        for _ in range(config.PENALTY_TTL + 1):
            simulacion.tick()
        self.assertEqual(len(simulacion.penalties), 0)


# --- El modo es un parametro -------------------------------------------------


class TestModo(unittest.TestCase):
    """La politica entra por nombre, y sale en el snapshot."""

    def test_por_defecto_sigue_siendo_la_baseline(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 2)
        self.assertEqual(simulacion.mode, "baseline")
        self.assertEqual(simulacion.get_snapshot()["mode"], "baseline")

    def test_el_nombre_monta_la_politica(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 2, policy="baseline")
        self.assertIsInstance(simulacion.policy, conflicts.BaselinePolicy)

    @sin_modelo
    def test_qlearning_por_nombre_carga_el_modelo_y_se_ata(self) -> None:
        simulacion = simulation.Simulation(
            warehouse(), 4, policy="qlearning", model=MODELO
        )
        self.assertEqual(simulacion.mode, "qlearning")
        self.assertEqual(simulacion.get_snapshot()["mode"], "qlearning")
        self.assertGreater(len(simulacion.policy.q), 0, "la Q-table venia vacia")
        # Atada: sin `bind()` el estado saldria del LocalState y `queue_ahead`
        # iria aproximado, o sea que no seria el estado con el que se entreno.
        self.assertIs(simulacion.policy._simulation, simulacion)

    def test_un_objeto_politica_sigue_entrando_por_el_constructor(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 2, policy=Quieta())
        self.assertEqual(simulacion.mode, "quieta")
        self.assertEqual(simulacion.stats()["policy"], "quieta")

    def test_una_politica_que_no_existe_lo_dice(self) -> None:
        with self.assertRaises(ValueError) as roto:
            simulation.Simulation(warehouse(), 2, policy="turbo")
        self.assertIn("turbo", str(roto.exception))

    def test_qlearning_sin_modelo_no_sirve_una_tabla_vacia(self) -> None:
        # Una Q-table a ceros siempre avanza: pareceria funcionar sin haber
        # aprendido nada, y eso es peor que un error.
        with self.assertRaises(ValueError) as roto:
            simulation.Simulation(
                warehouse(), 2, policy="qlearning", model="/no/existe.json"
            )
        self.assertIn("train", str(roto.exception))


class TestUnicaVariable(unittest.TestCase):
    """CRITERIO 5: entre los dos modos **solo** cambia la politica."""

    @sin_modelo
    def test_los_dos_modos_arrancan_con_el_mismo_escenario(self) -> None:
        def foto(modo: str) -> list[tuple]:
            simulacion = simulation.Simulation(
                warehouse(), 6, seed=7, policy=modo, model=MODELO
            )
            return [
                (a.id, a.current_node, a.target_node, tuple(a.path), a.task)
                for a in simulacion.agents
            ]

        self.assertEqual(foto("baseline"), foto("qlearning"))

    @sin_modelo
    def test_el_snapshot_tiene_la_misma_forma_en_los_dos_modos(self) -> None:
        # Unity recibe exactamente los mismos campos con una politica y con la
        # otra: lo que cambia es el valor de `mode` y lo que cada AGV decide,
        # nunca la forma de lo que llega por el cable.
        formas = []
        for modo in ("baseline", "qlearning"):
            simulacion = simulation.Simulation(
                warehouse(), 4, seed=7, policy=modo, model=MODELO
            )
            instantanea = simulacion.get_snapshot()
            self.assertEqual(instantanea["mode"], modo)
            formas.append(
                (
                    sorted(instantanea),
                    [sorted(uno) for uno in instantanea["agents"]],
                    sorted(instantanea["stats"]),
                )
            )
        self.assertEqual(formas[0], formas[1])

    def test_el_desatasco_es_del_motor_y_no_de_la_politica(self) -> None:
        # Si el desatasco solo corriera con una politica, la comparacion de la
        # fase 10 mediria el desatasco y no la politica. El mismo escenario
        # atascado con dos politicas distintas: las dos acaban desatascadas.
        for politica in ("baseline", Quieta()):
            simulacion = simulation.Simulation(
                graph.simple_graph(),
                routes=[("A", "C"), ("B", "A")],
                policy=politica,
            )
            for _ in range(config.DEADLOCK_TICKS):
                simulacion.tick()
            nombre = simulacion.mode
            self.assertGreater(
                simulacion.stats()["forced"], 0, f"la politica {nombre} no desatasco"
            )
            self.assertIsNone(
                simulacion.finished_reason, f"la politica {nombre} acabo en deadlock"
            )


# --- La accion en el snapshot ------------------------------------------------


class TestAccionEnElSnapshot(unittest.TestCase):
    """CRITERIO: Unity tiene que poder pintar lo que cada AGV esta decidiendo."""

    def test_todos_los_agentes_traen_accion_valida_en_cada_tick(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 6)
        for _ in range(120):
            instantanea = simulacion.get_snapshot()
            for uno in instantanea["agents"]:
                self.assertIn(uno["action"], conflicts.INTENTS)
                self.assertIsInstance(uno["blocked"], bool)

    def test_el_de_media_travesia_esta_ejecutando_un_avance(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 1)
        instantanea = simulacion.get_snapshot()
        uno = instantanea["agents"][0]
        self.assertGreater(simulacion.agents[0].progress, 0.0)
        self.assertEqual(uno["action"], "advance")

    def test_el_que_llego_no_esta_haciendo_nada(self) -> None:
        simulacion = simulation.Simulation(graph.simple_graph(), 1)
        for _ in range(40):
            simulacion.tick()
        self.assertEqual(simulacion.agents[0].state, agent.STATE_DONE)
        self.assertEqual(simulacion.get_snapshot()["agents"][0]["action"], "wait")

    def test_el_recuento_de_la_corrida_sale_en_stats(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 4)
        for _ in range(50):
            simulacion.tick()
        recuento = simulacion.stats()["actions"]
        self.assertEqual(set(recuento), set(conflicts.INTENTS))
        self.assertGreater(recuento["advance"], 0)
        simulacion.reset()
        self.assertEqual(simulacion.stats()["actions"]["advance"], 0)


class TestIntencionNoGarantia(unittest.TestCase):
    """CRITERIO 2 de la fase: ADVANCE es lo que quiero, no lo que va a pasar."""

    def test_el_motor_niega_el_paso_y_lo_deja_por_escrito(self) -> None:
        # Los dos quieren el nodo del otro: el motor desempata y el perdedor se
        # queda donde estaba, con `blocked` puesto. Ese es el -20 al entrenar.
        simulacion = simulation.Simulation(
            graph.simple_graph(), routes=[("A", "C"), ("B", "A")], policy=Temeraria()
        )
        simulacion.tick()
        registros = [simulacion.action_record(uno.id) for uno in simulacion.agents]
        self.assertTrue(all(r.action == "advance" for r in registros))
        self.assertTrue(
            any(r.blocked for r in registros),
            "los dos pidieron pasar y el motor no pudo darselo a los dos",
        )

    def test_una_politica_temeraria_no_rompe_la_invariante(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 6, policy=Temeraria())
        for _ in range(200):
            simulacion.tick()
            nodos = [uno.current_node for uno in simulacion.agents]
            self.assertEqual(len(set(nodos)), len(nodos), f"colision: {nodos}")


# --- El desatasco ------------------------------------------------------------


class TestDesatasco(unittest.TestCase):
    """CRITERIO 4: el sistema nunca se queda trabado."""

    def test_si_todos_esperan_el_motor_fuerza_el_paso(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 1, policy=Quieta())
        for _ in range(config.DEADLOCK_FORCE_TICKS):
            simulacion.tick()
        self.assertEqual(simulacion.agents[0].current_node, "S1", "todavia no toca")

        simulacion.tick()
        self.assertGreater(simulacion.stats()["forced"], 0)
        self.assertTrue(
            simulacion.action_record(1).forced,
            "el forzado queda marcado, que es lo que se cobra a -20",
        )
        self.assertGreater(simulacion.agents[0].progress, 0.0, "no se movio")

    def test_salta_antes_de_que_la_corrida_muera(self) -> None:
        self.assertLess(config.DEADLOCK_FORCE_TICKS, config.DEADLOCK_TICKS)

    def test_a_cero_el_desatasco_se_apaga(self) -> None:
        with mock.patch.object(config, "DEADLOCK_FORCE_TICKS", 0):
            simulacion = simulation.Simulation(warehouse(), 1, policy=Quieta())
            for _ in range(config.DEADLOCK_TICKS + 2):
                simulacion.tick()
            self.assertEqual(simulacion.stats()["forced"], 0)
            self.assertEqual(simulacion.finished_reason, simulation.FINISHED_DEADLOCK)

    def test_el_cara_a_cara_deja_de_ser_un_callejon_sin_salida(self) -> None:
        # El mismo escenario que en la fase 5 acababa en deadlock: uno de los dos
        # acaba rodeando en vez de morirse esperando.
        simulacion = simulation.Simulation(
            graph.simple_graph(), routes=[("A", "C"), ("B", "A")]
        )
        for _ in range(200):
            simulacion.tick()
            if simulacion.done:
                break
        self.assertIsNone(
            simulacion.finished_reason, "el cara a cara volvio a acabar en deadlock"
        )
        self.assertTrue(
            all(uno.state == agent.STATE_DONE for uno in simulacion.agents),
            [(uno.id, uno.state, uno.current_node) for uno in simulacion.agents],
        )

    def test_un_agv_que_termino_encima_de_G_se_aparta(self) -> None:
        # G es nodo de articulacion: aparcado ahi no hay ruta alternativa que
        # valga, y sin apartarlo el bloqueo seria para siempre.
        simulacion = simulation.Simulation(
            warehouse(), routes=[("S1", "G"), ("S6", "S3")]
        )
        for _ in range(300):
            simulacion.tick()
            if simulacion.done:
                break
        self.assertIsNone(simulacion.finished_reason)
        self.assertEqual(
            simulacion.agents[1].current_node, "S3", "el segundo nunca llego a cruzar"
        )
        self.assertGreater(simulacion.stats()["forced"], 0)

    def test_la_invariante_aguanta_con_el_desatasco_encendido(self) -> None:
        # El apartarse es una ruta de un tramo, no un movimiento especial: pasa
        # por el gate como cualquier otro y no puede meter dos AGVs en un nodo.
        simulacion = simulation.Simulation(warehouse(), 6)
        for _ in range(500):
            simulacion.tick()
            nodos = [uno.current_node for uno in simulacion.agents]
            self.assertEqual(len(set(nodos)), len(nodos), f"colision: {nodos}")
            for uno in simulacion.agents:
                self.assertEqual(simulacion.occupancy.get(uno.current_node), uno.id)

    def test_las_rutas_siguen_siendo_recorribles_despues_de_apartarse(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 6, seed=3)
        for _ in range(400):
            simulacion.tick()
            for uno in simulacion.agents:
                if not uno.path:
                    continue
                self.assertEqual(uno.path[uno.path_index], uno.current_node)
                for anterior, siguiente in zip(uno.path, uno.path[1:]):
                    self.assertTrue(simulacion.graph.has_edge(anterior, siguiente))


# --- SET_MODE ----------------------------------------------------------------


class TestSetMode(unittest.TestCase):
    """CRITERIO 3: cambiar de modo por socket, en caliente y sin reiniciar."""

    @sin_modelo
    def test_cambia_la_politica_y_arranca_una_corrida_limpia(self) -> None:
        simulacion = simulation.Simulation(
            warehouse(), 4, policy="baseline", model=MODELO
        )
        for _ in range(20):
            simulacion.tick()
        corrida = simulacion.run

        self.assertEqual(simulacion.set_mode("qlearning"), "qlearning")
        self.assertEqual(simulacion.mode, "qlearning")
        self.assertEqual(simulacion.step, 0)
        self.assertEqual(simulacion.run, corrida + 1)
        self.assertEqual(simulacion.stats()["conflicts"], 0)
        self.assertEqual(len(simulacion.penalties), 0)
        self.assertEqual(simulacion.get_snapshot()["mode"], "qlearning")

    def test_los_deadlocks_de_la_sesion_sobreviven_al_cambio(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 4)
        simulacion.deadlocks = 3
        simulacion.set_mode("baseline")
        self.assertEqual(simulacion.stats()["deadlocks"], 3)

    def test_un_modo_que_no_existe_no_toca_la_corrida(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 4)
        for _ in range(10):
            simulacion.tick()
        with self.assertRaises(ValueError):
            simulacion.set_mode("turbo")
        self.assertEqual(simulacion.mode, "baseline")
        self.assertEqual(simulacion.step, 10)

    @sin_modelo
    def test_el_comando_del_protocolo(self) -> None:
        simulacion = simulation.Simulation(
            warehouse(), 4, policy="baseline", model=MODELO
        )
        respuesta = protocol.handle_line("SET_MODE qlearning\n", simulacion)
        self.assertTrue(respuesta.endswith("\n"))
        self.assertEqual(respuesta.count("\n"), 1)
        self.assertIn('"ok":true', respuesta)
        self.assertIn('"mode":"qlearning"', respuesta)
        self.assertEqual(simulacion.mode, "qlearning")

    def test_no_distingue_mayusculas_ni_le_molesta_el_crlf(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 2)
        self.assertIn(
            '"ok":true', protocol.handle_line("set_mode BASELINE\r\n", simulacion)
        )

    def test_un_modo_desconocido_responde_y_sigue(self) -> None:
        simulacion = simulation.Simulation(warehouse(), 2)
        for linea in ("SET_MODE turbo", "SET_MODE"):
            respuesta = protocol.handle_line(linea, simulacion)
            self.assertIn(protocol.ERROR_BAD_MODE, respuesta)
            self.assertIn("baseline", respuesta)
        self.assertEqual(simulacion.mode, "baseline")

    def test_una_simulacion_que_no_cambia_de_modo_no_revienta(self) -> None:
        class Tonta:
            def get_snapshot(self):
                return {"step": 1}

            def reset(self):
                pass

        respuesta = protocol.handle_line("SET_MODE qlearning", Tonta())
        self.assertIn(protocol.ERROR_MODE_NOT_SUPPORTED, respuesta)

    def test_sin_modelo_lo_dice_en_vez_de_tumbar_al_cliente(self) -> None:
        simulacion = simulation.Simulation(
            warehouse(), 2, policy="baseline", model="/no/existe.json"
        )
        respuesta = protocol.handle_line("SET_MODE qlearning", simulacion)
        self.assertIn(protocol.ERROR_SET_MODE_FAILED, respuesta)
        self.assertEqual(simulacion.mode, "baseline")


# --- Los criterios de aceptacion, tal cual -----------------------------------


@sin_modelo
class TestCriteriosDeAceptacion(unittest.TestCase):
    """Los cuatro criterios de la fase, medidos como estan escritos."""

    def test_1_con_la_qtable_entrenada_se_completan_todas_las_tareas(self) -> None:
        simulacion = simulation.Simulation(
            warehouse(), 4, policy="qlearning", model=MODELO
        )
        for _ in range(600):
            simulacion.tick()
            if simulacion.done:
                break
        sin_terminar = [
            (uno.id, uno.state, uno.current_node, uno.target_node)
            for uno in simulacion.agents
            if uno.state != agent.STATE_DONE
        ]
        self.assertEqual(sin_terminar, [], f"en {simulacion.step} ticks")

    def test_2_se_ven_wait_y_reroute_y_no_solo_advance(self) -> None:
        simulacion = simulation.Simulation(
            warehouse(), 6, policy="qlearning", model=MODELO
        )
        vistas: set[str] = set()
        for _ in range(400):
            simulacion.tick()
            vistas.update(
                simulacion.action_record(uno.id).action for uno in simulacion.agents
            )
        recuento = simulacion.stats()["actions"]
        self.assertEqual(vistas, set(conflicts.INTENTS), f"recuento: {recuento}")
        for accion in conflicts.INTENTS:
            self.assertGreater(recuento[accion], 0, f"no salio ni un {accion}")

    def test_4_cero_deadlocks_permanentes(self) -> None:
        # 10 corridas de 1000 ticks con 6 AGVs. Las semillas van distintas: con
        # una sola, las diez corridas serian la misma corrida diez veces.
        for semilla in range(10):
            simulacion = simulation.Simulation(
                warehouse(), 6, seed=semilla, policy="qlearning", model=MODELO
            )
            for _ in range(1000):
                simulacion.tick()
                self.assertIsNone(
                    simulacion.finished_reason,
                    f"deadlock en la semilla {semilla}, paso {simulacion.step}",
                )
            self.assertEqual(simulacion.deadlocks, 0)

    def test_4_bis_tampoco_se_traba_la_baseline_con_el_mismo_motor(self) -> None:
        # El desatasco es del motor, no de la politica: si solo salvara al
        # Q-Learning, la comparacion de la fase 10 no mediria la politica.
        for semilla in range(10):
            simulacion = simulation.Simulation(warehouse(), 6, seed=semilla)
            for _ in range(1000):
                simulacion.tick()
            self.assertEqual(
                simulacion.deadlocks, 0, f"la baseline se trabo con la semilla {semilla}"
            )


if __name__ == "__main__":
    unittest.main()
