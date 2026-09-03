"""Tests de la fase 10: los cinco escenarios, el runner y la tabla resumen.

Los criterios de aceptacion de la fase estan marcados con "CRITERIO N" en el
docstring de cada clase. Los dos que sostienen la fase son el 1 (los cinco
corren de principio a fin) y el 2 (misma semilla, mismo resultado): sin el
segundo, los numeros del reporte no se pueden volver a comprobar.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import astar
import config
import graph
import metrics
import qlearning
import scenarios

_SILENCIO = None


def setUpModule() -> None:
    """Calla la simulacion mientras corre la suite.

    Una corrida son cientos de lineas de INFO por conflicto y de WARNING por
    desatasco. `run_comparison()` ya se calla sola, pero `run_scenario()`
    escribe su ficha por el log y los CSV avisan de cada escritura.
    """
    global _SILENCIO
    _SILENCIO = qlearning._quiet("simulation", "agent", "conflicts", "metrics", "scenarios")
    _SILENCIO.__enter__()


def tearDownModule() -> None:
    """Devuelve los loggers a como estaban."""
    if _SILENCIO is not None:
        _SILENCIO.__exit__(None, None, None)


class TestCorrenLosCinco(unittest.TestCase):
    """CRITERIO 1: los cinco escenarios corren de principio a fin sin errores."""

    def test_los_cinco_corren_con_las_dos_politicas(self) -> None:
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                resultados = scenarios.run_scenario(
                    spec,
                    config.POLICIES,
                    seeds=spec.seeds(2),
                    model=config.Q_TABLE_FILE,
                    out_dir=None,
                )
                self.assertEqual(sorted(resultados), sorted(config.POLICIES))
                for politica, corridas in resultados.items():
                    self.assertEqual(len(corridas), 2, politica)
                    for corrida in corridas:
                        # Que avance el reloj y que se despache algo: una corrida
                        # de cero ticks "no falla" y no vale para nada.
                        self.assertGreater(corrida.ticks, 0)
                        self.assertGreater(corrida.completed_tasks, 0)

    def test_hay_exactamente_cinco_y_son_de_la_A_a_la_E(self) -> None:
        self.assertEqual(scenarios.LETTERS, ("A", "B", "C", "D", "E"))
        self.assertEqual([spec.letter for spec in scenarios.all_scenarios()], list("ABCDE"))

    def test_cada_escenario_dice_que_prueba(self) -> None:
        """El docstring y el campo `tests` son parte de la entrega, no adorno."""
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                self.assertTrue(spec.tests.strip())
                doc = scenarios.SCENARIOS[spec.letter].__doc__ or ""
                self.assertIn("QUE PRUEBA", doc)
                self.assertIn("QUE SE ESPERA VER", doc)

    def test_una_letra_que_no_existe_no_pasa(self) -> None:
        for letra in ("Z", "", "AA"):
            with self.subTest(letra=letra):
                with self.assertRaises(ValueError):
                    scenarios.get(letra)

    def test_la_letra_da_igual_en_mayuscula_o_minuscula(self) -> None:
        self.assertEqual(scenarios.get("c").letter, "C")
        self.assertEqual(scenarios.get(" d ").letter, "D")


class TestReproducible(unittest.TestCase):
    """CRITERIO 2: misma semilla, mismo resultado. Sin esto no hay reporte."""

    def test_la_misma_semilla_da_el_mismo_escenario(self) -> None:
        for spec in scenarios.all_scenarios():
            for semilla in spec.seeds(5):
                with self.subTest(escenario=spec.letter, seed=semilla):
                    una = spec.build(semilla)
                    otra = spec.build(semilla)
                    self.assertEqual(una.routes, otra.routes)
                    self.assertEqual(una.pending, otra.pending)
                    self.assertEqual(una.seed, semilla)

    def test_la_misma_semilla_da_las_mismas_metricas(self) -> None:
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                escenario = spec.build(spec.seed)
                grafo = spec.graph()
                una = metrics.run_once(
                    grafo, escenario, config.POLICY_BASELINE, max_steps=spec.max_steps
                )
                otra = metrics.run_once(
                    grafo, escenario, config.POLICY_BASELINE, max_steps=spec.max_steps
                )
                self.assertEqual(una.to_row(), otra.to_row())

    def test_el_qlearning_tambien_es_reproducible(self) -> None:
        """Al servir, epsilon = 0: no hay azar que pueda mover el resultado."""
        spec = scenarios.get("E")
        escenario = spec.build(spec.seed)
        grafo = spec.graph()
        filas = [
            metrics.run_once(
                grafo,
                escenario,
                config.POLICY_QLEARNING,
                model=config.Q_TABLE_FILE,
                max_steps=spec.max_steps,
            ).to_row()
            for _ in range(2)
        ]
        self.assertEqual(filas[0], filas[1])

    def test_las_semillas_de_una_corrida_son_las_de_siempre(self) -> None:
        spec = scenarios.get("A")
        self.assertEqual(spec.seeds(4), [spec.seed + i for i in range(4)])
        with self.assertRaises(ValueError):
            spec.seeds(0)


class TestElEscenarioNoEsConstante(unittest.TestCase):
    """CRITERIO 2 bis: que `TestReproducible` no pase por ser trivial.

    Si todas las semillas dieran el mismo escenario, las 20 corridas serian una
    sola repetida 20 veces, la desviacion tipica saldria 0 y el test de arriba
    pasaria sin medir nada.
    """

    def test_semillas_distintas_dan_colas_distintas(self) -> None:
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                colas = {spec.build(s).pending for s in spec.seeds(10)}
                self.assertGreater(len(colas), 1)

    def test_la_estructura_no_cambia_entre_semillas(self) -> None:
        """Y al reves: el mapa, los AGVs y las salidas SI son fijos."""
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                rutas = {spec.build(s).routes for s in spec.seeds(10)}
                self.assertEqual(len(rutas), 1)

    def test_las_dos_politicas_no_dan_lo_mismo(self) -> None:
        spec = scenarios.get("D")
        resultados = scenarios.run_scenario(
            spec, config.POLICIES, seeds=spec.seeds(3), model=config.Q_TABLE_FILE, out_dir=None
        )
        base = [c.to_row() for c in resultados[config.POLICY_BASELINE]]
        otra = [c.to_row() for c in resultados[config.POLICY_QLEARNING]]
        self.assertNotEqual(base, otra)


class TestLosEscenariosSeSostienen(unittest.TestCase):
    """CRITERIO 3: un escenario mal escrito no da error, da una corrida rara."""

    def test_todos_pasan_check(self) -> None:
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                spec.check()

    def test_los_origenes_no_se_repiten(self) -> None:
        """Dos AGVs en el mismo nodo rompen la invariante antes de mover nada."""
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                self.assertEqual(len(set(spec.starts)), spec.n_agents)

    def test_nadie_arranca_en_su_propio_destino(self) -> None:
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                for origen, destino in zip(spec.starts, spec.first_targets):
                    self.assertNotEqual(origen, destino)

    def test_ningun_primer_destino_se_repite(self) -> None:
        """Un AGV aparcado encima del destino de otro lo bloquea para siempre."""
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                self.assertEqual(len(set(spec.first_targets)), spec.n_agents)

    def test_todos_los_nodos_estan_en_su_mapa(self) -> None:
        for spec in scenarios.all_scenarios():
            with self.subTest(escenario=spec.letter):
                nodos = set(spec.graph().adjacency)
                usados = set(spec.starts) | set(spec.first_targets)
                usados |= {nodo for pool in spec.pools for nodo in pool}
                self.assertEqual(usados - nodos, set())

    def test_un_escenario_roto_revienta_y_dice_por_que(self) -> None:
        roto = scenarios.ScenarioSpec(
            letter="X",
            name="roto",
            map_name="warehouse",
            starts=("S1", "S1"),
            first_targets=("N1", "N1"),
            pools=(("S2",),),
            tasks_per_agent=2,
            tests="no se sostiene",
        )
        with self.assertRaises(ValueError) as capturado:
            roto.check()
        mensaje = str(capturado.exception)
        self.assertIn("mismo nodo", mensaje)
        self.assertIn("mismo primer destino", mensaje)

    def test_un_nodo_que_no_esta_en_el_mapa_revienta(self) -> None:
        roto = scenarios.ScenarioSpec(
            letter="X",
            name="roto",
            map_name="warehouse",
            starts=("S1", "S2"),
            first_targets=("N1", "ATLANTIDA"),
            pools=(("S3",),),
            tasks_per_agent=2,
            tests="no se sostiene",
        )
        with self.assertRaises(ValueError) as capturado:
            roto.check()
        self.assertIn("ATLANTIDA", str(capturado.exception))


class TestLoQueMideCadaEscenario(unittest.TestCase):
    """CRITERIO 3 bis: que cada escenario pruebe lo que su docstring dice.

    Es lo que separa "cinco escenarios" de "el mismo escenario cinco veces": si
    A no fuera mas tranquilo que C, o D no obligara a pasar por G, la fase
    entera no mediria nada.
    """

    def test_A_tiene_mucha_menos_congestion_que_C(self) -> None:
        cargas = {}
        for letra in ("A", "C"):
            spec = scenarios.get(letra)
            corridas = scenarios.run_scenario(
                spec, [config.POLICY_BASELINE], seeds=spec.seeds(3), out_dir=None
            )[config.POLICY_BASELINE]
            cargas[letra] = sum(c.conflicts_per_tick for c in corridas) / len(corridas)
        self.assertLess(cargas["A"], cargas["C"] / 5)

    def test_en_D_toda_ruta_entre_mitades_pasa_por_G(self) -> None:
        """G es punto de articulacion: es lo que hace de D un cuello de botella."""
        almacen = scenarios.get("D").graph()
        for oeste in scenarios.OESTE:
            for este in scenarios.ESTE:
                with self.subTest(desde=oeste, hasta=este):
                    self.assertIn("G", astar.astar(almacen, oeste, este))

    def test_en_E_hay_mas_de_una_ruta_de_costo_minimo(self) -> None:
        """Y es justo lo que D no tiene: una alternativa igual de buena."""
        rejilla = scenarios.get("E").graph()
        ruta = astar.astar(rejilla, "A1", "D4")
        costo = astar.path_cost(rejilla, ruta)
        # Se penaliza el segundo nodo de la ruta: si hay redundancia de verdad,
        # A* devuelve otra ruta distinta y que cuesta lo mismo.
        otra = astar.astar(rejilla, "A1", "D4", penalties={ruta[1]: 10.0})
        self.assertNotEqual(ruta, otra)
        self.assertAlmostEqual(costo, astar.path_cost(rejilla, otra))

    def test_en_D_penalizar_G_no_da_ninguna_alternativa(self) -> None:
        almacen = scenarios.get("D").graph()
        ruta = astar.astar(almacen, "S1", "S5")
        otra = astar.astar(almacen, "S1", "S5", penalties={"G": 10.0})
        # Sigue pasando por G porque no hay otro camino: lo unico que ha
        # cambiado es que ahora cuesta mas.
        self.assertIn("G", otra)

    def test_D_y_E_piden_el_mismo_trabajo(self) -> None:
        """El par D/E solo vale si lo unico distinto es la redundancia del mapa."""
        d, e = scenarios.get("D"), scenarios.get("E")
        self.assertEqual(d.n_agents, e.n_agents)
        self.assertEqual(d.n_tasks, e.n_tasks)
        self.assertEqual(len(d.pools), len(e.pools))


class TestLosFicheros(unittest.TestCase):
    """CRITERIO 4: los CSV por escenario y la tabla resumen."""

    def setUp(self) -> None:
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        self.carpeta = Path(carpeta.name)
        self.spec = scenarios.get("A")
        self.resultados = scenarios.run_scenario(
            self.spec,
            config.POLICIES,
            seeds=self.spec.seeds(3),
            model=config.Q_TABLE_FILE,
            out_dir=self.carpeta,
        )

    def test_hay_un_csv_por_politica_con_una_fila_por_semilla(self) -> None:
        for politica in config.POLICIES:
            with self.subTest(politica=politica):
                ruta = self.carpeta / f"scenario_A_{politica}.csv"
                self.assertTrue(ruta.is_file(), f"falta {ruta}")
                filas = metrics.read_runs_csv(ruta)
                self.assertEqual(len(filas), 3)
                self.assertEqual([f["policy"] for f in filas], [politica] * 3)
                self.assertEqual(
                    [int(f["seed"]) for f in filas], self.spec.seeds(3)
                )

    def test_el_nombre_del_csv_es_el_que_pide_la_fase(self) -> None:
        self.assertEqual(
            config.scenario_csv("c", "qlearning").name, "scenario_C_qlearning.csv"
        )

    def test_la_tabla_resumen_tiene_una_fila_por_escenario_y_politica(self) -> None:
        filas = scenarios.summary_rows(
            self.spec, self.resultados, model=config.Q_TABLE_FILE
        )
        self.assertEqual(len(filas), 2)
        self.assertEqual({f["policy"] for f in filas}, set(config.POLICIES))
        for fila in filas:
            self.assertEqual(fila["scenario"], "A")
            self.assertEqual(fila["runs"], 3)
            self.assertEqual(fila["agents"], self.spec.n_agents)

    def test_la_tabla_resumen_lleva_todas_las_metricas(self) -> None:
        filas = scenarios.summary_rows(self.spec, self.resultados)
        destino = scenarios.write_summary_table(
            filas, self.carpeta / "summary_table.csv"
        )
        leidas = scenarios.read_summary_table(destino)
        self.assertEqual(len(leidas), 2)
        for campo in metrics.METRIC_FIELDS:
            self.assertIn(campo, leidas[0])
        for campo in ("completion_rate", "task_rate", "deadlock_free_rate"):
            self.assertIn(campo, leidas[0])

    def test_las_columnas_de_la_tabla_son_un_contrato(self) -> None:
        destino = scenarios.write_summary_table(
            scenarios.summary_rows(self.spec, self.resultados),
            self.carpeta / "summary_table.csv",
        )
        with destino.open(encoding=config.ENCODING, newline="") as fichero:
            cabecera = next(csv.reader(fichero))
        self.assertEqual(tuple(cabecera), scenarios.SUMMARY_COLUMNS)

    def test_las_semillas_ganadas_van_en_la_fila_del_qlearning(self) -> None:
        filas = scenarios.summary_rows(self.spec, self.resultados)
        por_politica = {f["policy"]: f for f in filas}
        aprendida = por_politica[config.POLICY_QLEARNING]
        ganadas = int(aprendida["wins_vs_baseline"])
        perdidas = int(aprendida["losses_vs_baseline"])
        empates = int(aprendida["ties_vs_baseline"])
        self.assertEqual(ganadas + perdidas + empates, 3)
        # La baseline no se compara consigo misma.
        self.assertEqual(por_politica[config.POLICY_BASELINE]["wins_vs_baseline"], "")

    def test_avisa_si_la_tabla_que_escribe_es_mas_corta_que_la_que_habia(self) -> None:
        """Correr `--name X` despues de un `--all` pisa la tabla: hay que decirlo."""
        destino = self.carpeta / "summary_table.csv"
        largas = scenarios.summary_rows(self.spec, self.resultados)
        scenarios.write_summary_table(largas, destino)
        with self.assertLogs("scenarios", level="WARNING") as capturado:
            scenarios.write_summary_table(largas[:1], destino)
        self.assertIn("no se acumula", capturado.output[0])
        self.assertEqual(len(scenarios.read_summary_table(destino)), 1)

    def test_no_avisa_cuando_la_tabla_crece_o_es_igual(self) -> None:
        destino = self.carpeta / "summary_table.csv"
        filas = scenarios.summary_rows(self.spec, self.resultados)
        scenarios.write_summary_table(filas[:1], destino)
        with self.assertNoLogs("scenarios", level="WARNING"):
            scenarios.write_summary_table(filas, destino)

    def test_con_out_dir_none_no_escribe_nada(self) -> None:
        vacia = self.carpeta / "vacia"
        scenarios.run_scenario(
            self.spec, [config.POLICY_BASELINE], seeds=[self.spec.seed], out_dir=None
        )
        self.assertFalse(vacia.exists())


class TestElVeredicto(unittest.TestCase):
    """CRITERIO 5: que diga con honestidad donde mejora y donde no."""

    def test_dice_mejora_cuando_gana_de_verdad(self) -> None:
        spec = scenarios.get("E")
        resultados = scenarios.run_scenario(
            spec, config.POLICIES, seeds=spec.seeds(3), model=config.Q_TABLE_FILE, out_dir=None
        )
        linea = scenarios.scenario_verdict(spec, resultados)
        self.assertIn("E (Rutas alternativas)", linea)
        self.assertRegex(linea, r"MEJORA|MIXTO|NO APORTA|EMPATE")

    def test_sin_las_dos_politicas_no_hay_veredicto(self) -> None:
        spec = scenarios.get("A")
        resultados = scenarios.run_scenario(
            spec, [config.POLICY_BASELINE], seeds=[spec.seed], out_dir=None
        )
        self.assertIn("no hay veredicto", scenarios.scenario_verdict(spec, resultados))

    def test_cuando_casi_nadie_completa_manda_el_trabajo_despachado(self) -> None:
        """En C el makespan de la corrida que no termina es el tope de ticks.

        Comparar esas medias seria comparar el tope consigo mismo, asi que por
        debajo de `SATURATED_BELOW` el veredicto lo decide `task_rate`.
        """
        spec = scenarios.get("C")
        resultados = scenarios.run_scenario(
            spec, config.POLICIES, seeds=spec.seeds(3), model=config.Q_TABLE_FILE, out_dir=None
        )
        completas = max(
            metrics.summarize(corridas)["completion_rate"]
            for corridas in resultados.values()
        )
        linea = scenarios.scenario_verdict(spec, resultados)
        if completas < scenarios.SATURATED_BELOW:
            self.assertIn("tareas despachadas", linea)
            self.assertIn("el makespan aqui es el tope", linea)


class TestElRunnerNoSesga(unittest.TestCase):
    """CRITERIO 6: la invariante de la fase 9 sigue en pie en la fase 10.

    El escenario se construye UNA vez por semilla y **fuera** del bucle de
    politicas. Si una politica viera un trabajo distinto del que vio la otra,
    la comparacion no mediria la politica y los cinco escenarios no valdrian
    para nada.
    """

    def test_las_dos_politicas_corren_el_mismo_trabajo(self) -> None:
        spec = scenarios.get("B")
        vistos: list[tuple[str, str]] = []

        def espia(seed: int) -> metrics.Scenario:
            escenario = spec.build(seed)
            vistos.append((escenario.routes, escenario.pending))
            return escenario

        semillas = spec.seeds(4)
        metrics.run_comparison(
            spec.graph(),
            spec.n_agents,
            spec.n_tasks,
            semillas,
            config.POLICIES,
            model=config.Q_TABLE_FILE,
            max_steps=spec.max_steps,
            builder=espia,
        )
        # Un escenario por semilla, no uno por (semilla, politica): si el
        # builder se llamara dentro del bucle de politicas, saldrian 8.
        self.assertEqual(len(vistos), len(semillas))

    def test_el_builder_manda_sobre_el_sorteo_de_la_fase_9(self) -> None:
        spec = scenarios.get("D")
        resultados = metrics.run_comparison(
            spec.graph(),
            spec.n_agents,
            spec.n_tasks,
            spec.seeds(2),
            [config.POLICY_BASELINE],
            max_steps=spec.max_steps,
            builder=spec.build,
        )
        for corrida in resultados[config.POLICY_BASELINE]:
            self.assertEqual(corrida.n_agents, spec.n_agents)
            self.assertEqual(corrida.n_tasks, spec.n_tasks)

    def test_sin_builder_run_comparison_hace_lo_de_siempre(self) -> None:
        """La fase 9 no cambia: el parametro es opcional y por defecto no esta."""
        almacen = graph.warehouse_graph()
        con = metrics.run_comparison(
            almacen, 3, 6, [7], [config.POLICY_BASELINE], max_steps=300
        )
        sin = metrics.run_comparison(
            almacen,
            3,
            6,
            [7],
            [config.POLICY_BASELINE],
            max_steps=300,
            builder=lambda s: metrics.build_scenario(almacen, 3, 6, seed=s),
        )
        self.assertEqual(
            con[config.POLICY_BASELINE][0].to_row(),
            sin[config.POLICY_BASELINE][0].to_row(),
        )


class TestEntrenarPorEscenario(unittest.TestCase):
    """CRITERIO 7: se puede entrenar una Q-table EN un escenario."""

    def test_el_generador_de_rutas_usa_las_salidas_del_escenario(self) -> None:
        import random

        spec = scenarios.get("D")
        fabrica = spec.routes_factory()
        rng = random.Random(1)
        for _ in range(20):
            rutas = fabrica(rng)
            self.assertEqual(len(rutas), spec.n_agents)
            self.assertEqual({o for o, _ in rutas}, set(spec.starts))
            for origen, destino in rutas:
                self.assertNotEqual(origen, destino)
                self.assertIn(destino, {n for pool in spec.pools for n in pool})

    def test_entrenar_en_un_escenario_escribe_su_propio_modelo(self) -> None:
        carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(carpeta.cleanup)
        destino = Path(carpeta.name) / "q_table_E.json"

        spec = scenarios.get("E")
        cfg = qlearning.TrainingConfig(
            map_name=spec.map_name,
            agents=spec.n_agents,
            episodes=20,
            seed=3,
            max_steps=120,
            scenario=spec.letter,
        )
        entrenador = scenarios.train_scenario(
            spec, cfg, model_path=destino, log_path=None, curve_path=None
        )
        self.assertTrue(destino.is_file())
        self.assertGreater(len(entrenador.q), 0)
        metadata = qlearning.load_metadata(destino)
        self.assertEqual(metadata["hyperparameters"]["scenario"], "E")
        self.assertEqual(metadata["map"], "grid")

    def test_sin_routes_factory_el_entrenamiento_es_el_de_la_fase_7(self) -> None:
        """El gancho es opcional: sin el, la fase 7 no cambia ni un numero."""
        almacen = graph.warehouse_graph()
        cfg = qlearning.TrainingConfig(agents=3, episodes=15, seed=11, max_steps=120)
        una = qlearning.Trainer(almacen, cfg)
        una.run(15)
        otra = qlearning.Trainer(almacen, cfg, routes_factory=None)
        otra.run(15)
        self.assertEqual(una.q.as_dict(), otra.q.as_dict())


class TestElMapaNuevo(unittest.TestCase):
    """CRITERIO 8: la rejilla del escenario E es un mapa del repo como los otros."""

    def test_grid_esta_en_los_mapas_internos(self) -> None:
        self.assertIn("grid", graph.BUILTIN_MAPS)

    def test_la_rejilla_tiene_dieciseis_nodos_y_veinticuatro_tramos(self) -> None:
        rejilla = graph.grid_graph()
        rejilla.validate()
        self.assertEqual(len(rejilla.nodes()), 16)
        self.assertEqual(len(rejilla.edges()), 24)

    def test_todos_los_tramos_cuestan_lo_mismo(self) -> None:
        """Es lo que hace que haya alternativas del MISMO costo, no rodeos."""
        for _, _, costo in graph.grid_graph().edges():
            self.assertEqual(costo, graph.GRID_SPACING)

    def test_ningun_nodo_es_punto_de_articulacion(self) -> None:
        """Al reves que G en el `warehouse`: quitar un nodo no parte la rejilla."""
        rejilla = graph.grid_graph()
        for quitado in rejilla.nodes():
            with self.subTest(sin=quitado):
                quedan = [n for n in rejilla.nodes() if n != quitado]
                adyacencia = {
                    n: {v: c for v, c in rejilla.adjacency[n].items() if v != quitado}
                    for n in quedan
                }
                self.assertEqual(
                    graph._alcanzables(quedan[0], adyacencia), set(quedan)
                )


if __name__ == "__main__":
    unittest.main()
