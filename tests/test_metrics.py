"""Tests de la fase 9: las metricas, el runner pareado y los ficheros de results/.

Los criterios de aceptacion de la fase estan marcados con "CRITERIO N" en el
docstring de cada clase. El que de verdad sostiene la fase es el CRITERIO 1: si
el escenario del baseline y el del Q-Learning no son el mismo, todo lo demas
mide ruido.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import conflicts
import graph
import metrics
import qlearning
import simulation


_SILENCIO = None


def setUpModule() -> None:
    """Calla la simulacion mientras corre la suite.

    Una corrida son cientos de lineas de INFO por conflicto y de WARNING por
    desatasco. `run_comparison()` ya se calla sola, pero los tests que llaman a
    `run_once()` directo no, y entonces el resultado de la suite no se lee.
    """
    global _SILENCIO
    _SILENCIO = qlearning._quiet("simulation", "agent", "conflicts")
    _SILENCIO.__enter__()


def tearDownModule() -> None:
    """Devuelve los loggers a como estaban."""
    if _SILENCIO is not None:
        _SILENCIO.__exit__(None, None, None)


def almacen() -> graph.WarehouseGraph:
    """El mapa con el cuello de botella, que es donde hay algo que medir."""
    return graph.warehouse_graph()


def corrida(seed: int = 1, **extra) -> metrics.RunMetrics:
    """Una corrida corta del baseline, para lo que no necesita las dos politicas."""
    mapa = almacen()
    escenario = metrics.build_scenario(mapa, extra.pop("agents", 3), extra.pop("tasks", 6), seed=seed)
    return metrics.run_once(mapa, escenario, config.POLICY_BASELINE, max_steps=extra.pop("max_steps", 400))


class TestMismasCondiciones(unittest.TestCase):
    """CRITERIO 1: las dos politicas corren EXACTAMENTE el mismo trabajo.

    Es el test que sostiene la fase entera. Si una politica viera un escenario
    mas facil que la otra, la comparacion no mediria la politica y el argumento
    del proyecto se caeria.
    """

    def test_la_misma_semilla_da_el_mismo_escenario(self) -> None:
        mapa = almacen()
        for semilla in range(1, 21):
            with self.subTest(seed=semilla):
                una = metrics.build_scenario(mapa, 4, 16, seed=semilla)
                otra = metrics.build_scenario(mapa, 4, 16, seed=semilla)
                self.assertEqual(una.routes, otra.routes)
                self.assertEqual(una.pending, otra.pending)

    def test_las_tareas_son_identicas_entre_las_dos_politicas(self) -> None:
        """El corazon de la fase: mismo mapa, mismos AGVs, mismos origenes y destinos."""
        mapa = almacen()
        for semilla in range(1, 21):
            with self.subTest(seed=semilla):
                escenario = metrics.build_scenario(mapa, 4, 16, seed=semilla)
                montadas = {
                    nombre: simulation.Simulation(
                        mapa,
                        routes=list(escenario.routes),
                        policy=nombre,
                        seed=escenario.seed,
                    )
                    for nombre in (config.POLICY_BASELINE, config.POLICY_QLEARNING)
                }
                base = montadas[config.POLICY_BASELINE]
                otra = montadas[config.POLICY_QLEARNING]

                self.assertEqual(len(base.agents), len(otra.agents))
                for uno, dos in zip(base.agents, otra.agents):
                    self.assertEqual(uno.id, dos.id)
                    self.assertEqual(uno.start_node, dos.start_node)
                    self.assertEqual(uno.current_node, dos.current_node)
                    self.assertEqual(uno.target_node, dos.target_node)
                    self.assertEqual(uno.path, dos.path)
                self.assertEqual(base.seed, otra.seed)
                # Y lo unico que las distingue es la politica.
                self.assertNotEqual(base.mode, otra.mode)

    def test_la_cola_de_tareas_pendientes_es_la_misma_para_las_dos(self) -> None:
        mapa = almacen()
        escenario = metrics.build_scenario(mapa, 4, 16, seed=3)
        # La cola vive en el escenario, no en la corrida: las dos politicas
        # reciben el mismo objeto inmutable y ninguna puede tocarlo.
        self.assertEqual(len(escenario.pending), 12)
        with self.assertRaises(Exception):
            escenario.pending[0] = "N1"  # type: ignore[index]

    def test_run_comparison_corre_las_dos_sobre_las_mismas_semillas(self) -> None:
        resultados = metrics.run_comparison(
            almacen(), 3, 6, range(1, 6), (config.POLICY_BASELINE, config.POLICY_QLEARNING),
            max_steps=300,
        )
        self.assertEqual(sorted(resultados), [config.POLICY_BASELINE, config.POLICY_QLEARNING])
        semillas_base = [c.seed for c in resultados[config.POLICY_BASELINE]]
        semillas_otra = [c.seed for c in resultados[config.POLICY_QLEARNING]]
        self.assertEqual(semillas_base, [1, 2, 3, 4, 5])
        self.assertEqual(semillas_base, semillas_otra)
        for una, otra in zip(*resultados.values()):
            self.assertEqual(una.n_tasks, otra.n_tasks)
            self.assertEqual(una.n_agents, otra.n_agents)


class TestElEscenarioNoEsConstante(unittest.TestCase):
    """CRITERIO 2: el criterio 1 no pasa por ser trivial.

    Un generador que devolviera siempre lo mismo cumpliria el criterio 1 sin
    medir nada. Estos dos tests son los que le dan valor al otro.
    """

    def test_semillas_distintas_dan_escenarios_distintos(self) -> None:
        mapa = almacen()
        escenarios = {
            (metrics.build_scenario(mapa, 4, 16, seed=s).routes,
             metrics.build_scenario(mapa, 4, 16, seed=s).pending)
            for s in range(1, 21)
        }
        self.assertGreater(len(escenarios), 15)

    def test_las_dos_politicas_no_dan_el_mismo_resultado(self) -> None:
        """Si dieran lo mismo, la politica no estaria cambiando nada."""
        mapa = almacen()
        escenario = metrics.build_scenario(mapa, 4, 16, seed=7)
        base = metrics.run_once(mapa, escenario, config.POLICY_BASELINE, max_steps=800)
        otra = metrics.run_once(mapa, escenario, config.POLICY_QLEARNING, max_steps=800)
        self.assertNotEqual(base.makespan, otra.makespan)


class TestLasCuentas(unittest.TestCase):
    """CRITERIO 3: cada metrica cuadra con lo que dice el motor."""

    def test_la_distancia_es_la_suma_de_los_tramos_pisados(self) -> None:
        mapa = almacen()
        escenario = metrics.build_scenario(mapa, 2, 2, seed=5)
        simulacion = simulation.Simulation(
            mapa, routes=list(escenario.routes), policy=config.POLICY_BASELINE, seed=5
        )
        medidas = metrics.RunMetrics(
            policy="baseline", seed=5, map_name="warehouse", n_agents=2, n_tasks=2
        )
        medidas.start(simulacion)

        a_mano = 0.0
        anterior = {a.id: a.current_node for a in simulacion.agents}
        while simulacion.step < 200 and not simulacion.done:
            simulacion.tick()
            medidas.observe(simulacion)
            for agente in simulacion.agents:
                if agente.current_node != anterior[agente.id]:
                    a_mano += mapa.cost(anterior[agente.id], agente.current_node)
                    anterior[agente.id] = agente.current_node

        self.assertAlmostEqual(medidas.total_distance, a_mano, places=6)
        self.assertGreater(medidas.total_distance, 0.0)

    def test_la_espera_y_los_conflictos_son_los_del_motor(self) -> None:
        mapa = almacen()
        escenario = metrics.build_scenario(mapa, 4, 8, seed=11)
        medidas = metrics.run_once(mapa, escenario, config.POLICY_BASELINE, max_steps=400)
        self.assertEqual(sum(medidas.wait_by_agent.values()), medidas.total_wait_time)
        self.assertEqual(sum(medidas.conflicts_by_type.values()), medidas.conflicts_total)
        self.assertEqual(sorted(medidas.conflicts_by_type), sorted(conflicts.TYPES))

    def test_la_espera_sobrevive_a_una_tarea_nueva(self) -> None:
        """`assign_task()` pone `wait_time` a 0 y el runner tiene que restaurarlo.

        Sin esto, la espera acumulada se borraria en cada tarea y la metrica con
        la que se comparan las politicas dejaria de significar nada.
        """
        mapa = almacen()
        simulacion = simulation.Simulation(
            mapa, routes=[("S1", "S2")], policy=config.POLICY_BASELINE
        )
        agente = simulacion.agents[0]
        agente.wait_time = 37
        medidas = metrics.RunMetrics(
            policy="baseline", seed=0, map_name="warehouse", n_agents=1, n_tasks=2
        )
        metrics._asigna(agente, "N1", 10, medidas)

        self.assertEqual(agente.wait_time, 37)
        self.assertEqual(agente.target_node, "N1")

    def test_makespan_y_throughput(self) -> None:
        medidas = corrida(seed=2)
        if medidas.all_completed:
            self.assertEqual(medidas.makespan, medidas.last_completion)
            self.assertLessEqual(medidas.makespan, medidas.ticks)
        else:
            self.assertEqual(medidas.makespan, medidas.ticks)
        self.assertAlmostEqual(
            medidas.throughput, 100.0 * medidas.completed_tasks / medidas.ticks, places=6
        )

    def test_una_tarea_completada_no_se_cuenta_dos_veces(self) -> None:
        """El AGV que se queda `done` con la cola vacia no suma una tarea por tick."""
        medidas = metrics.RunMetrics(
            policy="baseline", seed=0, map_name="x", n_agents=1, n_tasks=1
        )
        medidas.start_task(1, 0)
        self.assertTrue(medidas.complete_task(1, 5))
        self.assertFalse(medidas.complete_task(1, 6))
        self.assertFalse(medidas.complete_task(1, 7))
        self.assertEqual(medidas.completed_tasks, 1)

    def test_se_despacha_la_cola_entera_cuando_da_tiempo(self) -> None:
        mapa = almacen()
        escenario = metrics.build_scenario(mapa, 2, 6, seed=4)
        medidas = metrics.run_once(mapa, escenario, config.POLICY_BASELINE, max_steps=800)
        self.assertEqual(medidas.n_tasks, 6)
        self.assertEqual(medidas.completed_tasks, 6)
        self.assertTrue(medidas.all_completed)
        self.assertEqual(len(medidas.task_times), 6)


class TestReproducible(unittest.TestCase):
    """CRITERIO 4: la misma semilla y la misma politica dan la misma corrida."""

    def test_dos_corridas_iguales_dan_los_mismos_numeros(self) -> None:
        for politica in (config.POLICY_BASELINE, config.POLICY_QLEARNING):
            with self.subTest(policy=politica):
                mapa = almacen()
                escenario = metrics.build_scenario(mapa, 3, 9, seed=13)
                una = metrics.run_once(mapa, escenario, politica, max_steps=400)
                otra = metrics.run_once(mapa, escenario, politica, max_steps=400)
                self.assertEqual(una.to_row(), otra.to_row())


class TestLosFicheros(unittest.TestCase):
    """CRITERIO 5: los CSV y el JSON salen solos y se pueden volver a leer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.resultados = metrics.run_comparison(
            almacen(), 3, 6, range(1, 21), (config.POLICY_BASELINE, config.POLICY_QLEARNING),
            max_steps=300,
        )

    def test_el_csv_tiene_cabecera_y_una_fila_por_semilla(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            for nombre, corridas in self.resultados.items():
                destino = metrics.write_runs_csv(corridas, Path(carpeta) / f"{nombre}.csv")
                filas = metrics.read_runs_csv(destino)
                self.assertEqual(len(filas), 20)
                self.assertEqual([f["seed"] for f in filas], [str(s) for s in range(1, 21)])
                self.assertEqual({f["policy"] for f in filas}, {nombre})

    def test_el_csv_se_lee_con_el_csv_de_la_libreria_estandar(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            destino = metrics.write_runs_csv(
                self.resultados[config.POLICY_BASELINE], Path(carpeta) / "b.csv"
            )
            with destino.open(encoding=config.ENCODING, newline="") as fichero:
                filas = list(csv.DictReader(fichero))
            self.assertEqual(len(filas), 20)
            for columna in ("makespan", "avg_task_time", "total_wait_time",
                            "total_distance", "deadlocks", "reroutes", "throughput",
                            "conflicts_vertex", "conflicts_edge", "conflicts_congestion"):
                self.assertIn(columna, filas[0])
            # Nada anidado: cada celda es un valor suelto.
            for valor in filas[0].values():
                self.assertNotIn("{", valor)

    def test_el_json_trae_media_desviacion_y_mediana(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            destino = metrics.write_comparison_json(
                self.resultados, Path(carpeta) / "comparison.json",
                header={"map": "warehouse", "agents": 3},
            )
            contenido = json.loads(destino.read_text(encoding=config.ENCODING))
            self.assertEqual(contenido["experiment"]["map"], "warehouse")
            for nombre in self.resultados:
                resumen = contenido["policies"][nombre]
                self.assertEqual(resumen["runs"], 20)
                for campo in metrics.METRIC_FIELDS:
                    for estadistico in ("mean", "stdev", "median", "min", "max"):
                        self.assertIn(estadistico, resumen["metrics"][campo])

    def test_una_sola_corrida_no_revienta_la_desviacion(self) -> None:
        resumen = metrics.summarize([corrida(seed=1)])
        self.assertEqual(resumen["runs"], 1)
        self.assertEqual(resumen["metrics"]["makespan"]["stdev"], 0.0)


class TestElReporte(unittest.TestCase):
    """CRITERIO 6: el reporte no maquilla un Q-Learning que pierde."""

    def _falsas(self, makespans, *, policy, tasks=4):
        corridas = []
        for numero, valor in enumerate(makespans, start=1):
            medidas = metrics.RunMetrics(
                policy=policy, seed=numero, map_name="warehouse", n_agents=1, n_tasks=tasks
            )
            medidas.ticks = valor
            medidas.completed_tasks = tasks
            medidas.last_completion = valor
            medidas.task_times = [valor]
            corridas.append(medidas)
        return corridas

    def test_dice_que_no_cuando_el_qlearning_pierde(self) -> None:
        resultados = {
            config.POLICY_BASELINE: self._falsas([100, 100, 100], policy="baseline"),
            config.POLICY_QLEARNING: self._falsas([200, 200, 200], policy="qlearning"),
        }
        lineas = metrics.comparison_lines(resultados)
        texto = "\n".join(lineas)
        self.assertIn("¿Mejora?", texto)
        self.assertIn("NO", texto)
        self.assertIn("VEREDICTO: el Q-Learning NO mejora", texto)
        self.assertIn("+100.0%", texto)

    def test_dice_que_si_cuando_gana(self) -> None:
        resultados = {
            config.POLICY_BASELINE: self._falsas([200, 200, 200], policy="baseline"),
            config.POLICY_QLEARNING: self._falsas([100, 100, 100], policy="qlearning"),
        }
        texto = "\n".join(metrics.comparison_lines(resultados))
        self.assertIn("VEREDICTO: el Q-Learning MEJORA", texto)
        self.assertIn("-50.0%", texto)

    def test_avisa_cuando_la_media_y_el_reparto_no_dicen_lo_mismo(self) -> None:
        """Gana casi siempre pero se cuelga una vez: la media sola engana."""
        resultados = {
            config.POLICY_BASELINE: self._falsas([100, 100, 100, 100], policy="baseline"),
            config.POLICY_QLEARNING: self._falsas([90, 90, 90, 800], policy="qlearning"),
        }
        texto = "\n".join(metrics.comparison_lines(resultados))
        self.assertIn("VEREDICTO: mixto", texto)
        self.assertIn("la media sola engana", texto)

    def test_las_semillas_ganadas_se_cuentan_pareadas(self) -> None:
        base = self._falsas([100, 100, 100], policy="baseline")
        otra = self._falsas([90, 110, 100], policy="qlearning")
        self.assertEqual(metrics.paired_wins(base, otra, "makespan"), (1, 1, 1))

    def test_no_compara_corridas_desparejadas(self) -> None:
        base = self._falsas([100, 100], policy="baseline")
        otra = self._falsas([100, 100], policy="qlearning")
        otra[1].seed = 99
        with self.assertRaises(ValueError):
            metrics.paired_wins(base, otra)

    def test_el_reporte_lleva_los_cuatro_tipos_de_conflicto(self) -> None:
        texto = "\n".join(
            metrics.comparison_lines(
                {
                    config.POLICY_BASELINE: self._falsas([100], policy="baseline"),
                    config.POLICY_QLEARNING: self._falsas([100], policy="qlearning"),
                }
            )
        )
        for etiqueta in ("de nodo", "de arista", "de seguimiento", "de congestion"):
            self.assertIn(etiqueta, texto)


class TestLasGraficas(unittest.TestCase):
    """Las graficas son opcionales: sin matplotlib se avisa y se sigue."""

    def test_sin_matplotlib_devuelve_none_y_avisa(self) -> None:
        resultados = {config.POLICY_BASELINE: [corrida(seed=1)]}
        with mock.patch.dict("sys.modules", {"matplotlib": None}):
            with self.assertLogs("metrics", level="WARNING") as capturado:
                self.assertIsNone(
                    metrics.save_comparison_plot(resultados, "/tmp/no-se-escribe.png")
                )
        self.assertIn("matplotlib", capturado.output[0])

    def test_sin_corridas_no_dibuja(self) -> None:
        with self.assertLogs("metrics", level="WARNING"):
            self.assertIsNone(metrics.save_comparison_plot({}, "/tmp/no-se-escribe.png"))


class TestElEscenario(unittest.TestCase):
    """Lo que `build_scenario()` no deja pasar."""

    def test_no_caben_menos_tareas_que_agentes(self) -> None:
        with self.assertRaises(ValueError):
            metrics.build_scenario(almacen(), 4, 3)

    def test_sin_agentes_no_hay_escenario(self) -> None:
        with self.assertRaises(ValueError):
            metrics.build_scenario(almacen(), 0, 4)

    def test_por_defecto_son_cuatro_tareas_por_agente(self) -> None:
        escenario = metrics.build_scenario(almacen(), 3)
        self.assertEqual(escenario.n_tasks, 3 * config.BENCHMARK_TASKS_PER_AGENT)
        self.assertEqual(escenario.n_agents, 3)

    def test_sin_semillas_no_hay_comparacion(self) -> None:
        with self.assertRaises(ValueError):
            metrics.run_comparison(almacen(), 2, 4, [], (config.POLICY_BASELINE,))


if __name__ == "__main__":
    unittest.main()
