"""CRITERIOS DE LA FASE 7: el entrenamiento del Q-Learning.

Lo que se demuestra aqui:

1. La actualizacion es **Bellman**, con la formula del enunciado y nada mas.
2. TRAIN y EVALUATE estan **separados**: uno explora y escribe, el otro es
   greedy puro, carga del disco y no toca la tabla.
3. El entrenamiento **aprende**: la recompensa sube y los conflictos bajan.
4. La Q-table se guarda y se carga **con la metadata** de la corrida, y evaluar
   con la tabla del disco da lo mismo que con la de memoria.
5. Es **reproducible**: la misma semilla da la misma corrida, hasta la ultima
   celda de la tabla.
6. El entrenamiento corre **sin servidor y sin Unity**.
7. El CSV tiene las columnas del enunciado, una fila por episodio.
"""

import csv
import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import agent
import config
import conflicts
import graph
import qlearning
import server
import simulation

# 300 episodios bastan para ver la tendencia y tardan ~1.5 s; 1000 se corren a
# mano con `main.py train`, no dentro de la bateria de tests.
EPISODIOS = 300

# Cuantas veces hay que haber actualizado un estado para poder decir que lo que
# hay en su fila es politica aprendida y no la fila de ceros con la que nacio.
MINIMO_VISITAS = 50


def ajustes(**cambios: object) -> qlearning.TrainingConfig:
    """Una TrainingConfig corta, con lo que se le quiera cambiar."""
    return replace(
        qlearning.TrainingConfig(episodes=EPISODIOS, report_every=100), **cambios
    )


def entrena(grafo: graph.WarehouseGraph, **cambios: object) -> qlearning.Trainer:
    """Corre un entrenamiento corto y devuelve el Trainer."""
    entrenador = qlearning.Trainer(grafo, ajustes(**cambios))
    entrenador.run()
    return entrenador


def bloque(historia: list[qlearning.EpisodeStats], desde: int, hasta: int) -> dict[str, float]:
    """Las medias de un tramo de episodios."""
    tramo = historia[desde:hasta]
    return {
        "recompensa": sum(f.total_reward for f in tramo) / len(tramo),
        "conflictos": sum(f.conflicts for f in tramo) / len(tramo),
    }


class TestBellman(unittest.TestCase):
    """CRITERIO 1: la formula es la del enunciado, sin sorpresas."""

    def test_es_exactamente_la_formula(self) -> None:
        tabla = qlearning.QTable()
        estado, siguiente = (0, 0, 0, 0, 1), (1, 0, 0, 0, 0)
        tabla.set_value(estado, qlearning.Action.ADVANCE, 2.0)
        tabla.set_value(siguiente, qlearning.Action.WAIT, 10.0)

        nuevo = tabla.update(
            estado, qlearning.Action.ADVANCE, 5.0, siguiente, alpha=0.2, gamma=0.95
        )
        # 2.0 + 0.2 * (5.0 + 0.95 * 10.0 - 2.0) = 4.5
        self.assertAlmostEqual(nuevo, 4.5)
        self.assertAlmostEqual(tabla.value(estado, qlearning.Action.ADVANCE), 4.5)

    def test_terminal_no_descuenta_futuro(self) -> None:
        tabla = qlearning.QTable()
        estado, siguiente = (0, 0, 0, 0, 1), (1, 0, 0, 0, 0)
        tabla.set_value(siguiente, qlearning.Action.WAIT, 100.0)

        nuevo = tabla.update(
            estado,
            qlearning.Action.ADVANCE,
            5.0,
            siguiente,
            alpha=0.5,
            gamma=0.95,
            terminal=True,
        )
        # 0.0 + 0.5 * (5.0 + 0 - 0.0) = 2.5: el 100 del siguiente no cuenta.
        self.assertAlmostEqual(nuevo, 2.5)

    def test_el_maximo_se_limita_a_las_acciones_permitidas(self) -> None:
        tabla = qlearning.QTable()
        estado, siguiente = (0, 0, 0, 0, 1), (1, 0, 0, 0, 0)
        tabla.set_value(siguiente, qlearning.Action.REROUTE, 100.0)

        con_reroute = qlearning.QTable(
            {siguiente: {qlearning.Action.REROUTE: 100.0}}
        ).update(estado, qlearning.Action.ADVANCE, 0.0, siguiente, alpha=1.0, gamma=1.0)
        sin_reroute = qlearning.QTable(
            {siguiente: {qlearning.Action.REROUTE: 100.0}}
        ).update(
            estado,
            qlearning.Action.ADVANCE,
            0.0,
            siguiente,
            alpha=1.0,
            gamma=1.0,
            among=qlearning.enabled_actions(False),
        )
        self.assertAlmostEqual(con_reroute, 100.0)
        self.assertAlmostEqual(sin_reroute, 0.0)


class TestEpsilon(unittest.TestCase):
    """CRITERIO 2: el decaimiento es exponencial y tiene suelo."""

    def test_decae_multiplicando(self) -> None:
        entrenador = qlearning.Trainer(
            graph.simple_graph(), ajustes(epsilon_start=1.0, epsilon_decay=0.9)
        )
        self.assertAlmostEqual(entrenador.epsilon, 1.0)
        self.assertAlmostEqual(entrenador.decay_epsilon(), 0.9)
        self.assertAlmostEqual(entrenador.decay_epsilon(), 0.81)

    def test_nunca_baja_del_suelo(self) -> None:
        entrenador = qlearning.Trainer(
            graph.simple_graph(),
            ajustes(epsilon_start=1.0, epsilon_decay=0.5, epsilon_end=0.05),
        )
        for _ in range(100):
            entrenador.decay_epsilon()
        self.assertAlmostEqual(entrenador.epsilon, 0.05)

    def test_el_log_guarda_el_epsilon_de_cada_episodio(self) -> None:
        entrenador = entrena(graph.warehouse_graph(), episodes=20)
        epsilons = [fila.epsilon for fila in entrenador.history]
        self.assertEqual(epsilons[0], config.EPSILON_START)
        self.assertEqual(epsilons, sorted(epsilons, reverse=True))


class TestDosModos(unittest.TestCase):
    """CRITERIO 2: TRAIN explora y escribe; EVALUATE es greedy y no toca nada."""

    def setUp(self) -> None:
        self.grafo = graph.warehouse_graph()

    def test_train_escribe_en_la_tabla(self) -> None:
        entrenador = entrena(self.grafo, episodes=30)
        self.assertGreater(len(entrenador.q), 0)
        self.assertTrue(
            any(
                valor != 0.0
                for estado in entrenador.q
                for valor in entrenador.q[estado].values()
            )
        )

    def test_evaluate_no_toca_la_tabla(self) -> None:
        entrenador = entrena(self.grafo, episodes=60)
        antes = json.dumps(entrenador.q.as_dict(), sort_keys=True)

        evaluador = qlearning.Trainer(
            self.grafo, ajustes(), q_table=entrenador.q, learn=False
        )
        evaluador.run(30)

        self.assertEqual(json.dumps(evaluador.q.as_dict(), sort_keys=True), antes)

    def test_evaluate_arranca_con_epsilon_cero(self) -> None:
        evaluador = qlearning.Trainer(self.grafo, ajustes(), learn=False)
        self.assertEqual(evaluador.epsilon, 0.0)
        evaluador.run(5)
        self.assertEqual(evaluador.policy.epsilon, 0.0)
        self.assertTrue(all(fila.epsilon == 0.0 for fila in evaluador.history))

    def test_evaluate_repetido_da_exactamente_lo_mismo(self) -> None:
        entrenador = entrena(self.grafo, episodes=60)

        def evalua() -> list[dict[str, object]]:
            evaluador = qlearning.Trainer(
                self.grafo, ajustes(), q_table=entrenador.q, learn=False
            )
            evaluador.run(30)
            return [fila.as_row() for fila in evaluador.history]

        self.assertEqual(evalua(), evalua())


class TestAprende(unittest.TestCase):
    """CRITERIO 3: la recompensa sube y los conflictos bajan."""

    @classmethod
    def setUpClass(cls) -> None:
        # Un solo entrenamiento para las tres comprobaciones: es lo que tarda.
        cls.entrenador = entrena(graph.warehouse_graph())
        cls.primero = bloque(cls.entrenador.history, 0, 100)
        cls.ultimo = bloque(cls.entrenador.history, -100, len(cls.entrenador.history))

    def test_la_recompensa_media_sube(self) -> None:
        self.assertGreater(
            self.ultimo["recompensa"],
            self.primero["recompensa"],
            "los ultimos 100 episodios tenian que dar mas recompensa que los primeros",
        )

    def test_deja_de_ser_azar(self) -> None:
        # Con epsilon 1.0 la politica es un dado; si al final saca lo mismo, no
        # aprendio nada. El margen es amplio a proposito: lo que se comprueba es
        # que hay una diferencia grande, no un numero concreto.
        azar = qlearning.Trainer(
            graph.warehouse_graph(),
            ajustes(episodes=100, epsilon_start=1.0, epsilon_end=1.0, epsilon_decay=1.0),
        )
        azar.run()
        aleatoria = sum(f.total_reward for f in azar.history) / len(azar.history)
        self.assertGreater(self.ultimo["recompensa"], aleatoria + 100.0)

    def test_los_conflictos_por_episodio_bajan(self) -> None:
        self.assertLess(
            self.ultimo["conflictos"],
            self.primero["conflictos"],
            "los conflictos por episodio tenian que bajar con el entrenamiento",
        )

    def test_aprende_a_no_meterse_donde_no_cabe(self) -> None:
        # Lo unico que se le pide a la politica: si el nodo de delante esta
        # ocupado, no intentes entrar. Es la mitad del espacio de estados.
        #
        # Solo se miran los estados con datos detras: una celda visitada dos
        # veces sigue casi en el cero con el que nacio, y ahi gana ADVANCE por el
        # desempate de `best_action()`, no porque se haya aprendido nada. El
        # contador esta en `Trainer.visits` y va a la metadata del modelo justo
        # para poder distinguir una cosa de la otra.
        mirados = 0
        for estado in self.entrenador.q:
            if estado[0] != 1 or self.entrenador.visits[estado] < MINIMO_VISITAS:
                continue
            mirados += 1
            with self.subTest(estado=qlearning.encode_state(estado)):
                self.assertIsNot(
                    self.entrenador.q.best_action(estado),
                    qlearning.Action.ADVANCE,
                    "con el nodo siguiente ocupado, ADVANCE no puede ser la mejor",
                )
        self.assertGreaterEqual(mirados, 8, "casi ningun estado tuvo datos detras")

    def test_los_estados_con_datos_son_la_mayoria(self) -> None:
        # Si casi ninguna celda se visitara lo suficiente, la tabla seria ruido
        # con forma de politica por mucho que la recompensa suba.
        con_datos = sum(
            1 for estado in self.entrenador.q
            if self.entrenador.visits[estado] >= MINIMO_VISITAS
        )
        self.assertGreater(con_datos, len(self.entrenador.q) * 0.6)


class TestReproducible(unittest.TestCase):
    """CRITERIO 5: la misma semilla, la misma corrida."""

    def test_dos_entrenamientos_con_la_misma_semilla_son_iguales(self) -> None:
        uno = entrena(graph.warehouse_graph(), episodes=120, seed=42)
        otro = entrena(graph.warehouse_graph(), episodes=120, seed=42)

        self.assertEqual(
            [fila.as_row() for fila in uno.history],
            [fila.as_row() for fila in otro.history],
        )
        self.assertEqual(uno.q.as_dict(), otro.q.as_dict())

    def test_con_otra_semilla_sale_otra_cosa(self) -> None:
        uno = entrena(graph.warehouse_graph(), episodes=120, seed=42)
        otro = entrena(graph.warehouse_graph(), episodes=120, seed=7)
        self.assertNotEqual(
            [fila.as_row() for fila in uno.history],
            [fila.as_row() for fila in otro.history],
        )

    def test_los_escenarios_se_repiten_con_la_semilla(self) -> None:
        grafo = graph.warehouse_graph()
        primeros = qlearning.random_routes(grafo, 4, random.Random(42))
        segundos = qlearning.random_routes(grafo, 4, random.Random(42))
        self.assertEqual(primeros, segundos)


class TestEscenarios(unittest.TestCase):
    """Los repartos de tareas de cada episodio se sostienen."""

    def test_origenes_y_destinos_sin_repetir_y_nadie_ya_en_casa(self) -> None:
        grafo = graph.warehouse_graph()
        rng = random.Random(1)
        for _ in range(200):
            rutas = qlearning.random_routes(grafo, 4, rng)
            origenes = [origen for origen, _ in rutas]
            destinos = [destino for _, destino in rutas]
            self.assertEqual(len(set(origenes)), 4)
            self.assertEqual(len(set(destinos)), 4)
            for origen, destino in rutas:
                self.assertNotEqual(origen, destino)
                self.assertIn(origen, grafo.adjacency)
                self.assertIn(destino, grafo.adjacency)

    def test_no_caben_mas_agentes_que_nodos(self) -> None:
        with self.assertRaises(ValueError):
            qlearning.random_routes(graph.simple_graph(), 99, random.Random(0))

    def test_cada_episodio_es_un_escenario_distinto(self) -> None:
        entrenador = qlearning.Trainer(graph.warehouse_graph(), ajustes(episodes=30))
        vistos = set()
        for _ in range(30):
            entrenador.env.reset()
            vistos.add(
                tuple(
                    (agente.start_node, agente.target_node)
                    for agente in entrenador.env.sim.agents
                )
            )
        self.assertGreater(len(vistos), 20, "los episodios no pueden ser todos el mismo")


class TestRecompensaDelEntorno(unittest.TestCase):
    """La recompensa que reparte `TrainingEnv` sale de `config.py` y de nada mas."""

    def test_el_advance_que_cruza_un_tramo_cobra_el_progreso(self) -> None:
        # Un tramo del warehouse cuesta 4-8 ticks y el AGV no decide mientras lo
        # cruza: si la transicion se cerrara en el mismo tick, todo ADVANCE
        # valdria 0 y no habria nada que aprender.
        grafo = graph.warehouse_graph()
        politica = qlearning.QLearningPolicy()
        entorno = qlearning.TrainingEnv(grafo, 1, politica, seed=3, max_steps=60)
        entorno.reset()

        cerradas: list[qlearning.Transition] = []
        while not entorno.done:
            cerradas.extend(entorno.step())
        cerradas.extend(entorno.close_pending())

        avances = [
            transicion
            for transicion in cerradas
            if transicion.action is qlearning.Action.ADVANCE
        ]
        self.assertTrue(avances, "un AGV solo en el mapa tenia que avanzar")
        self.assertTrue(
            any(transicion.reward > 0.0 for transicion in avances),
            "algun ADVANCE tenia que cobrar el +2 de cruzar un tramo",
        )

    def test_llegar_paga_la_tarea_completa(self) -> None:
        grafo = graph.warehouse_graph()
        politica = qlearning.QLearningPolicy()
        entorno = qlearning.TrainingEnv(grafo, 1, politica, seed=3, max_steps=200)
        entorno.reset()

        cerradas: list[qlearning.Transition] = []
        while not entorno.done:
            cerradas.extend(entorno.step())
        cerradas.extend(entorno.close_pending())

        terminales = [t for t in cerradas if t.terminal]
        self.assertTrue(terminales)
        self.assertGreaterEqual(
            max(t.reward for t in terminales), config.REWARD_TASK_COMPLETE
        )

    def test_los_numeros_salen_de_config(self) -> None:
        grafo = graph.warehouse_graph()
        with mock.patch.object(config, "REWARD_TASK_COMPLETE", 1000.0):
            politica = qlearning.QLearningPolicy()
            entorno = qlearning.TrainingEnv(grafo, 1, politica, seed=3, max_steps=200)
            entorno.reset()
            cerradas: list[qlearning.Transition] = []
            while not entorno.done:
                cerradas.extend(entorno.step())
            cerradas.extend(entorno.close_pending())
        self.assertGreaterEqual(max(t.reward for t in cerradas if t.terminal), 1000.0)

    def test_la_invariante_del_almacen_aguanta_todo_el_entrenamiento(self) -> None:
        entrenador = qlearning.Trainer(graph.warehouse_graph(), ajustes(episodes=25))
        for _ in range(25):
            entrenador.env.reset()
            while not entrenador.env.done:
                entrenador.env.step()
                nodos = [a.current_node for a in entrenador.env.sim.agents]
                self.assertEqual(
                    len(set(nodos)),
                    len(nodos),
                    f"dos AGVs en el mismo nodo: {nodos}",
                )


class TestBaselineAdapter(unittest.TestCase):
    """La baseline se mide con la misma vara, pero decide lo mismo que en la fase 5."""

    def test_decide_igual_que_la_baseline_de_la_fase_5(self) -> None:
        grafo = graph.warehouse_graph()
        rng = random.Random(11)

        def corre(politica: object, rutas: list[tuple[str, str]]) -> list[tuple]:
            simulacion = simulation.Simulation(
                grafo, len(rutas), routes=rutas, policy=politica
            )
            if hasattr(politica, "bind"):
                politica.bind(simulacion)
            while not simulacion.done and simulacion.step < 200:
                simulacion.tick()
            return [
                (a.id, a.current_node, a.path_index, a.state, a.wait_time)
                for a in simulacion.agents
            ]

        for _ in range(20):
            rutas = qlearning.random_routes(grafo, 4, rng)
            self.assertEqual(
                corre(conflicts.BaselinePolicy(), rutas),
                corre(qlearning.BaselineAdapter(), rutas),
            )

    def test_cumple_el_contrato_de_politica(self) -> None:
        self.assertIsInstance(qlearning.BaselineAdapter(), conflicts.Policy)


class TestModelo(unittest.TestCase):
    """CRITERIO 4: la tabla se guarda con su metadata y se carga sin perder nada."""

    def test_guarda_la_metadata_de_la_corrida(self) -> None:
        entrenador = entrena(graph.warehouse_graph(), episodes=20, agents=3, seed=5)
        with tempfile.TemporaryDirectory() as carpeta:
            destino = entrenador.save(Path(carpeta) / "q_table.json")
            crudo = json.loads(destino.read_text(encoding="utf-8"))
            metadata = qlearning.load_metadata(destino)

        self.assertEqual(crudo["metadata"], metadata)
        self.assertEqual(metadata["map"], "warehouse")
        self.assertEqual(metadata["agents"], 3)
        self.assertEqual(metadata["seed"], 5)
        self.assertEqual(metadata["episodes_run"], 20)
        self.assertTrue(metadata["shared_q_table"])
        self.assertIn("trained_at", metadata)
        self.assertEqual(
            sum(metadata["visits"].values()), sum(entrenador.visits.values())
        )
        for clave in ("alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay"):
            self.assertIn(clave, metadata["hyperparameters"])

    def test_una_tabla_sin_metadata_se_sigue_cargando(self) -> None:
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "vieja.json"
            qlearning.QTable().save(destino)
            self.assertEqual(qlearning.load_metadata(destino), {})
            self.assertEqual(len(qlearning.QTable.load(destino)), 0)

    def test_la_tabla_del_disco_evalua_igual_que_la_de_memoria(self) -> None:
        grafo = graph.warehouse_graph()
        entrenador = entrena(grafo, episodes=60)

        def evalua(tabla: qlearning.QTable) -> list[dict[str, object]]:
            evaluador = qlearning.Trainer(grafo, ajustes(), q_table=tabla, learn=False)
            evaluador.run(25)
            return [fila.as_row() for fila in evaluador.history]

        with tempfile.TemporaryDirectory() as carpeta:
            destino = entrenador.save(Path(carpeta) / "q_table.json")
            del_disco = qlearning.QTable.load(destino)

        self.assertEqual(del_disco.as_dict(), entrenador.q.as_dict())
        self.assertEqual(evalua(del_disco), evalua(entrenador.q))


class TestRegistro(unittest.TestCase):
    """CRITERIO 7: el CSV tiene las columnas del enunciado."""

    def test_las_columnas_son_las_del_enunciado(self) -> None:
        self.assertEqual(
            qlearning.LOG_COLUMNS,
            (
                "episode",
                "epsilon",
                "total_reward",
                "avg_reward",
                "conflicts",
                "deadlocks",
                "completed_tasks",
                "makespan",
                "total_wait",
                "states_visited",
            ),
        )

    def test_una_fila_por_episodio_y_la_cabecera(self) -> None:
        entrenador = entrena(graph.warehouse_graph(), episodes=25)
        with tempfile.TemporaryDirectory() as carpeta:
            destino = qlearning.write_training_log(
                entrenador.history, Path(carpeta) / "training_log.csv"
            )
            with destino.open(encoding=config.ENCODING, newline="") as fichero:
                lector = csv.DictReader(fichero)
                self.assertEqual(tuple(lector.fieldnames or ()), qlearning.LOG_COLUMNS)
                filas = list(lector)

        self.assertEqual(len(filas), 25)
        self.assertEqual([int(fila["episode"]) for fila in filas], list(range(1, 26)))
        for fila in filas:
            self.assertLessEqual(int(fila["completed_tasks"]), 4)
            self.assertIn(int(fila["deadlocks"]), (0, 1))
            self.assertGreaterEqual(int(fila["makespan"]), 0)
            self.assertLessEqual(int(fila["states_visited"]), qlearning.state_space_size())
            float(fila["total_reward"])
            float(fila["avg_reward"])

    def test_el_resumen_tiene_una_fila_por_bloque(self) -> None:
        entrenador = entrena(graph.warehouse_graph(), episodes=30)
        lineas = qlearning.summary_lines(entrenador.history, 10)
        # cabecera del bloque + cabecera de columnas + regla + 3 bloques + pie
        self.assertEqual(len(lineas), 7)
        self.assertIn("1-10", lineas[3])
        self.assertIn("21-30", lineas[5])

    def test_la_comparacion_pone_las_dos_politicas(self) -> None:
        entrenador = entrena(graph.warehouse_graph(), episodes=20)
        referencia = qlearning._run_baseline(
            graph.warehouse_graph(), entrenador.cfg, 20
        )
        lineas = qlearning.compare_lines(entrenador.history, referencia)
        self.assertTrue(any("q-learning" in linea for linea in lineas))
        self.assertTrue(any("conflictos/tick" in linea for linea in lineas))


class TestCurvaDeAprendizaje(unittest.TestCase):
    """El PNG es opcional: sin matplotlib se avisa y se sigue."""

    def test_sin_matplotlib_no_revienta(self) -> None:
        entrenador = entrena(graph.warehouse_graph(), episodes=10)
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta) / "learning_curve.png"
            with mock.patch.dict(sys.modules, {"matplotlib": None}):
                with self.assertLogs(level="WARNING") as capturado:
                    self.assertIsNone(
                        qlearning.save_learning_curve(entrenador.history, destino)
                    )
            self.assertFalse(destino.exists())
        self.assertIn("matplotlib", "\n".join(capturado.output))

    def test_la_media_movil_no_cambia_el_largo(self) -> None:
        valores = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertEqual(qlearning.moving_average(valores, 1), valores)
        suavizado = qlearning.moving_average(valores, 3)
        self.assertEqual(len(suavizado), len(valores))
        self.assertAlmostEqual(suavizado[0], 1.0)
        self.assertAlmostEqual(suavizado[-1], 4.0)


class TestSinServidor(unittest.TestCase):
    """CRITERIO 6: entrenar no levanta el servidor ni habla con Unity."""

    def test_el_entrenamiento_no_toca_el_servidor(self) -> None:
        with mock.patch.object(server, "serve_forever") as falso:
            with mock.patch("socket.socket") as socket_falso:
                entrena(graph.warehouse_graph(), episodes=20)
        falso.assert_not_called()
        socket_falso.assert_not_called()


class TestCLI(unittest.TestCase):
    """Los dos subcomandos, de punta a punta."""

    def test_train_escribe_modelo_csv_y_devuelve_cero(self) -> None:
        import main

        with tempfile.TemporaryDirectory() as carpeta:
            modelo = Path(carpeta) / "q_table.json"
            registro = Path(carpeta) / "training_log.csv"
            codigo = main.main(
                [
                    "train",
                    "--map", "warehouse",
                    "--agents", "3",
                    "--episodes", "20",
                    "--seed", "42",
                    "--model", str(modelo),
                    "--log", str(registro),
                    "--no-curve",
                ]
            )
            self.assertEqual(codigo, 0)
            self.assertTrue(modelo.is_file())
            self.assertTrue(registro.is_file())
            self.assertEqual(qlearning.load_metadata(modelo)["agents"], 3)
            self.assertEqual(len(qlearning.read_training_log(registro)), 20)

    def test_evaluate_carga_el_modelo_y_devuelve_cero(self) -> None:
        import main

        with tempfile.TemporaryDirectory() as carpeta:
            modelo = Path(carpeta) / "q_table.json"
            main.main(
                ["train", "--episodes", "20", "--model", str(modelo), "--no-curve",
                 "--log", str(Path(carpeta) / "log.csv")]
            )
            with self.assertLogs(level="INFO") as capturado:
                codigo = main.main(
                    ["evaluate", "--model", str(modelo), "--episodes", "10"]
                )
        self.assertEqual(codigo, 0)
        self.assertIn("q-learning", "\n".join(capturado.output))

    def test_evaluate_sin_modelo_devuelve_dos(self) -> None:
        import main

        with tempfile.TemporaryDirectory() as carpeta:
            with self.assertLogs(level="ERROR") as capturado:
                codigo = main.main(
                    ["evaluate", "--model", str(Path(carpeta) / "no_esta.json")]
                )
        self.assertEqual(codigo, 2)
        self.assertIn("no existe el modelo", "\n".join(capturado.output))


if __name__ == "__main__":
    unittest.main()
