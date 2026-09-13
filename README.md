# agentesAGV — Simulación multiagente de AGVs

La simulación de un almacén con AGVs: las rutas con A\*, el reparto de trabajo por
subasta y la coordinación con Q-Learning. Aquí vive toda la lógica; Unity solo dibuja.

El contrato HTTP con Unity está en [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Requisitos

Python **3.10 o superior** y nada más: todo es librería estándar, el Q-Learning
incluido. No hay dependencias que instalar.

```bash
python3 --version
```

En macOS el `/usr/bin/python3` del sistema es 3.9 y **no vale** (el código usa
`X | Y` en las anotaciones). Con `brew install python@3.12` tienes uno en
`/opt/homebrew/bin/python3.12`.

## Cómo se corre

Todo sale de `python/main.py`, desde la raíz del repo:

```bash
python3 python/main.py --help
```

| Subcomando | Qué hace |
|---|---|
| `serve` | Levanta el servidor HTTP y atiende a Unity |
| `map` | Muestra el mapa lógico y lo valida |
| `simulate` | Corre la simulación sin servidor, paso a paso por el log |
| `train` | Entrena la Q-table, sin servidor y sin Unity |
| `evaluate` | Carga una Q-table y la juega, contra la baseline |

Para verlo moverse sin abrir Unity, que es lo más rápido:

```bash
python3 python/main.py simulate --agents 5 --steps 600 --deliveries
```

Con `--deliveries` el trabajo sale de la subasta; añade `--bus` y escupe además la
negociación entera, mensaje a mensaje.

Para servirle a Unity:

```bash
python3 python/main.py serve --agents 5 --port 5055 --policy qlearning
```

Por defecto son `127.0.0.1:5000`, 1 AGV y política `baseline`. Con `--agents 1` no
hay con quién chocar y `stats.conflicts` sale siempre 0: para ver conflictos hacen
falta varios. `Ctrl+C` cierra limpio.

En macOS el receptor de AirPlay se queda con el puerto 5000; si algo va raro, usa
`--port`.

## Entrenar y evaluar

```bash
python3 python/main.py train --episodes 1000 --agents 4 --seed 42
python3 python/main.py evaluate --agents 4
```

`train` tarda unos 8 segundos y deja la Q-table en `python/models/q_table.json` y el
log en `results/training_log.csv`. La semilla es 42 por defecto, así que la misma
orden da la misma corrida. `evaluate` la juega greedy contra la baseline sobre los
mismos episodios.

Todo sale por `stderr` con `logging`, nunca con `print`. Con `-v` se activa `DEBUG`.

## Con Unity

Este repo no se abre desde Unity. El proyecto de Unity tiene su propio adaptador
(`PythonBridge/serve_unity.py`) que importa estos módulos y se arranca desde allí
con `./Iniciar_AGV.command`; está explicado en el README de ese repo.

Si tu clon no se llama `MultiAgents/`, hay que pasarle la ruta:

```bash
./Iniciar_AGV.command --project ~/ruta/a/este/repo
```

## Qué hay dentro

```
python/
  main.py         los cinco subcomandos
  simulation.py   el almacen en marcha: ticks, conflictos, desatasco y snapshot
  agent.py        el AGV: su ruta, su carga y su bateria
  graph.py        el mapa como grafo, y A*
  missions.py     las misiones, el bus de mensajes y la subasta
  conflicts.py    los tipos de choque y quien cede el paso
  obstacles.py    los obstaculos que aparecen y estorban
  qlearning.py    el estado, la recompensa y la Q-table
  server.py       el servidor HTTP, sin logica de almacen
  config.py       constantes, recompensas y logging
  maps/           los mapas en JSON
  models/         las Q-tables entrenadas
docs/PROTOCOL.md  el contrato con Unity
results/          lo que sale de train y evaluate (no se versiona)
```
