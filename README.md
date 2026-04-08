<p align="center"><h1 align="center">Echo State Networks - Experimental Phase</h1></p>
<p align="center">
    <em>Repositorio de experimentacion sobre hiperparametros y topologias de reservorio en Echo State Networks (ESN) con senales sinteticas y ERA5.</em>
    <br>
    <em>Experimental repository on hyperparameters and reservoir topologies in Echo State Networks (ESN) with synthetic signals and ERA5 data.</em>
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Jupyter-F37626.svg?style=default&logo=Jupyter&logoColor=white" alt="Jupyter">
    <img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/NetworkX-000000.svg?style=default" alt="NetworkX">
    <img src="https://img.shields.io/badge/ReservoirPy-2C7FB8.svg?style=default" alt="ReservoirPy">
</p>

---

## Language

- [Espanol](#espanol)
- [English](#english)

---

## Espanol

### Indice

- [Vision general](#vision-general)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Caracteristicas principales](#caracteristicas-principales)
- [Primeros pasos](#primeros-pasos)
  - [Requisitos previos](#requisitos-previos)
  - [Instalacion y ejecucion](#instalacion-y-ejecucion)
- [Resultados y salidas](#resultados-y-salidas)
- [Reproducibilidad](#reproducibilidad)
- [Contribuir](#contribuir)
- [Licencia](#licencia)
- [Reconocimientos](#reconocimientos)

### Vision general

Este repositorio concentra la fase de experimentacion de una linea de investigacion sobre Echo State Networks (ESN), centrada en responder tres preguntas clave:

- Que hiperparametros impactan mas el rendimiento en series temporales.
- Como cambia el comportamiento del modelo segun la topologia del reservorio:
  Erdos-Renyi (aleatoria), Watts-Strogatz (small-world) y Barabasi-Albert (scale-free).
- Que arquitectura resiste mejor cuando la senal se vuelve mas compleja, ruidosa y no estacionaria.

La experimentacion se organiza en cuatro notebooks:

1. Notebook 1: estudio en senal seno sintetica controlada (Phase A + Phase B).
2. Notebook 2: estudio equivalente sobre datos reales de ERA5.
3. Notebook 3: comparacion arquitectonica en una bateria sintetica de dificultad creciente.
4. Notebook 4: estudio completo sobre benchmark sintetico (analogo al flujo del Notebook 2).

### Estructura del proyecto

```sh
2_fase_experimentacion/
├── .gitignore
├── README.md
├── 1 - Echo State Networks on Sine analysis - Hyperparameter and Reservoir.ipynb
├── 2 - Echo State Networks on ERA5 - Hyperparameter and Reservoir.ipynb
├── 3 - Echo State Networks - Architecture Comparison on Increasingly Difficult Synthetic Signals.ipynb
├── 4 - Echo State Networks on Synthetic Benchmark - Hyperparameter and Reservoir.ipynb
├── data.grib
├── data.grib.5b7b6.idx
├── single_hp_outputs_1/
├── single_hp_outputs_2/
├── single_hp_outputs_3/
└── single_hp_outputs_4/
```

### Caracteristicas principales

- Evaluacion temporal rigurosa con TimeSeriesSplit para evitar fugas de informacion.
- Busqueda de hiperparametros en dos fases: exploracion inicial y refinamiento.
- Single Hyperparameter Study (OAT) para N, sr, lr, ridge y washout.
- Comparativa topologica explicita en ER, WS y BA.
- Benchmark de dificultad creciente para tensionar memoria, robustez al ruido y adaptacion a no estacionariedad.
- Salidas listas para tesis/articulo en CSV y PNG.

### Primeros pasos

#### Requisitos previos

- Python 3.10+
- pip
- Jupyter (VS Code o Jupyter Lab/Notebook)

Paquetes clave:

- numpy, pandas, scipy, xarray
- matplotlib, seaborn
- scikit-learn
- networkx
- reservoirpy

#### Instalacion y ejecucion

1. Clona y entra al repositorio:

```sh
git clone <url-del-repositorio>
cd 2_fase_experimentacion
```

2. Crea y activa entorno virtual:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

3. Instala dependencias:

```sh
pip install -U pip
pip install numpy pandas scipy xarray matplotlib seaborn scikit-learn networkx reservoirpy jupyter
```

4. Abre los notebooks y selecciona el kernel de .venv.

5. Orden recomendado de ejecucion:

```text
1 -> 2 -> 3 -> 4
```

Notas:

- data.grib se usa en el flujo con ERA5.
- single_hp_outputs_* contiene resultados generados por cada notebook.

### Resultados y salidas

Cada notebook exporta artefactos en su carpeta single_hp_outputs_X:

- CSV con metricas por pliegue, sensibilidad y comparativas topologicas.
- Heatmaps (por ejemplo k x p_rewire en Watts-Strogatz).
- Graficas de prediccion del mejor modelo.
- Resumenes finales de RMSE por senal y arquitectura.

### Reproducibilidad

- Mantener semilla fija en NumPy/ReservoirPy.
- Ejecutar cada notebook de arriba a abajo sin saltar celdas.
- Conservar CSV de salida como artefacto canonico de cada corrida.
- Documentar cambios de malla de hiperparametros junto al output.

### Contribuir

1. Crea una rama con nombre descriptivo.
2. Manten estructura de experimentacion por fases.
3. Verifica ejecucion completa de notebooks y exportacion de resultados.
4. Abre una PR explicando objetivo, hipotesis y resultados esperados.

### Licencia

Este proyecto se distribuye bajo la licencia MIT.
Consulta el archivo [LICENSE](LICENSE) para mas detalles.

### Reconocimientos

Trabajo experimental desarrollado en el contexto del TFM sobre topologias de reservorio en Echo State Networks.

Autor: Antonio Macias Ferrera.

---

## English

### Table of contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Main features](#main-features)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation and execution](#installation-and-execution)
- [Results and outputs](#results-and-outputs)
- [Reproducibility](#reproducibility-1)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

### Overview

This repository contains an ESN experimental phase focused on three key questions:

- Which hyperparameters drive performance in time-series forecasting.
- How reservoir topology affects behavior:
  Erdos-Renyi (random), Watts-Strogatz (small-world), and Barabasi-Albert (scale-free).
- Which architecture is more robust under higher complexity, stronger noise, and non-stationarity.

The work is organized into four notebooks:

1. Notebook 1: controlled synthetic sine signal study (Phase A + Phase B).
2. Notebook 2: equivalent study on ERA5 real-world data.
3. Notebook 3: architecture comparison on progressively harder synthetic signals.
4. Notebook 4: full synthetic benchmark study (analogous to Notebook 2 workflow).

### Project structure

```sh
2_fase_experimentacion/
├── .gitignore
├── README.md
├── 1 - Echo State Networks on Sine analysis - Hyperparameter and Reservoir.ipynb
├── 2 - Echo State Networks on ERA5 - Hyperparameter and Reservoir.ipynb
├── 3 - Echo State Networks - Architecture Comparison on Increasingly Difficult Synthetic Signals.ipynb
├── 4 - Echo State Networks on Synthetic Benchmark - Hyperparameter and Reservoir.ipynb
├── data.grib
├── data.grib.5b7b6.idx
├── single_hp_outputs_1/
├── single_hp_outputs_2/
├── single_hp_outputs_3/
└── single_hp_outputs_4/
```

### Main features

- Strict temporal evaluation with TimeSeriesSplit to avoid leakage.
- Two-stage hyperparameter search: broad exploration plus refinement.
- Single Hyperparameter Study (OAT) for N, sr, lr, ridge, and washout.
- Explicit topology comparison across ER, WS, and BA.
- Increasing-difficulty benchmark to stress memory, noise robustness, and adaptation.
- Thesis/paper-ready outputs in CSV and PNG.

### Getting started

#### Prerequisites

- Python 3.10+
- pip
- Jupyter (VS Code or Jupyter Lab/Notebook)

Core packages:

- numpy, pandas, scipy, xarray
- matplotlib, seaborn
- scikit-learn
- networkx
- reservoirpy

#### Installation and execution

1. Clone and enter the repository:

```sh
git clone <repository-url>
cd 2_fase_experimentacion
```

2. Create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```sh
pip install -U pip
pip install numpy pandas scipy xarray matplotlib seaborn scikit-learn networkx reservoirpy jupyter
```

4. Open notebooks and select the .venv kernel.

5. Recommended execution order:

```text
1 -> 2 -> 3 -> 4
```

Notes:

- data.grib is used in the ERA5 workflow.
- single_hp_outputs_* stores generated notebook artifacts.

### Results and outputs

Each notebook exports artifacts into single_hp_outputs_X folders:

- CSV files with fold-level metrics, sensitivity analyses, and topology comparisons.
- Heatmaps (for example k x p_rewire in Watts-Strogatz).
- Best-model prediction plots.
- Final RMSE summaries by signal and architecture.

### Reproducibility

- Keep fixed random seeds in NumPy/ReservoirPy.
- Execute notebooks top-to-bottom without skipping cells.
- Preserve output CSV files as canonical run artifacts.
- Document hyperparameter grid changes together with generated outputs.

### Contributing

1. Create a descriptive branch.
2. Keep the phase-based experimental structure.
3. Verify full notebook execution and output generation.
4. Open a PR with goals, hypothesis, and expected results.

### License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for full details.

### Acknowledgements

Experimental work developed within a Master's thesis on reservoir topologies in Echo State Networks.

Author: Antonio Macias Ferrera.
