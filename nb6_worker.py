"""Funciones de trabajo del Notebook 6 (en un módulo para poder paralelizar con joblib)."""
import time
from functools import lru_cache
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.linear_model import Ridge as SkRidge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge

N_FOLDS = 10
LAG     = 10
T       = 15_000
t_axis_unused = None

# ---------------------------------------------------------------------------
# Señales S1-S10 de NB3 con semilla controlable
# ---------------------------------------------------------------------------
def make_signals(seed, noiseless=False):
    """Las diez señales de NB3 (celda 6) con semilla de ruido controlada."""
    randn = (lambda n: np.zeros(n)) if noiseless else np.random.RandomState(seed).randn
    # ---------------------------------------------------------------------------
    # Generar las diez señales
    # ---------------------------------------------------------------------------
    t_axis = np.arange(T)
    BASELINE_PERIODS = (50, 127, 310)

    # Núcleo periódico limpio compartido de Notebook 1
    baseline_nb1_clean = (
          np.sin(2 * np.pi * t_axis / BASELINE_PERIODS[0])
        + 0.50 * np.sin(2 * np.pi * t_axis / BASELINE_PERIODS[1])
        + 0.30 * np.sin(2 * np.pi * t_axis / BASELINE_PERIODS[2])
    )

    # ---- Señal S1: núcleo periódico exacto de Notebook 1 + sigma=0.05 ----
    s1 = baseline_nb1_clean + 0.05 * randn(T)

    # ---- Señal S2: 5 frec. + pareja batiente de periodos cercanos (antes S3)
    s2 = (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.22 * randn(T)
    )

    # ---- Señal S3: 7 frec. + AM fuerte (antes S5) ---------------------------
    am3 = 1.0 + 0.50 * np.sin(2 * np.pi * t_axis / 3000)
    s3 = am3 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
    ) + 0.56 * randn(T)

    # ---- Señal S4: 9 frec. + batido multiescala caótico (antes S7) ----------
    am4 = 1.0 + 0.80 * np.sin(2 * np.pi * t_axis / 1500)
    s4 = am4 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
        + 0.10 * np.sin(2 * np.pi * t_axis / 19)
        + 0.08 * np.sin(2 * np.pi * t_axis / 71)
    ) + 0.85 * randn(T)

    # ---- Señal S5: 11 frec. + AM doble (antes S9) ---------------------------
    am5 = (1.0 + 0.60 * np.sin(2 * np.pi * t_axis / 3500)) * (1.0 + 0.40 * np.sin(2 * np.pi * t_axis / 500))
    s5 = am5 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
        + 0.10 * np.sin(2 * np.pi * t_axis / 19)
        + 0.08 * np.sin(2 * np.pi * t_axis / 71)
        + 0.07 * np.sin(2 * np.pi * t_axis / 29)
        + 0.06 * np.sin(2 * np.pi * t_axis / 13)
    ) + 1.20 * randn(T)

    # ---- Señal S6: 12 frec. + envolvente de triple AM -----------------------
    # Añade una tercera capa rápida de AM (T_AM=120) sobre la doble AM de S5,
    # creando una estructura de amplitud no estacionaria con tres escalas temporales.
    am6 = (
        (1.0 + 0.60 * np.sin(2 * np.pi * t_axis / 3500))
      * (1.0 + 0.40 * np.sin(2 * np.pi * t_axis / 500))
      * (1.0 + 0.25 * np.sin(2 * np.pi * t_axis / 120))
    )
    s6 = am6 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
        + 0.10 * np.sin(2 * np.pi * t_axis / 19)
        + 0.08 * np.sin(2 * np.pi * t_axis / 71)
        + 0.07 * np.sin(2 * np.pi * t_axis / 29)
        + 0.06 * np.sin(2 * np.pi * t_axis / 13)
        + 0.05 * np.sin(2 * np.pi * t_axis / 97)
    ) + 1.50 * randn(T)

    # ---- Señal S7: 13 frec. + componente FM + triple AM ---------------------
    # Introduce modulación de frecuencia: un componente tiene frecuencia instantánea
    # variable en el tiempo (deriva tipo chirp), haciendo no estacionaria la estructura espectral.
    fm_comp7 = 0.18 * np.sin(2 * np.pi * t_axis / 45 + 1.5 * np.sin(2 * np.pi * t_axis / 1200))
    am7 = (
        (1.0 + 0.65 * np.sin(2 * np.pi * t_axis / 3500))
      * (1.0 + 0.45 * np.sin(2 * np.pi * t_axis / 500))
      * (1.0 + 0.30 * np.sin(2 * np.pi * t_axis / 120))
    )
    s7 = am7 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
        + 0.10 * np.sin(2 * np.pi * t_axis / 19)
        + 0.08 * np.sin(2 * np.pi * t_axis / 71)
        + 0.07 * np.sin(2 * np.pi * t_axis / 29)
        + 0.06 * np.sin(2 * np.pi * t_axis / 13)
        + 0.05 * np.sin(2 * np.pi * t_axis / 97)
        + fm_comp7
    ) + 1.80 * randn(T)

    # ---- Señal S8: 14 frec. + cambios de régimen ----------------------------
    # Una envolvente lenta de onda cuadrada (T_sw=6000) crea cambios bruscos de régimen de amplitud,
    # alternando entre dos estados de amplitud cada ~3000 instantes.
    regime8 = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * t_axis / 6000)))  # 0 or 1
    fm_comp8 = 0.18 * np.sin(2 * np.pi * t_axis / 45 + 1.5 * np.sin(2 * np.pi * t_axis / 1200))
    am8 = (
        (1.0 + 0.65 * np.sin(2 * np.pi * t_axis / 3500))
      * (1.0 + 0.45 * np.sin(2 * np.pi * t_axis / 500))
      * (1.0 + 0.30 * np.sin(2 * np.pi * t_axis / 120))
      * (0.6 + 0.8 * regime8)  # régimen: amplitud 0.6 (baja) o 1.4 (alta)
    )
    s8 = am8 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
        + 0.10 * np.sin(2 * np.pi * t_axis / 19)
        + 0.08 * np.sin(2 * np.pi * t_axis / 71)
        + 0.07 * np.sin(2 * np.pi * t_axis / 29)
        + 0.06 * np.sin(2 * np.pi * t_axis / 13)
        + 0.05 * np.sin(2 * np.pi * t_axis / 97)
        + fm_comp8
        + 0.04 * np.sin(2 * np.pi * t_axis / 61)
    ) + 2.20 * randn(T)

    # ---- Señal S9: 15 frec. + ruido heteroscedástico multiplicativo ----------
    # La propia varianza del ruido oscila lentamente, haciendo heteroscedástica la señal:
    # alternan épocas de ruido bajo y alto, independientemente de la amplitud de la señal.
    regime9 = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * t_axis / 6000)))
    fm_comp9 = 0.18 * np.sin(2 * np.pi * t_axis / 45 + 1.5 * np.sin(2 * np.pi * t_axis / 1200))
    am9 = (
        (1.0 + 0.65 * np.sin(2 * np.pi * t_axis / 3500))
      * (1.0 + 0.45 * np.sin(2 * np.pi * t_axis / 500))
      * (1.0 + 0.30 * np.sin(2 * np.pi * t_axis / 120))
      * (0.6 + 0.8 * regime9)
    )
    mult_noise9 = 1.0 + 0.50 * np.abs(np.sin(2 * np.pi * t_axis / 900))
    s9 = am9 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
        + 0.10 * np.sin(2 * np.pi * t_axis / 19)
        + 0.08 * np.sin(2 * np.pi * t_axis / 71)
        + 0.07 * np.sin(2 * np.pi * t_axis / 29)
        + 0.06 * np.sin(2 * np.pi * t_axis / 13)
        + 0.05 * np.sin(2 * np.pi * t_axis / 97)
        + fm_comp9
        + 0.04 * np.sin(2 * np.pi * t_axis / 61)
        + 0.04 * np.sin(2 * np.pi * t_axis / 41)
    ) + 2.60 * mult_noise9 * randn(T)

    # ---- Señal S10: 16 frec. + complejidad no lineal completa ---------------
    # Combina todos los mecanismos: triple AM, FM, cambios de régimen, ruido multiplicativo,
    # más un par en cuadratura acoplado en fase que crea un atractor 2D rotatorio.
    regime10 = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * t_axis / 6000)))
    fm_comp10 = 0.18 * np.sin(2 * np.pi * t_axis / 45 + 1.5 * np.sin(2 * np.pi * t_axis / 1200))
    phase_drift = 2 * np.pi * t_axis / 800
    coupled_pair10 = (
        0.12 * np.sin(2 * np.pi * t_axis / 23 + np.sin(phase_drift))
      + 0.12 * np.cos(2 * np.pi * t_axis / 23 + np.sin(phase_drift))
    )
    am10 = (
        (1.0 + 0.70 * np.sin(2 * np.pi * t_axis / 4000))
      * (1.0 + 0.50 * np.sin(2 * np.pi * t_axis / 400))
      * (1.0 + 0.35 * np.sin(2 * np.pi * t_axis / 110))
      * (0.5 + 0.9 * regime10)
    )
    mult_noise10 = 1.0 + 0.65 * np.abs(np.sin(2 * np.pi * t_axis / 750))
    s10 = am10 * (
          baseline_nb1_clean
        + 0.25 * np.sin(2 * np.pi * t_axis / 83)
        + 0.20 * np.sin(2 * np.pi * t_axis / 53)
        + 0.15 * np.sin(2 * np.pi * t_axis / 200)
        + 0.12 * np.sin(2 * np.pi * t_axis / 37)
        + 0.10 * np.sin(2 * np.pi * t_axis / 19)
        + 0.08 * np.sin(2 * np.pi * t_axis / 71)
        + 0.07 * np.sin(2 * np.pi * t_axis / 29)
        + 0.06 * np.sin(2 * np.pi * t_axis / 13)
        + 0.05 * np.sin(2 * np.pi * t_axis / 97)
        + fm_comp10
        + 0.04 * np.sin(2 * np.pi * t_axis / 61)
        + 0.04 * np.sin(2 * np.pi * t_axis / 41)
        + coupled_pair10
    ) + 3.00 * mult_noise10 * randn(T)


    return {f"S{i}": v for i, v in enumerate([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10], 1)}


@lru_cache(maxsize=64)
def get_signal(seed, name):
    # Devuelve (señal, suelo de ruido). Suelo = RMS del término de ruido: ningún predictor
    # que solo vea el pasado puede bajar de ese RMSE en un paso (el ruido es blanco).
    sig   = make_signals(seed)[name]
    clean = make_signals(seed, noiseless=True)[name]
    return sig, float(np.sqrt(np.mean((sig - clean) ** 2)))

def build_dataset(sig, lag=LAG):
    X = np.column_stack([sig[i : len(sig) - lag + i] for i in range(lag)])
    return X, sig[lag:].reshape(-1, 1)



# ---------------------------------------------------------------------------
# Reservorio, evaluación y ejecución en paralelo
# ---------------------------------------------------------------------------
def make_W(topo, N, tp, seed, rescale_to=None):
    # Misma construcción que NB1-NB4 (pesos randn sobre la máscara de aristas, grafos dirigidos)
    # pero con semilla. Si rescale_to es None NO se reescala (protocolo de NB1-NB4).
    if topo == "NONE":
        return np.zeros((N, N)), 0, 0.0
    if topo == "ER":
        G = nx.erdos_renyi_graph(N, tp["p"], seed=seed, directed=True)
    elif topo == "WS":
        k = min(max(int(tp["k"]), 2), N - 1)
        k -= k % 2
        G = nx.watts_strogatz_graph(N, k, tp["p_rw"], seed=seed).to_directed()
    elif topo == "BA":
        G = nx.barabasi_albert_graph(N, min(max(1, int(tp["m"])), N - 1), seed=seed).to_directed()
    mask = nx.to_numpy_array(G) > 0
    W = np.zeros((N, N))
    W[mask] = np.random.RandomState(seed).randn(mask.sum())
    rho = float(np.max(np.abs(np.linalg.eigvals(W)))) if mask.any() else 0.0
    if rescale_to is not None and rho > 1e-12:
        W *= rescale_to / rho
    return W, int(mask.sum()), rho


def run_job(job):
    t0 = time.time()
    rec = dict(job)
    try:
        sig, floor = get_signal(job["sig_seed"], job["signal"])
        X, y = build_dataset(sig)
        topo = job["topo"]
        W, n_edges, rho = None, 0, np.nan
        if topo not in ("PERS", "LIN"):
            tp = {k: job[k] for k in ("p", "k", "p_rw", "m") if not pd.isna(job[k])}
            W, n_edges, rho = make_W(topo, int(job["N"]), tp, job["topo_seed"],
                                     job["sr"] if job["variant"] == "scaled" else None)
        rm = []
        for tr, te in TimeSeriesSplit(n_splits=N_FOLDS).split(X):
            if topo == "PERS":                       # y(t) = s(t-1)
                pred = X[te][:, -1]
            else:
                sc = StandardScaler()
                Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
                if topo == "LIN":                    # ridge lineal sobre los retardos
                    pred = SkRidge(alpha=1.0).fit(Xtr, y[tr]).predict(Xte).ravel()
                else:
                    res = Reservoir(units=int(job["N"]), W=W, lr=job["lr"], seed=int(job["topo_seed"]))
                    mod = res >> Ridge(ridge=job["ridge"])
                    mod.fit(Xtr, y[tr], warmup=int(job["washout"]))
                    pred = np.asarray(mod.run(Xte)).ravel()
            rm.append(float(np.sqrt(np.mean((y[te].ravel() - pred) ** 2))))
        rec.update(rmse=float(np.mean(rm)), rmse_fold_std=float(np.std(rm, ddof=1)),
                   floor=floor, n_edges=n_edges, rho_raw=rho)
    except Exception as exc:
        rec.update(rmse=np.nan, rmse_fold_std=np.nan, floor=np.nan, n_edges=-1, rho_raw=np.nan, error=str(exc)[:80])
    rec["secs"] = time.time() - t0
    return rec


