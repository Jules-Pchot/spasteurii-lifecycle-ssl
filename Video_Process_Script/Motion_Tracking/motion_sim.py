"""Synthetic S. pasteurii videos with a known ground truth, in physical units (µm, s), for the concept notebooks.

Numbers come from the project's Research Report (see `PARAMETER_SOURCES`). No kinematic value has been measured for
S. pasteurii: swimming values are B. subtilis / E. coli proxies, the diffusion coefficient is a Stokes-Einstein
calculation, and the phase presets (`PHASES`) turn the report's qualitative, mostly inferred expectations into
illustrative numbers.

Three kinds of cells:
  - brownian: passive cell (non-motile, flagella off), thermal diffusion only, carried by the fluid flow;
  - swimmer:  active cell, run-and-tumble swimming (+ optional circling near the glass) + diffusion + flow;
  - stuck:    cell attached to the glass, chain, aggregate or spore stuck in place: does not follow the fluid.
Every object also moves with the stage drift (the whole image shifts).
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

KINDS = ("brownian", "swimmer", "stuck")
KIND_LABELS = {"brownian": "passive (brownienne)", "swimmer": "active (nageuse)", "stuck": "collée / chaîne / agrégat"}
KIND_COLORS = {"brownian": "#3987e5", "swimmer": "#d95926", "stuck": "#199e70"}

CELL_LENGTH = 3.0  # µm, typical S. pasteurii length (0.5-1.2 x 1.3-4.0 µm; ~3 µm observed)
CELL_WIDTH = 1.0   # µm

# Your two videos. Pixel size ESTIMATED from the measured cell length (19 px at x40, 50 px at x100) and a 3 µm
# cell: to calibrate with a stage micrometer.
VIDEOS = {"Expo (×40)": {"dt": 1 / 3.8, "um_per_px": 3.0 / 19, "field": (1920 * 3.0 / 19, 1080 * 3.0 / 19)},
          "Stationary (×100)": {"dt": 1 / 2.9, "um_per_px": 3.0 / 50, "field": (3584 * 3.0 / 50, 2746 * 3.0 / 50)}}

PARAMETER_SOURCES = [
    ("Forme, taille", "bâtonnet 0,5–1,2 × 1,3–4,0 µm, ~3 µm observé", "S. pasteurii", "établi", "Yoon 2001 ; Bergey's ; Bang 2001"),
    ("Flagelles", "péritriches, 39 gènes flagellaires", "S. pasteurii", "établi (image isolée)", "Bergey's ; Ghosh 2019 ; Pei 2021"),
    ("Vitesse de nage", "15–25 µm/s", "B. subtilis, E. coli", "proxy", "Hamby 2018"),
    ("Vitesse fin de phase log", "6–13 µm/s", "bactéries de l'étude", "proxy", "Hamby 2018"),
    ("Intervalle entre tumbles", "~3 s", "E. coli, B. subtilis", "proxy", "Hamby 2018"),
    ("Diffusion brownienne", "0,2–0,3 µm²/s", "bâtonnet 1 × 3 µm", "calcul (Stokes-Einstein)", "Research Report"),
    ("Mobilité vs CaCl₂", "restreinte et floculation à 0,5 M", "S. pasteurii", "établi", "Xiao 2022"),
    ("Cristaux CaCO₃", "17–46 µm, immobiles", "MICP en puce", "établi", "Elmaloglou 2022"),
    ("Profondeur de champ ×100", "< 1 µm : les nageuses sortent du plan", "optique", "physique", "Research Report"),
]


@dataclass(frozen=True)
class SimConfig:
    density: float = 6.0            # cells per 1000 µm²
    n_frames: int = 60
    width: float = 150.0            # simulated field (µm)
    height: float = 100.0
    dt: float = 1 / 3.8             # s between frames (Expo 0.26, Stationary 0.35)
    diffusion: float = 0.25         # µm²/s, Stokes-Einstein for a 1 x 3 µm rod
    swimmer_fraction: float = 0.3
    stuck_fraction: float = 0.1
    swim_speed: float = 20.0        # µm/s (proxy B. subtilis / E. coli: 15-25)
    speed_spread: float = 0.25      # relative spread of the swimming speed between cells
    tumble_rate: float = 1 / 3      # tumbles per second (proxy: one every ~3 s)
    circle_rate: float = 0.0        # rad/s, clockwise circling of swimmers near the glass
    focus_lifetime: float = 0.0     # s, mean time a swimmer stays in the focal plane (0 = always in focus)
    drift: tuple = (0.0, 0.0)       # stage drift (µm/s): moves everything
    flow: tuple = (0.0, 0.0)        # capillary / evaporation flow (µm/s): moves the free cells, not the stuck ones
    detection_noise: float = 0.1    # localization error (µm)
    miss_rate: float = 0.0          # probability that an in-focus cell is not detected on a frame
    seed: int = 0

    @property
    def n_cells(self) -> int:
        return max(2, round(self.density * self.width * self.height / 1000))


# Growth-phase presets. The kinematic trends follow the report's phase table (mostly INFERRED from Bacillus
# biology, an hypothesis to test); the exact fractions and densities are illustrative choices, not measurements.
PHASES = {
    "Lag": {"config": dict(density=2.0, swimmer_fraction=0.3, stuck_fraction=0.1, swim_speed=15.0),
            "kinematics": "Faible densité ; fraction mobile variable (flagelles à resynthétiser)",
            "scene": "Cellules isolées, éventuelles spores résiduelles en germination",
            "evidence": "Inféré (proxys Bacillus)"},
    "Exponentielle": {"config": dict(density=6.0, swimmer_fraction=0.6, stuck_fraction=0.1, swim_speed=20.0,
                                     circle_rate=0.3),
                      "kinematics": "Fraction mobile maximale ; trajectoires balistiques, cercles près des surfaces ; "
                                    "mélange cellules mobiles + chaînes",
                      "scene": "Bâtonnets isolés / doublets, divisions",
                      "evidence": "Proxy B. subtilis (Kearns & Losick 2005)"},
    "Fin exponentielle (fenêtre MICP)": {"config": dict(density=10.0, swimmer_fraction=0.3, stuck_fraction=0.15,
                                                        swim_speed=10.0),
                                         "kinematics": "Ralentissement des vitesses ; hausse de la part de cellules "
                                                       "browniennes",
                                         "scene": "Densité élevée ; uréase au plateau",
                                         "evidence": "Proxy (Hamby 2018) + Lapierre & Huber 2024"},
    "Stationnaire": {"config": dict(density=10.0, swimmer_fraction=0.1, stuck_fraction=0.3, swim_speed=6.0),
                     "kinematics": "Majorité browniennes ; nage résiduelle",
                     "scene": "Spores réfringentes en sporanges renflés, chaînes, agrégats",
                     "evidence": "Inféré (biologie des Bacillus)"},
    "Déclin": {"config": dict(density=8.0, swimmer_fraction=0.02, stuck_fraction=0.4, swim_speed=6.0),
               "kinematics": "Essentiellement brownien et dérive",
               "scene": "Spores libres, cellules lysées (« fantômes »), débris",
               "evidence": "Inféré"},
}


def phase_config(phase: str, **overrides) -> SimConfig:
    """SimConfig of a growth-phase preset, with any field overridden."""
    return replace(SimConfig(**PHASES[phase]["config"]), **overrides)


def simulate(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ground truth and detections of a synthetic video (positions in µm, times in s).

    truth:      frame, t, true_id, kind, x, y, inside (in the field), in_focus
    detections: frame, x, y (+ true_id, kind for evaluation only), shuffled within each frame
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_cells
    p_swim, p_stuck = cfg.swimmer_fraction, cfg.stuck_fraction
    kinds = rng.choice(KINDS, n, p=[1 - p_swim - p_stuck, p_swim, p_stuck])
    position = rng.uniform([0, 0], [cfg.width, cfg.height], (n, 2))
    heading = rng.uniform(0, 2 * np.pi, n)
    speed = cfg.swim_speed * np.clip(1 + cfg.speed_spread * rng.normal(size=n), 0.2, None)
    free, swimmer = kinds != "stuck", kinds == "swimmer"
    in_focus = np.ones(n, dtype=bool)
    p_tumble = 1 - np.exp(-cfg.tumble_rate * cfg.dt)
    p_switch = 1 - np.exp(-cfg.dt / cfg.focus_lifetime) if cfg.focus_lifetime > 0 else 0.0

    frames = []
    for frame in range(cfg.n_frames):
        frames.append(pd.DataFrame({"frame": frame, "t": frame * cfg.dt, "true_id": np.arange(n), "kind": kinds,
                                    "x": position[:, 0], "y": position[:, 1], "in_focus": in_focus.copy()}))
        step = np.zeros((n, 2))
        step[free] += rng.normal(0, np.sqrt(2 * cfg.diffusion * cfg.dt), (free.sum(), 2))
        step[free] += np.asarray(cfg.flow) * cfg.dt
        tumble = swimmer & (rng.random(n) < p_tumble)
        heading[tumble] = rng.uniform(0, 2 * np.pi, tumble.sum())
        heading[swimmer] += cfg.circle_rate * cfg.dt
        step[swimmer] += (speed * cfg.dt)[:, None][swimmer] * np.c_[np.cos(heading), np.sin(heading)][swimmer]
        step += np.asarray(cfg.drift) * cfg.dt
        position = position + step
        switch = swimmer & (rng.random(n) < p_switch)  # swimmers leave / re-enter the focal plane
        in_focus[switch] = ~in_focus[switch]
    truth = pd.concat(frames, ignore_index=True)
    truth["inside"] = truth.x.between(0, cfg.width) & truth.y.between(0, cfg.height)

    seen = truth[truth.inside & truth.in_focus & (rng.random(len(truth)) >= cfg.miss_rate)].copy()
    seen[["x", "y"]] += rng.normal(0, cfg.detection_noise, (len(seen), 2))
    detections = (seen.sample(frac=1, random_state=cfg.seed).sort_values("frame", kind="stable")
                  [["frame", "x", "y", "true_id", "kind"]].reset_index(drop=True))
    return truth, detections
