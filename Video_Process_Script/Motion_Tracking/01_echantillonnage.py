import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(mo.notebook_dir()))
    import reference as ref
    import tutor
    from motion_sim import CELL_LENGTH, KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, phase_config, simulate
    return CELL_LENGTH, KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, mo, np, phase_config, plt, ref, simulate, tutor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 — Échantillonnage temporel : peut-on suivre *S. pasteurii* sur tes vidéos ?

    Pour relier une bactérie d'une image à la suivante, elle ne doit **pas avoir bougé plus loin que sa voisine la
    plus proche**. Sinon, on ne sait plus qui est qui. On le résume par un rapport :

    $$r = \frac{\text{déplacement entre deux images}}{\text{distance à la plus proche voisine}}$$

    - $r < 0.5$ : facile ;
    - $0.5 < r < 1$ : faisable avec une bonne association (notebook 02) et une prédiction du mouvement (03) ;
    - $r > 1$ : ambigu.

    Le rapport donne aussi une règle plus stricte : un déplacement par image **inférieur à la moitié de la longueur
    d'une cellule**, soit 1,5 µm pour *S. pasteurii*.
    """)
    return


@app.cell(hide_code=True)
def _(mo, ref):
    nn_distance = ref.nearest_neighbor_distance
    mo.md(r"""
    ## Les deux grandeurs en jeu

    - **Déplacement entre deux images.** Une nageuse parcourt $v\,dt$ : à 20 µm/s, c'est 5 µm pour Expo (0,26 s) et
      7 µm pour Stationary (0,35 s). Une cellule passive diffuse de $\sqrt{4 D\,dt}$ seulement : avec D = 0,25 µm²/s,
      environ 0,5 µm.
    - **Distance à la plus proche voisine.** Elle ne dépend que de la **densité** de cellules, pas du temps.
    """)
    return (nn_distance,)


@app.cell(hide_code=True)
def _(PHASES, VIDEOS, mo):
    phase = mo.ui.dropdown(list(PHASES), value="Exponentielle", label="Phase (préréglage)")
    video = mo.ui.dropdown(list(VIDEOS), value="Expo (×40)", label="Vidéo")
    fps = mo.ui.slider(1, 60, step=1, value=4, label="images par seconde (pour comparer)", show_value=True)
    mo.vstack([mo.md("### Réglages"), phase, video,
               mo.md("Le curseur ci-dessous ne sert qu'à la dernière figure : tes vidéos sont à 3,8 et 2,9 images/s."),
               fps])
    return fps, phase, video


@app.cell(hide_code=True)
def _(VIDEOS, phase, phase_config, simulate, video):
    cfg = phase_config(phase.value, n_frames=20, dt=VIDEOS[video.value]["dt"])
    truth, detections = simulate(cfg)
    return cfg, detections, truth


@app.cell(hide_code=True)
def _(CELL_LENGTH, KIND_COLORS, KIND_LABELS, VIDEOS, cfg, detections, mo, np, nn_distance, plt, truth, video):
    _um_per_px = VIDEOS[video.value]["um_per_px"]
    _inside = truth[truth.inside].sort_values(["true_id", "frame"])
    _next = _inside.groupby("true_id")[["frame", "x", "y"]].shift(-1)
    _ok = _next.frame == _inside.frame + 1
    _step = np.hypot((_next.x - _inside.x)[_ok], (_next.y - _inside.y)[_ok])
    _kind = _inside.kind[_ok]
    _neighbor = np.median(np.concatenate([nn_distance(g[["x", "y"]].to_numpy())
                                          for _, g in detections.groupby("frame") if len(g) > 1]))
    _fig, _ax = plt.subplots(figsize=(9, 3.8))
    _rows = []
    for _i, _k in enumerate(["brownian", "swimmer", "stuck"]):
        _d = _step[_kind == _k]
        if len(_d):
            _m = np.median(_d)
            _ax.barh(_i, _m, color=KIND_COLORS[_k])
            _ax.text(_m, _i, f"  {_m:.1f} µm = {_m / _um_per_px:.0f} px", va="center", fontsize=9)
            _rows.append((_k, _m))
    _ax.axvline(_neighbor, color="black", lw=1.2)
    _ax.text(_neighbor, 2.45, " plus proche voisine", fontsize=8)
    _ax.axvline(CELL_LENGTH / 2, color="gray", ls="--", lw=1)
    _ax.text(CELL_LENGTH / 2, -0.55, " ½ cellule (règle du rapport)", fontsize=8, color="gray")
    _ax.set(yticks=range(3), yticklabels=[KIND_LABELS[k] for k in ["brownian", "swimmer", "stuck"]],
            xlabel="déplacement médian entre deux images (µm)",
            title=f"{video.value}, {cfg.dt:.2f} s entre images — voisine à {_neighbor:.1f} µm")
    _ax.set_xlim(0, max(_neighbor, max(m for _, m in _rows)) * 1.35)
    _fig.tight_layout()
    _r = {k: m / _neighbor for k, m in _rows}
    _swim = _r.get("swimmer")
    _kind_callout = "success" if (_swim or 0) < 0.5 else "warn" if (_swim or 0) < 1 else "danger"
    mo.vstack([mo.md("## Déplacement par image, à l'échelle de tes vidéos"), _fig,
               mo.callout(mo.md(f"r passives = **{_r.get('brownian', 0):.2f}** · r nageuses = "
                                f"**{_swim:.2f}**" if _swim is not None else
                                f"r passives = **{_r.get('brownian', 0):.2f}** · pas de nageuses dans ce préréglage"),
                          kind=_kind_callout)])
    return


@app.cell(hide_code=True)
def _(cfg, detections, mo, plt):
    _a, _b = detections[detections.frame == 0], detections[detections.frame == 1]
    _fig, _ax = plt.subplots(figsize=(9, 6))
    _ax.scatter(_a.x, _a.y, s=16, color="#3987e5", label="image t")
    _ax.scatter(_b.x, _b.y, s=16, color="#d95926", label="image t+1")
    _ax.set(xlim=(0, cfg.width), ylim=(cfg.height, 0), aspect="equal", xlabel="µm", ylabel="µm",
            title="Deux images superposées : saurais-tu dire qui est qui ?")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("*Les cellules passives se superposent presque, les nageuses ont sauté de plusieurs µm.*"), _fig])
    return


@app.cell(hide_code=True)
def _(CELL_LENGTH, cfg, fps, mo, np, plt, tutor):
    _fps = np.linspace(1, 60, 300)
    _fig, _ax = plt.subplots(figsize=(8, 3.6))
    _ax.plot(_fps, np.sqrt(4 * cfg.diffusion / _fps), color="#3987e5", label="passive (D = 0,25 µm²/s)")
    for _v, _ls in [(20, "-"), (10, "--"), (6, ":")]:
        _ax.plot(_fps, _v / _fps, color="#d95926", ls=_ls, label=f"nageuse à {_v} µm/s")
    _ax.axhline(CELL_LENGTH / 2, color="gray", lw=1)
    _ax.text(60, CELL_LENGTH / 2, "½ cellule ", ha="right", va="bottom", fontsize=8, color="gray")
    for _x, _name in [(2.9, "Stationary"), (3.8, "Expo"), (30, "conseil du rapport")]:
        _ax.axvline(_x, color="black", lw=0.8, ls="--")
        _ax.text(_x, 9.5, f" {_name}", rotation=90, va="top", fontsize=8)
    _ax.axvline(fps.value, color="#199e70", lw=2)
    _ax.set(xlabel="images par seconde", ylabel="déplacement par image (µm)", ylim=(0, 10), xscale="log",
            title="Quelle cadence pour suivre chaque type de mouvement ?")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig,
               tutor.question(mo, "Avec tes vidéos (2,9 et 3,8 images/s), quelles cellules peut-on suivre image par "
                                  "image ?",
                              "Les **passives** et les **collées** sans problème : environ 0,5 µm par image, bien "
                              "sous la ½ cellule. Les **nageuses** de phase exponentielle sont **à la limite** : "
                              "20 µm/s donnent 5 à 7 µm par image, presque la distance entre voisines (r ≈ 0,8). "
                              "Dans la simulation, on les suit encore (rappel ~0,9 avec un gating de 9 µm et le "
                              "Kalman, notebooks 02 et 03). Mais c'est fragile : une zone plus dense, une sortie du "
                              "plan focal ou un tumble suffisent à perdre l'identité. La règle de la ½ cellule est "
                              "violée d'un facteur 4. Les nageuses lentes (6 µm/s en stationnaire, ~2 µm par image) "
                              "sont faciles. Pour suivre la nage de façon fiable, le rapport conseille **~30 "
                              "images/s** sur des clips courts, quitte à passer en 1080p ou en binning pour garder "
                              "la cadence."),
               tutor.question(mo, "Si on ne peut pas suivre les nageuses, peut-on quand même mesurer la fraction "
                                  "mobile ?",
                              "Oui, par des méthodes **sans suivi individuel** : la **DDM** (microscopie dynamique "
                              "différentielle), qui analyse les différences entre images dans l'espace de Fourier, "
                              "ou le **flux optique**. On peut aussi simplement compter les cellules qui « "
                              "disparaissent » d'une image à l'autre. Le rapport recommande la DDM comme descripteur "
                              "de phase, car elle ne demande pas de segmentation.")])
    return


if __name__ == "__main__":
    app.run()
