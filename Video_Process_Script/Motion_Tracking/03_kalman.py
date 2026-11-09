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
    from motion_sim import PHASES, VIDEOS, SimConfig, phase_config, simulate
    return PHASES, SimConfig, VIDEOS, mo, np, phase_config, plt, ref, simulate, tutor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 — Filtre de Kalman : chercher la nageuse là où elle va

    Au notebook 02, on cherchait chaque cellule **là où on l'avait vue**. Une nageuse de *S. pasteurii* en phase
    exponentielle (proxy : ~20 µm/s) est déjà 5 à 7 µm plus loin à l'image suivante. Le filtre de Kalman retient
    pour chaque cellule sa **position**, sa **vitesse**, et **à quel point il en est sûr**. À chaque image, il
    alterne deux étapes :

    1. **Prédiction** : « vu sa vitesse, elle devrait être là ». L'incertitude grandit, car la cellule a pu tourner.
    2. **Correction** : la détection arrive. On place la cellule **entre** la prédiction et la mesure, plus près de
       celle à laquelle on fait le plus confiance, et on met la vitesse à jour.

    La nage des bactéries à flagelles péritriches est faite de **runs** en ligne presque droite, entrecoupés de
    **tumbles** : une réorientation brève et aléatoire. Le proxy donne un tumble toutes les ~3 s. Près de la lamelle,
    les trajectoires deviennent des **cercles** (sens horaire). Pendant un run, la vitesse prédit très bien la
    position suivante. Au tumble, la prédiction se trompe.
    """)
    return


@app.cell(hide_code=True)
def _(mo, ref):
    mo.accordion({"Pour aller plus loin : les formules": mo.md(r"""
    État $x = (x, y, v_x, v_y)$, incertitude $P$ (matrice de covariance), mesure $z = (x, y)$.

    - Prédiction : $x^- = F x$, $\;P^- = F P F^\top + Q$. La matrice $F$ fait avancer la position de $v\,dt$.
    - Correction : $K = P^- H^\top (H P^- H^\top + R)^{-1}$, $\;x = x^- + K (z - H x^-)$, $\;P = (I - K H) P^-$.

    $K$ est le **gain de Kalman** : proche de 1, on suit la mesure ; proche de 0, on suit la prédiction.
    $R$ est le bruit de mesure (~0,1 µm ici), $Q$ la variabilité du mouvement (les tumbles et les virages).
    """)})
    return


@app.cell(hide_code=True)
def _(VIDEOS, mo):
    video = mo.ui.dropdown(list(VIDEOS), value="Expo (×40)", label="Cadence de la vidéo")
    swim_speed = mo.ui.slider(5, 30, step=1, value=20, label="vitesse de nage (µm/s)", show_value=True)
    tumble_interval = mo.ui.slider(0.5, 10, step=0.5, value=3, label="temps moyen entre tumbles (s)", show_value=True)
    circle_rate = mo.ui.slider(0, 1.5, step=0.1, value=0.0, label="rotation près de la lamelle (rad/s)", show_value=True)
    mo.vstack([mo.md("## Une nageuse, image par image"), video, swim_speed, tumble_interval, circle_rate])
    return circle_rate, swim_speed, tumble_interval, video


@app.cell(hide_code=True)
def _(SimConfig, VIDEOS, circle_rate, mo, np, plt, ref, simulate, swim_speed, tumble_interval, video):
    _cfg = SimConfig(density=0.1, width=400, height=400, n_frames=50, dt=VIDEOS[video.value]["dt"],
                     swimmer_fraction=1.0, stuck_fraction=0.0, swim_speed=swim_speed.value, speed_spread=0.0,
                     tumble_rate=1 / tumble_interval.value, circle_rate=circle_rate.value, seed=3)
    _truth, _ = simulate(_cfg)
    _z = _truth[_truth.true_id == 0][["x", "y"]].to_numpy()
    _z = _z + np.random.default_rng(4).normal(0, _cfg.detection_noise, _z.shape)
    _m = ref.kalman_matrices(_cfg.dt, 10.0, _cfg.detection_noise)
    _x, _P = np.r_[_z[0], 0, 0], np.diag([_cfg.detection_noise ** 2] * 2 + [400.0] * 2)
    _err_last, _err_kalman = [], []
    for _k in range(1, len(_z)):
        _xp, _Pp = ref.kalman_predict(_x, _P, _m["F"], _m["Q"])
        _err_last.append(np.linalg.norm(_z[_k] - _z[_k - 1]))
        _err_kalman.append(np.linalg.norm(_z[_k] - _xp[:2]))
        _x, _P = ref.kalman_update(_xp, _Pp, _z[_k], _m["H"], _m["R"])
    _t = np.arange(1, len(_z)) * _cfg.dt
    _fig, (_path, _err) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1, 1.6]})
    _path.plot(*_z.T, "-o", ms=2.5, color="#d95926", lw=1)
    _path.plot(*_z[0], "ks", ms=6)
    _path.set(aspect="equal", xlabel="µm", ylabel="µm", title="Trajectoire (carré = départ)")
    _path.invert_yaxis()
    _err.plot(_t, _err_last, color="#e03131", label=f"« là où on l'a vue » (médiane {np.median(_err_last):.1f} µm)")
    _err.plot(_t, _err_kalman, color="#3987e5",
              label=f"Kalman (médiane {np.median(_err_kalman[3:]):.1f} µm après 3 images)")
    _err.axhline(1.5, color="gray", ls="--", lw=1)
    _err.text(_t[-1], 1.5, "½ cellule ", ha="right", va="bottom", fontsize=8, color="gray")
    _err.set(xlabel="temps (s)", ylabel="distance à l'endroit prédit (µm)",
             title="Où cherche-t-on la cellule à l'image suivante ?")
    _err.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(r"""
    *À lire :* la courbe rouge est le gating minimal qu'il faudrait sans prédiction, la bleue avec. Les pics bleus
    sont les **tumbles** : la direction change et la prédiction se trompe pendant une ou deux images. Avec une
    rotation près de la lamelle, la prédiction « en ligne droite » est toujours un peu en retard sur le cercle.
    """)])
    return


@app.cell(hide_code=True)
def _(PHASES, mo):
    phase = mo.ui.dropdown(list(PHASES), value="Exponentielle", label="Phase (préréglage)")
    max_distance = mo.ui.slider(1, 12, step=0.5, value=6, label="gating (µm)", show_value=True)
    mo.vstack([mo.md("## Kalman dans le tracker\n\nOn compare le tracker du notebook 02 et le même tracker avec "
                     "Kalman, sur les **nageuses** seulement."), phase, max_distance])
    return max_distance, phase


@app.cell(hide_code=True)
def _(VIDEOS, max_distance, mo, np, phase, phase_config, plt, ref, simulate, tutor, video):
    _cfg = phase_config(phase.value, n_frames=40, dt=VIDEOS[video.value]["dt"])
    _, _det = simulate(_cfg)
    _rows = []
    if (_det.kind == "swimmer").any():
        _model = ref.KalmanModel(dt=_cfg.dt, process_noise=10.0, measurement_noise=_cfg.detection_noise,
                                 initial_speed_std=_cfg.swim_speed)
        for _name, _motion in [("sans prédiction", None), ("Kalman", _model)]:
            _t = ref.track(_det, max_distance.value, motion=_motion)
            _rows.append((_name, *ref.link_scores(_t[_t.kind == "swimmer"])))
    _fig, _ax = plt.subplots(figsize=(6, 3.2))
    _pos = np.arange(len(_rows))
    _ax.bar(_pos - 0.18, [r[1] for r in _rows], 0.36, color="#3987e5", label="précision")
    _ax.bar(_pos + 0.18, [r[2] for r in _rows], 0.36, color="#d95926", label="rappel")
    _ax.set(xticks=_pos, xticklabels=[r[0] for r in _rows], ylim=(0, 1.05),
            title=f"Nageuses à {_cfg.swim_speed:g} µm/s : {_cfg.swim_speed * _cfg.dt:.1f} µm par image")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(r"""
    **Limite à comprendre :** une nouvelle trajectoire démarre avec une vitesse inconnue. Son **premier** lien doit
    donc passer le gating sans aide. Avec un gating de 3 µm en phase exponentielle, presque aucune nageuse ne démarre.
    Essaie 3, puis 6, puis 9 µm.
    """),
               tutor.question(mo, "Le Kalman aide-t-il pour les cellules passives (browniennes) ?",
                              "Non. Leur mouvement change de direction au hasard à chaque image, donc leur « vitesse » "
                              "ne prédit rien. Il ne sert que pour un mouvement **persistant** : les runs des "
                              "nageuses, la dérive, le flux d'évaporation. Pour les passives, chercher autour de la "
                              "dernière position suffit, avec un petit gating (~1 µm)."),
               tutor.question(mo, "À ×100, une nageuse sort du plan focal pendant 2 images puis revient. Que "
                                  "gagne-t-on avec la prédiction ?",
                              "Pendant son absence, le filtre continue de **prédire** sa position, en devenant plus "
                              "incertain. Quand elle revient, on la cherche là où son run l'a menée (~10 µm plus loin "
                              "à 20 µm/s), pas là où on l'a vue en dernier. C'est la mémoire du notebook 02, rendue "
                              "utilisable pour des cellules rapides."),
               tutor.question(mo, "Si le temps entre tumbles devenait plus court que le temps entre deux images, à "
                                  "quoi servirait encore le Kalman ?",
                              "À presque rien : la direction serait déjà oubliée d'une image à l'autre et la nage "
                              "ressemblerait à une **diffusion rapide**. Avec un tumble toutes les ~3 s (proxy) et "
                              "0,26 à 0,35 s entre images, une nageuse garde sa direction pendant ~10 images : le "
                              "Kalman a de quoi travailler.")])
    return


if __name__ == "__main__":
    app.run()
