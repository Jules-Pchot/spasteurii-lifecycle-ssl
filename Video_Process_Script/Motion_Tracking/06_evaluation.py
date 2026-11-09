import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(mo.notebook_dir()))
    import reference as ref
    import tutor
    from motion_sim import KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, phase_config, simulate
    return KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, mo, np, pd, phase_config, plt, ref, simulate, tutor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 — Évaluer un tracker quand personne ne connaît la réponse

    Pour *S. pasteurii*, il n'existe **aucune donnée cinématique publiée** : ni vitesse, ni durée de run, ni
    coefficient de diffusion. On ne peut donc pas comparer ses mesures à la littérature. Il faut évaluer le tracker
    **sur des simulations** calées sur les ordres de grandeur du rapport, avant de lui faire confiance sur les vraies
    vidéos.

    Les mesures, au niveau des **liens** (une détection reliée à la suivante de sa trajectoire) :

    - **précision** : part des liens faits qui relient bien la même cellule ;
    - **rappel** : part des vrais liens que le tracker a faits ;
    - **changements d'identité** : combien de fois une vraie cellule change de numéro de trajectoire.

    Les benchmarks combinent ces erreurs dans des scores globaux : MOTA, IDF1, HOTA, et ceux du Cell Tracking
    Challenge, qui comptent aussi les divisions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Un exemple concret

    La cellule A est détectée aux images 1, 2, 3, 4. Le tracker relie 1→2 et 2→3, puis la perd et démarre une
    nouvelle trajectoire à l'image 4.

    - Vrais liens : 3 (1→2, 2→3, 3→4). Liens faits pour A : 2, tous justes.
    - Pour A : **précision 2/2 = 1**, **rappel 2/3 ≈ 0,67**, **1 changement d'identité**.

    **Quelle erreur est la plus grave pour la mobilité ?** Une trajectoire **coupée** raccourcit les MSD : α devient
    moins précis (notebook 04). Une **identité échangée** colle deux cellules ensemble : elle fabrique un saut, donc
    une fausse nageuse. Pour mesurer une fraction mobile, mieux vaut un tracker **prudent** (bonne précision) qu'un
    tracker qui relie tout.
    """)
    return


@app.cell(hide_code=True)
def _(PHASES, VIDEOS, mo):
    phase = mo.ui.dropdown(list(PHASES), value="Exponentielle", label="Phase (préréglage)")
    video = mo.ui.dropdown(list(VIDEOS), value="Stationary (×100)", label="Cadence de la vidéo")
    max_distance = mo.ui.slider(1, 12, step=0.5, value=8, label="gating (µm)", show_value=True)
    mo.vstack([mo.md("## Sortie du plan focal et mémoire\n\nÀ ×100, la profondeur de champ est inférieure au micron : "
                     "une nageuse quitte le plan au bout de quelques secondes, puis revient parfois. Le tracker peut "
                     "tolérer quelques images manquées (la **mémoire**, ou *gap closing*)."),
               phase, video, max_distance])
    return max_distance, phase, video


@app.cell(hide_code=True)
def _(VIDEOS, max_distance, mo, np, pd, phase, phase_config, plt, ref, simulate, tutor, video):
    _rows = []
    for _life in [0.5, 1, 2, 4, 8, 0]:
        _cfg = phase_config(phase.value, n_frames=40, dt=VIDEOS[video.value]["dt"], focus_lifetime=_life)
        _, _det = simulate(_cfg)
        _model = ref.KalmanModel(dt=_cfg.dt, process_noise=10.0, measurement_noise=_cfg.detection_noise,
                                 initial_speed_std=_cfg.swim_speed)
        for _memory in (0, 2):
            _t = ref.track(_det, max_distance.value, memory=_memory, motion=_model)
            _sw = _t[_t.kind == "swimmer"]
            if len(_sw):
                _p, _r = ref.link_scores(_sw)
                _rows.append({"life": _life if _life else 20, "memory": _memory, "precision": _p, "recall": _r,
                              "switches": ref.id_switches(_sw) / max(_sw.true_id.nunique(), 1)})
    _df = pd.DataFrame(_rows)
    _fig, (_a, _b) = plt.subplots(1, 2, figsize=(12, 3.8))
    for _memory, _color in [(0, "#e03131"), (2, "#199e70")]:
        _g = _df[_df.memory == _memory].sort_values("life")
        _a.plot(_g.life, _g.recall, "-o", color=_color, label=f"mémoire {_memory} — rappel")
        _a.plot(_g.life, _g.precision, "--o", color=_color, label=f"mémoire {_memory} — précision")
        _b.plot(_g.life, _g.switches, "-o", color=_color, label=f"mémoire {_memory}")
    for _ax in (_a, _b):
        _ax.set_xscale("log")
        _ax.set_xticks([0.5, 1, 2, 4, 8, 20])
        _ax.set_xticklabels(["0,5", "1", "2", "4", "8", "toujours"])
        _ax.set_xlabel("temps moyen d'une nageuse dans le plan focal (s)")
    _a.set(ylabel="score", ylim=(0, 1.02), title="Liens des nageuses (Kalman)")
    _b.set(ylabel="par nageuse", title="Changements d'identité")
    _a.legend(fontsize=7, ncol=2)
    _b.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, tutor.question(mo, "Le rapport conseille des « statistiques robustes aux trajectoires courtes, ou "
                                        "la DDM ». Pourquoi ?",
                                    "Parce qu'à ×100, même avec de la mémoire, les nageuses donnent surtout des "
                                    "**morceaux** de trajectoires. Un α estimé sur 5 images est très bruité (notebook "
                                    "04). On peut regrouper les déplacements de tous les morceaux, par exemple dans un "
                                    "histogramme des déplacements, ou utiliser la **DDM**, qui mesure la mobilité sur "
                                    "l'image entière sans rien suivre.")])
    return


@app.cell(hide_code=True)
def _(mo):
    duplicate_rate = mo.ui.slider(0, 0.3, step=0.02, value=0.1, label="part d'images dupliquées", show_value=True)
    mo.vstack([mo.md(r"""
    ## Un piège d'acquisition : les images dupliquées

    Les caméras en 4K à cadence variable répètent parfois une image : l'image $k$ est une copie de l'image $k-1$.
    Si on suppose un temps régulier entre images, la cellule semble **s'arrêter net** pendant un intervalle, puis
    faire un **double saut**. Le rapport conseille de vérifier les horodatages et de décoder la vidéo une seule fois,
    en PNG ou TIFF.
    """), duplicate_rate])
    return (duplicate_rate,)


@app.cell(hide_code=True)
def _(KIND_COLORS, KIND_LABELS, duplicate_rate, mo, np, phase_config, plt, simulate, tutor):
    _cfg = phase_config("Exponentielle", n_frames=80)
    _truth, _ = simulate(_cfg)
    _rng = np.random.default_rng(12)
    _dup = _rng.random(_cfg.n_frames) < duplicate_rate.value
    _dup[0] = False
    _source = np.arange(_cfg.n_frames)
    for _f in range(1, _cfg.n_frames):  # a duplicated frame repeats what the previous frame shows
        if _dup[_f]:
            _source[_f] = _source[_f - 1]
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _bins = np.linspace(0, 45, 46)
    _text = []
    for _kind in ["swimmer", "brownian"]:
        _speeds = []
        for _, _g in _truth[_truth.kind == _kind].groupby("true_id"):
            _xy = _g.sort_values("frame")[["x", "y"]].to_numpy()[_source]
            _speeds.append(np.hypot(*np.diff(_xy, axis=0).T) / _cfg.dt)
        _speeds = np.concatenate(_speeds)
        _ax.hist(_speeds, _bins, alpha=0.6, color=KIND_COLORS[_kind], label=KIND_LABELS[_kind])
        _text.append(f"{KIND_LABELS[_kind]} : {np.mean(_speeds < 0.5):.0%} des vitesses < 0,5 µm/s")
    _ax.set(xlabel="vitesse instantanée apparente (µm/s)", ylabel="intervalles", title="Effet des images dupliquées")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.callout(mo.md("  \n".join(_text)), kind="info"),
               tutor.question(mo, "Pourquoi une image dupliquée peut-elle fausser la fraction mobile ?",
                              "Elle crée de **faux arrêts** : une nageuse à vitesse nulle sur un intervalle, puis à "
                              "vitesse double sur le suivant. La vitesse instantanée est faussée dans les deux sens. "
                              "Si on classait par vitesse instantanée, certaines nageuses passeraient pour passives. "
                              "C'est une raison de plus de classer par **α**, calculé sur plusieurs délais, et de "
                              "contrôler les horodatages."),
               tutor.question(mo, "Sur les vraies vidéos, sans vérité terrain, comment vérifier le tracker ?",
                              "En combinant : (1) des **simulations** calées sur les ordres de grandeur du rapport "
                              "(densité, cadence, 15–25 µm/s, D ≈ 0,25 µm²/s) ; (2) un **témoin non mobile** "
                              "(cellules fixées) qui doit donner α ≈ 1 et aucune nageuse ; (3) une **vérification "
                              "manuelle** d'un échantillon de trajectoires dans napari, surtout les nageuses et les "
                              "outliers ; (4) des **contrôles de cohérence** : longueurs des trajectoires, sauts entre "
                              "images, trajectoires qui naissent ou meurent au milieu du champ."),
               tutor.question(mo, "Comment savoir si les descripteurs de mouvement apportent vraiment quelque chose au "
                                  "modèle auto-supervisé ?",
                              "Par **ablation**, comme le recommande le rapport : comparer les performances avec et "
                              "sans les descripteurs de mouvement, à densité et morphologie égales. Si la fraction "
                              "mobile chute nettement entre le milieu et la fin de la phase exponentielle, elle "
                              "devient un candidat direct pour signaler la fenêtre de déclenchement MICP.")])
    return


if __name__ == "__main__":
    app.run()
