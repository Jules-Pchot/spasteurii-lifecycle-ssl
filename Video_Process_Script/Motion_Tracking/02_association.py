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
    from motion_sim import KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, phase_config, simulate
    return KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, mo, np, phase_config, plt, ref, simulate, tutor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 — Association entre images : qui est qui ?

    Le tracking « par détection » relie chaque cellule détectée à l'image $t+1$ à une cellule de l'image $t$.
    Deux questions :

    1. **Affectation** : quelles paires former ? Le *plus proche voisin glouton* prend les paires une par une, la
       plus courte d'abord. L'**algorithme hongrois** minimise la **somme** des distances sur toutes les paires à la
       fois.
    2. **Gating** : à partir de quelle distance refuser un lien ? Avec *S. pasteurii*, le choix est délicat : les
       cellules passives bougent de ~0,5 µm par image, les nageuses de 5 à 7 µm. Aucun gating ne convient aux deux.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, plt):
    # Two cells A and B swimming right by 7 µm (a 20 µm/s swimmer at Stationary's 0.35 s per frame)
    _before = np.array([[0.0, 0.0], [10.0, 0.0]])
    _after = np.array([[7.0, 0.0], [17.0, 0.0]])
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 2.4), sharey=True)
    for _ax, _pairs, _title in [(_axes[0], [(1, 0)], "Glouton : B prend A' (3 µm), A n'a plus que B' à 17 µm"),
                                (_axes[1], [(0, 0), (1, 1)], "Hongrois : A→A', B→B' (7 + 7 = 14 µm)")]:
        _ax.scatter(*_before.T, s=80, color="#3987e5", zorder=3)
        _ax.scatter(*_after.T, s=80, color="#d95926", zorder=3)
        for _name, (_x, _y) in zip("AB", _before):
            _ax.annotate(_name, (_x, _y), xytext=(0, 9), textcoords="offset points", ha="center")
        for _name, (_x, _y) in zip(["A'", "B'"], _after):
            _ax.annotate(_name, (_x, _y), xytext=(0, -16), textcoords="offset points", ha="center")
        for _i, _j in _pairs:
            _ax.annotate("", _after[_j], _before[_i], arrowprops=dict(arrowstyle="->", lw=1.5, color="#199e70"))
        _ax.set(title=_title, xlim=(-3, 20), ylim=(-1.5, 1.5), yticks=[], xlabel="µm")
    _fig.tight_layout()
    mo.vstack([mo.md(r"""
    ## Glouton contre hongrois, sur deux nageuses

    Deux nageuses côte à côte, à 10 µm l'une de l'autre, avancent de 7 µm, comme à 20 µm/s dans Stationary.
    Le glouton prend d'abord la paire la plus courte, B → A' (3 µm). A n'a plus que B' à 17 µm : sa trajectoire est
    coupée, ou reliée à la mauvaise cellule. Le hongrois choisit l'ensemble de paires de somme minimale et trouve les
    deux bons liens.
    """), _fig])
    return


@app.cell(hide_code=True)
def _(PHASES, VIDEOS, mo):
    phase = mo.ui.dropdown(list(PHASES), value="Fin exponentielle (fenêtre MICP)", label="Phase (préréglage)")
    video = mo.ui.dropdown(list(VIDEOS), value="Expo (×40)", label="Cadence de la vidéo")
    max_distance = mo.ui.slider(0.5, 10, step=0.5, value=2.0, label="gating : distance max d'un lien (µm)", show_value=True)
    memory = mo.ui.slider(0, 3, step=1, value=0, label="mémoire : images manquées tolérées", show_value=True)
    focus = mo.ui.slider(0, 10, step=0.5, value=0, label="temps moyen d'une nageuse dans le plan focal (s, 0 = toujours)",
                         show_value=True)
    mo.vstack([mo.md("### Réglages"), phase, video, max_distance, memory, focus])
    return focus, max_distance, memory, phase, video


@app.cell(hide_code=True)
def _(VIDEOS, focus, phase, phase_config, simulate, video):
    cfg = phase_config(phase.value, n_frames=40, dt=VIDEOS[video.value]["dt"], focus_lifetime=focus.value)
    truth, detections = simulate(cfg)
    return cfg, detections


@app.cell(hide_code=True)
def _(cfg, detections, max_distance, mo, np, plt, ref):
    _a = detections[detections.frame == 0].reset_index(drop=True)
    _b = detections[detections.frame == 1].reset_index(drop=True)
    _pairs = ref.link_frames(_a[["x", "y"]].to_numpy(), _b[["x", "y"]].to_numpy(), max_distance.value)
    _fig, _ax = plt.subplots(figsize=(9, 6))
    _ax.scatter(_a.x, _a.y, s=12, color="#3987e5", label="image t")
    _ax.scatter(_b.x, _b.y, s=12, color="#d95926", label="image t+1")
    _good = 0
    for _i, _j in _pairs:
        _ok = _a.true_id[_i] == _b.true_id[_j]
        _good += _ok
        _ax.plot([_a.x[_i], _b.x[_j]], [_a.y[_i], _b.y[_j]], color="#199e70" if _ok else "#e03131", lw=1 if _ok else 2)
    _ax.set(xlim=(0, cfg.width), ylim=(cfg.height, 0), aspect="equal", xlabel="µm", ylabel="µm",
            title=f"{len(_pairs)} liens : {_good} justes (vert), {len(_pairs) - _good} faux (rouge) ; "
                  f"{len(_b) - len(_pairs)} cellules sans lien")
    _ax.legend(loc="upper right", fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md("## Les liens entre les deux premières images"), _fig])
    return


@app.cell(hide_code=True)
def _(KIND_COLORS, KIND_LABELS, detections, max_distance, memory, mo, np, plt, ref):
    _gates = np.arange(0.5, 10.5, 0.5)
    _fig, _ax = plt.subplots(figsize=(8.5, 3.8))
    _tracks = [ref.track(detections, _g, memory=memory.value) for _g in _gates]
    for _kind in ["brownian", "swimmer"]:
        _scores = np.array([ref.link_scores(_t[_t.kind == _kind]) if (_t.kind == _kind).any() else (np.nan, np.nan)
                            for _t in _tracks])
        _ax.plot(_gates, _scores[:, 1], color=KIND_COLORS[_kind], label=f"{KIND_LABELS[_kind]} — rappel")
        _ax.plot(_gates, _scores[:, 0], color=KIND_COLORS[_kind], ls="--", label=f"{KIND_LABELS[_kind]} — précision")
    _ax.axvline(max_distance.value, color="gray", lw=1)
    _ax.set(xlabel="gating (µm)", ylabel="score", ylim=(0, 1.02), title="Qualité des liens selon le gating, par type")
    _ax.legend(fontsize=8, ncol=2)
    _fig.tight_layout()
    mo.vstack([mo.md(r"""
    ## Un gating pour qui ?

    - Le **rappel** est la part des vrais liens retrouvés.
    - La **précision** est la part des liens faits qui sont justes.

    Regarde les deux types séparément : les cellules passives sont bien suivies dès ~1 µm de gating, les nageuses
    demandent un gating de plusieurs µm. Ce gating large ouvre la porte aux erreurs quand la densité monte.
    """), _fig])
    return


@app.cell(hide_code=True)
def _(KIND_COLORS, cfg, detections, max_distance, memory, mo, plt, ref, tutor):
    tracked = ref.track(detections, max_distance.value, memory=memory.value)
    _precision, _recall = ref.link_scores(tracked)
    _fig, _ax = plt.subplots(figsize=(9, 6))
    for _tid, _g in tracked.groupby("track_id"):
        _ax.plot(_g.x, _g.y, lw=1, color=KIND_COLORS[_g.kind.mode()[0]])
    _ax.set(xlim=(0, cfg.width), ylim=(cfg.height, 0), aspect="equal", xlabel="µm", ylabel="µm",
            title=f"{tracked.track_id.nunique()} trajectoires pour {detections.true_id.nunique()} cellules — "
                  f"précision {_precision:.2f}, rappel {_recall:.2f}, {ref.id_switches(tracked)} changements d'identité")
    _fig.tight_layout()
    mo.vstack([mo.md("## Le tracking complet (couleur = vrai type de la cellule)"), _fig,
               mo.md("*À essayer :* mets le temps dans le plan focal à 2 s. À ×100, la profondeur de champ est "
                     "inférieure au micron : les nageuses entrent et sortent du plan. Monte ensuite la mémoire à 2."),
               tutor.question(mo, "En phase exponentielle, une cellule se divise. Que fait ce tracker ?",
                              "Il relie la mère à **une seule** des deux filles (la plus proche) et démarre une "
                              "nouvelle trajectoire pour l'autre : la division n'est pas représentée. Il faudrait un "
                              "tracker qui gère les « splits » (framework LAP de TrackMate, btrack). C'est utile ici, "
                              "car les divisions sont un marqueur de la phase exponentielle."),
               tutor.question(mo, "Une chaîne de 4 cellules se déplace d'un bloc. Combien de trajectoires obtient-on ?",
                              "Si le détecteur sépare les 4 cellules, on obtient 4 trajectoires **parfaitement "
                              "parallèles**. C'est une signature utile : des cellules voisines qui bougent ensemble "
                              "forment une chaîne ou un agrégat. Le rapport attend plus de chaînes en fin de phase log "
                              "(hypothèse inférée de *B. subtilis*) et plus d'agrégats avec le CaCl₂."),
               tutor.question(mo, "Pourquoi ne pas simplement prendre un gating très grand pour ne rater aucune nageuse ?",
                              "Parce que la précision s'effondre. Une cellule qui sort du plan ou du champ peut alors "
                              "être reliée à une autre, lointaine, ce qui fabrique un faux saut, donc une fausse "
                              "nageuse. Une solution plus fine : un gating **adapté au type de mouvement**, petit "
                              "pour les cellules lentes, grand pour celles qui ont déjà montré une vitesse (avec le "
                              "Kalman du notebook 03).")])
    return


if __name__ == "__main__":
    app.run()
