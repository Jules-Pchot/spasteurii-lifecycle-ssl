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
    from motion_sim import KIND_COLORS, KIND_LABELS, KINDS, PHASES, VIDEOS, phase_config, simulate
    return KINDS, KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, mo, np, pd, phase_config, plt, ref, simulate, tutor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 — Bouger « par rapport à son environnement »

    Sous la lamelle, deux mouvements emportent les cellules sans rien dire de leur physiologie (rapport, section
    « Pièges à contrôler ») :

    - la **dérive** de la platine ou de la lame : toute l'image glisse, cellules collées et cristaux compris ;
    - les **flux capillaires et d'évaporation** : le liquide entraîne toutes les cellules **libres** de façon
      cohérente. Les cellules collées à la lamelle, les agrégats fixés et les cristaux de CaCO₃ n'en suivent pas.

    Le rapport conseille de **soustraire le mouvement médian, ou celui des objets immobiles, avant de calculer des
    vitesses**. La méthode :

    1. vitesse moyenne de chaque trajectoire (moyenner dans le temps réduit le bruit) ;
    2. moins la **médiane des trajectoires voisines** : c'est la vitesse relative au liquide qui l'entoure ;
    3. une vitesse relative trop grande signale une cellule qui bouge différemment de son environnement : une
       **nageuse** (elle avance par elle-même) ou une cellule **collée** (elle ne suit pas le flux).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Deux façons de décider « trop grand »

    **A. Le z-score robuste**, une règle statistique. On compare chaque cellule aux autres :
    $z = (\text{score} - \text{médiane}) / (1{,}4826 \times \text{MAD})$. La médiane et la MAD résistent aux
    outliers… **tant qu'ils restent minoritaires**.

    **B. Un témoin non mobile**, recommandé par le rapport : des cellules fixées au glutaraldéhyde ou tuées par la
    chaleur, filmées dans les mêmes conditions. Leur score ne vient que du mouvement brownien et du bruit. On fixe le
    seuil au-dessus de presque toutes leurs valeurs (99e centile). Ici, le témoin est simulé : même densité, même
    cadence, mêmes dérive et flux, mais aucune nageuse ni cellule collée.
    """)
    return


@app.cell(hide_code=True)
def _(PHASES, VIDEOS, mo):
    phase = mo.ui.dropdown(list(PHASES), value="Fin exponentielle (fenêtre MICP)", label="Phase (préréglage)")
    video = mo.ui.dropdown(list(VIDEOS), value="Expo (×40)", label="Cadence de la vidéo")
    flow = mo.ui.slider(0, 10, step=0.5, value=2, label="flux d'évaporation vers la droite (µm/s)", show_value=True)
    drift = mo.ui.slider(0, 5, step=0.25, value=0.5, label="dérive de la lame (µm/s)", show_value=True)
    k = mo.ui.slider(3, 30, step=1, value=10, label="k voisines pour la référence", show_value=True)
    z_threshold = mo.ui.slider(1, 6, step=0.25, value=3, label="A : seuil sur z", show_value=True)
    source = mo.ui.radio(["vraies identités", "identités du tracker (Kalman, gating 6 µm)"], value="vraies identités",
                         label="trajectoires utilisées")
    mo.vstack([mo.md("### Réglages (les valeurs de flux et de dérive sont illustratives)"), phase, video, flow, drift, k,
               z_threshold, source])
    return drift, flow, k, phase, source, video, z_threshold


@app.cell(hide_code=True)
def _(VIDEOS, drift, flow, k, np, phase, phase_config, ref, simulate, source, video):
    def relative_motion(cfg, use_tracker):
        """Per-track mean velocity, its local reference and the relative speed (µm/s)."""
        truth, detections = simulate(cfg)
        if use_tracker:
            model = ref.KalmanModel(dt=cfg.dt, process_noise=10.0, measurement_noise=cfg.detection_noise,
                                    initial_speed_std=cfg.swim_speed)
            tracks, id_column = ref.track(detections, 6.0, memory=1, motion=model), "track_id"
        else:
            tracks, id_column = truth[truth.inside], "true_id"
        v = ref.velocities(tracks, cfg.dt, id_column)
        per = v.groupby(id_column).agg(x=("x", "mean"), y=("y", "mean"), vx=("vx", "mean"), vy=("vy", "mean"),
                                       n=("vx", "size"), kind=("kind", lambda s: s.mode()[0]))
        per = per[per.n >= 10].copy()  # too short: mean velocity too noisy
        local = ref.local_reference_velocity(per[["x", "y"]].to_numpy(), per[["vx", "vy"]].to_numpy(), k.value)
        per["rx"], per["ry"] = per.vx - local[:, 0], per.vy - local[:, 1]
        per["score"] = np.hypot(per.rx, per.ry)
        return per

    _common = dict(n_frames=60, dt=VIDEOS[video.value]["dt"], flow=(flow.value, 0.0),
                   drift=(drift.value, drift.value * 0.5))
    _tracker = source.value != "vraies identités"
    per_track = relative_motion(phase_config(phase.value, **_common), _tracker)
    control = relative_motion(phase_config(phase.value, swimmer_fraction=0.0, stuck_fraction=0.0, seed=11, **_common),
                              _tracker)
    control_threshold = float(np.percentile(control.score, 99))
    return control, control_threshold, per_track


@app.cell(hide_code=True)
def _(control_threshold, per_track, ref, z_threshold):
    _median = per_track.score.median()
    _mad = 1.4826 * (per_track.score - _median).abs().median()
    z_cut = _median + z_threshold.value * _mad  # score above which z > threshold
    per_track["z"] = ref.robust_z(per_track.score.to_numpy())
    per_track["by_z"] = per_track.z > z_threshold.value
    per_track["by_control"] = per_track.score > control_threshold
    return (z_cut,)


@app.cell(hide_code=True)
def _(KINDS, KIND_COLORS, KIND_LABELS, PHASES, control, control_threshold, mo, np, pd, per_track, phase, plt, z_cut):
    _fig, (_raw, _rel, _hist) = plt.subplots(1, 3, figsize=(16, 4.6))
    for _kind in KINDS:
        _g = per_track[per_track.kind == _kind]
        _raw.quiver(_g.x, _g.y, _g.vx, -_g.vy, color=KIND_COLORS[_kind], angles="xy", scale_units="xy", scale=0.5,
                    width=0.004, label=KIND_LABELS[_kind])
        _rel.quiver(_g.x, _g.y, _g.rx, -_g.ry, color=KIND_COLORS[_kind], angles="xy", scale_units="xy", scale=0.5,
                    width=0.004)
    _o = per_track[per_track.by_control]
    _rel.scatter(_o.x, _o.y, s=80, facecolors="none", edgecolors="black", lw=1, label="détecté (témoin)")
    for _ax, _title in [(_raw, "Vitesse moyenne brute"), (_rel, "Vitesse relative au liquide voisin")]:
        _ax.set(xlim=(0, 150), ylim=(100, 0), aspect="equal", xlabel="µm", title=_title)
        _ax.legend(fontsize=7, loc="lower right")
    _bins = np.logspace(-2, np.log10(max(per_track.score.max(), 1) * 1.2), 40)
    _hist.hist(control.score, _bins, color="#c8c8c8", label="témoin non mobile")
    for _kind in KINDS:
        _hist.hist(per_track.score[per_track.kind == _kind], _bins, alpha=0.6, color=KIND_COLORS[_kind],
                   label=KIND_LABELS[_kind])
    _hist.axvline(control_threshold, color="black", lw=1.5, label="B : seuil du témoin")
    _hist.axvline(z_cut, color="#e03131", ls="--", lw=1.5, label="A : seuil z robuste")
    _hist.set(xscale="log", xlabel="vitesse relative moyenne (µm/s)", ylabel="trajectoires", title="Score par trajectoire")
    _hist.legend(fontsize=7)
    _fig.tight_layout()
    _true = per_track.kind.isin(["swimmer", "stuck"])
    _rows = []
    for _name, _col in [("A : z robuste", "by_z"), ("B : témoin non mobile", "by_control")]:
        _hit = (per_track[_col] & _true).sum()
        _rows.append({"Méthode": _name, "Détectées": int(per_track[_col].sum()),
                      "Précision": round(_hit / max(per_track[_col].sum(), 1), 2),
                      "Rappel": round(_hit / max(_true.sum(), 1), 2),
                      "Nageuses trouvées": f"{per_track[_col][per_track.kind == 'swimmer'].mean():.0%}",
                      "Collées trouvées": f"{per_track[_col][per_track.kind == 'stuck'].mean():.0%}"})
    _share = _true.mean()
    mo.vstack([_fig, mo.callout(mo.md(f"Dans ce préréglage, **{_share:.0%} des trajectoires** sont des nageuses ou des "
                                      f"cellules collées : est-ce encore « rare » ?"), kind="info"),
               mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False)])
    return


@app.cell(hide_code=True)
def _(mo, tutor):
    mo.vstack([
        mo.md(r"""
    *À essayer :*

    1. **Phase exponentielle** : 70 % de cellules « anormales ». La médiane tombe parmi les nageuses et la règle A
       casse, alors que la règle B, calée sur la physique du mouvement passif, tient.
    2. **Flux à 0** : les cellules collées deviennent indétectables par la vitesse relative. Sans flux, elles ne
       diffèrent des passives que par l'absence de diffusion. Il faut l'α du notebook 04.
    3. **Dérive seule** (flux 0, dérive 5) : la vitesse brute change partout, la vitesse relative non.
    4. **Identités du tracker** : en phase exponentielle, les nageuses mal suivies donnent des trajectoires courtes ou
       mélangées, et le rappel baisse.
    5. **Déclin** : 40 % de cellules collées. La médiane des voisines n'est plus « le liquide » : elle est tirée vers
       les cellules immobiles, et même la règle B perd en précision. Il faudrait calculer la référence seulement sur
       les cellules classées passives (α ≈ 1, notebook 04).
    """),
        tutor.question(mo, "Vaut-il mieux prendre comme référence la médiane des voisines ou les objets immobiles "
                           "(cristaux, cellules collées) ?",
                       "Elles répondent à deux questions différentes. Les **objets immobiles** ne bougent qu'avec la "
                       "lame : soustraire leur mouvement retire la **dérive** et laisse voir le flux du liquide plus "
                       "la nage. La **médiane des voisines libres** retire dérive **et** flux : il ne reste que ce que "
                       "la cellule fait par elle-même. Pour la mobilité active, c'est la seconde. Les cristaux de CaCO₃ "
                       "(17–46 µm, immobiles) restent une excellente mesure de la dérive seule."),
        tutor.question(mo, "Pourquoi un témoin non mobile vaut-il mieux qu'une règle statistique pour *S. pasteurii* ?",
                       "Parce qu'aucune phase ne garantit que les cellules actives sont rares : en phase exponentielle, "
                       "elles seraient majoritaires (hypothèse du rapport). Le témoin fixe ce qu'est un mouvement "
                       "**passif dans ta géométrie** (épaisseur de liquide, bruit, cadence), indépendamment de la "
                       "proportion de nageuses. Il calibre aussi le plancher de bruit du notebook 04."),
        tutor.question(mo, "Un agrégat formé par le CaCl₂ (floculation) traverse le champ porté par le flux. Est-il "
                           "détecté ?",
                       "Non s'il est **libre** : il suit le liquide comme ses voisines, sa vitesse relative est faible. "
                       "Oui s'il est **accroché** à la lamelle : il ne suit plus le flux. Le mouvement seul ne dit donc "
                       "pas « agrégat » : il faut la morphologie (taille, forme) pour le reconnaître. D'où l'idée du "
                       "rapport de combiner descripteurs de mouvement et de forme."),
    ])
    return


if __name__ == "__main__":
    app.run()
