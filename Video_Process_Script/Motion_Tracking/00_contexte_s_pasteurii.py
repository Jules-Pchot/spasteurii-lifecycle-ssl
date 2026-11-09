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
    import tutor
    from motion_sim import KIND_COLORS, KIND_LABELS, KINDS, PARAMETER_SOURCES, PHASES, VIDEOS, phase_config, simulate
    return (KIND_COLORS, KIND_LABELS, KINDS, PARAMETER_SOURCES, PHASES, VIDEOS, mo, np, pd, phase_config, plt,
            simulate, tutor)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 00 — Ce que l'on sait du mouvement de *S. pasteurii*

    *D'après le Research Report du projet.*

    ***S. pasteurii*** est un **bâtonnet** (0,5–1,2 × 1,3–4,0 µm, ~3 µm observé), Gram+, décrit comme **mobile par
    des flagelles péritriches** : des flagelles répartis sur toute la surface. Son génome porte 39 gènes
    flagellaires. Ils sont exprimés davantage en carence d'azote et **réprimés pendant la minéralisation** (MICP).
    Un CaCl₂ élevé (0,5 M) restreint aussi la mobilité et fait floculer les cellules.

    **Aucune vitesse de nage, durée de « run », fréquence de tumble ni coefficient de diffusion n'a été publié pour
    cette espèce.** Les seuls repères sont :

    - des **proxys** : *B. subtilis* et *E. coli*, qui nagent à 15–25 µm/s et font un tumble toutes les ~3 s ;
    - un **calcul physique** : le mouvement brownien d'un bâtonnet de 1 × 3 µm, D ≈ 0,2–0,3 µm²/s.

    D'après le rapport, le signal de phase le plus prometteur n'est donc **pas la vitesse absolue**. C'est la
    **fraction de cellules activement mobiles** et la **nature du mouvement** : balistique (nage) contre brownien.
    Tous les notebooks suivants utilisent ces ordres de grandeur.
    """)
    return


@app.cell(hide_code=True)
def _(PARAMETER_SOURCES, mo, pd):
    mo.vstack([mo.md("## Les chiffres disponibles et leur niveau de preuve"),
               mo.ui.table(pd.DataFrame(PARAMETER_SOURCES, columns=["Paramètre", "Valeur", "Espèce", "Statut", "Source"]),
                           selection=None, show_column_summaries=False)])
    return


@app.cell(hide_code=True)
def _(PHASES, mo):
    phase = mo.ui.dropdown(list(PHASES), value="Exponentielle", label="Phase de croissance")
    mo.vstack([mo.md(r"""
    ## Ce qu'on s'attend à voir à chaque phase

    Le tableau du rapport est **en grande partie inféré** de la biologie des *Bacillus* : c'est une hypothèse à
    tester sur tes vidéos, pas un fait. Pour les simulations, j'ai traduit ces tendances en proportions
    **illustratives** (densité, part de nageuses, vitesse). Elles servent à comprendre les concepts, pas à prédire
    tes mesures.
    """), phase])
    return (phase,)


@app.cell(hide_code=True)
def _(KIND_COLORS, KIND_LABELS, KINDS, PHASES, mo, phase, phase_config, plt, simulate):
    _info = PHASES[phase.value]
    _cfg = phase_config(phase.value, n_frames=20)
    _truth, _ = simulate(_cfg)
    _fig, _ax = plt.subplots(figsize=(9, 6))
    for _tid, _g in _truth[_truth.inside].groupby("true_id"):
        _kind = _g.kind.iloc[0]
        _ax.plot(_g.x, _g.y, color=KIND_COLORS[_kind], lw=1.2)
        _ax.plot(_g.x.iloc[-1], _g.y.iloc[-1], "o", ms=3, color=KIND_COLORS[_kind])
    for _kind in KINDS:
        _ax.plot([], [], color=KIND_COLORS[_kind], label=KIND_LABELS[_kind])
    _ax.set(xlim=(0, _cfg.width), ylim=(_cfg.height, 0), aspect="equal", xlabel="µm", ylabel="µm",
            title=f"{phase.value} : 5 s de mouvement simulé (le point marque la fin)")
    _ax.legend(fontsize=8, loc="upper right")
    _fig.tight_layout()
    _c = _info["config"]
    mo.vstack([
        mo.callout(mo.md(f"**Cinématique attendue :** {_info['kinematics']}  \n"
                         f"**Morphologie / scène :** {_info['scene']}  \n"
                         f"**Niveau de preuve :** {_info['evidence']}"), kind="info"),
        mo.md(f"*Paramètres illustratifs de la simulation :* {_c['density']:g} cellules / 1000 µm², "
              f"{_c['swimmer_fraction']:.0%} de nageuses à {_c['swim_speed']:g} µm/s, "
              f"{_c['stuck_fraction']:.0%} de cellules collées, chaînes ou agrégats."),
        _fig])
    return


@app.cell(hide_code=True)
def _(VIDEOS, mo, pd):
    _rows = [{"Vidéo": name, "Images / s": f"{1 / v['dt']:.1f}", "Temps entre images": f"{v['dt']:.2f} s",
              "Taille du pixel (estimée)": f"{v['um_per_px']:.3f} µm", "Champ": f"{v['field'][0]:.0f} × {v['field'][1]:.0f} µm",
              "Cellule de 3 µm": f"{3 / v['um_per_px']:.0f} px"} for name, v in VIDEOS.items()]
    mo.vstack([mo.md(r"""
    ## Tes deux vidéos, en unités physiques

    La taille du pixel n'est pas calibrée. Je l'estime à partir de la longueur des bactéries mesurée sur tes vidéos
    (19 px en ×40, 50 px en ×100) et d'une longueur de 3 µm. C'est cohérent avec un capteur à pixels de ~6,5 µm.
    **À confirmer avec un micromètre objet** (première recommandation du rapport).
    """), mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False)])
    return


@app.cell(hide_code=True)
def _(mo, tutor):
    mo.vstack([
        tutor.question(mo, "Pourquoi le rapport conseille-t-il de mesurer la fraction de cellules mobiles plutôt que "
                           "leur vitesse ?",
                       "Parce que la vitesse dépend fortement du milieu (urée, ammonium, pH, CaCl₂, oxygène) et "
                       "qu'aucune valeur de référence n'existe pour l'espèce. La **proportion** de cellules qui nagent, "
                       "et la nature balistique ou brownienne du mouvement, sont des différences **qualitatives** "
                       "entre phases, plus robustes que des µm/s. Exemple : 60 % de nageuses en phase exponentielle "
                       "contre presque aucune en déclin, dans les préréglages."),
        tutor.question(mo, "Entre lame et lamelle, pourquoi la nage peut-elle s'arrêter en quelques minutes ?",
                       "*S. pasteurii* est **aérobie** et ne pousse pas sans oxygène. Au centre d'une préparation "
                       "scellée, l'O₂ s'épuise vite et la nage ralentit ou s'arrête. C'est inféré d'autres bactéries "
                       "aérobies flagellées, non mesuré pour *S. pasteurii*. D'où le conseil de filmer **moins de "
                       "2 min après le montage**, toujours avec le même délai."),
        tutor.question(mo, "Les cristaux de CaCO₃ font 17 à 46 µm et ne bougent pas. Est-ce un problème ou une aide ?",
                       "Les deux. Ce sont des **faux positifs** possibles pour la détection : objets sombres, 10 fois "
                       "plus gros que les cellules. Mais comme ils sont **immobiles**, ils donnent une référence "
                       "parfaite pour mesurer la dérive de la lame (notebook 05)."),
    ])
    return


if __name__ == "__main__":
    app.run()
