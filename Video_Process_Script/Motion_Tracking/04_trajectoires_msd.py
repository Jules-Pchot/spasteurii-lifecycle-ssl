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
    from motion_sim import KIND_COLORS, KIND_LABELS, KINDS, PHASES, VIDEOS, phase_config, simulate
    return KINDS, KIND_COLORS, KIND_LABELS, PHASES, VIDEOS, mo, np, phase_config, plt, ref, simulate, tutor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 — Nager ou diffuser : le MSD et l'exposant α

    C'est l'outil central du rapport. Ne **jamais** classer une cellule par sa vitesse instantanée : le bruit de
    localisation gonfle les vitesses aux petits pas de temps. On regarde plutôt **comment le déplacement grandit avec
    le temps**.

    Le **déplacement quadratique moyen** (MSD) mesure de combien une cellule s'éloigne en moyenne après un délai
    $\tau$. Sa forme, $\mathrm{MSD} \propto \tau^{\alpha}$, dit le type de mouvement :

    - **cellule passive** (brownienne) : $\mathrm{MSD} = 4 D \tau$, donc $\alpha \approx 1$. Avec D ≈ 0,25 µm²/s,
      elle s'éloigne d'environ 1 µm en 1 s ;
    - **nageuse** pendant un run : $\mathrm{MSD} = v^2 \tau^2$, donc $\alpha \approx 2$. À 20 µm/s, elle s'éloigne
      de 20 µm en 1 s ;
    - **cellule collée**, spore ou agrégat fixé : seul le bruit de mesure reste, donc $\alpha \approx 0$.

    L'écart d'un **ordre de grandeur** entre nage et diffusion rend la séparation robuste, à condition de regarder α
    et pas seulement une vitesse.
    """)
    return


@app.cell(hide_code=True)
def _(mo, ref):
    msd_fn, fit_fn = ref.msd, ref.fit_alpha
    mo.md(r"""
    ## Comment on le lit

    Pour un délai de 3 images, on prend toutes les paires de positions séparées de 3 images sur la trajectoire, on
    mesure la distance entre elles, on l'élève au carré et on fait la moyenne. On recommence pour 1, 2, 4… images.
    En **échelle log-log**, une loi $\tau^{\alpha}$ devient une **droite de pente α**.

    **Pourquoi le carré ?** Une cellule passive s'éloigne dans toutes les directions : en moyenne, son déplacement vaut
    zéro. Son déplacement au carré, lui, grandit régulièrement.
    """)
    return fit_fn, msd_fn


@app.cell(hide_code=True)
def _(PHASES, VIDEOS, mo):
    phase = mo.ui.dropdown(list(PHASES), value="Exponentielle", label="Phase (préréglage)")
    video = mo.ui.dropdown(list(VIDEOS), value="Expo (×40)", label="Cadence de la vidéo")
    track_length = mo.ui.slider(8, 120, step=4, value=24, label="longueur des trajectoires (images)", show_value=True)
    noise = mo.ui.slider(0.02, 0.5, step=0.02, value=0.1, label="bruit de localisation (µm)", show_value=True)
    threshold = mo.ui.slider(1.0, 1.9, step=0.05, value=1.4, label="seuil : active si α >", show_value=True)
    mo.vstack([mo.md("### Réglages\n\nÀ ×100, la profondeur de champ est inférieure au micron : les nageuses sortent "
                     "vite du plan et les trajectoires sont **courtes**. La longueur se règle ci-dessous."),
               phase, video, track_length, noise, threshold])
    return noise, phase, threshold, track_length, video


@app.cell(hide_code=True)
def _(VIDEOS, np, phase_config, simulate, track_length, video):
    def noisy_tracks(phase_name, noise_um, seed=0):
        """Ground-truth tracks of a phase preset, cut to the chosen length, with localization noise (µm)."""
        cfg = phase_config(phase_name, n_frames=track_length.value, dt=VIDEOS[video.value]["dt"], seed=seed)
        truth, _ = simulate(cfg)
        truth[["x", "y"]] += np.random.default_rng(seed + 7).normal(0, noise_um, (len(truth), 2))
        return cfg, truth

    max_lag = max(3, track_length.value // 4)
    return max_lag, noisy_tracks


@app.cell(hide_code=True)
def _(fit_fn, max_lag, msd_fn, noise, noisy_tracks, np, phase, threshold):
    def alphas_of(phase_name, seed=0):
        """(cfg, truth, table of (true_id, kind, alpha, msd curve)) for one phase preset."""
        cfg, truth = noisy_tracks(phase_name, noise.value, seed)
        lag_s = np.arange(1, max_lag + 1) * cfg.dt
        rows = []
        for tid, g in truth.groupby("true_id"):
            m = msd_fn(g[["x", "y"]].to_numpy(), max_lag)
            rows.append((tid, g.kind.iloc[0], fit_fn(lag_s, m)[0], m))
        return cfg, truth, lag_s, rows

    cfg, truth, lag_seconds, rows = alphas_of(phase.value)
    active_estimate = np.mean([a > threshold.value for _, _, a, _ in rows])
    return active_estimate, alphas_of, cfg, lag_seconds, rows, truth


@app.cell(hide_code=True)
def _(KINDS, KIND_COLORS, KIND_LABELS, active_estimate, cfg, lag_seconds, mo, np, plt, rows, threshold, truth):
    _fig, (_tracks, _curves, _hist) = plt.subplots(1, 3, figsize=(15, 4.4))
    for _tid, _g in list(truth.groupby("true_id"))[:40]:
        _xy = _g[["x", "y"]].to_numpy()
        _tracks.plot(*(_xy - _xy[0]).T, color=KIND_COLORS[_g.kind.iloc[0]], lw=0.9)
    _tracks.set(aspect="equal", xlabel="µm", ylabel="µm", title="Trajectoires (départ ramené à 0)")
    for _tid, _kind, _a, _m in rows:
        _curves.loglog(lag_seconds, _m, color=KIND_COLORS[_kind], alpha=0.25, lw=0.8)
    _curves.loglog(lag_seconds, 4 * cfg.diffusion * lag_seconds, "k--", lw=1.2, label="4 D τ (passive, D = 0,25)")
    _curves.loglog(lag_seconds, (cfg.swim_speed * lag_seconds) ** 2, "k:", lw=1.2,
                   label=f"(v τ)² (nage à {cfg.swim_speed:g} µm/s)")
    _curves.set(xlabel="délai τ (s)", ylabel="MSD (µm²)", title="MSD de chaque trajectoire")
    _curves.legend(fontsize=7)
    for _kind in KINDS:
        _a = [a for _, k, a, _ in rows if k == _kind]
        if _a:
            _hist.hist(_a, np.linspace(-0.5, 2.5, 31), alpha=0.6, color=KIND_COLORS[_kind],
                       label=f"{KIND_LABELS[_kind]} (médiane {np.median(_a):.2f})")
    _hist.axvline(threshold.value, color="black", lw=1)
    _hist.set(xlabel="α", ylabel="cellules", title="Exposant α par type")
    _hist.legend(fontsize=7)
    _fig.tight_layout()
    _true = cfg.swimmer_fraction
    mo.vstack([_fig, mo.callout(mo.md(f"**Fraction mobile estimée (α > {threshold.value:g}) : {active_estimate:.0%}** "
                                      f"— vraie part de nageuses dans ce préréglage : {_true:.0%}"), kind="info")])
    return


@app.cell(hide_code=True)
def _(PHASES, alphas_of, mo, np, plt, threshold):
    _names, _est, _true = [], [], []
    for _p in PHASES:
        _cfg, _, _, _rows = alphas_of(_p, seed=1)
        _names.append(_p.replace(" (fenêtre MICP)", "\n(MICP)"))
        _est.append(np.mean([a > threshold.value for _, _, a, _ in _rows]))
        _true.append(_cfg.swimmer_fraction)
    _fig, _ax = plt.subplots(figsize=(9, 3.6))
    _pos = np.arange(len(_names))
    _ax.bar(_pos - 0.18, _true, 0.36, color="#c8c8c8", label="vraie part de nageuses (préréglage)")
    _ax.bar(_pos + 0.18, _est, 0.36, color="#d95926", label=f"fraction mobile estimée (α > {threshold.value:g})")
    _ax.set(xticks=_pos, xticklabels=_names, ylabel="fraction", ylim=(0, 1),
            title="Le signal de phase proposé par le rapport : la fraction mobile")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(r"""
    ## La fraction mobile, phase par phase

    Le rapport propose la **fraction de cellules actives** comme descripteur de phase : maximale en phase
    exponentielle, en baisse en fin de phase log, faible en stationnaire et en déclin. C'est une hypothèse inférée
    des *Bacillus*. Voici ce que l'on retrouverait avec la méthode α sur les préréglages, à la cadence et avec la
    longueur de trajectoire choisies plus haut.
    """), _fig])
    return


@app.cell(hide_code=True)
def _(mo, tutor):
    mo.vstack([
        mo.md(r"""
    *À essayer :*

    - raccourcis les trajectoires à 8 images : les α s'étalent, et la fraction estimée devient moins fiable. C'est le
      problème des nageuses qui sortent du plan à ×100 ;
    - monte le bruit à 0,5 µm : les cellules passives voient leur α baisser (le bruit écrase les petits déplacements),
      et la séparation reste nette tant que le seuil est autour de 1,4 ;
    - passe en **Fin exponentielle** : les nageuses à 10 µm/s gardent un α élevé, mais leur MSD est 4 fois plus bas.
    """),
        tutor.question(mo, "Une spore collée devrait avoir un MSD nul. Pourquoi sa courbe est-elle plate mais pas à zéro ?",
                       "À cause du **bruit de localisation** : sa position détectée tremble de ~0,1 µm autour de la "
                       "vraie. Ce tremblement ne grandit pas avec le temps, d'où une courbe plate (α ≈ 0) à une hauteur "
                       "≈ 4 × bruit² (0,04 µm² ici). C'est le **plancher de bruit** : tout mouvement plus petit est "
                       "invisible. Le rapport conseille de le mesurer sur un **témoin non mobile** (cellules fixées ou "
                       "tuées par la chaleur)."),
        tutor.question(mo, "Pourquoi une nageuse avec un tumble toutes les ~3 s a-t-elle α ≈ 2 aux temps courts, mais "
                           "α → 1 aux temps longs ?",
                       "Sur moins de ~3 s, elle avance en ligne droite : balistique, α ≈ 2. Au-delà, ses runs partent "
                       "dans des directions aléatoires : à grande échelle, c'est une **marche aléatoire** (diffusion "
                       "« active », 1000 fois plus rapide que la diffusion thermique). Avec tes cadences et des "
                       "trajectoires de quelques secondes, on reste surtout dans le régime α ≈ 2."),
        tutor.question(mo, "Pourquoi le rapport préfère-t-il la fraction mobile à la vitesse moyenne ?",
                       "La vitesse moyenne mélange deux choses : **combien** de cellules nagent et **à quelle vitesse**. "
                       "Une baisse peut venir de moins de nageuses ou de nageuses plus lentes. La fraction mobile "
                       "sépare ces deux effets, et α n'a pas besoin de calibration µm/px pour classer actif / passif "
                       "(c'est une pente, sans unité)."),
    ])
    return


if __name__ == "__main__":
    app.run()
