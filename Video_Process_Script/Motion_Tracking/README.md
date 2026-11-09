# Piste mouvement : tracking des bactéries et détection des outliers

## Idée
Détecter toutes les bactéries sur chaque frame, les **suivre dans le temps** (une trajectoire par bactérie), puis
repérer les **outliers** : les bactéries qui bougent différemment de leur environnement (voisines, fond, dérive
de la lame).

Pourquoi le tracking plutôt que l'optical flow : l'optical flow mesure la déformation globale des pixels, sans
notion d'objet. Il est difficile à interpréter ici :
- l'éclairage saute ;
- les bactéries floues créent des halos qui bougent ;
- le fond est en grande partie uniforme, donc peu de texture à suivre.

L'optical flow peut tout de même servir à estimer le **mouvement de référence de l'environnement** (dérive
globale), auquel on compare chaque trajectoire.

```
détection par frame → association entre frames (tracks) → mouvement de référence local
                    → vitesse relative de chaque bactérie → score d'outlier
```

## Notebooks de concepts (marimo)
Pour comprendre les concepts **sans écrire de code**, avec les ordres de grandeur de *S. pasteurii* tirés du
**Research Report** du projet. Ces valeurs sont surtout des proxys (*B. subtilis*, *E. coli*) ou des calculs : aucune
donnée cinématique n'est publiée pour l'espèce.

Le simulateur produit des cellules **passives** (D ≈ 0,25 µm²/s), des **nageuses** (15–25 µm/s, un tumble toutes les
~3 s) et des cellules **collées / chaînes / agrégats**, avec la dérive de la lame et le flux d'évaporation. Tout est
en µm et en secondes, aux cadences de tes vidéos (Expo 0,26 s, Stationary 0,35 s entre images).

Des préréglages par phase de croissance (lag → déclin) traduisent le tableau du rapport, **inféré** des *Bacillus*,
en proportions **illustratives**.

La taille du pixel est estimée à ~0,16 µm (×40) et ~0,06 µm (×100), à partir d'une cellule de 3 µm. Elle reste à
calibrer.

Le code est masqué. Lancement en mode application, depuis la racine du projet :
```bash
.venv/bin/marimo run Video_Process_Script/Motion_Tracking/00_contexte_s_pasteurii.py
```

| Notebook | Concepts, appliqués à *S. pasteurii* |
|---|---|
| `00_contexte_s_pasteurii.py` | Ce qu'on sait (et ne sait pas) du mouvement de l'espèce ; phases attendues ; tes vidéos en µm |
| `01_echantillonnage.py` | Déplacement par image contre voisine ; règle de la ½ cellule ; quelle cadence pour suivre la nage |
| `02_association.py` | Glouton contre hongrois ; un gating pour les passives et un pour les nageuses ; divisions et chaînes |
| `03_kalman.py` | Prédire la nageuse pendant ses runs ; tumbles et cercles près de la lamelle ; limite du premier lien |
| `04_trajectoires_msd.py` | MSD et exposant α : actif contre passif ; plancher de bruit ; la fraction mobile comme signal de phase |
| `05_reference_outliers.py` | Dérive et flux d'évaporation ; z robuste contre témoin non mobile ; pourquoi les « outliers » ne sont pas rares |
| `06_evaluation.py` | Précision, rappel, identités ; sortie du plan focal à ×100 ; images dupliquées ; ablation |

Modules communs (pas besoin de les ouvrir) :
- `motion_sim.py` : le simulateur, les préréglages et les sources ;
- `reference.py` : les méthodes ;
- `tutor.py` : l'affichage des questions.

## Compétences à travailler en amont
Classées dans l'ordre où elles serviront. Pour chacune : ce qu'il faut savoir, et un exercice court.

### 1. Échantillonnage temporel et ordres de grandeur
- Frame rate : Expo 3.8 fps, Stationary 2.9 fps, soit 0.26 s et 0.35 s entre deux frames. Le tracking n'est fiable
  que si le **déplacement entre deux frames est petit devant la distance entre bactéries voisines**.
- Unités : vitesse en px/frame → px/s. La conversion en µm/s demande la taille du pixel, qui est encore inconnue
  (pas d'image de calibration).
- *Exercice :* sur 20 frames consécutives, mesurer à la main le déplacement de 10 bactéries et la distance à leur
  plus proche voisine. Le rapport des deux dit si le tracking est faisable tel quel.

### 2. Détection fiable frame par frame (tracking-by-detection)
- Le tracking hérite des erreurs de détection : une bactérie manquée sur une frame coupe sa trajectoire.
- Sources possibles : le détecteur YOLO (avec sa classe de netteté), ou les formes de la pipeline v2.
- Notions : seuil de confiance, NMS (suppression des doublons), centroïde vs boîte, détections à la sortie du plan
  focal.

### 3. Association entre frames (cœur du tracking)
- **Problème d'affectation** : coût = distance entre détections, résolu par l'algorithme hongrois
  (`scipy.optimize.linear_sum_assignment`).
- **Gating** : distance maximale au-delà de laquelle on refuse un lien.
- **Gap closing** : relier une trajectoire interrompue sur 1 ou 2 frames.
- **Split / merge** : division cellulaire, bactéries qui se croisent ou se collent.
- Référence : le framework LAP de Jaqaman *et al.* (2008), utilisé par TrackMate.
- *Exercice :* coder un tracker plus-proche-voisin + hongrois sur des points synthétiques, puis compter les
  échanges d'identité quand la densité augmente.

### 4. Modèle de mouvement et filtrage
- **Filtre de Kalman** (position + vitesse) : prédire où sera la bactérie à la frame suivante, pour mieux associer
  quand elle bouge vite.
- Mouvement brownien vs mouvement dirigé (nage). Une bactérie immobile ou collée « bouge » quand même par le bruit
  de détection.
- Notions : état, bruit de processus, bruit de mesure.

### 5. Outils de tracking existants
À comparer avant de coder soi-même :
- **trackpy** (Crocker-Grier, très utilisé pour les particules en microscopie) ;
- **laptrack** (LAP en Python, proche de TrackMate) ;
- **btrack** (bayésien, gère les divisions) ;
- **TrackMate** (Fiji, référence en bio-imagerie) ;
- les trackers intégrés à ultralytics (**ByteTrack**, **BoT-SORT**), pensés pour la vidéo classique.

Pour visualiser : la couche **Tracks** de napari.

### 6. Analyse de trajectoires
- Déplacement, vitesse instantanée et moyenne, angle de rotation, persistance.
- **MSD (déplacement quadratique moyen)** en fonction du délai τ : MSD ∝ τ^α. α ≈ 1 = diffusion (brownien),
  α ≈ 2 = mouvement dirigé, α < 1 = confiné ou collé.
- Coefficient de diffusion ; rapport entre la distance nette et la longueur du chemin parcouru.
- *Exercice :* simuler des trajectoires browniennes et dirigées, calculer leur MSD et retrouver α.

### 7. Mouvement de référence de l'environnement
- **Dérive globale** (lame, platine) : recalage par corrélation de phase (`skimage.registration.phase_cross_correlation`).
- **Écoulement local du fluide** : vitesse médiane des voisines (k plus proches voisins, rayon fixe ou triangulation
  de Delaunay).
- **Optical flow** (Farnebäck, TV-L1 avec `skimage.registration.optical_flow_tvl1`) comme carte du mouvement de fond,
  en connaissant ses limites (éclairage, halos, zones sans texture).
- Vitesse **relative** = vitesse de la bactérie − vitesse de référence locale.

### 8. Détection d'outliers
- **Statistiques robustes** : médiane, MAD, z-score robuste (déjà utilisés pour le bruit dans la pipeline v2).
- Méthodes multivariées : distance de Mahalanobis, Isolation Forest, LOF (Local Outlier Factor), DBSCAN
  (scikit-learn).
- Piège principal : une **erreur de tracking** (échange d'identité, saut) ressemble à un outlier. Il faut filtrer les
  trajectoires trop courtes ou incohérentes avant de chercher les outliers.

### 9. Évaluation sans vérité terrain
- Vérifier à la main un échantillon de trajectoires dans napari.
- **Données synthétiques** : simuler des bactéries avec un mouvement connu (quelques outliers injectés) pour mesurer
  le tracker et le détecteur d'outliers.
- Métriques de tracking : MOTA, IDF1, HOTA, et celles du Cell Tracking Challenge.

### 10. Biologie du mouvement bactérien
- *S. pasteurii* est décrite comme mobile (flagelles) : à vérifier dans la littérature, ainsi que l'évolution de la
  mobilité entre les phases de croissance.
- Nage « run-and-tumble », adhésion aux surfaces, mouvement brownien des cellules non mobiles.
- Lien avec le projet : la mobilité pourrait être un signal de la phase de croissance, en plus de la forme.

### 11. Outils transverses
- **pandas** pour les tables de trajectoires (une ligne = une détection : frame, id, x, y, mesures).
- Lecture vidéo par blocs (les vidéos ne tiennent pas en mémoire), vectorisation NumPy.
- Visualisation : trajectoires colorées par vitesse, histogrammes de vitesse, cartes de vitesse relative.

## Contraintes connues des données
- Sauts de lampe → normaliser chaque frame (déjà fait dans la pipeline v2).
- Changement d'objectif dans Stationary (frames 360–367) → couper les trajectoires à cet endroit.
- Bactéries qui sortent du plan focal → les trajectoires s'interrompent (la classe de netteté de YOLO aide à le voir).
- Pas de calibration µm/px → vitesses en px/s tant qu'il n'y a pas d'image de calibration.
