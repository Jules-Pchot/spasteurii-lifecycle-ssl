# Microscopy AI — Fichier de contexte

Document de référence pour le projet Claude. Rassemble le contexte du projet d'ingénieur *Microscopy AI*, l'état de la présentation (anglais, ton recherche) et les questions/réponses préparées pour la soutenance du **mercredi**.

---

## 1. Le projet en une phrase

Un CNN léger classe la phase de croissance de la bactérie *Sporosarcina pasteurii* à partir d'une photo au microscope, pour déclencher au bon moment un procédé de bio-cimentation (MICP) sur le terrain — sans connexion internet, sur smartphone.

## 2. Équipe et rôles

| Nom | Rôle |
|---|---|
| Jules Pichot | Chef de projet ; squelette de l'application ; approche technique définie avec Kylian |
| Kylian Tarde | AI Engineer — a codé le notebook (preprocessing, entraînement, évaluation du CNN) |
| Issam Azlouk | Software engineering (à confirmer le détail du rôle) |
| Rihem Boutemine | Literature review (à confirmer) |
| Simon Lucas | Data collection (à confirmer) |

Contexte : projet d'école en équipe sur 3 mois ; à l'issue d'une première phase d'évaluation, les 6 meilleurs projets ont été sélectionnés.

## 3. Biologie — MICP

- *Sporosarcina pasteurii* précipite de la calcite (CaCO₃) à partir de calcium et d'urée, via la production d'**uréase**.
- Mécanisme : uréase → hydrolyse de l'urée → hausse du pH → précipitation de calcite → la calcite lie les grains de sable entre eux → solidification du sol.
- **La phase optimale de cimentation est la fin de la phase exponentielle** (pas la phase stationnaire) : c'est le pic de production d'uréase. Passé ce pic, la production décline — ce n'est pas un arrêt net mais une décroissance après le pic.
- 4 phases biologiques réelles : Lag → Exponentielle → Stationnaire → Mort/Déclin. Le modèle actuel n'en classe que 3 (pas de classe Lag dans le dataset).
- Alternative de mesure envisageable : densité optique (OD600) au spectrophotomètre — écartée car non portable sur le terrain, contrairement à une photo au microscope + smartphone. C'est l'argument à donner si on demande "pourquoi une image plutôt qu'une mesure de densité".

## 4. Cahier des charges (« project requirements »)

| Volet | Exigence | Matériel/contrainte |
|---|---|---|
| Données | ≥ 2000 images | Microscope photonique entrée de gamme + caméra SWIFTCAM SC1003 |
| Modèle | Précision ≥ 95% | Léger, inférence faible latence sur smartphone |
| Application | Photo prise ou envoyée | Cross-OS, fonctionnement **hors ligne** |

Point non résolu : l'inférence tourne-t-elle réellement on-device, ou sur un serveur ? Si c'est côté serveur, il y a une contradiction avec l'exigence "offline" — à trancher avant toute question sur le déploiement.

## 5. Acquisition et labellisation des données

- 25 vidéos, 4K, microscope optique ×100, caméra SWIFTCAM SC1003, logiciel Swift Imaging 3.0.
- Échantillon : culture bactérienne prélevée en surface, montée entre lame et lamelle. Trois récipients sources : deux boîtes de Petri (même culture) + un tube avec eau et bactérie.
- Collecte étalée sur ~2 mois, à raison d'1 jour/semaine (~8-9 sessions) — explique la rareté de la classe Mort (293 images brutes) : le temps et l'accès au matériel étaient les principales contraintes.
- **Méthode de labellisation** : un superviseur du projet a déterminé la phase de chaque vidéo source ; corroboré par 2 doctorants. **Reste à préciser : sur quel critère mesurable (ex. OD600, courbe de croissance de référence, temps écoulé) cette détermination a été faite.** C'est la première question attendue d'un public recherche — à clarifier avec le superviseur avant mercredi.
- Le label de la vidéo est propagé à toutes ses frames extraites (pas d'annotation image par image).
- Nettoyage post-extraction : suppression des frames sans bactérie visible à l'écran, pour éviter un biais d'apprentissage sur des frames vides (ex. si le modèle apprenait à associer "frame vide" à une classe surreprésentée en frames vides).
  - Point de vigilance non résolu : si les frames vides étaient plus fréquentes dans certaines phases que d'autres, ce nettoyage a pu lui-même contribuer au déséquilibre 87/11/2%, en plus de le corriger.
- Généralisation : le projet ne vise pas une généralisation inter-laboratoires (usage prévu sur le terrain, pas en labo). Il reste néanmoins vrai que **caméra et microscope sont restés constants sur toute la collecte** — la diversité testée est celle des récipients (2 boîtes de Petri + 1 tube), pas celle du matériel d'acquisition. À ne pas confondre avec "pas besoin de généraliser du tout", qui est une réponse trop large pour un contexte de déploiement terrain.
- Résolution source des images : **incohérence non résolue** entre le README (1280×720) et les fichiers réels du dossier `Img/Stationary/` (1920×1080 constatés). À vérifier avant de citer un chiffre en présentation.

### Répartition brute des classes
- Exponentielle : 10 200 (87%)
- Stationnaire : 1 242 (11%)
- Mort : 293 (2%)

## 6. Pipeline de prétraitement (ordre d'exécution)

1. **Réduction qualité vidéo** (HandBrake).
2. **Extraction des frames** (script Python) → suppression des doublons → conversion en niveaux de gris → 11 735 images brutes.
3. **Sélection manuelle** : retrait des frames sans bactérie visible.
4. **Labellisation** : label = phase de la vidéo source.
5. **Split train/test** : 80/20, seed = 42, actuellement **au niveau frame** (voir limite ci-dessous).
6. **Augmentation** (sur le train set uniquement, après le split) : flips horizontal/vertical, classes minoritaires seulement.
   - Mort : 234 → 702
   - Stationnaire : 993 → 2017
   - Exponentielle : non augmentée
7. **Redimensionnement** : `transforms.Resize((256, 256))` — resize, **pas de crop**. Étire chaque axe indépendamment → déformation du ratio d'aspect (16:9 → 1:1), sans perte de contenu aux bords. Réduction du volume de données ≈ 85% (mesure empirique disque ; le calcul théorique en pixels donne plutôt ≈ 93%).
8. **Normalisation** : `Normalize(mean, std)` calculés empiriquement sur le train set. Dernier run : **mean ≈ 0.656, std ≈ 0.146**.

**Tenseur final envoyé au modèle** : grayscale → resize 256×256 (déformé) → normalisation (μ, σ du train) → image 1×256×256.

**Point de vigilance si le split change** (ex. passage à un split par vidéo) : recalculer μ/σ uniquement sur les nouvelles frames de train, et revoir les cibles d'augmentation si la distribution de classes change.

## 7. Modèle — `BacteriaCNN`

- 3 blocs Conv2d → BatchNorm2d → ReLU → MaxPool2d (32 → 64 → 128 canaux).
- `AdaptiveAvgPool2d((1,1))` → `Flatten` → `Linear(128, 3)`.
- **93 507 paramètres** — volontairement compact pour l'inférence mobile.
- `CrossEntropyLoss` pondérée par classe : `[1.0, 4, 11.6]`.
  - **Non vérifié** : à quel ratio ces poids correspondent-ils exactement — ratio sur les comptes **bruts** (10200/1242, 10200/293) ou sur les comptes **après augmentation** (10200/2017, 10200/702) ? À calculer pour pouvoir justifier le choix en soutenance.
- Adam, lr = 1e-3, batch size 64, early stopping (patience = 5).
- Choix CNN "maison" plutôt que transfer learning (MobileNet) : justifié par (1) le nombre de paramètres bien plus faible que MobileNet (~1M+), important pour l'inférence mobile ; (2) l'écart de domaine entre ImageNet (photos naturelles couleur) et les images de microscopie en niveaux de gris, qui réduit l'intérêt du pré-entraînement pour un petit dataset déséquilibré.

## 8. Run officiel et résultats

- **Run retenu : early stopping à l'epoch 5** (et non epoch 11 — l'ancienne version des slides utilisait l'autre run, à ne plus utiliser).
- Meilleure val accuracy dès l'epoch 1 ≈ 98%.
- Accuracy test globale 98% ; precision/recall/F1 pondérés 98%.
- Référence de comparaison : toujours prédire "Exponentielle" donnerait ≈ 87% d'accuracy sur la distribution brute — donc le gain réel du modèle par rapport à une prédiction triviale est plus modeste que "98%" ne le laisse penser.
- **Limite majeure non résolue** : split actuel au niveau frame, pas au niveau vidéo. Des frames quasi-identiques (même vidéo) peuvent se retrouver à la fois en train et en test → risque de fuite de données → 98% probablement optimiste. Un ré-entraînement avec split par vidéo est prévu en début de stage, pas avant la présentation de mercredi.
- Argument à avoir en tête si on objecte "mais l'utilisateur final envoie une photo, pas une vidéo" : une nouvelle photo sur le terrain ressemble bien plus à une **vidéo jamais vue** (nouvelle session d'acquisition) qu'à une frame isolée d'une vidéo déjà partiellement vue à l'entraînement — ce qui va dans le sens d'un split par vidéo, pas contre.
- Métriques par classe (precision/recall/F1 pour Exponentielle, Stationnaire, Mort séparément) : **non encore extraites** — nécessaires pour répondre sérieusement à "que cache la moyenne pondérée à 98%".

## 9. Application mobile

- Front end : React Native — UI/UX, portabilité cross-OS, assistance à la capture, zoom.
- Back end : Python — traitement d'image, appel au classifieur, historique des phases, seuil de confiance ajustable.
- 5 pages : Photo upload, GO, NO-GO, Historique, Réglages.
- GO = Exponentielle ; NO-GO = Stationnaire ou Mort.
- Latence mesurée ≈ 2 s sur les téléphones de test ; non mesurée sur du matériel plus ancien (hypothèse conservée : 2 s).
- **Non résolu** : l'inférence tourne-t-elle on-device ou sur un serveur ? Conditionne la cohérence avec l'exigence "sans connexion internet".

## 10. Limites assumées (à présenter, pas à cacher)

- Split frame-level plutôt que video-level → risque de fuite de données, 98% probablement optimisé à la hausse.
- Généralisation non testée à d'autres microscopes/caméras.
- Classes grossières : le modèle ne distingue pas début/fin de phase exponentielle, alors que le pic d'uréase se situe en fin de cette phase.
- Pas de données pour la phase Lag ; seulement 293 images brutes pour Mort.
- Le resize déforme le ratio d'aspect.

## 11. Objectifs futurs du stage (EPFL Center for Imaging, Dr. Daniel Sage)

Rupture avec l'approche actuelle (CNN supervisé, 3 classes) vers une approche **self-supervised** (SimCLR ou Masked Autoencoder), sur 4 phases biologiques (lag/log/stationnaire/déclin), avec séquences time-lapse en plus des snapshots.

- Ré-évaluer le CNN actuel avec un focus sur la segmentation des phases, on passerait soit à de petite vidéo de 3scd avec les moments important de chaque phase soit des images fixe de la bactérie
- Collecter les 4 phases, avec si possible des séquences time-lapse.
- Entraîner un modèle SSL avec une **temporal ordering loss** (pénalise un ordre incohérent des embeddings le long d'une séquence time-lapse).
- Analyser l'espace latent (UMAP, t-SNE ; silhouette score, Davies-Bouldin index).
- Définir un critère de décision dans l'espace latent pour le stade optimal de MICP.
- Benchmarker contre le CNN supervisé actuel.

**Argument de choix SimCLR vs Masked Autoencoder — non tranché** : SimCLR a besoin d'une notion claire de "deux vues du même objet", ce que les séquences time-lapse pourraient fournir naturellement ; un Masked Autoencoder n'a pas besoin de cette paire mais demande généralement plus de données pour bien pré-entraîner. À trancher en fonction de la taille réelle du futur dataset.

**Validation sans labels — piste non résolue** : envisager de garder un petit sous-ensemble labellisé, non utilisé pour l'entraînement, uniquement pour vérifier que les clusters de l'espace latent correspondent réellement aux phases biologiques.