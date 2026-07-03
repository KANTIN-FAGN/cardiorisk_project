# Trame oral final — CardioRisk Explorer (15 min + 5 min questions)

> Fil rouge du storytelling : partir du chiffre choc mondial, zoomer jusqu'au patient individuel,
> finir sur la prédiction personnalisée. Du global → à l'individu → à l'action.

---

## 0. Accroche & problématique — 1 min 30

- **Chiffre choc** : 18 millions de morts par an, 1ʳᵉ cause de mortalité mondiale (OMS).
- Problématique : *Quels sont les principaux facteurs associés aux maladies cardiovasculaires,
  à l'échelle individuelle et mondiale, et le Machine Learning peut-il prédire le risque ?*
- Annonce du plan : données → analyses → modèles → démo → limites.
- 4 hypothèses (H1 comportements, H2 données cliniques > lifestyle, H3 disparités géographiques,
  H4 le ML prédit correctement) — dire qu'on y répondra en conclusion.

*Grille : introduction de la démarche.*

## 1. Les données — 2 min

- **3 sources ouvertes** (critère "Acquérir des données") :
  - Cardio Train (Kaggle) : 68 606 patients, examens médicaux réels (Russie), équilibré 50/50
  - BRFSS 2015 (Kaggle) : 229 781 répondants, enquête téléphonique CDC (USA), déséquilibré 10/90
  - OMS / Our World in Data : mortalité par pays, 194 pays, 1950–2023
- Insister sur la **complémentarité** : mesures cliniques vs déclaratif lifestyle → 2 modèles.

## 2. Préparation & nettoyage — 1 min 30

- Doublons supprimés, valeurs aberrantes filtrées : tensions impossibles (ap_hi ≤ ap_lo,
  valeurs hors bornes physiologiques), IMC hors [10-60].
- Encodage des catégorielles, conversion de l'âge (jours → années).
- **Feature engineering** : `bp_category` (seuils AHA), `bmi_category` (seuils OMS),
  `unhealthy_days` (santé mentale + physique cumulée).
- Chiffre à citer : ~1 400 lignes supprimées sur Cardio Train (~2%).

*Grille : "Préparer et nettoyer" (3) + création de variables.*

## 3. Exploration & analyses — 2 min 30 (DÉMO app, pages Analyse mondiale + patients)

- **Carte mondiale** (30 s) : disparités fortes → valide H3. Mortalité élevée Europe de l'Est /
  Asie centrale ; scatter PIB : gradient richesse-santé net.
- **Analyse patients** (2 min) : montrer 2-3 graphiques clés :
  - Risque par âge : 24% chez les 30-39 ans → 74% chez les 60-69 (Cardio Train)
  - Cholestérol / hypertension : associations les plus fortes
  - Dire : **chaque association est validée par un test statistique** (Chi² pour les
    catégorielles, Mann-Whitney pour les continues, p < 0.001) → valide H1
- Point d'honnêteté qui marque : l'alcool excessif ressort "protecteur" dans BRFSS —
  artefact déclaratif connu, on sait lire ses données de façon critique.

*Grille : "Explorer et analyser" (3) + "Visualiser" (3).*

## 4. Modélisation — 3 min (DÉMO page Modélisation)

- Pipeline : split stratifié 80/20 → standardisation → **4 modèles comparés**
  (Logistic Regression, Random Forest, XGBoost, LightGBM).
- **Gestion du déséquilibre** BRFSS : class_weight / scale_pos_weight — sinon un modèle
  "tout le monde est sain" ferait 90% d'accuracy.
- **Critère de choix : ROC-AUC** (indépendant du seuil, robuste au déséquilibre).
- **Validation croisée 5-fold stratifiée** : écart-types ±0.002 → performances stables,
  pas un coup de chance du découpage.
- Résultats à connaître par cœur :
  - Cardio Train → Random Forest, ROC-AUC 0.807 (CV 0.797 ± 0.002)
  - BRFSS → LightGBM, ROC-AUC 0.836 (CV 0.834 ± 0.002)
- Feature importance : tension artérielle domine le modèle clinique ; IMC, âge et gradient
  socio-économique dominent le lifestyle.
- **Réponse à H2 (moment fort)** : contre-intuitif, le modèle lifestyle fait mieux que le
  clinique — pas parce que le déclaratif est "meilleur", mais dataset 3× plus grand et tâche
  différente (dépistage populationnel vs diagnostic hospitalier). H2 non confirmée telle quelle.

*Grille : "Appliquer un modèle ML et l'évaluer" (4) — le critère le plus lourd avec le 6.*

## 5. Prédiction & recommandations — 2 min 30 (DÉMO page Prédiction)

- Cliquer un **profil d'exemple** (🔴 Malade) → le formulaire se remplit → Prédire.
- Montrer : score de risque, facteurs de risque identifiés, **recommandations** associées
  (activité physique, arrêt tabac, consultation médicale si risque élevé).
- Expliquer le choix des profils d'exemple : sélectionnés par le modèle lui-même
  (risque min / ~50% / max) pour illustrer toute la gamme.
- Rappeler l'avertissement médical : outil pédagogique, pas un diagnostic. → valide H4 (avec nuance)

*Grille : "Prédire et recommander" (4) + "Interface interactive" (2).*

## 6. Limites & améliorations — 1 min

- Limites : données déclaratives (sous-déclaration), biais géographique (Russie/USA),
  F1 bas sur BRFSS assumé (compromis dépistage : rappel > précision).
- Améliorations : validation sur cohorte externe, enrichissement socio-économique des
  données mondiales, déploiement en ligne.

*Grille : maturité + "Documenter" (2) — mentionner README, notebooks, dépôt